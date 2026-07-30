"""Audit how much genuine 1 mm information your "1 mm" volumes actually contain.

WHY THIS EXISTS
---------------
A 2 mm acquisition measures nothing above 0.25 cycles/voxel on a 1 mm grid. So
the entire learnable content of a 2 mm -> 1 mm task is the spectral energy your
targets carry ABOVE that line. If a target has almost none -- because it was
acquired at coarser resolution and resampled to a 1 mm grid, or is heavily
smoothed, or is motion-blurred -- then:

  * the simulated 2 mm input is already nearly identical to it, so the sinc
    baseline PSNR is enormous and the model looks terrible by comparison;
  * the pair contributes almost no training signal, because the residual the
    network is asked to predict is ~0;
  * and it silently inflates your reported baseline, making a genuinely useful
    model look like it is losing.

None of that is visible in a QC figure, because the *simulator* imposes a clean
cliff at 0.25 either way. You have to look at the target's own spectrum.

WHAT IT REPORTS
---------------
hf025      Fraction of spectral power above 0.25 cyc/voxel, i.e. the band a 2 mm
           scan cannot see. This is the headroom. Interpret it relatively:
           compare volumes against each other and against the cohort median.
f99_x/y/z  Per-axis frequency (cyc/voxel) below which 99% of that axis's power
           lies -- an estimate of the effective sampling actually achieved.
           f99 well under 0.25 on an axis means that axis carries no 2 mm-
           unreachable detail at all. Reported per axis on purpose: clinical
           and legacy infant data is often acquired anisotropically (e.g.
           1x1x3 mm) and resampled to isotropic, which leaves one direction
           with nothing to restore while the other two look fine.
eff_mm     0.5 / min(f99) scaled by voxel size -- a rough effective resolution
           in mm along the WORST axis. A volume stored at 1 mm with eff_mm near
           2 was, to a good approximation, never a 1 mm image.
cliff      Power in [0.25, 0.30) divided by power in [0.20, 0.25). Read this
           ONLY as a relative ranking. A real brain has a steeply falling
           (roughly power-law) spectrum, so cliff well below 1 is the NORMAL
           rolloff and is not by itself evidence of prior resampling. Only a
           value far below the rest of the cohort's is suspicious.

CALIBRATION WARNING
-------------------
f99 and eff_mm were validated against synthetic phantoms with near-white
texture, whose spectra are flat. Real anatomy is not: its power falls steeply
with frequency, so 99% of the power sits at low frequency even in a genuinely
1 mm image, and eff_mm therefore reads PESSIMISTIC (too coarse) on real brains.
Do not conclude from eff_mm ~= 2.5 that a volume is not 1 mm data.

Comparisons that survive this bias, and are the ones to actually use:
  * volume against volume WITHIN this cohort (same tissue, same rolloff);
  * axis against axis WITHIN one volume -- the anisotropy check, where the
    spectral shape cancels out;
  * --degrade_check, which measures headroom in dB directly by simulating the
    degradation and scoring the result. That number is self-calibrating and is
    the one that decides whether the task is worth doing on a given volume.

USAGE
-----
    python -m sr.audit_resolution --in_dir /path/to/originals
    python -m sr.audit_resolution --in_dir .../hr --csv audit.csv --flag_below 0.5

Prints a table sorted by hf025 (worst first) and a per-subgroup summary. Nothing
is modified or deleted; use --exclude_list to write the flagged names to a file
you can feed to make_folds.
"""
import argparse
import os
import sys

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sys.exit('SimpleITK is required: pip install SimpleITK')

from sr.naming import DEFAULT_NAME_SCHEMA, strip_ext, subgroup_of, age_of, subject_of
from sr.sr_metrics import hf_energy, brain_mask, psnr


def load(path):
    """Load to a z,y,x -> x,y,z float array plus its voxel spacing in mm."""
    im = sitk.ReadImage(path)
    a = np.transpose(sitk.GetArrayFromImage(im), (2, 1, 0)).astype(np.float64)
    return a, tuple(im.GetSpacing())


def axis_f99(a, axis, frac=0.99):
    """Frequency below which `frac` of this axis's spectral power lies.

    Averaged over the other two axes so it reflects the axis rather than one
    arbitrary line through the volume. DC is dropped: it carries most of the
    power in an MR image and would swamp the cumulative sum.
    """
    n = a.shape[axis]
    A = np.fft.fft(a - a.mean(), axis=axis)
    P = np.abs(A) ** 2
    other = tuple(i for i in range(3) if i != axis)
    P = P.mean(axis=other)
    f = np.fft.fftfreq(n)
    keep = f >= 0
    f, P = f[keep], P[keep]
    if len(f) < 3:
        return float('nan')
    P[0] = 0.0
    tot = P.sum()
    if tot <= 0:
        return float('nan')
    c = np.cumsum(P) / tot
    return float(np.interp(frac, c, f))


def band_power(a, lo, hi):
    A = np.fft.fftn(a - a.mean())
    P = np.abs(A) ** 2
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in a.shape], indexing='ij')
    kr = np.sqrt(sum(g ** 2 for g in grids))
    return float(P[(kr >= lo) & (kr < hi)].sum())


def headroom_db(path, target_spacing, mode='kspace', snr=0.0):
    """Measure the available headroom in dB, self-calibrating.

    Simulates the acquisition with the SAME forward model training uses, then
    scores the sinc-interpolated result against the original inside a foreground
    mask. This is the honest headroom: it is exactly the baseline the network has
    to beat, expressed in the units the network is scored in, and it depends on
    nothing but the volume itself -- no phantom calibration, no assumption about
    spectral shape.

    High PSNR means the simulated 2 mm input is already nearly the target, so
    there is little to learn and little to demonstrate on that volume.
    """
    from sr.kspace import degrade                     # local: keeps import cheap

    a, spacing = load(path)
    factors = tuple(float(t) / float(s) for t, s in zip(target_spacing, spacing))
    lo = degrade(a, factors, mode=mode, snr=snr)
    rng = max(float(np.ptp(a)), 1e-12)
    a01 = (a - a.min()) / rng
    l01 = np.clip((lo - a.min()) / rng, 0, 1)
    m = brain_mask(a01)
    if not m.any():
        return float('nan'), float('nan')
    return psnr(l01[m], a01[m]), float(m.mean())


def audit_one(path, cutoff=0.25, use_mask=True):
    a, spacing = load(path)
    if use_mask:
        # Crop to the head bounding box. Air contributes only noise, and a large
        # empty margin dilutes every spectral fraction by an amount that depends
        # on FOV rather than on image quality -- which would make volumes
        # incomparable across scanners and head sizes.
        # np.ptp(a), not a.ptp(): the ndarray method was removed in NumPy 2.0.
        lo = (a - a.min()) / max(float(np.ptp(a)), 1e-12)
        m = brain_mask(lo)
        if m.any():
            idx = np.where(m)
            sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
            a = a[sl]
    a = a / max(np.abs(a).max(), 1e-12)

    f99 = [axis_f99(a, ax) for ax in range(3)]
    below = band_power(a, 0.20, cutoff)
    above = band_power(a, cutoff, 0.30)
    worst = np.nanmin(f99) if not all(np.isnan(f99)) else float('nan')
    # 0.5 cyc/voxel is Nyquist for the stored grid; scale by that axis's spacing.
    worst_axis = int(np.nanargmin(f99)) if not all(np.isnan(f99)) else 0
    eff_mm = (0.5 / worst * spacing[worst_axis]) if worst and worst > 0 else float('nan')
    return {
        'hf025': hf_energy(a, cutoff),
        'f99_x': f99[0], 'f99_y': f99[1], 'f99_z': f99[2],
        'eff_mm': eff_mm,
        'worst_axis': 'xyz'[worst_axis],
        'cliff': (above / below) if below > 0 else float('nan'),
        'shape': 'x'.join(str(n) for n in a.shape),
        'spacing': '%.2f,%.2f,%.2f' % spacing,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_dir', required=True,
                   help='Directory of the high-resolution target volumes')
    p.add_argument('--cutoff', type=float, default=0.25,
                   help='Nyquist of the simulated acquisition, cyc/voxel (default '
                        '0.25 = 2 mm on a 1 mm grid)')
    p.add_argument('--csv', default='', help='Write the full table here')
    p.add_argument('--flag_below', type=float, default=0.5,
                   help='Flag volumes whose hf025 is below this MULTIPLE of the '
                        'cohort median (default 0.5, i.e. less than half the '
                        'typical headroom). Relative by design: the absolute '
                        'value depends on contrast, age and FOV.')
    p.add_argument('--exclude_list', default='',
                   help='Write flagged stems here, one per line')
    p.add_argument('--name_schema', default=DEFAULT_NAME_SCHEMA)
    p.add_argument('--no_mask', action='store_true',
                   help='Do not crop to the head bounding box first')
    p.add_argument('--limit', type=int, default=0, help='Audit only the first N')
    p.add_argument('--degrade_check', action='store_true',
                   help='Also simulate the degradation and report the masked sinc '
                        'PSNR per volume -- the headroom in dB, self-calibrating '
                        'and directly comparable to the [val] baseline. Slower '
                        '(one FFT round trip per volume) but this is the number '
                        'that decides whether a volume can demonstrate anything.')
    p.add_argument('--target_spacing', type=float, nargs=3, default=[2.0, 2.0, 2.0],
                   help='Simulated acquisition spacing for --degrade_check')
    # 40.0 rather than 39.0: on this cohort (median 37.1 dB, range 34.2-41.6) a
    # 39 dB cut excludes 16/50 volumes including 8 of the 14 t2w, leaving too few
    # t2w to report a subgroup on at all. 40 dB excludes 10/50 and keeps 10 t2w.
    # NOTE this is selection on task DIFFICULTY, measured from the targets alone
    # with no model involved, so it is not circular and can be fixed before any
    # results are seen -- but it must be fixed before, and stated.
    p.add_argument('--flag_db_above', type=float, default=40.0,
                   help='With --degrade_check, flag volumes whose measured sinc '
                        'baseline is at or above this many dB -- there is too '
                        'little left to restore for them to demonstrate anything. '
                        'Takes precedence over --flag_below, which is only a proxy. '
                        'Absolute and in the units the model is scored in, so it '
                        'transfers across cohorts, unlike the hf025 ratio.')
    a = p.parse_args(argv)

    names = sorted(f for f in os.listdir(a.in_dir)
                   if f.endswith(('.nii', '.nii.gz')))
    if a.limit:
        names = names[:a.limit]
    if not names:
        sys.exit('no NIfTI files in %s' % a.in_dir)

    rows = []
    for i, n in enumerate(names):
        stem = strip_ext(n)
        try:
            r = audit_one(os.path.join(a.in_dir, n), a.cutoff, not a.no_mask)
        except Exception as e:                                # noqa: BLE001
            print('  !! %s: %s' % (n, e), file=sys.stderr)
            continue
        r['stem'] = stem
        if a.degrade_check:
            try:
                r['sinc_db'], _ = headroom_db(os.path.join(a.in_dir, n),
                                              a.target_spacing)
            except Exception as e:                            # noqa: BLE001
                print('  !! headroom %s: %s' % (n, e), file=sys.stderr)
                r['sinc_db'] = float('nan')
        try:
            # Keyword args: the second POSITIONAL parameter of both helpers is
            # `regex`, not `schema`.
            r['weighting'] = subgroup_of(stem, schema=a.name_schema) or '?'
            r['age'] = age_of(stem, schema=a.name_schema)
            r['subject'] = subject_of(stem, schema=a.name_schema)
        except Exception:                                    # noqa: BLE001
            r['weighting'], r['age'], r['subject'] = '?', float('nan'), '?'
        rows.append(r)
        print('\r  audited %d/%d' % (i + 1, len(names)), end='', file=sys.stderr)
    print('', file=sys.stderr)

    if not rows:
        sys.exit('nothing audited')

    hf = np.array([r['hf025'] for r in rows])
    med = max(float(np.median(hf)), 1e-12)   # guard the rel.med division
    thresh = a.flag_below * med

    # Flag on MEASURED headroom when we have it. hf025 is a relative spectral
    # fraction and turned out to be a poor proxy: on this cohort it flagged
    # volumes at 37.2 and 37.6 dB while leaving others at 40.4 and 41.6 dB
    # unflagged -- i.e. it excluded volumes with MORE headroom than ones it
    # kept. sinc_dB is the baseline the model is actually scored against, so it
    # is the criterion that matters. hf025 remains the fallback when
    # --degrade_check was not run.
    use_db = a.degrade_check and any(np.isfinite(r.get('sinc_db', np.nan))
                                     for r in rows)
    if use_db:
        for r in rows:
            v = r.get('sinc_db', float('nan'))
            r['flag'] = 'LOW' if (np.isfinite(v) and v >= a.flag_db_above) else ''
    else:
        for r in rows:
            r['flag'] = 'LOW' if r['hf025'] < thresh else ''

    rows.sort(key=lambda r: (-r['sinc_db']) if use_db else r['hf025'])
    if use_db:
        db = np.array([r['sinc_db'] for r in rows if np.isfinite(r['sinc_db'])])
        print('\nflagging on MEASURED headroom: sinc baseline >= %.1f dB'
              % a.flag_db_above)
        print('cohort sinc baseline: median %.2f dB, range %.2f..%.2f  '
              '(hf025 median %.5f, shown but not used for the flag)\n'
              % (np.median(db), db.min(), db.max(), med))
    else:
        print('\ncohort median hf025 = %.5f   flag threshold = %.5f (%.2fx median)'
              % (med, thresh, a.flag_below))
        print('(proxy criterion -- rerun with --degrade_check to flag on the '
              'measured baseline instead)\n')
    dbcol = '%8s' % 'sinc_dB' if a.degrade_check else ''
    hdr = ('%-34s %4s %7s %8s %7s %7s %7s %7s %6s%s %5s'
           % ('stem', 'wt', 'hf025', 'rel.med', 'f99_x', 'f99_y', 'f99_z',
              'eff_mm', 'cliff', dbcol, 'flag'))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        dbv = ('%8.2f' % r['sinc_db']) if a.degrade_check else ''
        print('%-34s %4s %7.5f %8.2f %7.3f %7.3f %7.3f %7.2f %6.2f%s %5s'
              % (r['stem'][:34], r['weighting'], r['hf025'], r['hf025'] / med,
                 r['f99_x'], r['f99_y'], r['f99_z'], r['eff_mm'], r['cliff'],
                 dbv, r['flag']))

    print('\nper-weighting summary')
    for wt in sorted({r['weighting'] for r in rows}):
        sub = [r for r in rows if r['weighting'] == wt]
        v = np.array([r['hf025'] for r in sub])
        extra = ''
        if a.degrade_check:
            db = np.array([r['sinc_db'] for r in sub], dtype=float)
            db = db[np.isfinite(db)]
            if len(db):
                extra = '  sinc %.1f dB (%.1f..%.1f)' % (np.median(db), db.min(), db.max())
        print('  %-5s n=%-3d hf025 median %.5f  range %.5f..%.5f  flagged %d%s'
              % (wt, len(sub), np.median(v), v.min(), v.max(),
                 sum(1 for r in sub if r['flag']), extra))

    # ---- cohort composition -------------------------------------------------
    # Printed here because it constrains the split far more tightly than
    # anything spectral, and it is invisible in a per-volume table. With
    # --group_by subject a longitudinal subject is one indivisible block: if it
    # is a large share of the cohort it cannot be balanced across K folds at all.
    subs = {}
    for r in rows:
        subs.setdefault(r.get('subject', '?'), []).append(r)
    order = sorted(subs.items(), key=lambda kv: -len(kv[1]))
    print('\ncohort composition: %d volumes / %d subjects' % (len(rows), len(subs)))
    for name, rs in order[:5]:
        share = 100.0 * len(rs) / len(rows)
        wts = ','.join('%s=%d' % (w, sum(1 for x in rs if x['weighting'] == w))
                       for w in sorted({x['weighting'] for x in rs}))
        print('  %-16s %2d volumes  %5.1f%%  [%s]' % (name[:16], len(rs), share, wts))
    if len(order) > 5:
        print('  ... %d further subjects with 1-2 volumes'
              % sum(1 for _, rs in order[5:] if len(rs) <= 2))
    top_share = 100.0 * len(order[0][1]) / len(rows)
    if top_share > 20:
        print('\n  !! %s alone is %.0f%% of the cohort. With --group_by subject it is\n'
              '     one indivisible block, so K-fold CV cannot balance it: one fold\n'
              '     gets ~%d volumes and the rest ~%d. Every per-fold number will be\n'
              '     dominated by whether that subject was in train or test, and the\n'
              '     CV standard error will understate uncertainty badly.'
              % (order[0][0], top_share, len(order[0][1]),
                 (len(rows) - len(order[0][1])) // max(1, 4)))

    # hf025 is a RADIAL fraction, so a volume that is fine in-plane and coarse
    # through-plane keeps a healthy hf025 and escapes the flag entirely -- the
    # surviving x/y content masks the dead axis. Verified: a 1x1x3 mm phantom
    # scored 0.72x the cohort median (unflagged) while its f99_z was 0.151
    # against 0.479 in-plane. So check anisotropy separately.
    aniso = []
    for r in rows:
        f = [r['f99_x'], r['f99_y'], r['f99_z']]
        if all(np.isfinite(f)) and min(f) > 0 and max(f) / min(f) >= 1.5:
            aniso.append((r['stem'], 'xyz'[int(np.argmin(f))], max(f) / min(f)))
    if aniso:
        print('\nANISOTROPIC (one axis >=1.5x coarser than another) -- these can pass\n'
              'the hf025 flag while carrying no restorable detail along one axis:')
        for stem, ax, ratio in sorted(aniso, key=lambda t: -t[2]):
            print('  %-34s worst axis %s, %.1fx anisotropy' % (stem[:34], ax, ratio))
        print('Likely acquired anisotropically and resampled to isotropic. An\n'
              'isotropic 2 mm simulation removes almost nothing along the already-\n'
              'coarse axis, so those pairs teach the network to do nothing there.')

    nflag = sum(1 for r in rows if r['flag'])
    spread = hf.max() / max(hf.min(), 1e-12)
    print('\n%d/%d flagged LOW. Max/min headroom spread = %.1fx.' % (nflag, len(rows), spread))
    if spread > 3:
        print('A spread above ~3x means these volumes are not one population.\n'
              'Pooling them makes the mean baseline PSNR nearly meaningless --\n'
              'report per-volume paired deltas, and consider whether the low-\n'
              'headroom volumes belong in the TEST set at all (they cannot\n'
              'demonstrate anything) even if you keep them for training.')

    if a.csv:
        import csv as _csv
        keys = ['stem', 'subject', 'weighting', 'age', 'hf025', 'f99_x', 'f99_y',
                'f99_z', 'eff_mm', 'worst_axis', 'cliff', 'sinc_db', 'shape',
                'spacing', 'flag']
        with open(a.csv, 'w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print('wrote %s' % a.csv)

    if a.exclude_list:
        with open(a.exclude_list, 'w') as fh:
            for r in rows:
                if r['flag']:
                    fh.write(r['stem'] + '\n')
        print('wrote %s (%d stems)' % (a.exclude_list, nflag))


if __name__ == '__main__':
    main()
