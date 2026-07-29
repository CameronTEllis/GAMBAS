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
cliff      Power in [0.25, 0.30) divided by power in [0.20, 0.25). A sharp
           drop (<<1) is the signature of prior resampling or filtering at
           exactly the 2 mm Nyquist. Around 1 means a natural rolloff.

Neither f99 nor eff_mm is a calibrated physical measurement -- noise raises the
apparent high-frequency floor and biases them optimistic, so treat them as a
ranking over the cohort rather than a specification. hf025 and cliff are the
robust signals.

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

from sr.naming import DEFAULT_NAME_SCHEMA, strip_ext, subgroup_of, age_of
from sr.sr_metrics import hf_energy, brain_mask


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
        try:
            # Keyword args: the second POSITIONAL parameter of both helpers is
            # `regex`, not `schema`.
            r['weighting'] = subgroup_of(stem, schema=a.name_schema) or '?'
            r['age'] = age_of(stem, schema=a.name_schema)
        except Exception:                                    # noqa: BLE001
            r['weighting'], r['age'] = '?', float('nan')
        rows.append(r)
        print('\r  audited %d/%d' % (i + 1, len(names)), end='', file=sys.stderr)
    print('', file=sys.stderr)

    if not rows:
        sys.exit('nothing audited')

    hf = np.array([r['hf025'] for r in rows])
    med = max(float(np.median(hf)), 1e-12)   # guard the rel.med division
    thresh = a.flag_below * med
    for r in rows:
        r['flag'] = 'LOW' if r['hf025'] < thresh else ''

    rows.sort(key=lambda r: r['hf025'])
    print('\ncohort median hf025 = %.5f   flag threshold = %.5f (%.2fx median)\n'
          % (med, thresh, a.flag_below))
    hdr = ('%-34s %4s %7s %8s %7s %7s %7s %7s %6s %5s'
           % ('stem', 'wt', 'hf025', 'rel.med', 'f99_x', 'f99_y', 'f99_z',
              'eff_mm', 'cliff', 'flag'))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print('%-34s %4s %7.5f %8.2f %7.3f %7.3f %7.3f %7.2f %6.2f %5s'
              % (r['stem'][:34], r['weighting'], r['hf025'], r['hf025'] / med,
                 r['f99_x'], r['f99_y'], r['f99_z'], r['eff_mm'], r['cliff'],
                 r['flag']))

    print('\nper-weighting summary')
    for wt in sorted({r['weighting'] for r in rows}):
        sub = [r for r in rows if r['weighting'] == wt]
        v = np.array([r['hf025'] for r in sub])
        print('  %-5s n=%-3d hf025 median %.5f  range %.5f..%.5f  flagged %d'
              % (wt, len(sub), np.median(v), v.min(), v.max(),
                 sum(1 for r in sub if r['flag'])))

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
        keys = ['stem', 'weighting', 'age', 'hf025', 'f99_x', 'f99_y', 'f99_z',
                'eff_mm', 'worst_axis', 'cliff', 'shape', 'spacing', 'flag']
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
