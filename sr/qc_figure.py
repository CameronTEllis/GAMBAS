#!/usr/bin/env python3
"""
qc_figure.py
============
Visual QC for the simulated 2 mm volumes. Produces one PNG per subject with:

  row 1  mid-axial slice: 1 mm truth | simulated 2 mm (native grid) | 2 mm on the
         1 mm grid | absolute difference
  row 2  the same for a mid-sagittal slice
  row 3  a zoomed crop through cortex, where the resolution loss is visible
  row 4  radial power spectra of the truth and the simulation, with the 2 mm
         Nyquist marked. The simulation's curve should fall off a cliff exactly
         at that line -- if it does not, the forward model is not doing what it
         claims.

The spectrum panel is the one to actually look at. Two volumes can look similar
in a slice view and have completely different frequency content.

Usage:
    python -m sr.qc_figure --sim_dir /data/sr_sim --out_dir /data/sr_sim/qc_png
    python -m sr.qc_figure --sim_dir /data/sr_sim --out_dir /tmp/qc --n 5
"""

import argparse
import fnmatch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sr.naming import age_of, subgroup_of
from sr.sr_metrics import (radial_power_spectrum, axis_power_spectrum, psnr,
                           ssim3d)


def load(path):
    im = sitk.ReadImage(path, sitk.sitkFloat32)
    a = np.transpose(sitk.GetArrayFromImage(im), (2, 1, 0))
    return a, im


def norm(a, ref=None):
    r = ref if ref is not None else a
    lo, hi = np.percentile(r, 1), np.percentile(r, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def make_figure(name, hr_path, low_path, up_path, out_png, target_spacing=2.0,
                source_spacing=1.0):
    hr, hr_itk = load(hr_path)
    up, _ = load(up_path)
    low, low_itk = (load(low_path) if low_path and os.path.exists(low_path)
                    else (None, None))

    hrn = norm(hr)
    upn = norm(up, ref=hr)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.08)

    cx, cy, cz = [s // 2 for s in hr.shape]

    def show(ax, img, title, vmin=0, vmax=1, cmap='gray'):
        ax.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # --- row 1: axial ---------------------------------------------------------
    show(fig.add_subplot(gs[0, 0]), hrn[:, :, cz], '1 mm truth (axial)')
    if low is not None:
        ln = norm(low, ref=hr)
        show(fig.add_subplot(gs[0, 1]), ln[:, :, low.shape[2] // 2],
             '%g mm native grid %s' % (target_spacing, low.shape))
    else:
        fig.add_subplot(gs[0, 1]).axis('off')
    show(fig.add_subplot(gs[0, 2]), upn[:, :, cz],
         '%g mm on 1 mm grid (net input)' % target_spacing)
    d = np.abs(hrn - upn)
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(np.rot90(d[:, :, cz]), cmap='inferno', vmin=0,
                   vmax=np.percentile(d, 99.5), interpolation='nearest')
    ax.set_title('|difference|', fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # --- row 2: sagittal ------------------------------------------------------
    show(fig.add_subplot(gs[1, 0]), hrn[cx, :, :], '1 mm truth (sagittal)')
    if low is not None:
        show(fig.add_subplot(gs[1, 1]), ln[low.shape[0] // 2, :, :],
             '%g mm native grid' % target_spacing)
    else:
        fig.add_subplot(gs[1, 1]).axis('off')
    show(fig.add_subplot(gs[1, 2]), upn[cx, :, :], '%g mm on 1 mm grid' % target_spacing)
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(np.rot90(d[cx, :, :]), cmap='inferno', vmin=0,
              vmax=np.percentile(d, 99.5), interpolation='nearest')
    ax.set_title('|difference|', fontsize=9)
    ax.axis('off')

    # --- row 3: cortical zoom -------------------------------------------------
    # Pick the axial slice and quadrant with the most edge energy: that is where
    # resolution loss is most visible.
    gz = np.abs(np.diff(hrn, axis=0)).sum(axis=(0, 1))
    z = int(np.argmax(gz))
    h, w = hr.shape[0], hr.shape[1]
    y0, y1 = int(0.10 * h), int(0.45 * h)
    x0, x1 = int(0.25 * w), int(0.60 * w)
    show(fig.add_subplot(gs[2, 0]), hrn[y0:y1, x0:x1, z], '1 mm truth (zoom)')
    show(fig.add_subplot(gs[2, 1]), upn[y0:y1, x0:x1, z],
         '%g mm on 1 mm grid (zoom)' % target_spacing)
    ax = fig.add_subplot(gs[2, 2])
    prof_y = hr.shape[1] // 2
    ax.plot(hrn[y0:y1, prof_y, z], label='1 mm truth', lw=1.2)
    ax.plot(upn[y0:y1, prof_y, z], label='%g mm sim' % target_spacing, lw=1.2)
    ax.set_title('intensity profile through cortex', fontsize=9)
    ax.set_xlabel('voxel', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    txt = ['PSNR (sim vs truth)  %.2f dB' % psnr(upn, hrn),
           'SSIM (sim vs truth)  %.4f' % ssim3d(upn, hrn),
           'corr                 %.4f' % np.corrcoef(hrn.ravel(), upn.ravel())[0, 1],
           '',
           'truth shape  %s' % (hr.shape,),
           'input shape  %s' % (up.shape,)]
    if low is not None:
        txt.append('2 mm shape   %s' % (low.shape,))
        txt.append('2 mm spacing %s' % (tuple(round(s, 3) for s in low_itk.GetSpacing()),))
    ax.text(0.0, 0.95, '\n'.join(txt), va='top', family='monospace', fontsize=9)

    # --- row 4: spectra -------------------------------------------------------
    nyq = 0.5 * source_spacing / target_spacing

    # Left: per-axis spectrum. This is the panel that verifies the forward model.
    # A rectangular k-space truncation cuts each axis independently, so along one
    # axis the simulated curve must fall off a cliff exactly at the target
    # Nyquist. If it does not, the simulation is wrong.
    ax = fig.add_subplot(gs[3, :2])
    for axis, style in zip(range(3), ('-', '--', ':')):
        f, p_hr = axis_power_spectrum(hrn, axis=axis)
        _, p_up = axis_power_spectrum(upn, axis=axis)
        ax.semilogy(f, np.maximum(p_hr, 1e-20), style, color='C0', lw=1.2,
                    label='1 mm truth' if axis == 0 else None)
        ax.semilogy(f, np.maximum(p_up, 1e-20), style, color='C1', lw=1.2,
                    label='%g mm simulated' % target_spacing if axis == 0 else None)
    ax.axvline(nyq, color='r', ls='--', lw=1.2,
               label='%g mm Nyquist = %.3f cyc/voxel' % (target_spacing, nyq))
    ax.set_xlabel('spatial frequency (cycles per 1 mm voxel)', fontsize=9)
    ax.set_ylabel('mean power along axis', fontsize=9)
    ax.set_title('Per-axis spectrum (solid/dashed/dotted = x/y/z).\n'
                 'The orange curves MUST drop sharply at the red line.',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    ax.tick_params(labelsize=8)

    # Right: radial spectrum, for reference. Note that it does NOT show a clean
    # cliff, and that is expected -- see the note in the title.
    ax = fig.add_subplot(gs[3, 2:])
    f, p_hr = radial_power_spectrum(hrn, nbins=96)
    _, p_up = radial_power_spectrum(upn, nbins=96)
    ax.semilogy(f, np.maximum(p_hr, 1e-20), color='C0', label='1 mm truth', lw=1.5)
    ax.semilogy(f, np.maximum(p_up, 1e-20), color='C1',
                label='%g mm simulated' % target_spacing, lw=1.5)
    ax.axvline(nyq, color='r', ls='--', lw=1.2, label='%g mm Nyquist' % target_spacing)
    ax.axvline(nyq * np.sqrt(3), color='r', ls=':', lw=1.0,
               label='cube corner (Nyquist x sqrt3)')
    ax.set_xlabel('|k| (cycles per 1 mm voxel)', fontsize=9)
    ax.set_ylabel('radial mean power', fontsize=9)
    ax.set_title('Radial spectrum. No cliff expected: a rectangular k-space\n'
                 'window retains energy out to the cube corner at |k|=Nyquist*sqrt3.',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    ax.tick_params(labelsize=8)

    fig.suptitle(name, fontsize=12)
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sim_dir', required=True, help='simulate_lowres.py output dir')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--method', default=None,
                   help="Simulation method tag, e.g. 'kspace'. Selects "
                        "lowres-<method>/ and lowres-<method>-native/ (the "
                        "default --layout method of simulate_lowres.py). Omit for "
                        "the flat layout.")
    p.add_argument('--lr_dir', default=None, help='Override the on-HR-grid dir')
    p.add_argument('--native_dir', default=None, help='Override the native 2 mm dir')
    p.add_argument('--hr_dir', default=None, help='Override the HR target dir')
    p.add_argument('--n', type=int, default=5,
                   help='Cap on how many figures to make. 0 = all. Applied AFTER '
                        'the filters below.')
    p.add_argument('--include', nargs='+', default=None,
                   help='Only these volumes. Each value is matched against the '
                        'filename stem: a bare string matches as a substring '
                        '(--include 12345), and shell wildcards work '
                        "(--include '12345_*_t1w'). Repeatable.")
    p.add_argument('--weighting', default=None,
                   help="Only this weighting, e.g. t1w or t2w (case-insensitive)")
    p.add_argument('--age_range', type=float, nargs=2, default=None,
                   help='Only volumes whose age field falls in [LO, HI] months, '
                        'e.g. --age_range 0 6 for the youngest')
    p.add_argument('--extremes', action='store_true',
                   help='Just the youngest and oldest volume by age field. Handy '
                        'for checking FOV and normalisation across a wide '
                        'developmental range.')
    p.add_argument('--name_schema', default='id,session,age,weighting',
                   help='Field order, used by --weighting/--age_range/--extremes')
    p.add_argument('--target_spacing', type=float, default=2.0)
    p.add_argument('--source_spacing', type=float, default=1.0)
    args = p.parse_args(argv)

    if args.method:
        up_default = 'lowres-%s' % args.method
        low_default = 'lowres-%s-native' % args.method
    else:
        up_default, low_default = 'lowres_on_hr_grid', 'lowres_native'
    hr_dir = args.hr_dir or os.path.join(args.sim_dir, 'hr')
    up_dir = args.lr_dir or os.path.join(args.sim_dir, up_default)
    low_dir = args.native_dir or os.path.join(args.sim_dir, low_default)

    if not os.path.isdir(up_dir):
        available = sorted(d for d in os.listdir(args.sim_dir)
                           if d.startswith('lowres')) if os.path.isdir(args.sim_dir) else []
        sys.exit('%s missing. Available lowres dirs: %s -- pass --method or --lr_dir'
                 % (up_dir, available))
    if not os.path.isdir(hr_dir):
        sys.exit('%s missing -- rerun the simulator with --copy_hr' % hr_dir)

    names = sorted(f for f in os.listdir(up_dir) if not f.startswith('.'))
    total = len(names)
    stem_of = lambda f: f.replace('.nii.gz', '').replace('.nii', '')

    # ---- filters, applied before --n so the cap never hides your selection ---
    if args.include:
        keep = []
        for f in names:
            s = stem_of(f)
            for pat in args.include:
                # A pattern with no wildcard is treated as a substring, which is
                # what you want for "just show me participant 12345".
                hit = (fnmatch.fnmatch(s, pat)
                       if any(c in pat for c in '*?[') else pat in s)
                if hit:
                    keep.append(f)
                    break
        names = keep

    if args.weighting:
        want = args.weighting.lower()
        names = [f for f in names
                 if subgroup_of(stem_of(f), schema=args.name_schema) == want]

    if args.age_range or args.extremes:
        aged = []
        for f in names:
            _, num = age_of(stem_of(f), None, args.name_schema)
            if num is not None:
                aged.append((num, f))
        if not aged:
            sys.exit('--age_range/--extremes need a parseable age field; none '
                     'found with --name_schema %s' % args.name_schema)
        if args.age_range:
            lo, hi = sorted(args.age_range)
            aged = [(a, f) for a, f in aged if lo <= a <= hi]
        aged.sort()
        if args.extremes:
            aged = [aged[0]] + ([aged[-1]] if len(aged) > 1 else [])
        names = [f for _, f in aged]

    if not names:
        sys.exit('no volumes matched. %d available in %s; check --include / '
                 '--weighting / --age_range.' % (total, up_dir))

    n_matched = len(names)
    if args.n and not args.extremes:
        names = names[:args.n]
    print('%d of %d volume(s) matched; making %d figure(s)'
          % (n_matched, total, len(names)))
    os.makedirs(args.out_dir, exist_ok=True)

    for fn in names:
        hr_p = os.path.join(hr_dir, fn)
        if not os.path.exists(hr_p):
            print('skip %s (no HR)' % fn)
            continue
        stem = fn.replace('.nii.gz', '').replace('.nii', '')
        out = os.path.join(args.out_dir, stem + '_qc.png')
        make_figure(stem, hr_p, os.path.join(low_dir, fn),
                    os.path.join(up_dir, fn), out,
                    args.target_spacing, args.source_spacing)
        print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
