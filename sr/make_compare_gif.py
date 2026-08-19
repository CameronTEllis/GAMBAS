#!/usr/bin/env python3
"""Make a looping GIF that flickers between the 2 mm input, the 1 mm target, and
the model's super-resolution prediction, so you can SEE what the model restored.

The frame order is  low-res -> original -> prediction -> original -> (loop).
Putting the original between the other two lets your eye do the comparison: the
low-res -> original step shows what the 2 mm acquisition lost, and the
prediction -> original step shows how much of it the model put back. If the
prediction frame looks much more like the original than the low-res frame does,
the model is working.

Every frame shows three orthogonal views at once (sagittal, coronal, axial) at the
same voxel positions, and all frames share ONE intensity window (taken from the
target's foreground) so the brightness does not flicker -- only the detail does.
Display uses nearest-neighbour interpolation on purpose: bilinear would blur all
three equally and hide the very sharpness difference you are trying to judge.

USAGE
-----
    # Easy mode -- just the held-out index (from per_volume.csv). Paths are built
    # from $DATASET_DIR / $EVAL_DIR, so `source sr/cluster/config.sh` first:
    python -m sr.make_compare_gif 3
    python -m sr.make_compare_gif 3 --fold fold0 --mode cv     # a CV fold

    # Explicit mode -- give the three files yourself:
    python -m sr.make_compare_gif --lowres INPUT.nii.gz --target TRUTH.nii.gz \
                                  --pred PRED.nii.gz --out compare.gif

    # options: --ms 600 (per-frame), --panel_px 500, --slices X Y Z (override)
"""
import argparse
import os
import sys

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit('nibabel is required: pip install nibabel')
try:
    from PIL import Image
except ImportError:
    sys.exit('Pillow is required: pip install Pillow')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402


def load(path):
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.float64)
    spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    return arr, spacing


def window(arr, lo=1.0, hi=99.0):
    """Robust display window from foreground percentiles (skull/CSF saturate)."""
    fg = arr[arr > arr.max() * 0.02]
    if fg.size == 0:
        fg = arr.ravel()
    return float(np.percentile(fg, lo)), float(np.percentile(fg, hi))


def normalize01(arr, lo=1.0, hi=99.0):
    """Map arr to [0, 1] by its OWN foreground percentiles.

    Each volume is normalised independently because they are saved on different
    numeric scales -- the target/input keep the raw MRI intensities (hundreds to
    thousands) while the prediction is written as p01*255 (0..255). A single shared
    window would then make the prediction render near-black. Per-volume percentile
    normalisation puts all three on the same display range so only the sharpness
    flickers, not the brightness.
    """
    lo_v, hi_v = window(arr, lo, hi)
    if hi_v <= lo_v:
        hi_v = lo_v + 1e-6
    return np.clip((arr - lo_v) / (hi_v - lo_v), 0.0, 1.0)


def center_of_mass(arr):
    """Foreground centre of mass, for informative (not edge) slice positions."""
    m = arr > (arr.max() * 0.1)
    if not m.any():
        return [n // 2 for n in arr.shape]
    idx = np.nonzero(m)
    return [int(round(c.mean())) for c in idx]


def render_frame(arr, spacing, slices, vmin, vmax, label, panel_px, flip='v'):
    """One montage: sagittal | coronal | axial, titled with `label`. Returns RGB.

    `flip` reorients the displayed slices: 'v' flips vertically (the default, so
    superior is up for these volumes), 'h' horizontally, 'vh' both, 'none' as-is.
    """
    # (fixed axis, the two in-plane axes) for each orthogonal view
    views = [(0, 'Sagittal'), (1, 'Coronal'), (2, 'Axial')]
    dpi = 100
    fig, axes = plt.subplots(1, 3, figsize=(3 * panel_px / dpi, panel_px / dpi),
                             dpi=dpi, facecolor='black')
    for ax, (fixed, name) in zip(axes, views):
        sl = np.take(arr, slices[fixed], axis=fixed)          # 2D
        inplane = [a for a in range(3) if a != fixed]
        sa, sb = spacing[inplane[0]], spacing[inplane[1]]
        # Show as (rows=second in-plane axis, cols=first) with physical aspect.
        disp = sl.T
        if flip in ('v', 'vh'):
            disp = disp[::-1, :]          # flip the vertical (display-row) axis
        if flip in ('h', 'vh'):
            disp = disp[:, ::-1]          # flip the horizontal (display-col) axis
        ax.imshow(disp, cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
                  interpolation='nearest',
                  extent=[0, sl.shape[0] * sa, 0, sl.shape[1] * sb], aspect='equal')
        ax.set_title(name, color='0.8', fontsize=10)
        ax.axis('off')
    fig.suptitle(label, color='white', fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(rgb)


def resolve_by_index(index, mode, fold, test_dir, pred_dir):
    """Build (lowres, target, pred, out) paths for held-out volume `index`.

    Base dirs default to the standard cluster layout, derived from $DATASET_DIR
    and $EVAL_DIR (as set by config.sh):
        test_dir = ${DATASET_DIR}_<mode>/<fold>/test   (has images/ and labels/)
        pred_dir = ${EVAL_DIR}_<mode>/<fold>/predictions
    Override either with --test_dir/--pred_dir if your paths differ.
    """
    if test_dir is None:
        dd = os.environ.get('DATASET_DIR')
        if not dd:
            sys.exit('index mode needs $DATASET_DIR (run `source sr/cluster/'
                     'config.sh` first) or an explicit --test_dir.')
        test_dir = os.path.join('%s_%s' % (dd, mode), fold, 'test')
    if pred_dir is None:
        ed = os.environ.get('EVAL_DIR')
        if not ed:
            sys.exit('index mode needs $EVAL_DIR (run `source sr/cluster/config.sh` '
                     'first) or an explicit --pred_dir.')
        pred_dir = os.path.join('%s_%s' % (ed, mode), fold, 'predictions')

    lowres = os.path.join(test_dir, 'images', '%d.nii.gz' % index)
    target = os.path.join(test_dir, 'labels', '%d.nii.gz' % index)
    pred = os.path.join(pred_dir, '%d_pred.nii.gz' % index)
    for tag, pth in (('input', lowres), ('target', target), ('prediction', pred)):
        if not os.path.exists(pth):
            sys.exit('index %d: %s not found at %s\n(check the index against '
                     'per_volume.csv, and --mode/--fold/--test_dir/--pred_dir).'
                     % (index, tag, pth))
    return lowres, target, pred, 'compare_%s_%s_%d.gif' % (mode, fold, index)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('index', nargs='?', type=int, default=None,
                   help='Held-out volume index (from per_volume.csv). If given, the '
                        'three files are located automatically; --lowres/--target/'
                        '--pred are then optional.')
    p.add_argument('--mode', default='dev',
                   help='Split mode for index paths: dev or cv (default dev).')
    p.add_argument('--fold', default='folddev',
                   help='Fold tag for index paths: folddev, fold0, ... (default '
                        'folddev).')
    p.add_argument('--test_dir', default=None,
                   help='Override the dir holding images/ and labels/ (index mode).')
    p.add_argument('--pred_dir', default=None,
                   help='Override the dir holding <i>_pred.nii.gz (index mode).')
    p.add_argument('--lowres', help='2 mm input (network input)')
    p.add_argument('--target', help='1 mm truth (network target)')
    p.add_argument('--pred', help='model prediction')
    p.add_argument('--out', default=None, help='output .gif')
    p.add_argument('--ms', type=int, default=600, help='ms per frame (default 600)')
    p.add_argument('--panel_px', type=int, default=500,
                   help='pixels per orthogonal view (default 500; raise for a '
                        'higher-resolution gif)')
    p.add_argument('--slices', type=int, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='voxel slice indices; default = target centre of mass')
    p.add_argument('--clip', type=float, nargs=2, default=[1.0, 99.0],
                   metavar=('LO', 'HI'), help='display percentiles (default 1 99)')
    p.add_argument('--flip', choices=['v', 'h', 'vh', 'none'], default='v',
                   help="display flip: 'v' vertical (default), 'h' horizontal, "
                        "'vh' both, 'none' as-is.")
    a = p.parse_args(argv)

    # Index mode: derive the three paths (and a default output name).
    if a.index is not None:
        a.lowres, a.target, a.pred, derived_out = resolve_by_index(
            a.index, a.mode, a.fold, a.test_dir, a.pred_dir)
        if a.out is None:
            a.out = derived_out
    elif not (a.lowres and a.target and a.pred):
        p.error('give an index (e.g. `make_compare_gif 3`) OR all of '
                '--lowres/--target/--pred.')
    if a.out is None:
        a.out = 'compare.gif'

    low, sp_l = load(a.lowres)
    tgt, sp_t = load(a.target)
    prd, sp_p = load(a.pred)
    if not (low.shape == tgt.shape == prd.shape):
        sys.exit('shapes differ: lowres %s, target %s, pred %s -- all three must be '
                 'on the same grid.' % (low.shape, tgt.shape, prd.shape))

    slices = a.slices if a.slices else center_of_mass(tgt)
    slices = [int(np.clip(s, 0, n - 1)) for s, n in zip(slices, tgt.shape)]

    # Normalise each volume by its own foreground percentiles -> [0, 1], so the
    # different on-disk scales (raw MRI vs 0..255 prediction) don't matter.
    low = normalize01(low, a.clip[0], a.clip[1])
    tgt = normalize01(tgt, a.clip[0], a.clip[1])
    prd = normalize01(prd, a.clip[0], a.clip[1])
    vmin, vmax = 0.0, 1.0
    print('slices (x,y,z) = %s   (each volume normalised to [0,1] by its own '
          'foreground p%.0f-p%.0f)' % (slices, a.clip[0], a.clip[1]))

    # low-res -> original -> prediction -> original -> (loop)
    sequence = [
        (low, sp_l, 'LOW-RES (2 mm input)'),
        (tgt, sp_t, 'ORIGINAL (1 mm truth)'),
        (prd, sp_p, 'PREDICTION (super-res)'),
        (tgt, sp_t, 'ORIGINAL (1 mm truth)'),
    ]
    frames = [render_frame(arr, sp, slices, vmin, vmax, label, a.panel_px, a.flip)
              for arr, sp, label in sequence]

    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=a.ms, loop=0, disposal=2)
    print('wrote %s  (%d frames, %d ms each, looping)' % (a.out, len(frames), a.ms))


if __name__ == '__main__':
    main()
