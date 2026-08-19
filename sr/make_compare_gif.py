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

USAGE  (`source sr/cluster/config.sh` first so $DATASET_DIR/$EVAL_DIR are set)
-----
    # Batch -- one GIF per participant, across every fold, named by participant:
    python -m sr.make_compare_gif --all --mode cv
    python -m sr.make_compare_gif --all --mode cv --pred_tag adv_latest   # adversarial
    #   -> ${EVAL_DIR}_cv/gifs/<participant>.gif           (L1)
    #      ${EVAL_DIR}_cv/gifs-adv_latest/<participant>_adv_latest.gif  (adv, no clash)

    # Single held-out volume by index (from per_volume.csv):
    python -m sr.make_compare_gif 3 --mode cv --fold fold0
    python -m sr.make_compare_gif 3 --mode cv --fold fold0 --pred_tag adv_latest

    # Explicit -- give the three files yourself:
    python -m sr.make_compare_gif --lowres INPUT.nii.gz --target TRUTH.nii.gz \
                                  --pred PRED.nii.gz --out compare.gif

    # options: --ms 1500 (per-frame), --panel_px 500, --slices X Y Z, --flip
"""
import argparse
import csv
import glob
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


def _pred_subdir(pred_tag):
    """predictions/ or predictions-<tag>/ -- must match evaluate_sr --pred_tag."""
    return 'predictions' + (('-' + pred_tag) if pred_tag else '')


def build_gif(lowres, target, pred, out_path, clip=(1.0, 99.0), flip='v',
              ms=1500, panel_px=500, slices=None, title=None, quiet=False):
    """Render the low-res -> original -> prediction -> original loop for one volume.

    Raises ValueError on a grid mismatch so a batch caller can skip and continue.
    """
    low, sp_l = load(lowres)
    tgt, sp_t = load(target)
    prd, sp_p = load(pred)
    if not (low.shape == tgt.shape == prd.shape):
        raise ValueError('shapes differ: lowres %s, target %s, pred %s -- all three '
                         'must be on the same grid.' % (low.shape, tgt.shape, prd.shape))

    sl = slices if slices else center_of_mass(tgt)
    sl = [int(np.clip(s, 0, n - 1)) for s, n in zip(sl, tgt.shape)]
    # Per-volume normalisation so the different on-disk scales don't matter.
    low = normalize01(low, clip[0], clip[1])
    tgt = normalize01(tgt, clip[0], clip[1])
    prd = normalize01(prd, clip[0], clip[1])

    pre = ('%s\n' % title) if title else ''
    sequence = [
        (low, sp_l, pre + 'LOW-RES (2 mm input)'),
        (tgt, sp_t, pre + 'ORIGINAL (1 mm truth)'),
        (prd, sp_p, pre + 'PREDICTION (super-res)'),
        (tgt, sp_t, pre + 'ORIGINAL (1 mm truth)'),
    ]
    frames = [render_frame(arr, sp, sl, 0.0, 1.0, label, panel_px, flip)
              for arr, sp, label in sequence]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=ms, loop=0, disposal=2)
    if not quiet:
        print('  wrote %s' % out_path)


def resolve_by_index(index, mode, fold, test_dir, pred_dir, pred_tag):
    """Build (lowres, target, pred, out) paths for held-out volume `index`.

    Base dirs default to the standard cluster layout, derived from $DATASET_DIR
    and $EVAL_DIR (as set by config.sh):
        test_dir = ${DATASET_DIR}_<mode>/<fold>/test   (has images/ and labels/)
        pred_dir = ${EVAL_DIR}_<mode>/<fold>/predictions[-<pred_tag>]
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
        pred_dir = os.path.join('%s_%s' % (ed, mode), fold, _pred_subdir(pred_tag))

    lowres = os.path.join(test_dir, 'images', '%d.nii.gz' % index)
    target = os.path.join(test_dir, 'labels', '%d.nii.gz' % index)
    pred = os.path.join(pred_dir, '%d_pred.nii.gz' % index)
    for tag, pth in (('input', lowres), ('target', target), ('prediction', pred)):
        if not os.path.exists(pth):
            sys.exit('index %d: %s not found at %s\n(check the index against '
                     'per_volume.csv, and --mode/--fold/--pred_tag/--test_dir/'
                     '--pred_dir).' % (index, tag, pth))
    suffix = ('_' + pred_tag) if pred_tag else ''
    return lowres, target, pred, 'compare_%s_%s_%d%s.gif' % (mode, fold, index, suffix)


def run_batch(mode, pred_tag, out_dir, ds_root, eval_root, opts):
    """One GIF per held-out participant, across ALL folds, named by participant.

    Walks every ${EVAL_DIR}_<mode>/fold*/ directory, reads its per_volume.csv to
    map the on-disk index (0,1,...) back to the participant stem, finds the
    matching input/target/prediction, and writes <stem>[_<pred_tag>].gif. The
    pred_tag both selects predictions-<tag>/ AND is appended to every output name,
    so an adversarial run cannot overwrite the L1 GIFs.
    """
    if eval_root is None:
        ed = os.environ.get('EVAL_DIR')
        if not ed:
            sys.exit('--all needs $EVAL_DIR (source sr/cluster/config.sh) or '
                     '--eval_root.')
        eval_root = '%s_%s' % (ed, mode)
    if ds_root is None:
        dd = os.environ.get('DATASET_DIR')
        if not dd:
            sys.exit('--all needs $DATASET_DIR (source sr/cluster/config.sh) or '
                     '--ds_root.')
        ds_root = '%s_%s' % (dd, mode)

    fold_dirs = sorted(d for d in glob.glob(os.path.join(eval_root, 'fold*'))
                       if os.path.isdir(d))
    if not fold_dirs:
        sys.exit('no fold*/ directories under %s (check --mode and $EVAL_DIR).'
                 % eval_root)

    if out_dir is None:
        out_dir = os.path.join(eval_root, 'gifs' + (('-' + pred_tag) if pred_tag else ''))
    os.makedirs(out_dir, exist_ok=True)
    suffix = ('_' + pred_tag) if pred_tag else ''
    pred_sub = _pred_subdir(pred_tag)

    n_ok = n_skip = 0
    for fold_dir in fold_dirs:
        fold = os.path.basename(fold_dir)
        pv = os.path.join(fold_dir, 'per_volume.csv')
        if not os.path.exists(pv):
            print('  [skip fold] no per_volume.csv in %s' % fold_dir)
            continue
        test = os.path.join(ds_root, fold, 'test')
        pred_dir = os.path.join(fold_dir, pred_sub)
        with open(pv) as fh:
            rows = list(csv.DictReader(fh))
        print('%s: %d volumes  (predictions: %s)' % (fold, len(rows), pred_dir))
        for r in rows:
            idx = r.get('volume')
            stem = r.get('stem') or idx
            lowres = os.path.join(test, 'images', '%s.nii.gz' % idx)
            target = os.path.join(test, 'labels', '%s.nii.gz' % idx)
            pred = os.path.join(pred_dir, '%s_pred.nii.gz' % idx)
            missing = [p for p in (lowres, target, pred) if not os.path.exists(p)]
            if missing:
                print('  [skip] %s (idx %s): missing %s'
                      % (stem, idx, os.path.basename(missing[0])))
                n_skip += 1
                continue
            out = os.path.join(out_dir, '%s%s.gif' % (stem, suffix))
            try:
                build_gif(lowres, target, pred, out, title=stem, **opts)
                n_ok += 1
            except Exception as e:                               # noqa: BLE001
                print('  [fail] %s: %s' % (stem, e))
                n_skip += 1
    print('\ndone: %d gif(s) written to %s, %d skipped.' % (n_ok, out_dir, n_skip))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('index', nargs='?', type=int, default=None,
                   help='Held-out volume index (from per_volume.csv). If given, the '
                        'three files are located automatically; --lowres/--target/'
                        '--pred are then optional.')
    p.add_argument('--all', action='store_true',
                   help='Batch: one GIF per held-out participant across ALL folds '
                        'of --mode, named by participant. Uses $DATASET_DIR/$EVAL_DIR.')
    p.add_argument('--mode', default='dev',
                   help='Split mode: dev or cv (default dev). With --all, walks '
                        'every ${EVAL_DIR}_<mode>/fold*/.')
    p.add_argument('--fold', default='folddev',
                   help='Fold tag for single-index mode: folddev, fold0, ... '
                        '(default folddev). Ignored by --all.')
    p.add_argument('--pred_tag', default='',
                   help='Predictions subdir suffix, matching evaluate_sr --pred_tag: '
                        '"" -> predictions/, "adv_latest" -> predictions-adv_latest/. '
                        'Also appended to every output GIF name so an adversarial '
                        'run never overwrites the L1 GIFs.')
    p.add_argument('--out_dir', default=None,
                   help='Batch output dir (default ${EVAL_DIR}_<mode>/gifs[-<tag>]).')
    p.add_argument('--ds_root', default=None,
                   help='Override the ${DATASET_DIR}_<mode> root (--all).')
    p.add_argument('--eval_root', default=None,
                   help='Override the ${EVAL_DIR}_<mode> root (--all).')
    p.add_argument('--test_dir', default=None,
                   help='Override the dir holding images/ and labels/ (index mode).')
    p.add_argument('--pred_dir', default=None,
                   help='Override the dir holding <i>_pred.nii.gz (index mode).')
    p.add_argument('--lowres', help='2 mm input (network input)')
    p.add_argument('--target', help='1 mm truth (network target)')
    p.add_argument('--pred', help='model prediction')
    p.add_argument('--out', default=None, help='output .gif')
    p.add_argument('--ms', type=int, default=1500, help='ms per frame (default 1500)')
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
    render_opts = dict(clip=tuple(a.clip), flip=a.flip, ms=a.ms,
                       panel_px=a.panel_px, slices=a.slices)

    # Batch: one GIF per participant across all folds.
    if a.all:
        run_batch(a.mode, a.pred_tag, a.out_dir, a.ds_root, a.eval_root, render_opts)
        return

    # Single volume: by index (auto-located) or by explicit paths.
    out = a.out
    if a.index is not None:
        a.lowres, a.target, a.pred, derived_out = resolve_by_index(
            a.index, a.mode, a.fold, a.test_dir, a.pred_dir, a.pred_tag)
        out = out or derived_out
    elif not (a.lowres and a.target and a.pred):
        p.error('give an index (e.g. `make_compare_gif 3`), --all, OR all of '
                '--lowres/--target/--pred.')
    out = out or 'compare.gif'

    try:
        build_gif(a.lowres, a.target, a.pred, out, title=None, **render_opts)
    except ValueError as e:
        sys.exit(str(e))
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
