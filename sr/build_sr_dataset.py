#!/usr/bin/env python3
"""
build_sr_dataset.py
===================
Turn the output of `sr/simulate_lowres.py` into the exact folder layout that
GAMBAS's `utils/NiftiDataset.NiftiDataSet` expects, with a subject-level
train/val/test split.

GAMBAS layout (see README):

    <root>/train/images/0.nii.gz     <- network INPUT  ("real_A", the 2 mm sim)
    <root>/train/labels/0.nii.gz     <- network TARGET ("real_B", the 1 mm truth)
    <root>/val/images/... labels/...
    <root>/test/images/... labels/...

Two important facts about that dataloader, which drive the design here:

  1. `RandomCrop` crops `image` and `label` with a *single* RegionOfInterest
     index and size. The input and the target must therefore live on the SAME
     voxel grid and have the SAME array shape. That is why we feed it
     `lowres_on_hr_grid/` (the sinc-upsampled 2 mm volume) rather than the true
     2 mm volume. The network's job is to restore the high-frequency content
     that the 2 mm acquisition never measured -- it is not asked to change grid
     size.

  2. `Normalization()` rescales each volume independently to 0-255 by
     (x - mean)/std then min-max. This happens per-volume at load time, so the
     absolute intensity of what we write here does not matter much -- but the
     *shape* of the intensity histogram does. We therefore do NOT z-score or
     mask anything here; we hand over faithful volumes and let the dataloader do
     its thing.

`--link` uses symlinks instead of copies, which is what you want on a cluster
with a shared filesystem: the dataset directory becomes ~0 bytes.

Usage
-----
    python -m sr.build_sr_dataset \
        --sim_dir /data/sr_sim \
        --out_root /data/sr_dataset \
        --val_frac 0.1 --test_frac 0.1 --link

If your HR volumes were not written by the simulator (`--copy_hr` omitted),
point `--hr_dir` at the originals instead.
"""

import argparse
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sr.naming import DEFAULT_NAME_SCHEMA, strip_ext, subject_of

NIFTI_EXTS = ('.nii.gz', '.nii', '.mgz', '.mha', '.mhd', '.nrrd')


def index_dir(d):
    """{stem: full_path} for image files in `d`."""
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.startswith('.'):
            continue
        if any(fn.lower().endswith(e) for e in NIFTI_EXTS):
            out[strip_ext(fn)] = os.path.join(d, fn)
    return out


def subject_key(stem, pattern):
    """Group stems into subjects so repeated sessions/runs never straddle a split.

    Delegates to sr.naming so the definition of 'subject' is identical here, in
    make_folds.py and in evaluate_sr.py. `pattern` is an optional regex override;
    None means use the positional --name_schema.
    """
    return subject_of(stem, pattern, DEFAULT_NAME_SCHEMA)


def place(src, dst, mode):
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == 'link':
        os.symlink(os.path.abspath(src), dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Assemble a GAMBAS-format paired SR dataset.')
    p.add_argument('--sim_dir', required=True,
                   help='Output dir of simulate_lowres.py')
    p.add_argument('--out_root', required=True,
                   help='Dataset root to create (train/ val/ test/ inside)')
    p.add_argument('--method', default=None,
                   help="Simulation method tag, e.g. 'kspace'. Shorthand for "
                        "--lr_subdir lowres-<method>, matching "
                        "simulate_lowres.py's default --layout method.")
    p.add_argument('--lr_subdir', default=None,
                   help='Subdirectory of --sim_dir holding the network INPUT. '
                        "Defaults to 'lowres-<method>' when --method is given, "
                        "otherwise 'lowres_on_hr_grid' (the flat layout). Use the "
                        'on-HR-grid volumes, not the native 2 mm ones, unless you '
                        'have modified the dataloader -- it crops input and target '
                        'with one index, so they must share a grid.')
    p.add_argument('--hr_dir', default=None,
                   help='HR targets. Defaults to <sim_dir>/hr. Point this at the '
                        'hr/ written by simulate_lowres.py rather than your '
                        'originals/: --reorient/--conform change the grid, and the '
                        'pair must match exactly.')
    p.add_argument('--folds_json', default=None,
                   help='Split specification from sr/make_folds.py. When given, '
                        '--val_frac/--test_frac/--subject_regex are ignored and '
                        'the split is taken verbatim from the file. This is the '
                        'path you want for anything you intend to report, since '
                        'make_folds.py enforces subject grouping and subgroup '
                        'stratification.')
    p.add_argument('--fold', default=None,
                   help="Which fold from --folds_json ('dev', '0', '1', ...). "
                        "Required with --folds_json unless the file has one fold.")
    p.add_argument('--val_frac', type=float, default=0.10)
    p.add_argument('--test_frac', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--subject_regex', default=None,
                   help='Optional regex override for the subject id. Default is '
                        'the id field of the positional name schema (%s). Only '
                        'used for the random-split path and the manifest; with '
                        '--folds_json the split comes from the file.'
                        % DEFAULT_NAME_SCHEMA)
    p.add_argument('--link', dest='mode', action='store_const', const='link',
                   default='copy', help='Symlink instead of copying (recommended)')
    p.add_argument('--hardlink', dest='mode', action='store_const', const='hardlink')
    p.add_argument('--keep_names', action='store_true',
                   help='Keep original filenames instead of renumbering to 0,1,2... '
                        'The dataloader sorts both folders numerically, so names '
                        'must sort identically in images/ and labels/. Renumbering '
                        '(the default) removes any risk of mispairing.')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args(argv)

    lr_subdir = args.lr_subdir
    if lr_subdir is None:
        lr_subdir = ('lowres-%s' % args.method if args.method
                     else 'lowres_on_hr_grid')
    lr_dir = os.path.join(args.sim_dir, lr_subdir)
    lr = index_dir(lr_dir)
    hr_dir = args.hr_dir or os.path.join(args.sim_dir, 'hr')
    hr = index_dir(hr_dir)

    if not lr:
        available = sorted(d for d in os.listdir(args.sim_dir)
                           if d.startswith('lowres')) if os.path.isdir(args.sim_dir) else []
        sys.exit('No low-res volumes in %s.%s'
                 % (lr_dir,
                    ('\nAvailable lowres dirs in %s: %s\nPass --method or '
                     '--lr_subdir.' % (args.sim_dir, available)) if available
                    else ' Did simulate_lowres.py run?'))
    print('input  (LR): %s  [%d volumes]' % (lr_dir, len(lr)))
    print('target (HR): %s  [%d volumes]' % (hr_dir, len(hr)))
    if not hr:
        sys.exit('No high-res volumes in %s (pass --hr_dir, or rerun the simulator '
                 'with --copy_hr)' % hr_dir)

    stems = sorted(set(lr) & set(hr))
    missing_hr = sorted(set(lr) - set(hr))
    missing_lr = sorted(set(hr) - set(lr))
    if missing_hr:
        print('WARNING: %d LR volume(s) have no HR match, e.g. %s'
              % (len(missing_hr), missing_hr[:5]), file=sys.stderr)
    if missing_lr:
        print('WARNING: %d HR volume(s) have no LR match, e.g. %s'
              % (len(missing_lr), missing_lr[:5]), file=sys.stderr)
    if not stems:
        sys.exit('No paired volumes found -- LR and HR filenames must match.')

    # ---- decide the split ----------------------------------------------------
    if args.folds_json:
        with open(args.folds_json) as f:
            spec = json.load(f)
        folds = spec.get('folds', [])
        if not folds:
            sys.exit('%s contains no folds' % args.folds_json)
        if args.fold is None:
            if len(folds) != 1:
                sys.exit('%s has %d folds; pass --fold (one of %s)'
                         % (args.folds_json, len(folds),
                            [str(f['fold']) for f in folds]))
            chosen = folds[0]
        else:
            matches = [f for f in folds if str(f['fold']) == str(args.fold)]
            if not matches:
                sys.exit('fold %r not in %s (available: %s)'
                         % (args.fold, args.folds_json,
                            [str(f['fold']) for f in folds]))
            chosen = matches[0]

        split_stems = {s: sorted(chosen.get(s, [])) for s in
                       ('train', 'val', 'test')}
        # Every stem in the spec must be a pair we actually have on disk.
        spec_all = set().union(*[set(v) for v in split_stems.values()])
        absent = sorted(spec_all - set(stems))
        if absent:
            sys.exit('%d volume(s) named in %s are missing from %s, e.g. %s. '
                     'Regenerate the folds against this sim_dir.'
                     % (len(absent), args.folds_json, args.sim_dir, absent[:5]))
        unused = sorted(set(stems) - spec_all)
        if unused:
            print('NOTE: %d paired volume(s) on disk are not in the fold spec '
                  'and will be skipped, e.g. %s' % (len(unused), unused[:5]),
                  file=sys.stderr)
        print('using fold %r from %s' % (chosen['fold'], args.folds_json))
    else:
        subjects = {}
        for s in stems:
            subjects.setdefault(subject_key(s, args.subject_regex), []).append(s)
        subj_ids = sorted(subjects)
        rng = random.Random(args.seed)
        rng.shuffle(subj_ids)

        n = len(subj_ids)
        n_test = int(round(args.test_frac * n))
        n_val = int(round(args.val_frac * n))
        if n - n_val - n_test < 1:
            sys.exit('Split leaves no training subjects (%d subjects total).' % n)

        split_of = {}
        for i, sid in enumerate(subj_ids):
            if i < n_test:
                split_of[sid] = 'test'
            elif i < n_test + n_val:
                split_of[sid] = 'val'
            else:
                split_of[sid] = 'train'
        split_stems = {'train': [], 'val': [], 'test': []}
        for sid in sorted(subj_ids):
            split_stems[split_of[sid]].extend(sorted(subjects[sid]))
        print('%d subjects / %d volumes' % (n, len(stems)))

    # ---- assign output names -------------------------------------------------
    manifest = {'train': [], 'val': [], 'test': []}
    counters = {'train': 0, 'val': 0, 'test': 0}
    for split in ('train', 'val', 'test'):
        for stem in split_stems[split]:
            idx = counters[split]
            counters[split] += 1
            ext = '.nii.gz'
            name = stem + ext if args.keep_names else '%d%s' % (idx, ext)
            manifest[split].append({'index': idx, 'assigned_name': name,
                                    'subject': subject_key(stem, args.subject_regex),
                                    'stem': stem,
                                    'lr': lr[stem], 'hr': hr[stem]})

    print('train %d, val %d, test %d'
          % (counters['train'], counters['val'], counters['test']))

    if args.dry_run:
        print(json.dumps({k: v[:3] for k, v in manifest.items()}, indent=2))
        return 0

    for split, entries in manifest.items():
        img_d = os.path.join(args.out_root, split, 'images')
        lab_d = os.path.join(args.out_root, split, 'labels')
        os.makedirs(img_d, exist_ok=True)
        os.makedirs(lab_d, exist_ok=True)
        for e in entries:
            place(e['lr'], os.path.join(img_d, e['assigned_name']), args.mode)
            place(e['hr'], os.path.join(lab_d, e['assigned_name']), args.mode)

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, 'manifest.json'), 'w') as f:
        json.dump({'args': vars(args), 'splits': manifest}, f, indent=2)

    print('wrote %s' % os.path.join(args.out_root, 'manifest.json'))
    print('\nSet these in options/base_options.py (or pass on the CLI):')
    print('  --data_path %s' % os.path.join(args.out_root, 'train'))
    print('  --val_path  %s' % os.path.join(args.out_root, 'val'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
