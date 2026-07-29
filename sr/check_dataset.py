#!/usr/bin/env python3
"""
check_dataset.py
================
Pre-flight checks on an assembled dataset. Run this before burning GPU hours.

Catches the failure modes that otherwise show up as "the loss goes down but the
output is garbage":

  1. images/ and labels/ have different counts, or sort into a different order
     (the dataloader pairs them by sorted position, not by name).
  2. A pair has mismatched array shape -- `RandomCrop` would then crop different
     anatomy from the input and the target.
  3. A pair has mismatched spacing / origin / direction -- same problem, but
     silent, because the arrays can still be the same shape.
  4. A volume is smaller than the patch size along some axis.
  5. A volume is constant / all zeros / has NaNs.
  6. The pairing is inverted (labels are lower-resolution than images). Detected
     by comparing high-frequency energy: the label should have MORE.
  7. Mutual information / correlation between the pair is low, meaning they are
     not the same subject.

Exit code is non-zero if any hard error is found, so it can gate an sbatch chain.

Usage:
    python -m sr.check_dataset --root /data/sr_dataset --patch_size 128 128 128
    python -m sr.check_dataset --root /data/sr_dataset --n 4     # quick, 4 per split
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk

from sr.naming import lst_files, subgroup_of
from sr.sr_metrics import hf_energy


# Naming helpers live in sr/naming.py, which is stdlib-only, so this pre-flight
# check runs without torch / mamba_ssm.


def check_split(root, split, patch_size, n_check, errors, warnings):
    img_dir = os.path.join(root, split, 'images')
    lab_dir = os.path.join(root, split, 'labels')
    if not os.path.isdir(img_dir):
        warnings.append('%s: no images/ dir (skipping split)' % split)
        return
    imgs = lst_files(img_dir)
    labs = lst_files(lab_dir)

    print('\n=== %s: %d images, %d labels ===' % (split, len(imgs), len(labs)))
    if len(imgs) != len(labs):
        errors.append('%s: %d images vs %d labels' % (split, len(imgs), len(labs)))
        return
    if not imgs:
        warnings.append('%s: empty' % split)
        return

    # Check 1: names must correspond position-by-position.
    for a, b in zip(imgs, labs):
        if os.path.basename(a) != os.path.basename(b):
            errors.append('%s: sorted order differs -- images[%s] pairs with '
                          'labels[%s]. Rebuild with build_sr_dataset.py (which '
                          'renumbers) or fix the filenames.'
                          % (split, os.path.basename(a), os.path.basename(b)))
            break

    idxs = range(len(imgs)) if n_check <= 0 else \
        np.linspace(0, len(imgs) - 1, min(n_check, len(imgs))).astype(int)

    for i in idxs:
        name = os.path.basename(imgs[i])
        try:
            ia = sitk.ReadImage(imgs[i], sitk.sitkFloat32)
            ib = sitk.ReadImage(labs[i], sitk.sitkFloat32)
        except Exception as e:
            errors.append('%s/%s: unreadable: %s' % (split, name, e))
            continue

        # Checks 2 and 3: identical grid.
        if ia.GetSize() != ib.GetSize():
            errors.append('%s/%s: size %s (image) vs %s (label). The dataloader '
                          'crops both with one index -- they must match.'
                          % (split, name, ia.GetSize(), ib.GetSize()))
            continue
        for attr in ('GetSpacing', 'GetOrigin', 'GetDirection'):
            va, vb = getattr(ia, attr)(), getattr(ib, attr)()
            if not np.allclose(va, vb, atol=1e-3):
                errors.append('%s/%s: %s differs: %s vs %s'
                              % (split, name, attr[3:], np.round(va, 3),
                                 np.round(vb, 3)))

        # Check 4: patch fits.
        for ax, (s, p) in enumerate(zip(ia.GetSize(), patch_size)):
            if s < p:
                warnings.append('%s/%s: axis %d is %d < patch %d -- PadTo will '
                                'edge-pad it' % (split, name, ax, s, p))

        a = np.transpose(sitk.GetArrayFromImage(ia), (2, 1, 0)).astype(np.float32)
        b = np.transpose(sitk.GetArrayFromImage(ib), (2, 1, 0)).astype(np.float32)

        # Check 5: degenerate volumes.
        for tag, arr in (('image', a), ('label', b)):
            if not np.isfinite(arr).all():
                errors.append('%s/%s: %s has non-finite values' % (split, name, tag))
            if float(arr.max() - arr.min()) < 1e-6:
                errors.append('%s/%s: %s is constant' % (split, name, tag))

        # Normalise for the comparison checks (mirrors the dataloader).
        def norm(x):
            lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
            return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        an, bn = norm(a), norm(b)

        # Check 6: which one is sharper? Label must be.
        hf_a, hf_b = hf_energy(an), hf_energy(bn)
        ratio = hf_b / max(hf_a, 1e-12)
        # Check 7: same subject?
        corr = float(np.corrcoef(an.ravel(), bn.ravel())[0, 1])

        flag = ''
        if ratio < 1.02:
            errors.append('%s/%s: label is NOT higher-frequency than image '
                          '(HF label %.4f vs image %.4f). images/ and labels/ '
                          'are probably swapped.' % (split, name, hf_b, hf_a))
            flag = '  <-- SWAPPED?'
        if corr < 0.80:
            errors.append('%s/%s: correlation %.3f is too low -- image and label '
                          'may be different subjects or misregistered.'
                          % (split, name, corr))
            flag = '  <-- MISPAIRED?'

        print('  %-28s %s  HF img %.4f -> lab %.4f (x%.2f)  corr %.4f%s'
              % (name, tuple(ia.GetSize()), hf_a, hf_b, ratio, corr, flag))


def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', required=True, help='Dataset root (contains train/ val/ test/)')
    p.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128])
    p.add_argument('--n', type=int, default=6,
                   help='Volumes to inspect per split (0 = all)')
    p.add_argument('--splits', nargs='*', default=['train', 'val', 'test'])
    args = p.parse_args(argv)

    errors, warnings = [], []
    for split in args.splits:
        check_split(args.root, split, args.patch_size, args.n, errors, warnings)

    print('\n' + '=' * 70)
    if warnings:
        print('%d warning(s):' % len(warnings))
        for w in warnings:
            print('  ! ' + w)
    if errors:
        print('%d ERROR(s):' % len(errors))
        for e in errors:
            print('  X ' + e)
        print('\nFix these before training.')
        return 1
    print('all checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
