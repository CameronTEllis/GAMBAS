#!/usr/bin/env python3
"""
evaluate_sr.py
==============
Run a trained checkpoint over a test split and report per-volume and aggregate
metrics against the 1 mm ground truth, alongside baselines.

Baselines are the point of this script. A super-resolution number is meaningless
without them, because the input is already a very good approximation of the
target. The three reported comparators are:

  sinc      the input as-is (2 mm zero-fill-interpolated onto the 1 mm grid).
            This is what you get for free. If the model does not beat it, stop.
  bspline   BSpline resampling of the true 2 mm volume onto the 1 mm grid, the
            standard "just interpolate it" pipeline. Requires --lowres_native_dir.
  identity  ground truth vs itself, i.e. the ceiling (inf PSNR); reported only
            for the sharpness / hf_energy columns, where the target's own value
            is the number the model should be matching, not exceeding.

Outputs
-------
  <out_dir>/per_volume.csv    one row per test volume, all metrics
  <out_dir>/summary.csv       mean / std / median per metric
  <out_dir>/spectra.csv       radial power spectra (pred, target, input)
  <out_dir>/predictions/*.nii.gz   with --save_predictions

Usage
-----
    python -m sr.evaluate_sr \
        --test_path /data/sr_dataset/test \
        --checkpoints_dir /scratch/checkpoints --name sr_2mm_to_1mm \
        --which_epoch best \
        --out_dir /scratch/eval/sr_2mm_to_1mm --save_predictions
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk
import torch

import utils.NiftiDataset as NiftiDataset
from models import networks3D
from sr import sr_metrics
from sr.naming import (DEFAULT_NAME_SCHEMA, age_of, load_name_map, session_of,
                       subgroup_of, subject_of)
from sr.train_sr import sliding_window_predict


def load_generator(args, device):
    """Build the GAMBAS generator directly and load weights, avoiding the
    option-parsing dance in models/__init__.py."""
    path = os.path.join(args.checkpoints_dir, args.name,
                        '%s_net_G.pth' % args.which_epoch)
    if not os.path.exists(path):
        sys.exit('checkpoint not found: %s' % path)
    state = torch.load(path, map_location='cpu')
    if hasattr(state, '_metadata'):
        del state._metadata

    # The architecture is DETECTED from the checkpoint, not taken from a flag.
    # load_state_dict(strict=False) below would otherwise file an unmatched
    # 'res_scale' under `unexpected` and carry on, evaluating a residual model as
    # a non-residual one -- silently, and producing nonsense. Reading the key is
    # the only way this cannot be got wrong from the command line.
    residual = any(k.split('.')[-1] == 'res_scale' for k in state)
    if args.global_residual is not None and bool(args.global_residual) != residual:
        sys.exit('--global_residual=%s contradicts the checkpoint (res_scale %s '
                 'present in %s). Drop the flag and let it be detected.'
                 % (args.global_residual, 'is' if residual else 'is not', path))
    print('architecture: %s (from checkpoint)'
          % ('global residual y = x + s*G(x)' if residual else 'published, no long skip'))

    net = networks3D.define_G(
        args.input_nc, args.output_nc, args.ngf, args.netG, args.norm,
        not args.no_dropout, 'normal', 0.02, args.gpu_ids,
        **{'img_size': (args.imageSize, args.imageSize),
           'global_residual': residual})
    target = net.module if isinstance(net, torch.nn.DataParallel) else net
    # InstanceNorm running stats are absent in these checkpoints.
    state = {k: v for k, v in state.items()
             if not any(k.endswith(s) for s in
                        ('running_mean', 'running_var', 'num_batches_tracked'))}
    missing, unexpected = target.load_state_dict(state, strict=False)
    if unexpected:
        print('unexpected keys: %s' % list(unexpected)[:5])
    if missing:
        print('missing keys: %s' % list(missing)[:5])
    print('loaded %s' % path)
    net.to(device).eval()
    return net


def prep(path):
    im = sitk.ReadImage(path)
    im = NiftiDataset.Normalization(im)
    im = sitk.Cast(im, sitk.sitkFloat32)
    a = np.abs(sitk.GetArrayFromImage(im))
    a = np.transpose(a, (2, 1, 0)).astype(np.float32)
    return (a - 127.5) / 127.5, im


def to01(x):
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--test_path', required=True,
                   help='Split dir containing images/ and labels/')
    p.add_argument('--checkpoints_dir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--which_epoch', default='best',
                   help="'best', 'latest', or an epoch number")
    p.add_argument('--out_dir', required=True)

    p.add_argument('--lowres_native_dir', default=None,
                   help='simulate_lowres.py lowres_native/ dir, to add the '
                        'BSpline-interpolation baseline')

    p.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128])
    p.add_argument('--stride', type=int, nargs=3, default=None,
                   help='default: half the patch')
    p.add_argument('--masked', action='store_true', default=True,
                   help='Also report metrics restricted to a foreground mask')
    p.add_argument('--no_masked', dest='masked', action='store_false')
    p.add_argument('--save_predictions', action='store_true')
    p.add_argument('--pred_tag', default='',
                   help='Suffix for the predictions subdir: "" -> predictions/, '
                        '"adv" -> predictions-adv/. Use it so a second eval (e.g. '
                        'an adversarial-loss variant) does not overwrite the first '
                        "run's saved volumes.")
    p.add_argument('--max_volumes', type=int, default=0, help='0 = all')

    p.add_argument('--name_schema', default=DEFAULT_NAME_SCHEMA,
                   help='Field order in the ORIGINAL filename (resolved via the '
                        'dataset manifest, since the builder renumbers files). '
                        'Supplies the weighting label and the age column.')
    p.add_argument('--subgroup_regex', default=None,
                   help='Override the subgroup label (default: the weighting field '
                        'of --name_schema). Metrics are reported per subgroup as '
                        'well as pooled, because a pooled mean over subgroups with '
                        'different intrinsic difficulty describes neither.')
    p.add_argument('--subgroup_name', default='anatomical weighting (confounded with age)',
                   help='Label for the subgroup axis in the printed report')
    p.add_argument('--manifest', default=None,
                   help='Path to manifest.json (default: alongside --test_path)')
    p.add_argument('--fold', default='', help='Recorded in the CSVs, for CV pooling')

    # network construction (must match training)
    p.add_argument('--netG', default='gambas')
    p.add_argument('--exclude_list', default='',
                   help='File of stems (one per line, e.g. the --exclude_list '
                        'output of sr.audit_resolution) with too little headroom '
                        'above the simulated Nyquist to demonstrate anything. They '
                        'are still evaluated and still written to per_volume.csv '
                        'with low_headroom=1, but are held out of the headline '
                        'summary and reported in their own LOW_HEADROOM block. '
                        'Keeps them in training while stopping them from pulling '
                        'the reported baseline toward ~42 dB.')
    # Tri-state on purpose: None = detect from the checkpoint (the default and
    # the right answer); 0/1 only to assert an expectation and fail loudly if
    # the checkpoint disagrees.
    p.add_argument('--global_residual', type=int, default=None, choices=[0, 1],
                   help='Leave unset to detect from the checkpoint. Set only to '
                        'assert what you think the checkpoint is.')
    p.add_argument('--input_nc', type=int, default=1)
    p.add_argument('--output_nc', type=int, default=1)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--norm', default='instance')
    p.add_argument('--no_dropout', action='store_true', default=True)
    p.add_argument('--imageSize', type=int, default=256)
    p.add_argument('--gpu_ids', type=int, nargs='*', default=[0])
    p.add_argument('--amp', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stride is None:
        args.stride = [max(1, p // 2) for p in args.patch_size]

    if not torch.cuda.is_available():
        print('CUDA not available -- running on CPU (slow).')
        args.gpu_ids = []
    device = torch.device('cuda:%d' % args.gpu_ids[0]) if args.gpu_ids else torch.device('cpu')
    amp_dtype = torch.bfloat16 if args.amp else None

    net = load_generator(args, device)

    imgs = NiftiDataset.lstFiles(os.path.join(args.test_path, 'images'))
    labs = NiftiDataset.lstFiles(os.path.join(args.test_path, 'labels'))
    if len(imgs) != len(labs) or not imgs:
        sys.exit('images/labels mismatch: %d vs %d' % (len(imgs), len(labs)))
    if args.max_volumes:
        imgs, labs = imgs[:args.max_volumes], labs[:args.max_volumes]

    os.makedirs(args.out_dir, exist_ok=True)
    # Predictions go in 'predictions/' by default, or 'predictions-<tag>/' when a
    # tag is given -- so a second run (e.g. an adversarial variant) does not
    # overwrite the first run's saved volumes in the same out_dir.
    pred_subdir = 'predictions' + (('-' + args.pred_tag) if args.pred_tag else '')
    pred_dir = os.path.join(args.out_dir, pred_subdir)
    if args.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)
        print('saving predictions to %s' % pred_dir)

    # Split-aware: assigned names restart at 0 per split, so the split has
    # to be pinned or '0' resolves to the wrong volume.
    name_map = load_name_map(args.test_path, args.manifest,
                             warn=lambda m: print('WARNING: ' + m,
                                                  file=sys.stderr))
    if args.subgroup_regex and not name_map:
        print('NOTE: no manifest.json found, so subgroup labels are read from the '
              'on-disk filenames. If the dataset was renumbered, every volume '
              'will come out "unknown".', file=sys.stderr)

    # Volumes with negligible headroom above the simulated Nyquist. They stay in
    # training (they still produce gradient) but must not drive the reported
    # numbers: for them the sinc input is already ~the target, so they push the
    # baseline toward 42 dB and compress every delta toward zero regardless of
    # how good the model is. Marked and reported separately rather than deleted,
    # so the exclusion is visible in per_volume.csv instead of implicit.
    low_headroom = set()
    if args.exclude_list:
        if not os.path.exists(args.exclude_list):
            sys.exit('--exclude_list not found: %s' % args.exclude_list)
        with open(args.exclude_list) as fh:
            low_headroom = {ln.strip() for ln in fh if ln.strip()}
        print('low-headroom list: %d stem(s) from %s'
              % (len(low_headroom), args.exclude_list))

    rows = []
    spectra = []

    for i, (ip, lp) in enumerate(zip(imgs, labs)):
        name = os.path.basename(ip).replace('.nii.gz', '').replace('.nii', '')
        stem = name_map.get(name, name)
        sub = subgroup_of(stem, args.subgroup_regex, schema=args.name_schema)
        age_tok, age_num = age_of(stem, None, args.name_schema)
        lr, lr_itk = prep(ip)
        hr, _ = prep(lp)
        if lr.shape != hr.shape:
            print('[skip] %s shape %s vs %s' % (name, lr.shape, hr.shape))
            continue

        pred = sliding_window_predict(net, lr, list(args.patch_size),
                                      list(args.stride), device, amp_dtype)
        p01, h01, l01 = to01(pred), to01(hr), to01(lr)
        mask = sr_metrics.brain_mask(h01) if args.masked else None

        row = {'volume': name, 'stem': stem, 'subgroup': sub, 'fold': args.fold,
               'subject': subject_of(stem, None, args.name_schema),
               'session': session_of(stem, None, args.name_schema),
               'age': age_tok, 'age_num': ('' if age_num is None else age_num),
               'low_headroom': int(stem in low_headroom)}
        for tag, arr in (('pred', p01), ('sinc', l01)):
            m = sr_metrics.all_metrics(arr, h01, mask=mask)
            for k, v in m.items():
                if k.endswith('_target'):
                    row[k] = v          # written once, identical for both
                else:
                    row['%s_%s' % (tag, k)] = v

        # BSpline baseline from the true 2 mm volume, if available
        if args.lowres_native_dir:
            for ext in ('.nii.gz', '.nii'):
                cand = os.path.join(args.lowres_native_dir, name + ext)
                if os.path.exists(cand):
                    low = sitk.ReadImage(cand, sitk.sitkFloat32)
                    up = sitk.Resample(low, lr_itk, sitk.Transform(),
                                       sitk.sitkBSpline, 0.0, sitk.sitkFloat32)
                    up = NiftiDataset.Normalization(up)
                    b = np.transpose(np.abs(sitk.GetArrayFromImage(up)), (2, 1, 0))
                    b01 = to01((b.astype(np.float32) - 127.5) / 127.5)
                    if b01.shape == h01.shape:
                        m = sr_metrics.all_metrics(b01, h01, mask=mask)
                        for k, v in m.items():
                            if not k.endswith('_target'):
                                row['bspline_%s' % k] = v
                    break

        rows.append(row)

        f_p, s_p = sr_metrics.radial_power_spectrum(p01)
        _, s_h = sr_metrics.radial_power_spectrum(h01)
        _, s_l = sr_metrics.radial_power_spectrum(l01)
        spectra.append((name, f_p, s_p, s_h, s_l))

        if args.save_predictions:
            out = sitk.GetImageFromArray(np.transpose(p01 * 255.0, (2, 1, 0)))
            out.CopyInformation(lr_itk)
            sitk.WriteImage(out, os.path.join(pred_dir, name + '_pred.nii.gz'))

        print('[%d/%d] %-28s %-8s PSNR pred %.3f / sinc %.3f  SSIM %.4f / %.4f  '
              'HF %.4f / %.4f (gt %.4f)'
              % (i + 1, len(imgs), stem, sub, row['pred_psnr'], row['sinc_psnr'],
                 row['pred_ssim'], row['sinc_ssim'], row['pred_hf_energy_pred'],
                 row['sinc_hf_energy_pred'], row['hf_energy_target']), flush=True)

    if not rows:
        sys.exit('nothing evaluated')

    # ---- write per-volume ----------------------------------------------------
    # 'low_headroom' belongs in lead, not in metric_keys -- otherwise it would be
    # summarised as though it were a metric and appear in summary.csv as a mean.
    lead = ['volume', 'stem', 'subject', 'session', 'subgroup', 'age',
            'age_num', 'fold', 'low_headroom']
    keys = lead + sorted(k for k in rows[0] if k not in lead)
    with open(os.path.join(args.out_dir, 'per_volume.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    metric_keys = [k for k in keys if k not in lead]

    # ---- summary, pooled and per subgroup ------------------------------------
    # Headline numbers come from the volumes that HAVE headroom. The excluded
    # ones get their own 'LOW_HEADROOM' block so nothing is hidden.
    scored = [r for r in rows if not r['low_headroom']]
    dropped = [r for r in rows if r['low_headroom']]
    if dropped and not scored:
        print('WARNING: every test volume is on the low-headroom list; reporting '
              'all of them rather than nothing.')
        scored, dropped = rows, []
    if dropped:
        print('excluded %d/%d test volume(s) from the headline summary as '
              'low-headroom: %s' % (len(dropped), len(rows),
                                    ', '.join(r['stem'] for r in dropped)))

    subgroups = sorted({r['subgroup'] for r in scored})
    blocks = [('ALL', scored)]
    if len(subgroups) > 1:
        blocks += [(sg, [r for r in scored if r['subgroup'] == sg])
                   for sg in subgroups]
    if dropped:
        blocks.append(('LOW_HEADROOM', dropped))
    with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subgroup', 'metric', 'mean', 'std', 'median', 'n'])
        for sg, sel in blocks:
            for k in metric_keys:
                vals = np.array([r[k] for r in sel
                                 if k in r and np.isfinite(r[k])], dtype=float)
                if vals.size:
                    w.writerow([sg, k, '%.6f' % vals.mean(), '%.6f' % vals.std(),
                                '%.6f' % np.median(vals), vals.size])

    # ---- spectra -------------------------------------------------------------
    with open(os.path.join(args.out_dir, 'spectra.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['volume', 'freq_cycles_per_voxel', 'power_pred',
                    'power_target', 'power_input'])
        for name, fr, sp, sh, sl in spectra:
            for a, b, c, d in zip(fr, sp, sh, sl):
                w.writerow([name, '%.6f' % a, '%.6e' % b, '%.6e' % c, '%.6e' % d])

    # ---- headline ------------------------------------------------------------
    def mean(k, sel):
        v = [r[k] for r in sel if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float('nan')

    def block(title, sel):
        print('\n=== %s (%d volumes) ===' % (title, len(sel)))
        print('              PSNR      SSIM      MAE       HF-energy')
        for tag in ('pred', 'sinc', 'bspline'):
            if '%s_psnr' % tag not in rows[0]:
                continue
            print('%-8s  %8.3f  %8.4f  %8.5f  %8.4f'
                  % (tag, mean('%s_psnr' % tag, sel), mean('%s_ssim' % tag, sel),
                     mean('%s_mae' % tag, sel), mean('%s_hf_energy_pred' % tag, sel)))
        print('%-8s  %8s  %8s  %8s  %8.4f'
              % ('target', '-', '-', '-', mean('hf_energy_target', sel)))
        print('PSNR gain over sinc: %+.3f dB'
              % (mean('pred_psnr', sel) - mean('sinc_psnr', sel)))
        # Paired per-volume deltas. The mean delta equals the difference of the
        # means, but the range and the win count do not follow from it, and with
        # headroom varying several-fold across this cohort a positive mean can
        # hide volumes the model actively degraded.
        d = sorted(r['pred_psnr'] - r['sinc_psnr'] for r in sel
                   if np.isfinite(r.get('pred_psnr', np.nan))
                   and np.isfinite(r.get('sinc_psnr', np.nan)))
        if d:
            print('  per-volume delta: %+.3f .. %+.3f dB, %d/%d improved'
                  % (d[0], d[-1], sum(x > 0 for x in d), len(d)))

    block('ALL', scored)
    if len(subgroups) > 1:
        for sg in subgroups:
            block(sg, [r for r in scored if r['subgroup'] == sg])
        print('\nSubgroup axis: %s.' % args.subgroup_name)
        print('The pooled ALL block above is a weighted average over subgroups '
              'of unequal size and unequal intrinsic difficulty. Quote the '
              'per-subgroup blocks, not ALL.')
        ns = {sg: sum(1 for r in rows if r['subgroup'] == sg) for sg in subgroups}
        small = {k: v for k, v in ns.items() if v < 5}
        if small:
            print('WARNING: subgroup(s) %s have fewer than 5 test volumes here. '
                  'A mean over that many volumes has a very wide interval -- '
                  'pool across folds with sr/aggregate_cv.py before quoting it.'
                  % small)
    print('\nwrote %s' % args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
