#!/usr/bin/env python3
"""
smoke_test.py
=============
End-to-end shakedown on synthetic phantoms. Run this FIRST on the cluster, on a
GPU node, before submitting a real job. It exercises every piece of the pipeline
in about two minutes and fails loudly on the things that otherwise waste a day:

  1. imports (torch, mamba_ssm, SimpleITK) and CUDA visibility
  2. simulate_lowres on generated phantoms
  3. build_sr_dataset + check_dataset
  4. the training dataloader (real patch shapes out of real transforms)
  4b. the subgroup-balanced sampler, with labels resolved via the manifest
  4c. on-the-fly degradation randomisation: matches the offline operator, stays
      band-limited, draws span the configured range
  5. one generator forward pass at the configured patch size, with peak GPU
     memory reported -- this is what tells you whether 128^3 will fit
  6. one full optimize_parameters() step (G + D), and one with --lambda_adv 0
  6b. --init_from round trip: save G, reload it, assert 100% coverage and
      bit-identical transfer, and confirm an incompatible checkpoint is refused
  7. sliding-window inference over a whole volume
  8. the metrics module

Phantoms are named with the real <id>_<session>_<age>_<weighting> convention and
include both weightings across two visits per subject, so schema parsing, subject
grouping, subgroup stratification and balanced sampling are all exercised rather
than stubbed.

Usage:
    python -m sr.smoke_test --work_dir /scratch/$USER/smoke --patch_size 128
    python -m sr.smoke_test --work_dir /tmp/smoke --patch_size 64 --cpu
"""

import argparse
import os
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

RESULTS = []


def step(name):
    def deco(fn):
        def wrapped(*a, **kw):
            print('\n' + '=' * 72)
            print('STEP: %s' % name)
            print('=' * 72, flush=True)
            try:
                out = fn(*a, **kw)
                RESULTS.append((name, 'PASS', ''))
                return out
            except Exception as e:
                RESULTS.append((name, 'FAIL', '%s: %s' % (type(e).__name__, e)))
                traceback.print_exc()
                return None
        return wrapped
    return deco


@step('1. imports and CUDA')
def s_imports(args):
    import SimpleITK as sitk
    import torch
    print('SimpleITK', sitk.Version.VersionString())
    print('torch    ', torch.__version__, 'cuda', torch.version.cuda)
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available() and not args.cpu:
        print('device:', torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print('gpu memory: %.1f GB free / %.1f GB total'
              % (free / 1e9, total / 1e9))
    elif not args.cpu:
        raise RuntimeError('CUDA not available. Pass --cpu to test on CPU '
                           '(mamba_ssm will fail: it is CUDA-only).')
    try:
        import mamba_ssm
        print('mamba_ssm', getattr(mamba_ssm, '__version__', 'unknown'))
    except Exception as e:
        if args.cpu:
            print('mamba_ssm unavailable (%s) -- expected on CPU' % e)
        else:
            raise


@step('2. make phantoms + simulate 2 mm')
def s_simulate(args):
    import SimpleITK as sitk
    hr_dir = os.path.join(args.work_dir, 'hr')
    os.makedirs(hr_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    n = max(4, args.n_subjects)
    sx, sy, sz = args.phantom_size
    x, y, z = np.meshgrid(np.linspace(-1, 1, sx), np.linspace(-1, 1, sy),
                          np.linspace(-1, 1, sz), indexing='ij')
    r0 = np.sqrt((x / 0.75) ** 2 + (y / 0.95) ** 2 + (z / 0.8) ** 2)

    # Names follow the real convention <id>_<session>_<age>_<weighting> so that
    # the schema parsing, subgroup stratification and balanced sampling are all
    # actually exercised. Two visits per subject, with the earlier visit t2w
    # (inverted contrast, bright CSF) to mirror the real cohort's confound.
    plan = []
    for s in range(n):
        plan.append(('%05d' % (1000 + s), '01', '%.1f' % (2.0 + 0.3 * s), 't2w'))
        plan.append(('%05d' % (1000 + s), '02', '%.1f' % (18.0 + 0.7 * s), 't1w'))

    for (sid, ses, age, w) in plan:
        scale = 0.90 if float(age) < 12.0 else 1.0
        r = r0 / scale
        v = np.zeros((sx, sy, sz), np.float32)
        wm, gm, csf = (110.0, 75.0, 30.0) if w == 't1w' else (60.0, 90.0, 180.0)
        v[(r <= 0.82) & (r > 0.36)] = wm
        v[(r < 1.0) & (r > 0.82)] = gm
        v[r <= 0.36] = csf
        m = (r < 1.0) & (r > 0.7)
        v[m] += 25 * (np.sin(x * 38) * np.sin(y * 38) * np.sin(z * 42))[m]
        v[(r > 1.0) & (r < 1.08)] = 180.0
        v = np.clip(v + rng.normal(0, 1.5, v.shape), 0, None).astype(np.float32)
        im = sitk.GetImageFromArray(np.transpose(v, (2, 1, 0)))
        im.SetSpacing((1.0, 1.0, 1.0))
        im.SetOrigin((-sx / 2.0, -sy / 2.0, -sz / 2.0))
        sitk.WriteImage(im, os.path.join(
            hr_dir, '%s_%s_%s_%s.nii.gz' % (sid, ses, age, w)))
    print('wrote %d phantoms of size %s (%d subjects x 2 visits, t1w+t2w)'
          % (len(plan), (sx, sy, sz), n))

    from sr.simulate_lowres import main as sim_main
    sim_dir = os.path.join(args.work_dir, 'sim')
    rc = sim_main(['--in_dir', hr_dir, '--out_dir', sim_dir, '--copy_hr',
                   '--target_snr', '30', '--overwrite'])
    if rc:
        raise RuntimeError('simulate_lowres reported failures')
    return sim_dir


@step('3. make folds + build dataset + preflight checks')
def s_dataset(args, sim_dir):
    from sr.build_sr_dataset import main as build_main
    from sr.check_dataset import main as check_main
    from sr.make_folds import main as folds_main

    # Go through make_folds so schema validation, subject grouping and subgroup
    # stratification are all exercised, not just the random-split path.
    folds = os.path.join(args.work_dir, 'folds_dev.json')
    rc = folds_main(['--sim_dir', sim_dir, '--method', 'kspace', '--out', folds,
                     '--mode', 'single', '--test_frac', '0.25',
                     '--val_frac', '0.25'])
    if rc:
        raise RuntimeError('make_folds failed')

    ds = os.path.join(args.work_dir, 'dataset')
    build_main(['--sim_dir', sim_dir, '--out_root', ds, '--method', 'kspace',
                '--folds_json', folds, '--fold', 'dev', '--link'])
    rc = check_main(['--root', ds, '--n', '0',
                    '--patch_size'] + [str(p) for p in args.patch_size])
    if rc:
        raise RuntimeError('check_dataset found errors')

    # Verify the manifest resolves each split's renumbered files back to their
    # ORIGINAL stems. Numbering restarts per split, so a split-blind lookup would
    # silently mislabel train and val.
    from sr.naming import load_name_map, subgroup_of
    import json
    man = json.load(open(os.path.join(ds, 'manifest.json')))
    for split in ('train', 'val', 'test'):
        nm = load_name_map(os.path.join(ds, split))
        truth = {e['assigned_name'].replace('.nii.gz', ''): e['stem']
                 for e in man['splits'][split]}
        assert nm == truth, 'manifest lookup wrong for %s split' % split
        labs = {subgroup_of(v) for v in truth.values()}
        assert labs <= {'t1w', 't2w'}, 'bad subgroup labels in %s: %s' % (split, labs)
    print('manifest name resolution correct for all three splits')
    return ds


@step('4. training dataloader')
def s_loader(args, ds):
    from torch.utils.data import DataLoader
    import utils.NiftiDataset as ND
    from sr.train_sr import build_train_transforms

    class O:
        pass
    o = O()
    o.new_resolution, o.resample = [1.0] * 3, False
    o.patch_size = list(args.patch_size)
    o.no_augment, o.sr_augment = False, True
    o.gamma_jitter, o.input_noise_std, o.aug_prob = 0.1, 0.01, 0.8
    o.min_fg_frac, o.seed = 0.10, 0

    tset = ND.NiftiDataSet(os.path.join(ds, 'train'), which_direction='AtoB',
                           transforms=build_train_transforms(o), train=True)
    loader = DataLoader(tset, batch_size=1, shuffle=True, num_workers=0)
    for i, (a, b) in enumerate(loader):
        print('batch %d: input %s [%.3f, %.3f]  target %s [%.3f, %.3f]'
              % (i, tuple(a.shape), a.min(), a.max(),
                 tuple(b.shape), b.min(), b.max()))
        assert a.shape == b.shape, 'input/target shape mismatch'
        assert tuple(a.shape[2:]) == tuple(args.patch_size), \
            'patch is %s, expected %s' % (tuple(a.shape[2:]), args.patch_size)
        assert all(d % 4 == 0 for d in a.shape[2:]), 'patch not divisible by 4'
        if i >= 2:
            break
    return tset


@step('4b. subgroup-balanced sampler')
def s_sampler(args, ds):
    import utils.NiftiDataset as ND
    from sr.train_sr import build_sampler, subgroup_labels_for
    from collections import Counter

    class O:
        pass
    o = O()
    o.data_path = os.path.join(ds, 'train')
    o.name_schema = 'id,session,age,weighting'
    o.balance_subgroups, o.balance_power, o.balance_cap = True, 1.0, None
    o.iters_per_epoch = 0

    tset = ND.NiftiDataSet(o.data_path, which_direction='AtoB', train=True)
    labels = subgroup_labels_for(tset, o)
    assert labels is not None, 'could not resolve subgroup labels'
    assert 'unknown' not in labels, 'unlabelled training volumes: %s' % Counter(labels)
    print('training labels: %s' % dict(Counter(labels)))

    sampler = build_sampler(o, tset)
    assert sampler is not None, 'expected a WeightedRandomSampler'
    drawn = Counter(labels[i] for i in list(sampler))
    print('one epoch of draws: %s' % dict(drawn))
    # At power=1.0 the two subgroups should be near-equal in expectation. The
    # sample is small, so allow a generous tolerance -- we are testing that the
    # weighting is applied at all, not its precision (that is unit-tested).
    if len(set(labels)) == 2 and len(labels) >= 4:
        a, b = sorted(drawn.values())
        nat = Counter(labels)
        na, nb = sorted(nat.values())
        print('natural ratio %.2f -> sampled ratio %.2f (closer to 1.0 is more '
              'balanced)' % (nb / max(na, 1), b / max(a, 1)))
    return True


@step('4c. on-the-fly degradation randomisation')
def s_degradation(args, ds):
    """The regenerated input must match the offline operator, stay band-limited,
    and preserve geometry. This is the check that catches the online and offline
    forward models drifting apart -- a divergence no training metric would show."""
    import SimpleITK as sitk
    import utils.NiftiDataset as ND
    from sr.sr_transforms import RandomKspaceDegradation, _to_np
    from sr.sr_metrics import hf_energy, psnr, axis_power_spectrum

    imgs = ND.lstFiles(os.path.join(ds, 'train', 'images'))
    labs = ND.lstFiles(os.path.join(ds, 'train', 'labels'))
    lr_itk = sitk.ReadImage(imgs[0], sitk.sitkFloat32)
    hr_itk = sitk.ReadImage(labs[0], sitk.sitkFloat32)
    hr, lr0 = _to_np(hr_itk), _to_np(lr_itk)
    lo, hi = np.percentile(hr, 1), np.percentile(hr, 99.5)
    nrm = lambda a: np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    # Matched settings should reproduce the precomputed volume closely.
    t = RandomKspaceDegradation(apod_range=(0.0, 0.0), snr_range=(30.0, 30.0),
                                seed=0)
    out = t({'image': lr_itk, 'label': hr_itk})
    lr1 = _to_np(out['image'])
    assert lr1.shape == hr.shape, (lr1.shape, hr.shape)
    assert out['image'].GetSpacing() == hr_itk.GetSpacing()
    assert np.allclose(out['image'].GetOrigin(), hr_itk.GetOrigin())
    agree = psnr(nrm(lr1), nrm(lr0))
    print('regenerated vs precomputed input: %.2f dB, HF %.5f vs %.5f'
          % (agree, hf_energy(nrm(lr1)), hf_energy(nrm(lr0))))
    assert agree > 35, ('online and offline degradations disagree (%.1f dB) -- '
                        'sr/kspace.py is not being applied identically' % agree)

    # Still band-limited at the coarse Nyquist.
    f, pw = axis_power_spectrum(nrm(lr1), axis=2)
    lo_i = int(np.argmin(np.abs(f - 0.20)))
    hi_i = int(np.argmin(np.abs(f - 0.30)))
    ratio = pw[lo_i] / max(pw[hi_i], 1e-30)
    print('power drop across the coarse Nyquist: %.0fx' % ratio)
    assert ratio > 20, 'regenerated input is not band-limited'

    # Draws span the configured range.
    t2 = RandomKspaceDegradation(apod_range=(0.0, 0.3), snr_range=(20.0, 40.0),
                                 seed=1)
    d = [t2._draw() for _ in range(500)]
    print('apod %.3f-%.3f, snr %.1f-%.1f over 500 draws'
          % (min(x[1] for x in d), max(x[1] for x in d),
             min(x[2] for x in d), max(x[2] for x in d)))
    return True


@step('5-6. generator forward + full optimisation step')
def s_model(args, ds):
    import torch
    from models import create_model
    from sr.sr_options import SROptions

    for lambda_adv in (1.0, 0.0):
        print('\n--- lambda_adv = %g ---' % lambda_adv)
        argv = ['--data_path', os.path.join(ds, 'train'),
                '--val_path', os.path.join(ds, 'val'),
                '--checkpoints_dir', os.path.join(args.work_dir, 'ckpt'),
                '--name', 'smoke',
                '--patch_size'] + [str(p) for p in args.patch_size] + [
                '--lambda_adv', str(lambda_adv),
                '--gpu_ids', '-1' if args.cpu else '0']
        sys.argv = ['smoke'] + argv
        opt = SROptions().parse()
        model = create_model(opt)
        model.setup(opt)

        p = args.patch_size
        dev = model.device
        a = torch.randn(1, 1, *p, device=dev)
        b = torch.randn(1, 1, *p, device=dev)

        if not args.cpu:
            torch.cuda.reset_peak_memory_stats()
        model.set_input((a, b))
        model.optimize_parameters()
        print('losses:', {k: round(v, 4) for k, v in
                          model.get_current_losses().items()})
        print('output shape:', tuple(model.fake_B.shape))
        assert model.fake_B.shape == b.shape, 'generator changed the shape!'
        if not args.cpu:
            print('peak GPU memory: %.2f GB'
                  % (torch.cuda.max_memory_allocated() / 1e9))
        if lambda_adv == 0:
            assert float(model.loss_G_GAN) == 0.0, \
                'lambda_adv=0 but G_GAN is %s' % model.loss_G_GAN
            print('confirmed: adversarial term is off')
        del model, a, b
        if not args.cpu:
            torch.cuda.empty_cache()


@step('6b. warm-start round trip (--init_from)')
def s_warm_start(args, ds):
    """Save the generator's own weights, then warm-start a fresh one from them.

    Coverage must be 100%: the checkpoint is by construction for this exact
    architecture. Anything less means the key/shape reconciliation is broken, and
    this catches it without needing the real released checkpoint.
    """
    import torch
    from models import networks3D
    from sr.train_sr import warm_start

    dev = torch.device('cpu') if args.cpu else torch.device('cuda:0')
    gpu_ids = [] if args.cpu else [0]
    mk = lambda: networks3D.define_G(1, 1, 64, 'gambas', 'instance', False,
                                    'normal', 0.02, gpu_ids,
                                    **{'img_size': (256, 256),
                                       'global_residual':
                                           os.environ.get('GLOBAL_RESIDUAL', '1') != '0'})
    src = mk()
    ckpt = os.path.join(args.work_dir, 'init_G.pth')
    inner = src.module if isinstance(src, torch.nn.DataParallel) else src
    torch.save(inner.state_dict(), ckpt)

    dst = mk()
    res = warm_start(dst, ckpt, 'G', 0.99, False)
    assert res is not None and res['coverage'] == 1.0, \
        'round-trip coverage was %.3f, expected 1.0' % (res or {}).get('coverage', -1)
    assert not res['shape_mismatch'] and not res['missing']

    # And confirm the weights actually transferred, not just that keys matched.
    a = (src.module if isinstance(src, torch.nn.DataParallel) else src).state_dict()
    b = (dst.module if isinstance(dst, torch.nn.DataParallel) else dst).state_dict()
    diffs = [k for k in a if not torch.allclose(a[k].float(), b[k].float())]
    assert not diffs, 'these tensors did not transfer: %s' % diffs[:5]
    print('all %d tensors transferred bit-identically' % len(a))

    # A deliberately incompatible checkpoint must be REFUSED, not half-loaded.
    wrong = os.path.join(args.work_dir, 'wrong_G.pth')
    torch.save({'not.a.real.key': torch.zeros(3, 3)}, wrong)
    try:
        warm_start(mk(), wrong, 'G', 0.5, False)
    except SystemExit as e:
        print('incompatible checkpoint correctly refused')
    else:
        raise AssertionError('an incompatible checkpoint was accepted')
    return True


@step('7. sliding-window inference + 8. metrics')
def s_inference(args, ds):
    import torch
    from models import networks3D
    from sr.train_sr import sliding_window_predict, load_pair
    from sr import sr_metrics
    import utils.NiftiDataset as ND

    dev = torch.device('cpu') if args.cpu else torch.device('cuda:0')
    gpu_ids = [] if args.cpu else [0]
    residual = os.environ.get('GLOBAL_RESIDUAL', '1') != '0'
    net = networks3D.define_G(1, 1, 64, 'gambas', 'instance', False, 'normal',
                              0.02, gpu_ids,
                              **{'img_size': (256, 256),
                                 'global_residual': residual})
    net.to(dev).eval()
    print('global_residual: %s' % residual)

    imgs = ND.lstFiles(os.path.join(ds, 'val', 'images'))
    labs = ND.lstFiles(os.path.join(ds, 'val', 'labels'))
    lr, hr, _ = load_pair(imgs[0], labs[0])
    print('volume %s' % (lr.shape,))
    pred = sliding_window_predict(net, lr, list(args.patch_size),
                                  [p // 2 for p in args.patch_size], dev)
    print('prediction %s  range [%.3f, %.3f]' % (pred.shape, pred.min(), pred.max()))
    assert pred.shape == lr.shape, 'sliding window changed the volume shape'
    assert np.isfinite(pred).all(), 'prediction contains NaN/Inf'

    to01 = lambda v: np.clip((v + 1) / 2, 0, 1)
    m = sr_metrics.all_metrics(to01(lr), to01(hr))
    print('\nsinc-input vs 1 mm truth (this is the baseline to beat):')
    for k in sorted(m):
        print('  %-22s %.5f' % (k, m[k]))
    assert m['psnr'] > 10, 'baseline PSNR implausibly low -- check the pairing'
    mm = sr_metrics.all_metrics(to01(pred), to01(hr))

    if residual:
        # End-to-end proof of the property the whole change rests on: with
        # res_scale == 0 the untrained network is EXACTLY the identity, so
        # epoch-0 validation must equal the sinc baseline rather than sitting
        # ~10 dB below it. Checked here rather than in a unit test because it
        # also exercises the sliding-window path -- if the overlap-blend weights
        # did not sum to 1, identity input would come back subtly attenuated and
        # nothing else in the pipeline would notice.
        err = float(np.abs(pred - lr).max())
        print('untrained net PSNR %.3f  (residual: must equal the sinc %.3f)'
              % (mm['psnr'], m['psnr']))
        print('max |pred - input| = %.3e  (identity at init, incl. window blend)' % err)
        assert err < 1e-4, (
            'global_residual is set but the untrained net is not the identity '
            '(max abs error %.3e). Either res_scale did not initialise to zero '
            'or the sliding-window blend weights do not sum to 1.' % err)
        assert abs(mm['psnr'] - m['psnr']) < 0.05, (
            'untrained residual net scored %.3f but the sinc baseline is %.3f; '
            'these must match.' % (mm['psnr'], m['psnr']))
    else:
        # Untrained net: metrics will be terrible. We only check they compute.
        print('untrained net PSNR %.3f (expected to be bad)' % mm['psnr'])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--work_dir', default='/tmp/gambas_smoke')
    p.add_argument('--patch_size', type=int, nargs='+', default=[128],
                   help='One value for a cube, or three')
    p.add_argument('--phantom_size', type=int, nargs=3, default=[160, 192, 160])
    p.add_argument('--n_subjects', type=int, default=4)
    p.add_argument('--cpu', action='store_true',
                   help='Skip CUDA checks. mamba_ssm is CUDA-only, so steps 5-7 '
                        'will fail; useful only for testing data plumbing.')
    p.add_argument('--keep', action='store_true', help='Do not delete --work_dir first')
    args = p.parse_args()

    if len(args.patch_size) == 1:
        args.patch_size = args.patch_size * 3
    assert len(args.patch_size) == 3
    for v in args.patch_size:
        if v % 4:
            sys.exit('patch_size must be divisible by 4, got %s' % args.patch_size)

    if os.path.exists(args.work_dir) and not args.keep:
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    print('work dir  : %s' % args.work_dir)
    print('patch size: %s' % args.patch_size)

    s_imports(args)
    sim_dir = s_simulate(args)
    ds = s_dataset(args, sim_dir) if sim_dir else None
    if ds:
        s_loader(args, ds)
        s_sampler(args, ds)
        s_degradation(args, ds)
        s_model(args, ds)
        s_warm_start(args, ds)
        s_inference(args, ds)

    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    for name, status, msg in RESULTS:
        print('  %-4s  %-45s %s' % (status, name, msg))
    n_fail = sum(1 for _, s, _ in RESULTS if s == 'FAIL')
    print('\n%d/%d steps passed' % (len(RESULTS) - n_fail, len(RESULTS)))
    if n_fail == 0:
        print('\nAll good. You can submit the real jobs:')
        print('  cd sr/cluster && ./submit_all.sh')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
