#!/usr/bin/env python3
"""
train_sr.py
===========
Training loop for GAMBAS as a 2 mm -> 1 mm super-resolution model.

Differences from the repo's `train.py`, all of which matter on a cluster:

  * SR-safe augmentation (see sr/sr_transforms.py). The default pipeline blurs
    and re-noises the *label*, which directly fights an SR objective.
  * A real validation pass: full-volume sliding-window inference on held-out
    subjects, reporting PSNR / SSIM / L1 against the 1 mm ground truth, plus the
    same metrics for plain sinc interpolation so you always know whether the
    network is actually beating the trivial baseline.
  * `best_net_G.pth` saved on the validation metric, not just the last epoch.
  * Crash/preemption-safe resume: `--continue_train` restores generator,
    discriminator, optimiser state, scheduler state and epoch counter.
  * Deterministic seeding of Python/NumPy/Torch and of the dataloader workers.
  * Optional AMP and gradient accumulation.
  * TensorBoard scalars + mid-slice image grids.

Run:
    python -m sr.train_sr \
        --data_path /data/sr_dataset/train \
        --val_path  /data/sr_dataset/val \
        --checkpoints_dir /scratch/checkpoints \
        --name sr_2mm_to_1mm \
        --patch_size 128 --batch_size 1 --lambda_adv 0.0
"""

import os
import random
import sys
import time

# Allow both `python -m sr.train_sr` and `python sr/train_sr.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

import utils.NiftiDataset as NiftiDataset
from models import create_model
from sr.sr_options import SROptions
from sr import sr_transforms
from sr.checkpoint_utils import (balance_weights, format_balance, format_report,
                                match_state_dict)
from sr.naming import load_name_map, subgroup_of
from sr.sr_metrics import psnr, ssim3d, mae, brain_mask


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Mamba's selective-scan kernels are not deterministic; we do not force
    # torch.use_deterministic_algorithms, we just make the data pipeline
    # reproducible. cudnn.benchmark is a large speed win for fixed patch sizes.
    torch.backends.cudnn.benchmark = True


def worker_init(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def build_train_transforms(opt):
    tf = [NiftiDataset.Resample(opt.new_resolution, opt.resample),
          sr_transforms.PadTo(opt.patch_size, multiple_of=4)]

    input_noise_std = opt.input_noise_std

    if getattr(opt, 'randomize_degradation', False):
        # Must run on the WHOLE volume and BEFORE any cropping: k-space truncation
        # is global, so truncating a patch is a different operator with a
        # different PSF and ringing at the patch edges.
        #
        # It also has to run before SRAugmentation, because it *replaces* the
        # input wholesale -- anything SRAugmentation did to the image would be
        # discarded. And since the re-simulation already draws its own SNR, the
        # augmentation's separate input-noise jitter would double-count: that
        # noise would be full-bandwidth white on the fine grid, which is exactly
        # the artefact the coarse-grid noise placement exists to avoid.
        if input_noise_std > 0:
            print('note: --randomize_degradation draws its own SNR from '
                  '--snr_range, so --input_noise_std (%.3g) is disabled to avoid '
                  'adding a second, full-bandwidth noise term on the fine grid.'
                  % input_noise_std, flush=True)
            input_noise_std = 0.0
        tf.append(sr_transforms.RandomKspaceDegradation(
            target_spacing=opt.target_spacing,
            source_spacing=opt.source_spacing,
            apod_range=tuple(opt.apod_range),
            snr_range=tuple(opt.snr_range),
            modes=tuple(opt.degradation_modes),
            p=opt.degradation_p,
            seed=opt.seed,
            log_first=3))

    if not opt.no_augment:
        if opt.sr_augment:
            g = opt.gamma_jitter
            tf.append(sr_transforms.SRAugmentation(
                flip_axes=(0, 1),
                rot90_axes=(0, 1) if opt.patch_size[0] == opt.patch_size[1] else None,
                gamma_range=(1.0 - g, 1.0 + g),
                gain_range=(0.95, 1.05),
                input_noise_std=input_noise_std,
                p=opt.aug_prob,
                seed=opt.seed))
        else:
            tf.append(NiftiDataset.Augmentation())

    tf.append(sr_transforms.PairedRandomCrop(
        opt.patch_size, min_fg_frac=opt.min_fg_frac, seed=opt.seed))
    return tf


# --------------------------------------------------------------------------- #
# Warm start
# --------------------------------------------------------------------------- #

def warm_start(net, path, tag, min_coverage, allow_partial):
    """Initialise `net` from an arbitrary checkpoint file.

    The repo's own `base_model.load_networks` derives its path from
    <checkpoints_dir>/<name>/<epoch>_net_<X>.pth and loads G and D together, so it
    cannot be pointed at a single downloaded file. This can, and it reports what
    actually landed rather than silently half-loading.
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit('--init_from: no such file: %s' % path)

    target = net.module if isinstance(net, torch.nn.DataParallel) else net
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and not any(
            k.endswith('.weight') for k in state):
        state = state['state_dict']          # some releases nest it
    if hasattr(state, '_metadata'):
        del state._metadata

    ckpt_shapes = {k: tuple(v.shape) for k, v in state.items()
                   if hasattr(v, 'shape')}
    model_shapes = {k: tuple(v.shape) for k, v in target.state_dict().items()}
    res = match_state_dict(ckpt_shapes, model_shapes)
    print(format_report(res, tag), flush=True)

    if res['coverage_guard'] < min_coverage and not allow_partial:
        raise SystemExit(
            'Refusing to warm-start %s: only %.1f%% matched (%.1f%% of tensors, '
            '%.1f%% of parameters), below --init_min_coverage %.2f. Either this '
            'checkpoint is for a different architecture (check --ngf / --input_nc / '
            '--output_nc), or the path is wrong. Pass --init_allow_partial to '
            'proceed anyway.'
            % (tag, 100 * res['coverage_guard'], 100 * res['coverage'],
               100 * res['coverage_numel'], min_coverage))

    to_load = {}
    for ck, mk in res['rename'].items():
        to_load[mk] = state[ck]
    missing, unexpected = target.load_state_dict(to_load, strict=False)
    print('  applied %d tensor(s) from %s' % (len(to_load), path), flush=True)
    return res


# --------------------------------------------------------------------------- #
# Subgroup-balanced sampling
# --------------------------------------------------------------------------- #

def subgroup_labels_for(train_set, opt):
    """Subgroup label per training volume, in dataset order.

    The dataset builder renumbers files to 0.nii.gz, so the label has to come back
    through manifest.json. Returns None when it cannot be resolved, so the caller
    can fall back rather than silently balancing on garbage.
    """
    name_map = load_name_map(opt.data_path,
                             warn=lambda m: print('WARNING: ' + m,
                                                  file=sys.stderr))
    if not name_map:
        print('WARNING: --balance_subgroups needs manifest.json next to '
              '%s to recover the original filenames; none found. Falling back to '
              'unbalanced sampling.' % opt.data_path, file=sys.stderr)
        return None
    labels = []
    for p in train_set.images_list:
        stem_on_disk = os.path.basename(p)
        for e in ('.nii.gz', '.nii'):
            if stem_on_disk.endswith(e):
                stem_on_disk = stem_on_disk[: -len(e)]
                break
        original = name_map.get(stem_on_disk, stem_on_disk)
        labels.append(subgroup_of(original, schema=opt.name_schema))
    if len(set(labels)) < 2:
        print('WARNING: --balance_subgroups found only one subgroup (%s); nothing '
              'to balance.' % set(labels), file=sys.stderr)
        return None
    if 'unknown' in labels:
        n = labels.count('unknown')
        print('WARNING: %d training volume(s) have no subgroup label. Check '
              '--name_schema. They form their own "unknown" stratum.' % n,
              file=sys.stderr)
    return labels


def build_sampler(opt, train_set):
    """Return a DataLoader sampler, or None for plain shuffling."""
    n = len(train_set)
    num_samples = opt.iters_per_epoch if opt.iters_per_epoch > 0 else n

    if opt.balance_subgroups:
        labels = subgroup_labels_for(train_set, opt)
        if labels is not None:
            weights, info = balance_weights(labels, opt.balance_power,
                                            opt.balance_cap)
            print(format_balance(info), flush=True)
            return torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=num_samples, replacement=True)

    if opt.iters_per_epoch > 0:
        # Fixed number of optimiser steps per epoch regardless of volume count.
        return torch.utils.data.RandomSampler(train_set, replacement=True,
                                             num_samples=num_samples)
    return None


# --------------------------------------------------------------------------- #
# Sliding-window full-volume inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sliding_window_predict(net, vol, patch, stride, device, amp_dtype=None):
    """vol: float32 numpy (x, y, z) already scaled to [-1, 1]. Returns same shape.

    Gaussian-weighted blending across overlapping windows, which removes the
    seam artefacts that uniform averaging leaves at patch borders.
    """
    shape = vol.shape
    # Shrink the patch to fit small volumes, but keep every dimension a multiple
    # of 4: the generator has two stride-2 encoder convs and two transposed-conv
    # decoders, so a patch of e.g. 91 comes back out as 92 and the forward pass
    # fails. Round the volume size up to a multiple of 4 first, so we never
    # shrink the patch below what padding will make available.
    patch = [max(4, (min(p, ((n + 3) // 4) * 4) // 4) * 4)
             for p, n in zip(patch, shape)]
    # Pad up to at least one patch, and to a multiple of 4.
    pads = []
    for n, p in zip(shape, patch):
        need = max(0, p - n)
        need += (-(n + need)) % 4
        pads.append((0, need))
    v = np.pad(vol, tuple(pads), mode='edge')
    pshape = v.shape
    assert all(p % 4 == 0 for p in patch), patch
    assert all(n % 4 == 0 for n in pshape), pshape

    acc = np.zeros(pshape, dtype=np.float32)
    wsum = np.zeros(pshape, dtype=np.float32)

    # Separable Gaussian window, sigma = patch/6 (standard for MONAI-style blending)
    def gauss1d(n):
        x = np.arange(n) - (n - 1) / 2.0
        return np.exp(-0.5 * (x / (n / 6.0)) ** 2).astype(np.float32)
    w = (gauss1d(patch[0])[:, None, None]
         * gauss1d(patch[1])[None, :, None]
         * gauss1d(patch[2])[None, None, :])
    w = np.maximum(w, 1e-3)

    def starts(n, p, s):
        if n <= p:
            return [0]
        out = list(range(0, n - p + 1, s))
        if out[-1] != n - p:
            out.append(n - p)
        return out

    autocast = (torch.autocast(device_type='cuda', dtype=amp_dtype)
                if (amp_dtype is not None and device.type == 'cuda')
                else torch.autocast(device_type='cpu', enabled=False))

    for i in starts(pshape[0], patch[0], stride[0]):
        for j in starts(pshape[1], patch[1], stride[1]):
            for k in starts(pshape[2], patch[2], stride[2]):
                blk = v[i:i + patch[0], j:j + patch[1], k:k + patch[2]]
                t = torch.from_numpy(blk[None, None]).to(device)
                with autocast:
                    out = net(t)
                out = out.float().cpu().numpy()[0, 0]
                acc[i:i + patch[0], j:j + patch[1], k:k + patch[2]] += out * w
                wsum[i:i + patch[0], j:j + patch[1], k:k + patch[2]] += w

    pred = acc / np.maximum(wsum, 1e-8)
    sl = tuple(slice(0, n) for n in shape)
    return pred[sl]


def load_pair(img_path, lab_path):
    """Reproduce the dataloader's preprocessing exactly, so validation matches
    training: per-volume Normalization to 0-255, then (x-127.5)/127.5."""
    def prep(p):
        im = sitk.ReadImage(p)
        im = NiftiDataset.Normalization(im)
        im = sitk.Cast(im, sitk.sitkFloat32)
        a = np.abs(sitk.GetArrayFromImage(im))
        a = np.transpose(a, (2, 1, 0)).astype(np.float32)
        return (a - 127.5) / 127.5, im
    a, im_a = prep(img_path)
    b, _ = prep(lab_path)
    return a, b, im_a


@torch.no_grad()
def validate(net, opt, device, epoch, writer=None, amp_dtype=None):
    """Full-volume validation. Returns dict of mean metrics."""
    img_dir = os.path.join(opt.val_path, 'images')
    lab_dir = os.path.join(opt.val_path, 'labels')
    imgs = NiftiDataset.lstFiles(img_dir)
    labs = NiftiDataset.lstFiles(lab_dir)
    if not imgs or len(imgs) != len(labs):
        print('[val] skipped: %d images / %d labels in %s'
              % (len(imgs), len(labs), opt.val_path), flush=True)
        return {}

    n = min(len(imgs), opt.val_max_volumes)
    was_training = net.training
    net.eval()

    rows = []
    out_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_pred', 'epoch_%03d' % epoch)
    if opt.save_val_predictions:
        os.makedirs(out_dir, exist_ok=True)

    for idx in range(n):
        lr, hr, lr_itk = load_pair(imgs[idx], labs[idx])
        if lr.shape != hr.shape:
            print('[val] shape mismatch %s %s vs %s -- skipping'
                  % (os.path.basename(imgs[idx]), lr.shape, hr.shape), flush=True)
            continue
        pred = sliding_window_predict(net, lr, list(opt.val_patch_size),
                                      list(opt.val_stride), device, amp_dtype)
        # Metrics on the [-1, 1] scale mapped back to [0, 1] so PSNR uses
        # data_range = 1 and numbers are comparable to the literature.
        p01 = np.clip((pred + 1) / 2, 0, 1)
        h01 = np.clip((hr + 1) / 2, 0, 1)
        l01 = np.clip((lr + 1) / 2, 0, 1)
        # Metrics are FOREGROUND-MASKED by default. Only ~45-75% of one of these
        # volumes is head; in the air outside it the 2 mm input and the 1 mm
        # target are both ~0 and agree perfectly, which inflates whole-volume
        # PSNR by 1.5-3.5 dB and pushes SSIM toward 1. Selecting the best
        # checkpoint on that means partly selecting on empty space. The mask
        # comes from the ground truth so it is identical for model and baseline.
        # The unmasked numbers are kept as *_whole for comparability with papers
        # that report them (most do, usually without saying so).
        m = brain_mask(h01)
        rows.append({
            'psnr': psnr(p01[m], h01[m]),
            'ssim': ssim3d(p01, h01, mask=m),
            'l1': mae(p01[m], h01[m]),
            'psnr_baseline': psnr(l01[m], h01[m]),
            'ssim_baseline': ssim3d(l01, h01, mask=m),
            'l1_baseline': mae(l01[m], h01[m]),
            'psnr_whole': psnr(p01, h01),
            'psnr_baseline_whole': psnr(l01, h01),
            'fg_frac': float(m.mean()),
        })
        if opt.save_val_predictions:
            out = sitk.GetImageFromArray(np.transpose(p01 * 255.0, (2, 1, 0)))
            out.CopyInformation(lr_itk)
            sitk.WriteImage(out, os.path.join(
                out_dir, os.path.basename(imgs[idx]).replace('.nii', '_pred.nii')))

    if was_training:
        net.train()
    if not rows:
        return {}

    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    # Per-volume paired deltas. The mean delta equals the difference of means,
    # but the RANGE does not follow from it, and it is the useful number here:
    # the volumes differ several-fold in how much content actually sits above
    # the 2 mm Nyquist, so a positive mean can hide volumes made worse.
    d = sorted(r['psnr'] - r['psnr_baseline'] for r in rows)
    print('[val] epoch %d  n=%d  PSNR %.3f (sinc %.3f, %+.3f)  SSIM %.4f (sinc %.4f)  '
          'L1 %.5f  [masked, fg %.0f%%]  per-vol delta %+.2f..%+.2f, %d/%d improved'
          % (epoch, len(rows), means['psnr'], means['psnr_baseline'],
             means['psnr'] - means['psnr_baseline'], means['ssim'],
             means['ssim_baseline'], means['l1'], 100 * means['fg_frac'],
             d[0], d[-1], sum(x > 0 for x in d), len(d)), flush=True)
    if writer is not None:
        for k, v in means.items():
            writer.add_scalar('val/' + k, v, epoch)
    return means


# --------------------------------------------------------------------------- #
# Checkpointing (optimiser + scheduler + epoch, which the repo does not save)
# --------------------------------------------------------------------------- #

def save_trainer_state(model, opt, epoch, best):
    path = os.path.join(opt.checkpoints_dir, opt.name, 'trainer_state.pth')
    torch.save({'epoch': epoch,
                'best': best,
                'optimizers': [o.state_dict() for o in model.optimizers],
                'schedulers': [s.state_dict() for s in getattr(model, 'schedulers', [])]},
               path)


def load_trainer_state(model, opt):
    path = os.path.join(opt.checkpoints_dir, opt.name, 'trainer_state.pth')
    if not os.path.exists(path):
        return 1, None
    st = torch.load(path, map_location='cpu')
    for o, sd in zip(model.optimizers, st.get('optimizers', [])):
        o.load_state_dict(sd)
    for s, sd in zip(getattr(model, 'schedulers', []), st.get('schedulers', [])):
        s.load_state_dict(sd)
    print('resumed trainer state from epoch %d (best=%s)' % (st['epoch'], st.get('best')))
    return st['epoch'] + 1, st.get('best')


# --------------------------------------------------------------------------- #

def main():
    opt = SROptions().parse()
    seed_everything(opt.seed)

    device = (torch.device('cuda:%d' % opt.gpu_ids[0]) if opt.gpu_ids
              else torch.device('cpu'))
    amp_dtype = None
    if opt.amp:
        amp_dtype = torch.bfloat16 if opt.amp_dtype == 'bfloat16' else torch.float16

    # ---- data ---------------------------------------------------------------
    train_set = NiftiDataset.NiftiDataSet(
        opt.data_path, which_direction=opt.which_direction,
        transforms=build_train_transforms(opt), shuffle_labels=False, train=True)
    if len(train_set) == 0:
        sys.exit('No training volumes under %s/{images,labels}' % opt.data_path)
    print('train volumes: %d' % len(train_set), flush=True)

    sampler = build_sampler(opt, train_set)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=opt.workers, pin_memory=True,
                              drop_last=False, worker_init_fn=worker_init,
                              persistent_workers=opt.workers > 0)

    sample = train_set[0]
    print('patch shapes: input %s target %s' % (tuple(sample[0].shape),
                                                tuple(sample[1].shape)), flush=True)

    # ---- model --------------------------------------------------------------
    model = create_model(opt)
    model.setup(opt)

    start_epoch, best = 1, None
    if opt.continue_train:
        # Resuming takes priority over warm-starting: the resumed weights already
        # contain whatever the warm start contributed, and re-applying an
        # initialisation on top of a partly-trained network would discard progress.
        start_epoch, best = load_trainer_state(model, opt)
        if opt.init_from:
            print('note: --continue_train is resuming an existing run, so '
                  '--init_from is ignored.', flush=True)
    elif opt.init_from or opt.init_from_D:
        warm_start(model.netG, opt.init_from, 'G', opt.init_min_coverage,
                   opt.init_allow_partial)
        if opt.init_from_D and hasattr(model, 'netD'):
            warm_start(model.netD, opt.init_from_D, 'D', opt.init_min_coverage,
                       opt.init_allow_partial)
        elif opt.init_from_D:
            print('note: --init_from_D given but this model has no netD.',
                  flush=True)
        if opt.init_from and opt.lambda_adv == 0:
            print('note: lambda_adv=0, so the discriminator is constructed but '
                  'never used; only the generator warm start matters here.',
                  flush=True)

    writer = None
    if opt.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'tb'))
        except ImportError:
            print('tensorboard not installed; continuing without it', flush=True)

    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'train_log.txt')
    total_iters = 0
    n_epochs = opt.niter + opt.niter_decay

    for epoch in range(start_epoch, n_epochs + 1):
        t_epoch = time.time()
        model.netG.train()
        epoch_losses = {}

        for i, data in enumerate(train_loader):
            t_iter = time.time()
            model.set_input(data)
            model.optimize_parameters()
            total_iters += opt.batch_size

            losses = model.get_current_losses()
            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v)

            if total_iters % opt.print_freq == 0:
                msg = ('epoch %d/%d iter %d  %s  %.2fs/it'
                       % (epoch, n_epochs, i,
                          '  '.join('%s %.4f' % (k, v) for k, v in losses.items()),
                          time.time() - t_iter))
                print(msg, flush=True)
                with open(log_path, 'a') as f:
                    f.write(msg + '\n')
                if writer is not None:
                    for k, v in losses.items():
                        writer.add_scalar('train/' + k, v, total_iters)

            if opt.save_latest_freq and total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')
                save_trainer_state(model, opt, epoch - 1, best)

        if writer is not None:
            for k, v in epoch_losses.items():
                writer.add_scalar('train_epoch/' + k, float(np.mean(v)), epoch)

        print('end of epoch %d/%d  %.1fs' % (epoch, n_epochs, time.time() - t_epoch),
              flush=True)

        # ---- validation & checkpoints --------------------------------------
        if opt.val_freq and epoch % opt.val_freq == 0:
            net = model.netG.module if hasattr(model.netG, 'module') else model.netG
            means = validate(net, opt, device, epoch, writer, amp_dtype)
            if means:
                key = {'psnr': 'psnr', 'ssim': 'ssim', 'l1': 'l1'}[opt.val_metric]
                score = means[key]
                better = (best is None or
                          (score > best if opt.val_metric != 'l1' else score < best))
                if better:
                    best = score
                    model.save_networks('best')
                    print('[val] new best %s = %.5f -> best_net_G.pth'
                          % (opt.val_metric, best), flush=True)
                with open(os.path.join(opt.checkpoints_dir, opt.name,
                                       'val_metrics.csv'), 'a') as f:
                    if f.tell() == 0:
                        f.write('epoch,' + ','.join(sorted(means)) + '\n')
                    f.write('%d,' % epoch + ','.join('%.6f' % means[k]
                                                     for k in sorted(means)) + '\n')

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
            save_trainer_state(model, opt, epoch, best)

        model.update_learning_rate()

    model.save_networks('latest')
    save_trainer_state(model, opt, n_epochs, best)
    if writer is not None:
        writer.close()
    print('training complete. best %s = %s' % (opt.val_metric, best), flush=True)


if __name__ == '__main__':
    main()
