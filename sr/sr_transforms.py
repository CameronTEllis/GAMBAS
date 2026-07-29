#!/usr/bin/env python3
"""
sr_transforms.py
================
Augmentations that are safe for a *super-resolution* objective.

Why the repo's default `NiftiDataset.Augmentation` is wrong for this task
------------------------------------------------------------------------
`utils/NiftiDataset.Augmentation` picks one of 8 branches per sample. Several of
them are actively harmful when the target is a 1 mm volume whose high-frequency
content is the whole point:

  * branch 1 (additive Gaussian noise) runs the noise filter on the **label**
    as well as the image. You are then asking the generator to hallucinate a
    specific noise realisation -- unlearnable, and it caps achievable PSNR.
  * branch 2 (RecursiveGaussian blur) blurs the **label**. This destroys exactly
    the frequencies the network is supposed to restore. It is the single worst
    branch for an SR objective.
  * branches 6 and 7 (brightness / contrast) call `np.random.randint` separately
    inside the image call and the label call, so image and label receive
    *different* intensity shifts. The paired intensity relationship is broken.
  * branch 3/4 (rotation, BSpline deformation) resample both volumes with linear
    / BSpline interpolation, which low-pass filters the label. Small, but it is
    a systematic blur of the target on ~2/8 of samples.
  * branch 5 (flip) is a no-op: `NiftiDataset.flipit` builds `img` and then
    returns `image`. It has never flipped anything.

What this module provides instead
---------------------------------
`SRAugmentation`:
  * axis flips and 90-degree in-plane rotations -- exact voxel permutations, so
    zero interpolation blur, applied identically to image and label;
  * an intensity gamma/gain jitter applied with the *same* parameters to image
    and label, so the mapping the network must learn is unchanged;
  * optional acquisition-noise jitter applied to the **input only**, which makes
    the model robust to SNR variation across scanners without corrupting the
    target.

`PairedRandomCrop`: same contract as `NiftiDataset.RandomCrop` but
  * no `np.random.randint(10, ...)` lower bound that crashes when the volume is
    only slightly larger than the patch;
  * foreground fraction is measured on the crop rather than a binarised label,
    so `--min_fg_frac` is interpretable;
  * a bounded retry count, so a mostly-empty volume cannot hang a dataloader
    worker forever.

`CenterCrop` / `PadTo`: deterministic helpers for validation.

All classes take and return the repo's `{'image': sitk.Image, 'label': sitk.Image}`
sample dict, so they drop straight into the `transforms` list that
`NiftiDataSet` already accepts.
"""

import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sr import kspace


def _to_np(img):
    """sitk -> numpy in (x, y, z) order."""
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


def _to_sitk(arr, like):
    out = sitk.GetImageFromArray(np.transpose(np.ascontiguousarray(arr), (2, 1, 0)))
    out.SetSpacing(like.GetSpacing())
    out.SetOrigin(like.GetOrigin())
    out.SetDirection(like.GetDirection())
    return out


class SRAugmentation(object):
    """Interpolation-free geometric augmentation + paired intensity jitter.

    Args:
        flip_axes: axes eligible for random flipping. Default (0, 1) -- left/right
            and anterior/posterior. Flipping the third axis of a brain is
            anatomically implausible, so it is off by default.
        rot90_axes: plane in which to apply random k*90-degree rotations.
            Set to None to disable. Only valid if that plane is square, which is
            true once you are cropping cubic patches.
        gamma_range: (lo, hi) for a random gamma applied to BOTH volumes.
            (1.0, 1.0) disables it.
        gain_range: (lo, hi) multiplicative gain applied to BOTH volumes.
        input_noise_std: std of Gaussian noise added to the INPUT only, as a
            fraction of the input's dynamic range. 0 disables.
        p: probability that any augmentation is applied at all.
    """

    def __init__(self, flip_axes=(0, 1), rot90_axes=(0, 1),
                 gamma_range=(0.9, 1.1), gain_range=(0.95, 1.05),
                 input_noise_std=0.01, p=0.8, seed=None):
        self.name = 'SRAugmentation'
        self.flip_axes = tuple(flip_axes) if flip_axes else ()
        self.rot90_axes = tuple(rot90_axes) if rot90_axes else None
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.input_noise_std = float(input_noise_std)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.rng.random() > self.p:
            return sample

        a = _to_np(image).astype(np.float32)
        b = _to_np(label).astype(np.float32)

        # --- exact geometric ops (no interpolation) --------------------------
        for ax in self.flip_axes:
            if ax < a.ndim and self.rng.random() < 0.5:
                a = np.flip(a, axis=ax)
                b = np.flip(b, axis=ax)

        if self.rot90_axes is not None:
            ax0, ax1 = self.rot90_axes
            if a.shape[ax0] == a.shape[ax1]:
                k = int(self.rng.integers(0, 4))
                if k:
                    a = np.rot90(a, k=k, axes=(ax0, ax1))
                    b = np.rot90(b, k=k, axes=(ax0, ax1))

        # --- paired intensity ops (identical params for input and target) ----
        lo, hi = self.gamma_range
        if hi > lo or lo != 1.0:
            g = float(self.rng.uniform(lo, hi))
            if abs(g - 1.0) > 1e-6:
                def apply_gamma(v):
                    vmin, vmax = float(v.min()), float(v.max())
                    if vmax <= vmin:
                        return v
                    return (((v - vmin) / (vmax - vmin)) ** g
                            * (vmax - vmin) + vmin).astype(np.float32)
                # Same gamma for input and target, so the mapping the network
                # has to learn is unchanged by the augmentation.
                a, b = apply_gamma(a), apply_gamma(b)

        lo, hi = self.gain_range
        if hi > lo:
            gain = float(self.rng.uniform(lo, hi))
            a = a * gain
            b = b * gain

        # --- input-only acquisition noise ------------------------------------
        if self.input_noise_std > 0:
            rng_ = float(a.max() - a.min())
            if rng_ > 0:
                s = float(self.rng.uniform(0, self.input_noise_std)) * rng_
                a = np.sqrt((a + self.rng.normal(0, s, a.shape)) ** 2
                            + self.rng.normal(0, s, a.shape) ** 2).astype(np.float32)

        return {'image': _to_sitk(a, image), 'label': _to_sitk(b, label)}


class RandomKspaceDegradation(object):
    """Regenerate the network INPUT from the TARGET with a random degradation.

    This replaces the precomputed `lowres-<method>/` volume for training samples:
    each time a volume is drawn, the 2 mm acquisition is re-simulated from that
    volume's own 1 mm target with a freshly sampled apodisation and SNR. The
    precomputed files remain what validation and test use, so those stay on a
    single fixed condition.

    Why on the fly rather than pre-generating a grid of variants
    -----------------------------------------------------------
    * The degradation axis becomes continuous instead of a handful of discrete
      points, and every epoch sees new draws.
    * No disk multiplication: a 4-window x 3-SNR grid would be 12 copies of every
      volume.
    * It sidesteps a real hazard. Pre-generated variants share both anatomy AND
      filename across `lowres-*` directories, so a dataset built by unioning them
      needs every variant of one subject forced into the same CV fold. Get that
      wrong and variant-of-X trains while another variant-of-X tests -- a total
      anatomy leak that `check_dataset.py` cannot see, because each pair is
      individually valid.

    Why the WHOLE volume, before cropping
    -------------------------------------
    k-space truncation is a global operation. Truncating a 128^3 patch's k-space
    is a *different* operator: the PSF corresponds to the patch's own Nyquist, and
    the patch edges ring. So this transform must run before `PairedRandomCrop`,
    on the full (padded) volume. That costs 4 volume FFTs per sample, which the
    dataloader workers absorb.

    Args:
        target_spacing / source_spacing: mm, to set the downsampling factors.
        apod_range: (lo, hi) Tukey taper fraction, sampled uniformly. 0 is
            rectangular truncation, 1 is full Hann over the band. The default
            (0, 0.3) is deliberately narrow: the two extremes differ by ~87% in
            mean gradient magnitude, far more than plausible vendor variation, so
            randomising over the full interval would make the model hedge across a
            span reality does not contain.
        snr_range: (lo, hi) target SNR at source resolution, sampled uniformly.
            None or (0, 0) disables noise.
        modes: degradation families to sample from. Defaults to ('kspace',) --
            i.e. the rectangular family with continuous apodisation, which covers
            3D acquisitions. Do NOT include 'gaussian': it is a deliberately wrong
            acquisition model kept only for ablation, and training on it teaches
            the network to invert a blur kernel no scanner produces. Include
            'slab' only if your target protocol is genuinely 2D multi-slice.
        p: probability of re-simulating. With p < 1 the remaining samples keep the
            precomputed input, which mixes the fixed condition back in.
        log_first: print this many draws per worker, so you can confirm from the
            training log that randomisation is actually happening.
    """

    def __init__(self, target_spacing=(2.0, 2.0, 2.0),
                 source_spacing=(1.0, 1.0, 1.0),
                 apod_range=(0.0, 0.3), snr_range=(20.0, 40.0),
                 modes=('kspace',), slice_axis=2, slab_fwhm_factor=1.2,
                 snr_mode='voxel_gain', p=1.0, seed=None, log_first=0):
        self.name = 'RandomKspaceDegradation'
        self.target_spacing = tuple(float(s) for s in target_spacing)
        self.source_spacing = tuple(float(s) for s in source_spacing)
        self.apod_range = tuple(float(a) for a in apod_range)
        self.snr_range = (None if not snr_range
                          else tuple(float(s) for s in snr_range))
        self.modes = tuple(modes) if modes else ('kspace',)
        if 'gaussian' in self.modes:
            print('WARNING: RandomKspaceDegradation was given mode "gaussian", '
                  'which is a deliberately unphysical blur-then-decimate model. '
                  'Training on it injects a blur kernel no scanner produces.',
                  file=sys.stderr)
        self.slice_axis = int(slice_axis)
        self.slab_fwhm_factor = float(slab_fwhm_factor)
        self.snr_mode = snr_mode
        self.p = float(p)
        self.rng = np.random.default_rng(seed)
        self.log_first = int(log_first)
        self._logged = 0

    def _draw(self):
        apod = float(self.rng.uniform(*self.apod_range)) \
            if self.apod_range[1] > self.apod_range[0] else self.apod_range[0]
        snr = 0.0
        if self.snr_range and self.snr_range[1] > 0:
            snr = float(self.rng.uniform(*self.snr_range)) \
                if self.snr_range[1] > self.snr_range[0] else self.snr_range[0]
        mode = self.modes[int(self.rng.integers(0, len(self.modes)))]
        return mode, apod, snr

    def __call__(self, sample):
        label = sample['label']
        if self.rng.random() > self.p:
            return sample

        mode, apod, snr = self._draw()
        factors = np.array(self.target_spacing) / np.array(self.source_spacing)

        hr = _to_np(label).astype(np.float32)
        lr = kspace.degrade(
            hr, factors, mode=mode, apod=apod, snr=snr, rng=self.rng,
            slice_axis=self.slice_axis, slab_fwhm_factor=self.slab_fwhm_factor,
            src_spacing=self.source_spacing, tgt_spacing=self.target_spacing,
            snr_mode=self.snr_mode, magnitude=True, clip_negative=True)

        if self._logged < self.log_first:
            self._logged += 1
            print('[degradation] worker draw %d: mode=%s apod=%.3f snr=%.1f '
                  'shape=%s' % (self._logged, mode, apod, snr, hr.shape),
                  flush=True)

        return {'image': _to_sitk(lr, label), 'label': label}


class PairedRandomCrop(object):
    """Random cubic/cuboid crop applied identically to image and label.

    Args:
        output_size: (x, y, z) patch size.
        min_fg_frac: reject a crop if the fraction of label voxels above
            `fg_threshold_percentile` of the whole-volume intensity is below
            this. 0 accepts everything.
        max_tries: after this many rejections, accept the last crop rather than
            spinning forever.
    """

    def __init__(self, output_size, min_fg_frac=0.10, fg_threshold=None,
                 max_tries=25, seed=None):
        self.name = 'PairedRandomCrop'
        self.output_size = tuple(int(s) for s in output_size)
        assert len(self.output_size) == 3 and all(s > 0 for s in self.output_size)
        self.min_fg_frac = float(min_fg_frac)
        self.fg_threshold = fg_threshold
        self.max_tries = int(max_tries)
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        sx, sy, sz = self.output_size

        # Volumes must already be at least patch-size: chain NiftiDataset.Padding
        # (or PadTo below) before this transform.
        sx, sy, sz = (min(sx, size_old[0]), min(sy, size_old[1]), min(sz, size_old[2]))

        roi = sitk.RegionOfInterestImageFilter()
        roi.SetSize([sx, sy, sz])

        # Threshold for "is this voxel tissue" -- taken once from the label.
        if self.fg_threshold is None:
            lab_np = sitk.GetArrayFromImage(label)
            thr = float(np.percentile(lab_np, 50))
        else:
            thr = float(self.fg_threshold)

        best = None
        best_frac = -1.0
        for _ in range(max(1, self.max_tries)):
            start = [int(self.rng.integers(0, max(1, size_old[i] - self.output_size[i] + 1)))
                     for i in range(3)]
            roi.SetIndex(start)
            lab_crop = roi.Execute(label)
            frac = float(np.mean(sitk.GetArrayFromImage(lab_crop) > thr))
            if frac > best_frac:
                best_frac, best = frac, list(start)
            if frac >= self.min_fg_frac:
                break

        roi.SetIndex(best)
        return {'image': roi.Execute(image), 'label': roi.Execute(label)}


class PadTo(object):
    """Edge-replicate pad so both volumes are at least `output_size`.

    Unlike `NiftiDataset.Padding`, which resamples with a BSpline kernel (and so
    slightly blurs the label), this pads in the array domain and leaves every
    existing voxel bit-identical.
    """

    def __init__(self, output_size, multiple_of=None):
        self.name = 'PadTo'
        self.output_size = tuple(int(s) for s in output_size)
        self.multiple_of = multiple_of

    def _target(self, size):
        tgt = [max(s, o) for s, o in zip(size, self.output_size)]
        if self.multiple_of:
            m = int(self.multiple_of)
            tgt = [int(np.ceil(t / m) * m) for t in tgt]
        return tgt

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        tgt = self._target(image.GetSize())
        if list(image.GetSize()) == tgt:
            return sample

        out = {}
        for key, img in (('image', image), ('label', label)):
            arr = _to_np(img)
            pads = []
            for n, t in zip(arr.shape, tgt):
                total = max(0, t - n)
                pads.append((total // 2, total - total // 2))
            arr = np.pad(arr, tuple(pads), mode='edge')
            new = _to_sitk(arr, img)
            # Shift the origin so the padded voxels sit outside the old FOV
            # rather than silently translating the anatomy.
            idx = [-p[0] for p in pads]
            new.SetOrigin(img.TransformContinuousIndexToPhysicalPoint(
                [float(i) for i in idx]))
            out[key] = new
        return out


class CenterCrop(object):
    """Deterministic centre crop -- use for validation so the metric is stable."""

    def __init__(self, output_size):
        self.name = 'CenterCrop'
        self.output_size = tuple(int(s) for s in output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size = image.GetSize()
        sz = [min(o, s) for o, s in zip(self.output_size, size)]
        start = [(s - o) // 2 for s, o in zip(size, sz)]
        roi = sitk.RegionOfInterestImageFilter()
        roi.SetSize([int(s) for s in sz])
        roi.SetIndex([int(s) for s in start])
        return {'image': roi.Execute(image), 'label': roi.Execute(label)}


class Identity(object):
    def __init__(self):
        self.name = 'Identity'

    def __call__(self, sample):
        return sample
