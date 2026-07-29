#!/usr/bin/env python3
"""
kspace.py
=========
The 2 mm acquisition forward model, as pure NumPy. Shared by
`sr/simulate_lowres.py` (offline, writes files) and
`sr/sr_transforms.RandomKspaceDegradation` (online, regenerates the training
input per sample). One implementation so the two cannot drift apart -- if the
offline and online degradations differ, the model is trained on one operator and
evaluated against another, which is undetectable from the metrics.

Continuous apodisation
----------------------
`apod` is the **Tukey taper fraction** applied across the retained k-space band:

    apod = 0.0   rectangular truncation. Sinc PSF, full Gibbs ringing.
                 Equivalent to the old `--mode kspace`.
    apod = 1.0   Hann across the whole band. Minimal ringing, blurrier.
                 Equivalent to the old `--mode kspace_hann`.
    0 < apod < 1 taper only the outer `apod` fraction of the band.

Those endpoints are exact, not approximate: a Tukey window with taper fraction 1
IS the raised cosine `0.5 * (1 + cos(pi * f / k_max))` that `kspace_hann` used, and
with taper fraction 0 it is the indicator function that `kspace` used. So the two
original discrete modes are now the ends of one continuous axis, which is what
makes randomising over it meaningful: real vendor apodisation sits somewhere
between them, not at one end.

Measured on a phantom, those endpoints are ~87% apart in mean gradient magnitude
and ~3.9 dB apart in baseline PSNR. That is a much wider span than plausible
vendor variation, which is why the recommended randomisation range is narrow
(apod ~ U[0, 0.3]) rather than the full interval.

Where the noise goes
--------------------
Rician noise is added on the **native coarse grid**, before the volume is put back
on the fine grid. That ordering matters and is not cosmetic:

  * A real scanner's noise is band-limited to the acquired band. Adding noise on
    the fine grid instead would give full-bandwidth white noise, handing the
    network a high-frequency component that no 2 mm acquisition contains -- and
    since the task is precisely to synthesise high frequencies, that is a
    corrupting cue rather than a harmless one.
  * Magnitude (Rician) combination is nonlinear, so it does not commute with
    interpolation. It has to happen where the scanner does it.
"""

import hashlib

import numpy as np


# --------------------------------------------------------------------------- #
# Deterministic seeding
# --------------------------------------------------------------------------- #

def stable_seed(*parts):
    """A 32-bit seed derived from `parts`, stable across processes and machines.

    Do NOT build seeds from Python's `hash()`. String hashing is salted per
    process (PEP 456), so `hash(('seed', subject_id))` returns a different value
    on every invocation. A `--seed` flag built on it silently does nothing, which
    is the worst kind of reproducibility bug: the pipeline looks seeded and is not.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode('utf-8'))
        h.update(b'\x00')
    return int.from_bytes(h.digest()[:4], 'little')


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

def _centered_freq(n):
    """Normalised frequency in [-0.5, 0.5) for an fftshifted axis."""
    return np.fft.fftshift(np.fft.fftfreq(n))


def _tukey_profile(f, k_max, apod):
    """Tukey (tapered-cosine) profile over |f| <= k_max, zero outside.

    apod is the taper fraction: 0 -> rectangular, 1 -> Hann over the band.
    """
    keep = np.abs(f) <= k_max + 1e-12
    prof = np.zeros_like(f, dtype=np.float64)
    if apod <= 0:
        prof[keep] = 1.0
        return prof
    apod = min(float(apod), 1.0)
    r = np.abs(f[keep]) / (k_max + 1e-12)
    t = np.ones_like(r)
    edge = r > (1.0 - apod)
    t[edge] = 0.5 * (1.0 + np.cos(np.pi * (r[edge] - (1.0 - apod)) / apod))
    prof[keep] = t
    return prof


def _gaussian_profile(f, k_max, frac, fwhm_factor=1.0):
    """Gaussian PSF of FWHM = fwhm_factor * target voxel size, band-limited.

    Used for the slice-select direction of a 2D multi-slice acquisition, and for
    the `gaussian` ablation mode.
    """
    sigma_x = (fwhm_factor / frac) / 2.3548200450309493
    prof = np.exp(-2.0 * (np.pi * sigma_x * f) ** 2)
    return prof * (np.abs(f) <= k_max + 1e-12)


def kspace_window(shape, keep_fraction, mode='kspace', slice_axis=2,
                  slab_fwhm_factor=1.2, apod=None):
    """Separable multiplicative k-space window, fftshifted (DC at centre).

    keep_fraction[i] = source_spacing[i] / target_spacing[i], i.e. the fraction of
    the source band that the coarse acquisition retains along axis i.

    `mode` selects the family; `apod` overrides the taper fraction for the
    rectangular family. The legacy modes map onto apod as:
        kspace       -> apod 0.0
        kspace_hann  -> apod 1.0
    """
    if apod is None:
        apod = 1.0 if mode == 'kspace_hann' else 0.0

    win = np.ones(shape, dtype=np.float64)
    for ax, (n, frac) in enumerate(zip(shape, keep_fraction)):
        f = _centered_freq(n)
        k_max = 0.5 * frac

        if mode == 'gaussian':
            prof = _gaussian_profile(f, k_max, frac, 1.0)
        elif mode == 'slab' and ax == slice_axis:
            prof = _gaussian_profile(f, k_max, frac, slab_fwhm_factor)
        else:
            # 'kspace', 'kspace_hann', and the in-plane axes of 'slab'
            prof = _tukey_profile(f, k_max, apod)

        shp = [1] * len(shape)
        shp[ax] = n
        win = win * prof.reshape(shp)
    return win.astype(np.float32)


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #

def crop_kspace(K, target_shape):
    """Centre-crop an fftshifted k-space array to `target_shape`."""
    sl = []
    for n, m in zip(K.shape, target_shape):
        start = (n - m) // 2
        sl.append(slice(start, start + m))
    return K[tuple(sl)]


def zerofill_kspace(K, target_shape):
    """Centre-pad an fftshifted k-space array to `target_shape` (sinc interp)."""
    pads = []
    for n, m in zip(K.shape, target_shape):
        total = m - n
        pads.append((total // 2, total - total // 2))
    return np.pad(K, tuple(pads), mode='constant')


def pad_to_even_multiple(arr, factors):
    """Edge-replicate pad so each axis is even and divides by its factor.

    Edge replication rather than zeros: a hard step at the FOV boundary would ring
    across the whole volume once k-space is truncated.
    """
    before, after = [], []
    for n, f in zip(arr.shape, factors):
        f = int(max(1, round(f)))
        rem = (-n) % (2 * f)
        before.append(rem // 2)
        after.append(rem - rem // 2)
    if any(before) or any(after):
        arr = np.pad(arr, tuple(zip(before, after)), mode='edge')
    return arr, before, after


def target_shape_for(padded_shape, factors):
    """Even coarse-grid shape implied by `factors`."""
    ts = tuple(int(round(n / f)) for n, f in zip(padded_shape, factors))
    return tuple(n if n % 2 == 0 else n - 1 for n in ts)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #

def foreground_mean(arr, percentile=60):
    """Rough signal level: mean of voxels above `percentile`."""
    flat = arr[np.isfinite(arr)]
    if flat.size == 0:
        return 0.0
    thr = np.percentile(flat, percentile)
    fg = flat[flat > thr]
    if fg.size < 100:
        fg = flat
    return float(np.mean(fg))


def add_rician(arr, sigma, rng):
    """|(S + n_r) + i n_i| with n ~ N(0, sigma). MRI magnitude images are Rician."""
    if sigma <= 0:
        return arr
    real = arr + rng.normal(0.0, sigma, arr.shape)
    imag = rng.normal(0.0, sigma, arr.shape)
    return np.sqrt(real ** 2 + imag ** 2).astype(np.float32)


def sigma_for_snr(arr, snr, src_spacing, tgt_spacing, snr_mode='voxel_gain'):
    """Noise sigma achieving `snr` (foreground mean / sigma) at SOURCE resolution.

    With snr_mode='voxel_gain' the coarse acquisition is then given the SNR
    benefit of its larger voxels (SNR scales with voxel volume), which is what a
    real 2 mm scan gets for free. 'fixed' applies the number literally.
    """
    if not snr or snr <= 0:
        return 0.0
    sigma = foreground_mean(arr) / float(snr)
    if snr_mode == 'voxel_gain':
        gain = float(np.prod(np.asarray(tgt_spacing, dtype=float))
                     / np.prod(np.asarray(src_spacing, dtype=float)))
        sigma /= gain
    return float(sigma)


# --------------------------------------------------------------------------- #
# Full forward model
# --------------------------------------------------------------------------- #

def degrade(arr, factors, mode='kspace', apod=None, snr=0.0, rng=None,
            slice_axis=2, slab_fwhm_factor=1.2, src_spacing=(1.0, 1.0, 1.0),
            tgt_spacing=(2.0, 2.0, 2.0), snr_mode='voxel_gain',
            magnitude=True, clip_negative=True, return_native=False):
    """Full 1 mm -> coarse -> 1 mm-grid forward model on a numpy array.

    `arr` is (x, y, z) float. Returns the degraded volume on the SAME grid as
    `arr` (sinc / zero-fill interpolated back), which is what the network
    consumes. With `return_native=True` also returns the true coarse-grid volume.

    This is the operator `simulate_lowres.py` applies offline; calling it here per
    training sample guarantees the online and offline degradations are identical
    apart from the random draw.
    """
    if rng is None:
        rng = np.random.default_rng()
    orig_shape = arr.shape
    factors = np.asarray(factors, dtype=float)

    arr_p, pad_b, _ = pad_to_even_multiple(np.asarray(arr, dtype=np.float32),
                                           factors)
    padded_shape = arr_p.shape
    tgt = target_shape_for(padded_shape, factors)
    keep_fraction = [m / n for m, n in zip(tgt, padded_shape)]

    info = {'padded_shape': list(padded_shape),
            'pad_before': list(pad_b),
            'lowres_shape': list(tgt),
            'keep_fraction': list(keep_fraction),
            'mode': mode,
            'apod': (1.0 if mode == 'kspace_hann' else 0.0) if apod is None
                    else float(apod)}

    K = np.fft.fftshift(np.fft.fftn(arr_p))
    K = K * kspace_window(padded_shape, keep_fraction, mode, slice_axis,
                          slab_fwhm_factor, apod)

    # Coarse grid. numpy's ifftn normalises by 1/N, and N differs between grids,
    # so rescale to preserve absolute intensity.
    K_low = crop_kspace(K, tgt)
    scale = float(np.prod(tgt)) / float(np.prod(padded_shape))
    low = np.real(np.fft.ifftn(np.fft.ifftshift(K_low))) * scale
    if magnitude:
        low = np.abs(low)

    # Noise belongs HERE -- on the coarse grid, before interpolation.
    info['foreground_mean_prenoise'] = foreground_mean(low)
    sigma = sigma_for_snr(low, snr, src_spacing, tgt_spacing, snr_mode)
    info['noise_sigma'] = float(sigma)
    if sigma > 0:
        low = add_rician(low, sigma, rng)
    if clip_negative:
        low = np.clip(low, 0, None)

    # Back onto the fine grid by zero-filling: adds no new frequency content.
    Kl = np.fft.fftshift(np.fft.fftn(low))
    up = np.real(np.fft.ifftn(np.fft.ifftshift(zerofill_kspace(Kl, padded_shape))))
    up = up * (float(np.prod(padded_shape)) / float(np.prod(tgt)))
    sl = tuple(slice(b, b + n) for b, n in zip(pad_b, orig_shape))
    up = up[sl]
    if magnitude:
        up = np.abs(up)
    if clip_negative:
        up = np.clip(up, 0, None)
    up = up.astype(np.float32)

    if return_native:
        return up, low.astype(np.float32), info
    return up
