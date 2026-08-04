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

Vendor apodisation (GE Fermi) -- the physical default
-----------------------------------------------------
`mode='fermi'` (the default) reproduces the radial Fermi window GE applies to the
acquired k-space during reconstruction, translated directly from the vendor's
`fermi_win.m`:

    dv  = sqrt( sum_i (f_i / k_nyq_i)^2 )     # radius, in units of the coarse Nyquist
    win = 1 / (1 + exp((dv - p1) / p2))       # Fermi-Dirac roll-off

`p1` is the corner radius (the 0.5 point, in units of the coarse Nyquist) and `p2`
the transition width; GE typically uses p1/p2 ~ 0.9/0.1 or 0.8/0.2. This matters
for FIDELITY: a real native 2 mm GE acquisition is Fermi-apodised, so a rectangular
(brick-wall) truncation would imprint Gibbs ringing a real 2 mm scan does not have,
and the simulator would then teach the network to de-ring an artefact absent at
inference. The Fermi suppresses that ringing and, like the scanner, slightly lowers
the effective resolution near the cutoff. Being radial it also rolls off the
k-space corners, matching GE's recon.

Do NOT confuse this with the de-ZIP: `dezip_kspace.py` recovers data GE has ALREADY
Fermi-filtered at the native matrix, so it applies no window. Here we synthesise a
NEW, coarser acquisition that GE would itself apodise, so the window belongs.

Continuous apodisation (Tukey) -- ablation / alternative
--------------------------------------------------------
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


def _fermi_radial(shape, keep_fraction, p1, p2):
    """Radial (elliptical) Fermi window -- a faithful port of GE's fermi_win.m.

    The MATLAB routine computes, over the acquired k-space,

        dv  = |dm| / x                 (or |Re(dm)/Re(x) + i Im(dm)/Im(x)| for 2D)
        win = 1 / (1 + exp((dv - p1)/p2))

    where `dm` runs -x:x across the matrix, so `x` is the acquired half-matrix and
    `dv` is the normalised radius (0 at DC, 1 at the matrix edge). The complex form
    is just a per-axis normalisation, i.e. an elliptical radius. We reproduce that
    exactly on the FINE grid we are about to crop: for each axis the coarse Nyquist
    sits at |f_i| = k_nyq_i = 0.5 * keep_fraction[i], so f_i / k_nyq_i plays the role
    of dm/x and hits +-1 at the acquired matrix edge. The 3D radius extends the 2D
    ellipse of the .m file to the (3D) anatomical acquisitions.

        p1  corner radius (the 0.5 crossing), in units of the coarse Nyquist
        p2  transition width; smaller = sharper roll-off

    Being radial, the window also drives the k-space corners (dv up to sqrt(3))
    to ~0, which is what GE's radial apodisation does.
    """
    coords = []
    for n, frac in zip(shape, keep_fraction):
        f = _centered_freq(n)
        k_nyq = 0.5 * frac + 1e-12         # coarse Nyquist on the fine grid
        coords.append(f / k_nyq)           # -> +-1 at the acquired matrix edge
    grids = np.meshgrid(*coords, indexing='ij')
    dv = np.sqrt(sum(g ** 2 for g in grids))
    return 1.0 / (1.0 + np.exp((dv - float(p1)) / float(p2)))


def _gaussian_profile(f, k_max, frac, fwhm_factor=1.0):
    """Gaussian PSF of FWHM = fwhm_factor * target voxel size, band-limited.

    Used for the slice-select direction of a 2D multi-slice acquisition, and for
    the `gaussian` ablation mode.
    """
    sigma_x = (fwhm_factor / frac) / 2.3548200450309493
    prof = np.exp(-2.0 * (np.pi * sigma_x * f) ** 2)
    return prof * (np.abs(f) <= k_max + 1e-12)


def kspace_window(shape, keep_fraction, mode='fermi', slice_axis=2,
                  slab_fwhm_factor=1.2, apod=None,
                  fermi_p1=0.9, fermi_p2=0.1):
    """Multiplicative k-space window, fftshifted (DC at centre).

    keep_fraction[i] = source_spacing[i] / target_spacing[i], i.e. the fraction of
    the source band that the coarse acquisition retains along axis i.

    `mode` selects the family:
        fermi        -> radial GE Fermi window (the physical default); fermi_p1/p2
        kspace       -> rectangular, i.e. Tukey apod 0.0
        kspace_hann  -> Hann, i.e. Tukey apod 1.0
        gaussian/slab-> Gaussian PSF families (2D / slice-select ablations)
    `apod` overrides the Tukey taper fraction for the rectangular family.

    The Fermi window is radial, not separable, so it is built and returned whole
    rather than as a per-axis product.
    """
    if mode == 'fermi':
        return _fermi_radial(shape, keep_fraction, fermi_p1, fermi_p2).astype(
            np.float32)

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

def degrade(arr, factors, mode='fermi', apod=None, snr=0.0, rng=None,
            slice_axis=2, slab_fwhm_factor=1.2, src_spacing=(1.0, 1.0, 1.0),
            tgt_spacing=(2.0, 2.0, 2.0), snr_mode='voxel_gain',
            magnitude=True, clip_negative=True, return_native=False,
            fermi_p1=0.9, fermi_p2=0.1):
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
            'mode': mode}
    if mode == 'fermi':
        info['apod'] = None                     # keep the key present downstream
        info['fermi_p1'] = float(fermi_p1)
        info['fermi_p2'] = float(fermi_p2)
    else:
        info['apod'] = ((1.0 if mode == 'kspace_hann' else 0.0)
                        if apod is None else float(apod))

    K = np.fft.fftshift(np.fft.fftn(arr_p))
    K = K * kspace_window(padded_shape, keep_fraction, mode, slice_axis,
                          slab_fwhm_factor, apod, fermi_p1, fermi_p2)

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
