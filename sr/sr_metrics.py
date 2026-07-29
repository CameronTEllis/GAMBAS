#!/usr/bin/env python3
"""
sr_metrics.py
=============
Dependency-light image-quality metrics for 3D volumes. Pure NumPy + SciPy so it
runs in the same environment as the rest of the pipeline without pulling in
scikit-image.

All functions take arrays in the same shape and assume values in [0, 1] unless a
`data_range` is given.

Notes on interpretation for super-resolution
--------------------------------------------
  * PSNR rewards the conditional mean. A pure-L1 model will beat a GAN model on
    PSNR essentially always. That does not make the L1 model better-looking.
  * SSIM is computed here with the standard 11-voxel Gaussian window
    (sigma = 1.5) extended to 3D, matching skimage's defaults.
  * `sharpness` (mean gradient magnitude) and `hf_energy` (fraction of spectral
    energy above the 2 mm Nyquist) are the metrics that actually distinguish
    "restored detail" from "smooth interpolation". A model that scores well on
    PSNR but has hf_energy close to the sinc baseline has not learned anything
    beyond deblurring.
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def _as_f64(x):
    return np.asarray(x, dtype=np.float64)


def mae(a, b):
    return float(np.mean(np.abs(_as_f64(a) - _as_f64(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((_as_f64(a) - _as_f64(b)) ** 2)))


def psnr(a, b, data_range=1.0):
    mse = np.mean((_as_f64(a) - _as_f64(b)) ** 2)
    if mse <= 0:
        return float('inf')
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim3d(a, b, data_range=1.0, sigma=1.5, k1=0.01, k2=0.03, mask=None):
    """3D SSIM with a Gaussian window, matching skimage's gaussian_weights=True.

    If `mask` is given, the SSIM map is averaged over the mask only, which is
    the right thing to do for brain images -- otherwise a large air background
    (where both volumes are near-identical zeros) inflates the score.
    """
    a, b = _as_f64(a), _as_f64(b)
    if not _HAVE_SCIPY:
        # Fall back to a global (non-windowed) SSIM. Coarser, but no dependency.
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
        return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                     / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))

    f = lambda x: gaussian_filter(x, sigma, truncate=3.5, mode='nearest')
    mu_a, mu_b = f(a), f(b)
    saa = f(a * a) - mu_a * mu_a
    sbb = f(b * b) - mu_b * mu_b
    sab = f(a * b) - mu_a * mu_b
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    smap = (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        return float(smap[m].mean()) if m.any() else float('nan')
    return float(smap.mean())


def sharpness(a):
    """Mean gradient magnitude -- a simple, monotone proxy for perceived detail."""
    a = _as_f64(a)
    g = np.gradient(a)
    return float(np.mean(np.sqrt(sum(gi ** 2 for gi in g))))


def hf_energy(a, cutoff=0.25):
    """Fraction of total spectral power at |k| above `cutoff` cycles/voxel.

    With 1 mm voxels, a 2 mm acquisition measures nothing above 0.25
    cycles/voxel. So `hf_energy(pred, 0.25)` is literally "how much energy did
    the model put where the scanner had none". Compare it against the ground
    truth's value: matching it is the goal, exceeding it means the model is
    over-sharpening / inventing texture.
    """
    a = _as_f64(a)
    A = np.fft.fftn(a - a.mean())
    P = np.abs(A) ** 2
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in a.shape], indexing='ij')
    kr = np.sqrt(sum(g ** 2 for g in grids))
    tot = P.sum()
    if tot <= 0:
        return 0.0
    return float(P[kr > cutoff].sum() / tot)


def radial_power_spectrum(a, nbins=64):
    """Return (freq_centres, mean power) -- useful for a figure showing that the
    model restores the band the 2 mm acquisition threw away."""
    a = _as_f64(a)
    P = np.abs(np.fft.fftn(a - a.mean())) ** 2
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in a.shape], indexing='ij')
    kr = np.sqrt(sum(g ** 2 for g in grids)).ravel()
    P = P.ravel()
    edges = np.linspace(0, kr.max(), nbins + 1)
    idx = np.clip(np.digitize(kr, edges) - 1, 0, nbins - 1)
    sums = np.bincount(idx, weights=P, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        prof = sums / np.maximum(counts, 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, prof


def axis_power_spectrum(a, axis=0, nbins=None):
    """Power spectrum along ONE axis, averaged over the other two.

    This is the diagnostic to use when checking a rectangular k-space forward
    model. `radial_power_spectrum` bins by |k|, and a radial shell at |k| = 0.3
    contains directions whose individual components are all below 0.25 -- so a
    rectangular (per-axis) truncation at 0.25 does NOT produce a cliff in the
    radial profile. Along a single axis it does, exactly at the target Nyquist.

    Returns (freq, power) for the non-negative half of the axis.
    """
    a = _as_f64(a)
    A = np.fft.fft(a - a.mean(), axis=axis)
    P = (np.abs(A) ** 2)
    other = tuple(i for i in range(a.ndim) if i != axis)
    prof = P.mean(axis=other)
    n = a.shape[axis]
    f = np.fft.fftfreq(n)
    half = f >= 0
    f, prof = f[half], prof[half]
    order = np.argsort(f)
    f, prof = f[order], prof[order]
    if nbins:
        edges = np.linspace(0, f.max(), nbins + 1)
        idx = np.clip(np.digitize(f, edges) - 1, 0, nbins - 1)
        sums = np.bincount(idx, weights=prof, minlength=nbins)
        counts = np.bincount(idx, minlength=nbins)
        prof = sums / np.maximum(counts, 1)
        f = 0.5 * (edges[:-1] + edges[1:])
    return f, prof


def brain_mask(a, percentile=55):
    """Crude foreground mask for masked metrics. Replace with a real skull-strip
    (HD-BET, SynthStrip) if you are reporting numbers in a paper."""
    a = _as_f64(a)
    thr = np.percentile(a, percentile)
    m = a > thr
    if _HAVE_SCIPY:
        from scipy.ndimage import binary_closing, binary_fill_holes, label
        m = binary_closing(m, np.ones((3, 3, 3)))
        m = binary_fill_holes(m)
        lab, n = label(m)
        if n > 1:  # keep the largest connected component
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = lab == sizes.argmax()
    return m


def all_metrics(pred, target, mask=None, data_range=1.0):
    out = {
        'psnr': psnr(pred, target, data_range),
        'ssim': ssim3d(pred, target, data_range, mask=mask),
        'mae': mae(pred, target),
        'rmse': rmse(pred, target),
        'sharpness_pred': sharpness(pred),
        'sharpness_target': sharpness(target),
        'hf_energy_pred': hf_energy(pred),
        'hf_energy_target': hf_energy(target),
    }
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.any():
            out['psnr_masked'] = psnr(np.asarray(pred)[m], np.asarray(target)[m],
                                      data_range)
            out['mae_masked'] = mae(np.asarray(pred)[m], np.asarray(target)[m])
    return out
