#!/usr/bin/env python3
"""
simulate_lowres.py
==================
Simulate a *natively acquired* low-resolution (default 2 mm isotropic) anatomical
volume from a high-resolution (default 1 mm isotropic) source volume.

Why not just `resample --spacing 2 2 2`?
---------------------------------------
Trilinear/BSpline downsampling of an image on a fine grid is NOT what an MRI
scanner does. A scanner samples a finite region of k-space; the resulting voxel
is the convolution of the object with a *point spread function* (PSF) whose shape
is set by the acquisition, and the image is then sampled on the coarse grid. The
faithful forward model in image space is therefore:

    I_lowres = D_2mm { PSF_2mm * I_object } + noise

Since we only have I_1mm (already band-limited to the 1 mm Nyquist), the closest
achievable approximation is to reproduce the *bandwidth* and *PSF* of the 2 mm
acquisition, which is exactly what truncating k-space does. This script:

  1. FFTs the 1 mm volume.
  2. Applies the readout/phase-encode PSF as a k-space filter.
       - `--mode kspace`  : hard rectangular truncation to the 2 mm Nyquist band.
                            Equivalent to a sinc PSF in image space -> produces
                            the Gibbs ringing that real MRI shows at tissue
                            boundaries. This is the default and the most
                            physically defensible.
       - `--mode kspace_hann`: truncation with a Hann (raised-cosine) apodisation,
                            mimicking vendor filtering. Less ringing, slightly
                            blurrier.
       - `--mode slab`     : rectangular truncation in-plane, but a Gaussian slab
                            profile along `--slice_axis` with FWHM =
                            `--slab_fwhm_factor` x slice thickness. Models a 2D
                            multi-slice acquisition, where the slice-select RF
                            pulse gives a much broader (and non-sinc) PSF than
                            the frequency-encoded directions. Use this if your
                            target protocol is 2D, not 3D.
       - `--mode gaussian` : classic "blur then decimate" baseline. Included only
                            for ablation; it is *not* a good acquisition model.
  3. Crops k-space (or IFFTs and resamples) onto the coarse grid so that the
     result is a genuine 2 mm volume with no interpolation applied afterwards.
  4. Optionally adds Rician noise at a target SNR, and optionally scales the
     noise by the theoretical SNR gain of larger voxels (see --snr_mode).

Limitations we cannot fix without raw data (documented honestly):
  * Partial-Fourier, parallel-imaging (GRAPPA/SENSE) g-factor noise
    amplification, and elliptical k-space shutters are not modelled.
  * Motion during the (longer or shorter) acquisition is not modelled.
  * Contrast differences from a different TR/TE/flip angle at the coarser
    protocol are not modelled -- the tissue contrast is inherited from the 1 mm
    source.
  * The 1 mm source is itself already PSF-blurred at 1 mm, so our 2 mm output is
    marginally blurrier than a true 2 mm acquisition. This is unavoidable.
  * B1/gradient nonlinearity distortion differences are not modelled.

Outputs
-------
For an input `X.nii.gz` and `--out_dir DIR`:
  DIR/lowres_native/X.nii.gz   -- true 2 mm isotropic grid
  DIR/lowres_on_hr_grid/X.nii.gz -- the 2 mm volume put back on the exact 1 mm
                                    grid of the input (sinc/zero-fill upsampled
                                    by default). This is what GAMBAS trains on,
                                    because its dataloader crops the input and
                                    the target with identical voxel indices, so
                                    they must share a grid.
  DIR/hr/X.nii.gz              -- (with --copy_hr) the untouched 1 mm target,
                                    conformed if --conform was requested.
  DIR/qc/X.json                -- provenance + measured statistics.

Usage
-----
Single file:
    python -m sr.simulate_lowres --in_file /data/hr/sub-01_T1w.nii.gz \
        --out_dir /data/sr_dataset --copy_hr

Whole folder (serial):
    python -m sr.simulate_lowres --in_dir /data/hr --out_dir /data/sr_dataset --copy_hr

One shard of a SLURM array (0-indexed):
    python -m sr.simulate_lowres --in_dir /data/hr --out_dir /data/sr_dataset \
        --copy_hr --shard_index $SLURM_ARRAY_TASK_ID --num_shards $SLURM_ARRAY_TASK_COUNT
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared so there is ONE definition of "what is this file's stem". The local copy
# used to fall back to os.path.splitext, which mangles stems containing a decimal
# point -- and the age field is months with a decimal (e.g. 2.2).
from sr import kspace
from sr.kspace import stable_seed
from sr.naming import strip_ext


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #

NIFTI_EXTS = ('.nii', '.nii.gz', '.mgz', '.mha', '.mhd', '.nrrd')


def numerical_sort(value):
    parts = re.compile(r'(\d+)').split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def list_volumes(path):
    """Return a sorted list of image files directly under `path` (recursive)."""
    out = []
    for dirname, _, filelist in os.walk(path):
        for fn in filelist:
            if fn.startswith('.'):
                continue
            low = fn.lower()
            if any(low.endswith(e) for e in NIFTI_EXTS):
                out.append(os.path.join(dirname, fn))
    return sorted(out, key=numerical_sort)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def method_tag_of(args):
    """Short name identifying this simulation configuration."""
    if args.method_tag:
        return args.method_tag
    return args.mode


def output_dirs(out_dir, layout, tag):
    """Where each product goes.

    `method` layout (default) keeps one directory per simulation method as a
    sibling of your `originals/`, so you can generate several methods side by
    side and compare them without re-running anything:

        <out_dir>/lowres-<tag>/          2 mm on the 1 mm grid  <- training input
        <out_dir>/lowres-<tag>-native/   the true 2 mm volume
        <out_dir>/hr/                    reoriented 1 mm target (--copy_hr)
        <out_dir>/qc-<tag>/              per-volume provenance JSON

    `hr/` has no tag because it does not depend on the method. It is still
    written per run because --reorient / --conform change the target's grid, and
    the training pair must share a grid exactly -- pairing the simulated input
    against your untouched `originals/` would silently mismatch direction
    cosines. Point build_sr_dataset.py at this `hr/`, not at `originals/`.

    `flat` layout is the original scheme (lowres_on_hr_grid/, lowres_native/).
    """
    if layout == 'method':
        return {'up': os.path.join(out_dir, 'lowres-%s' % tag),
                'native': os.path.join(out_dir, 'lowres-%s-native' % tag),
                'hr': os.path.join(out_dir, 'hr'),
                'qc': os.path.join(out_dir, 'qc-%s' % tag)}
    return {'up': os.path.join(out_dir, 'lowres_on_hr_grid'),
            'native': os.path.join(out_dir, 'lowres_native'),
            'hr': os.path.join(out_dir, 'hr'),
            'qc': os.path.join(out_dir, 'qc')}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def reorient_to_lps(img):
    """Reorient to a canonical (LPS / 'RAI' in ITK terms) axis order.

    Working in a canonical orientation means `--slice_axis 2` reliably means
    'axial slices' rather than 'whatever the third stored axis happens to be'.
    """
    return sitk.DICOMOrient(img, 'RAI')


def conform_to_spacing(img, spacing, interpolator=sitk.sitkBSpline):
    """Resample onto an isotropic grid of `spacing`, preserving FOV and centre."""
    spacing = [float(s) for s in spacing]
    old_size = np.array(img.GetSize(), dtype=float)
    old_spacing = np.array(img.GetSpacing(), dtype=float)
    new_size = np.maximum(1, np.round(old_size * old_spacing / np.array(spacing))).astype(int)

    ref = sitk.Image([int(s) for s in new_size], img.GetPixelID())
    ref.SetSpacing(spacing)
    ref.SetDirection(img.GetDirection())
    # Keep physical centre fixed.
    old_centre = np.array(img.TransformContinuousIndexToPhysicalPoint(
        (old_size / 2.0).tolist()))
    ref.SetOrigin(img.GetOrigin())
    new_centre = np.array(ref.TransformContinuousIndexToPhysicalPoint(
        (new_size / 2.0).tolist()))
    ref.SetOrigin(tuple(np.array(img.GetOrigin()) + (old_centre - new_centre)))

    return sitk.Resample(img, ref, sitk.Transform(), interpolator, 0.0, img.GetPixelID())


def pad_to_even_and_multiple(arr, factors):
    """Zero-pad (edge-replicate) `arr` so arr.shape[i] % factors[i] == 0.

    Returns (padded, pad_before, pad_after). Edge replication rather than zeros
    avoids introducing a hard step at the FOV boundary, which would otherwise
    ring across the whole volume once we truncate k-space.
    """
    pad_before, pad_after = [], []
    for n, f in zip(arr.shape, factors):
        f = int(max(1, round(f)))
        rem = (-n) % (2 * f)  # keep it even too, so the FFT grid is symmetric
        pad_before.append(rem // 2)
        pad_after.append(rem - rem // 2)
    if any(pad_before) or any(pad_after):
        arr = np.pad(arr, tuple(zip(pad_before, pad_after)), mode='edge')
    return arr, pad_before, pad_after


# --------------------------------------------------------------------------- #
# Forward-model helpers now live in sr/kspace.py so the offline path here and the
# on-the-fly randomised degradation used during training are one operator.
# Re-exported for backwards compatibility with anything that imported them here.
# --------------------------------------------------------------------------- #

kspace_window = kspace.kspace_window
crop_kspace = kspace.crop_kspace
zerofill_kspace = kspace.zerofill_kspace
add_rician_noise = kspace.add_rician
pad_to_even_and_multiple = kspace.pad_to_even_multiple


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def simulate(img, args, rng):
    """img: sitk image at source (high) resolution. Returns (lowres_native,
    lowres_on_hr_grid, qc dict)."""
    qc = {}

    src_spacing = np.array(img.GetSpacing(), dtype=float)
    tgt_spacing = np.array([float(s) for s in args.target_spacing], dtype=float)
    qc['source_spacing'] = src_spacing.tolist()
    qc['target_spacing'] = tgt_spacing.tolist()

    if np.any(tgt_spacing < src_spacing - 1e-6):
        raise ValueError(
            'Target spacing %s is finer than source spacing %s along some axis; '
            'this script only downsamples.' % (tgt_spacing, src_spacing))

    # sitk GetArrayFromImage -> numpy index order is (z, y, x). Convert to
    # (x, y, z) so axis indices match --slice_axis / spacing ordering.
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.transpose(arr, (2, 1, 0))
    orig_shape = arr.shape

    factors = tgt_spacing / src_spacing               # e.g. 2.0, 2.0, 2.0
    qc['downsample_factors'] = factors.tolist()

    # ---- forward model ------------------------------------------------------
    # Delegated to sr/kspace.py so that this offline path and the on-the-fly
    # randomised degradation in sr_transforms.RandomKspaceDegradation are the
    # SAME operator. If they drifted apart, the model would be trained on one
    # forward model and evaluated against another, which no metric would reveal.
    up_arr, low, kinfo = kspace.degrade(
        arr, factors,
        mode=args.mode,
        apod=args.apod,
        snr=args.target_snr,
        rng=rng,
        slice_axis=args.slice_axis,
        slab_fwhm_factor=args.slab_fwhm_factor,
        src_spacing=src_spacing,
        tgt_spacing=tgt_spacing,
        snr_mode=args.snr_mode,
        magnitude=args.magnitude,
        clip_negative=args.clip_negative,
        return_native=True)

    padded_shape = tuple(kinfo['padded_shape'])
    target_shape = tuple(kinfo['lowres_shape'])
    pad_b = kinfo['pad_before']
    qc['padded_shape'] = kinfo['padded_shape']
    qc['pad_before'] = pad_b
    qc['lowres_shape'] = kinfo['lowres_shape']
    qc['keep_fraction'] = kinfo['keep_fraction']
    qc['apod'] = kinfo['apod']
    qc['foreground_mean_prenoise'] = kinfo['foreground_mean_prenoise']
    qc['noise_sigma'] = kinfo['noise_sigma']
    qc['snr_mode'] = args.snr_mode

    # ---- back to sitk, on a true coarse grid ------------------------------
    low_itk = sitk.GetImageFromArray(np.transpose(low, (2, 1, 0)))
    low_itk.SetDirection(img.GetDirection())
    low_itk.SetSpacing(tuple((src_spacing * np.array(padded_shape) /
                              np.array(target_shape)).tolist()))
    # Align physical centres of the padded source FOV and the coarse volume.
    src_centre = np.array(img.TransformContinuousIndexToPhysicalPoint(
        (np.array(orig_shape, dtype=float) / 2.0).tolist()))
    low_itk.SetOrigin(img.GetOrigin())
    low_centre = np.array(low_itk.TransformContinuousIndexToPhysicalPoint(
        (np.array(target_shape, dtype=float) / 2.0).tolist()))
    low_itk.SetOrigin(tuple(np.array(img.GetOrigin()) + (src_centre - low_centre)))
    qc['lowres_actual_spacing'] = list(low_itk.GetSpacing())

    # ---- put it back on the HR grid (what GAMBAS consumes) ----------------
    if args.upsample == 'sinc':
        # Zero-fill interpolation, already computed by kspace.degrade: the
        # canonical MRI way to view a low-res acquisition on a fine grid. Adds no
        # new frequency content.
        up_itk = sitk.GetImageFromArray(np.transpose(up_arr, (2, 1, 0)))
        up_itk.CopyInformation(img)
    else:
        interp = {'linear': sitk.sitkLinear,
                  'bspline': sitk.sitkBSpline,
                  'lanczos': sitk.sitkLanczosWindowedSinc}[args.upsample]
        up_itk = sitk.Resample(low_itk, img, sitk.Transform(), interp, 0.0,
                               sitk.sitkFloat32)
    qc['upsample'] = args.upsample

    return low_itk, up_itk, qc


def process_one(in_path, args, rng):
    t0 = time.time()
    name = strip_ext(in_path)
    img = sitk.ReadImage(in_path, sitk.sitkFloat32)

    qc = {'input': os.path.abspath(in_path),
          'input_spacing': list(img.GetSpacing()),
          'input_size': list(img.GetSize()),
          'mode': args.mode,
          'seed': args.seed}

    if args.reorient:
        img = reorient_to_lps(img)
        qc['reoriented'] = 'RAI'

    if args.conform:
        img = conform_to_spacing(img, args.conform_spacing)
        qc['conformed_to'] = list(img.GetSpacing())

    low_itk, up_itk, sim_qc = simulate(img, args, rng)
    qc.update(sim_qc)

    dirs = output_dirs(args.out_dir, args.layout, method_tag_of(args))
    qc['method_tag'] = method_tag_of(args)
    qc['output_dirs'] = dirs
    for key in ('up', 'qc'):
        ensure_dir(dirs[key])
    if not args.skip_native:
        ensure_dir(dirs['native'])

    ext = args.out_ext
    if not args.skip_native:
        sitk.WriteImage(low_itk, os.path.join(dirs['native'], name + ext))
    sitk.WriteImage(up_itk, os.path.join(dirs['up'], name + ext))
    if args.copy_hr:
        ensure_dir(dirs['hr'])
        sitk.WriteImage(img, os.path.join(dirs['hr'], name + ext))
    d_qc = dirs['qc']

    # Simple sanity metric: correlation between the LR-on-HR-grid and the HR
    # target. Should be high (>0.95 for T1w); a low value means something in the
    # geometry went wrong.
    a = sitk.GetArrayFromImage(img).ravel()
    b = sitk.GetArrayFromImage(up_itk).ravel()
    if a.size == b.size:
        qc['corr_lr_vs_hr'] = float(np.corrcoef(a, b)[0, 1])
    qc['seconds'] = round(time.time() - t0, 2)

    with open(os.path.join(d_qc, name + '.json'), 'w') as f:
        json.dump(qc, f, indent=2)

    print('[ok] %-40s  %s -> %s  corr=%.4f  %.1fs' % (
        name, qc['input_size'], qc.get('lowres_shape'),
        qc.get('corr_lr_vs_hr', float('nan')), qc['seconds']), flush=True)
    return qc


def build_parser():
    p = argparse.ArgumentParser(
        description='Simulate a native 2 mm acquisition from a 1 mm anatomical volume.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--in_dir', type=str, help='Folder of high-res volumes')
    src.add_argument('--in_file', type=str, help='Single high-res volume')
    src.add_argument('--in_list', type=str, help='Text file, one path per line')

    p.add_argument('--out_dir', type=str, required=True,
                   help='Parent directory. With the default --layout method, the '
                        'products land in siblings of your originals/: '
                        'lowres-<method>/, lowres-<method>-native/, hr/, qc-<method>/')
    p.add_argument('--out_ext', type=str, default='.nii.gz',
                   choices=['.nii.gz', '.nii'])
    p.add_argument('--layout', type=str, default='method',
                   choices=['method', 'flat'],
                   help='method: lowres-<method>/ per simulation method (default, '
                        'lets you generate several and compare). '
                        'flat: the original lowres_on_hr_grid/ + lowres_native/.')
    p.add_argument('--method_tag', type=str, default=None,
                   help='Override the <method> part of the directory names. Use '
                        'this to keep variants apart, e.g. --method_tag '
                        'kspace-snr20 vs kspace-snr40.')

    p.add_argument('--target_spacing', type=float, nargs=3, default=[2.0, 2.0, 2.0],
                   help='Simulated acquisition voxel size in mm (x y z)')
    p.add_argument('--mode', type=str, default='kspace',
                   choices=['kspace', 'kspace_hann', 'slab', 'gaussian'],
                   help='Forward model. See module docstring.')
    p.add_argument('--slice_axis', type=int, default=2, choices=[0, 1, 2],
                   help='Slice-select axis for --mode slab (2 = axial after --reorient)')
    p.add_argument('--slab_fwhm_factor', type=float, default=1.2,
                   help='Slice profile FWHM as a multiple of slice thickness (--mode slab)')
    p.add_argument('--apod', type=float, default=None,
                   help='Tukey taper fraction across the retained k-space band. '
                        '0 = rectangular truncation (identical to --mode kspace), '
                        '1 = Hann over the whole band (identical to --mode '
                        'kspace_hann). Values between give intermediate vendor-like '
                        'filtering. Default: derived from --mode.')

    p.add_argument('--target_snr', type=float, default=0.0,
                   help='Foreground-mean / noise-sigma at the SOURCE resolution. '
                        '0 disables noise. Typical 3T T1w: 20-40.')
    p.add_argument('--snr_mode', type=str, default='voxel_gain',
                   choices=['voxel_gain', 'fixed'],
                   help="voxel_gain: coarse voxels get the SNR benefit of their "
                        "larger volume (realistic). fixed: apply --target_snr as-is.")
    p.add_argument('--magnitude', action='store_true', default=True,
                   help='Take magnitude after the inverse FFT (MRI images are magnitude)')
    p.add_argument('--no_magnitude', dest='magnitude', action='store_false')
    p.add_argument('--clip_negative', action='store_true', default=True,
                   help='Clip negative values introduced by ringing')
    p.add_argument('--no_clip_negative', dest='clip_negative', action='store_false')

    p.add_argument('--upsample', type=str, default='sinc',
                   choices=['sinc', 'linear', 'bspline', 'lanczos'],
                   help='How to place the low-res volume back on the HR grid. '
                        '"sinc" = k-space zero-fill (recommended); "linear" is '
                        'what a naive pipeline would do and makes a fair baseline.')

    p.add_argument('--reorient', action='store_true', default=True,
                   help='Reorient to canonical RAI before processing')
    p.add_argument('--no_reorient', dest='reorient', action='store_false')
    p.add_argument('--conform', action='store_true',
                   help='Resample the source to --conform_spacing first (use if your '
                        'inputs are not already exactly isotropic)')
    p.add_argument('--conform_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0])

    p.add_argument('--copy_hr', action='store_true',
                   help='Also write the (possibly reoriented/conformed) HR target')
    p.add_argument('--skip_native', action='store_true',
                   help='Do not write the true 2 mm volume (saves disk)')

    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--shard_index', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.in_file:
        files = [args.in_file]
    elif args.in_list:
        with open(args.in_list) as f:
            files = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    else:
        files = list_volumes(args.in_dir)

    if not files:
        sys.exit('No input volumes found.')

    files = files[args.shard_index::args.num_shards]
    ensure_dir(args.out_dir)
    dirs = output_dirs(args.out_dir, args.layout, method_tag_of(args))

    print('shard %d/%d : %d volume(s)' % (args.shard_index, args.num_shards, len(files)),
          flush=True)
    print('method   : %s (mode=%s, snr=%s, upsample=%s)'
          % (method_tag_of(args), args.mode, args.target_snr, args.upsample))
    for k in ('up', 'native', 'hr', 'qc'):
        if k == 'native' and args.skip_native:
            continue
        if k == 'hr' and not args.copy_hr:
            continue
        print('  %-7s -> %s' % (k, dirs[k]))

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, fp in enumerate(files):
        name = strip_ext(fp)
        done = os.path.join(dirs['up'], name + args.out_ext)
        if os.path.exists(done) and not args.overwrite:
            print('[skip] %s already done' % name, flush=True)
            n_skip += 1
            continue
        # Per-subject seed: reproducible, but not the same noise field for everyone.
        #
        # Must NOT use Python's hash(): string hashing is salted per process
        # (PEP 456), so hash(('seed', name)) differs on every invocation and
        # --seed silently did nothing -- the noise realisation in the training
        # inputs was irreproducible run to run. blake2b is stable across
        # processes, interpreter versions and machines.
        rng = np.random.default_rng(stable_seed(args.seed, name))
        try:
            process_one(fp, args, rng)
            n_ok += 1
        except Exception as e:  # keep the array job alive
            n_fail += 1
            print('[FAIL] %s : %s: %s' % (name, type(e).__name__, e),
                  file=sys.stderr, flush=True)

    print('done: %d ok, %d skipped, %d failed' % (n_ok, n_skip, n_fail), flush=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
