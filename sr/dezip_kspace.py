#!/usr/bin/env python3
"""Undo GE's in-plane k-space zero-fill ("ZIP") to recover honest isotropic data.

THE PROBLEM
-----------
GE reconstructs the in-plane matrix to `ReconMatrixPE` (e.g. 256), which is finer
than what was actually acquired, `AcquisitionMatrixPE` (e.g. 180). It does this by
zero-filling k-space before the inverse FFT. The stored 0.703 mm voxels therefore
carry REAL information only out to the true acquisition Nyquist; the band above it
is exactly zero. The image looks 0.7 mm but is 1 mm of information on a finer grid.

THE FELICITOUS FIX
------------------
Zero-fill in k-space and centre-crop in k-space are exact inverses. So the correct
way back to honest isotropic resolution is:

    image  --FFT-->  k-space  --centre-crop each zero-filled axis-->  --IFFT-->

This recovers the native acquisition precisely. The only artefact is the Gibbs
ringing a true 1 mm acquisition would itself have shown (optionally softened with
--apod). 

WHAT IT DOES vs DOESN'T ASSUME
------------------------------
The per-axis crop target comes from the HEADER voxel spacing, not from a single
JSON field: any axis whose spacing is finer than the coarsest (native) axis was
interpolated, and is cropped by exactly its own factor. This is more robust than
reusing `AcquisitionMatrixPE` for the slice direction, which has its own ZIP
factor. The JSON is read only to CROSS-CHECK and warn -- it never overrides the
geometry.

LEGACY
------
To make this script backwards compatible, we have functionality for legacy versions
which will align the the new volume with an existing one (presumably the one from 
the original bad prep method). This is sub optimal but better than the alternative.

USAGE
-----
    python dezip_kspace.py IN.nii.gz IN.json OUT.nii.gz
    python dezip_kspace.py IN.nii.gz IN.json OUT.nii.gz --apod 0.3
    python dezip_kspace.py IN.nii.gz IN.json OUT.nii.gz --match_ref LEGACY.nii.gz

Writes OUT.nii.gz: isotropic, radiological (via FSL), with qform == sform. With
--match_ref the output's grid is copied from an existing anatomical so it stays
byte-for-byte aligned with legacy preprocessing (see that flag's help).
"""
import argparse
import json as _json
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit('nibabel is required: pip install nibabel')


# --------------------------------------------------------------------------- #
# k-space cropping
# --------------------------------------------------------------------------- #
def _tukey_1d(m, alpha):
    """Separable Tukey window of length m. alpha=0 rectangular, alpha=1 Hann."""
    if alpha <= 0 or m < 2:
        return np.ones(m)
    w = np.ones(m)
    edge = int(np.floor(alpha * (m - 1) / 2.0))
    if edge < 1:
        return w
    n = np.arange(edge + 1)
    taper = 0.5 * (1 + np.cos(np.pi * (2.0 * n / (alpha * (m - 1)) - 1)))
    w[:edge + 1] = taper
    w[-(edge + 1):] = taper[::-1]
    return w


def kspace_crop(arr, targets, apod=0.0):
    """Centre-crop each axis in k-space from arr.shape[a] to targets[a].

    Intensity is preserved: cropping keeps the DC term, and the inverse FFT
    divides by the new (smaller) sample count, so the result is rescaled by
    prod(new)/prod(old) to keep the mean fixed (verified on a constant image).

    Returns a real magnitude array of shape `targets`.
    """
    F = np.fft.fftshift(np.fft.fftn(arr.astype(np.float64)))
    slices = []
    for n, m in zip(arr.shape, targets):
        if m > n:
            raise ValueError('target %d exceeds source %d (this undoes zero-fill, '
                             'it does not add resolution)' % (m, n))
        start = (n - m) // 2          # symmetric window around the centred DC
        slices.append(slice(start, start + m))
    Fc = F[tuple(slices)]

    if apod > 0:
        for ax, m in enumerate(targets):
            shape = [1, 1, 1]
            shape[ax] = m
            Fc = Fc * _tukey_1d(m, apod).reshape(shape)

    out = np.fft.ifftn(np.fft.ifftshift(Fc))
    scale = float(np.prod(targets)) / float(np.prod(arr.shape))
    return np.abs(out) * scale


# --------------------------------------------------------------------------- #
# geometry (nibabel: data axes are voxel i,j,k; affine maps them to world mm)
# --------------------------------------------------------------------------- #
def voxel_sizes(affine):
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def regrid_affine(old_affine, old_shape, new_shape, new_spacing):
    """New 4x4 affine that keeps orientation and the physical CENTRE fixed.

    Each direction cosine (unit column of the rotation block) is retained and
    rescaled to the new spacing; the translation is recomputed so the centre
    voxel maps to the same world point despite the changed matrix -- this keeps
    the brain in the same location instead of drifting by half a FOV.
    """
    A = np.asarray(old_affine, dtype=np.float64)
    R = A[:3, :3]
    old_sp = voxel_sizes(A)
    units = R / old_sp                       # columns are unit direction cosines
    Rn = units * np.asarray(new_spacing, dtype=np.float64)

    c_old = (np.asarray(old_shape, dtype=np.float64) - 1) / 2.0
    world_c = A[:3, 3] + R @ c_old
    c_new = (np.asarray(new_shape, dtype=np.float64) - 1) / 2.0
    t = world_c - Rn @ c_new

    out = np.eye(4)
    out[:3, :3] = Rn
    out[:3, 3] = t
    return out



# --------------------------------------------------------------------------- #
# orientation (delegated to FSL, as in the user's existing pipeline)
# --------------------------------------------------------------------------- #
def to_radiological(path, log):
    """Force radiological (LAS) orientation via FSL, matching the raw pipeline.

    Neurological-convention volumes are flipped left-right (`fslswapdim -x y z`)
    and then stamped radiological (`fslorient -forceradiological`). Done with FSL
    rather than by editing the affine here so the result is byte-for-byte what the
    rest of the pipeline produces. No-op on already-radiological files; a clear
    skip (not a crash) if FSL isn't on PATH.
    """
    if shutil.which('fslorient') is None:
        log('  !! FSL not found on PATH; skipping radiological reorientation. '
            '`module load fsl` / activate the conda env, or pass --no_radiological.')
        return
    try:
        orient = subprocess.run(['fslorient', '-getorient', path],
                                capture_output=True, text=True, check=True
                                ).stdout.strip()
    except subprocess.CalledProcessError as e:                   # noqa: BLE001
        log('  !! fslorient failed (%s); leaving orientation untouched.' % e)
        return
    log('  orientation: %s' % orient)
    if orient == 'NEUROLOGICAL':
        log('  reorienting to radiological: fslswapdim -x y z, then '
            'forceradiological (the left-right "warning" is expected)')
        subprocess.run(['fslswapdim', path, '-x', 'y', 'z', path], check=True)
        subprocess.run(['fslorient', '-forceradiological', path], check=True)


# --------------------------------------------------------------------------- #
# backward compatibility with an existing (legacy) grid
# --------------------------------------------------------------------------- #
def match_reference(out_path, ref_path, log):
    """Copy an existing anatomical's grid onto the output for exact overlay.

    de-ZIP places the volume at the physically-correct FOV centre, which differs
    from a legacy flirt grid by a sub-voxel origin offset (flirt keeps voxel 0 and
    never accounted for the centre shift when 256@0.703 became 180@1.0). When
    downstream masks, segmentations or registrations were built on the legacy grid,
    the de-ZIP output must sit on that SAME grid to remain usable. This copies the
    reference's sform, qform and codes verbatim -- deliberately adopting its origin
    convention so the two overlay byte-for-byte.

    Guarded on purpose: the copy happens only if the two share the same matrix AND
    the same 3x3 (orientation + spacing). A rotation or handedness mismatch means
    the voxel arrays do NOT correspond, and pasting the affine would silently mirror
    or rotate the data -- so that case aborts loudly instead. Run this AFTER the
    radiological step, so both volumes are already in the same convention.
    """
    out = nib.load(out_path)
    ref = nib.load(ref_path)
    if tuple(out.shape[:3]) != tuple(ref.shape[:3]):
        sys.exit('--match_ref: output matrix %s != reference %s. Cannot paste a grid '
                 'onto a different matrix size; reconcile the crop with the legacy '
                 'matrix first (the de-ZIP crop target is derived from the header '
                 'spacing, the legacy one from AcquisitionMatrixPE -- they should '
                 'agree, but a 1-voxel rounding difference will trip this).'
                 % (tuple(out.shape[:3]), tuple(ref.shape[:3])))
    oa, ra = out.affine, ref.affine
    if not np.allclose(oa[:3, :3], ra[:3, :3], atol=1e-3):
        sys.exit('--match_ref: reference orientation/spacing differs from the output '
                 '(determinants %+.0f vs %+.0f). They are not the same grid up to a '
                 'translation, so copying the affine would mis-align or mirror the '
                 'data. Check both are radiological (the FSL reorient ran) and the '
                 'same resolution, then retry.'
                 % (np.sign(np.linalg.det(ra[:3, :3])),
                    np.sign(np.linalg.det(oa[:3, :3]))))
    off = float(np.linalg.norm(oa[:3, 3] - ra[:3, 3]))
    s, sc = ref.get_sform(coded=True)
    q, qc = ref.get_qform(coded=True)
    matched = nib.Nifti1Image(np.asanyarray(out.dataobj), ra, out.header)
    matched.set_sform(s, code=int(sc) if sc else 1)
    matched.set_qform(q if q is not None else s, code=int(qc) if qc else 1)
    nib.save(matched, out_path)
    log('  matched grid to %s: copied sform/qform (codes %s/%s), absorbing a '
        '%.3f mm origin offset so it overlays legacy data exactly.'
        % (os.path.basename(ref_path), sc, qc, off))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def dezip(in_file, json_file, out_file, apod=0.0, quiet=False,
          radiological=True, match_ref=None):
    img = nib.load(str(in_file))
    arr = np.asanyarray(img.dataobj).astype(np.float64)
    sp = voxel_sizes(img.affine)
    iso = float(sp.max())

    def log(*a):
        if not quiet:
            print(*a)

    # Per-axis crop target from the spacing ratio. Even targets keep the k-space
    # window symmetric about DC.
    targets, new_sp = [], []
    for n, s in zip(arr.shape, sp):
        if s < iso * (1 - 1e-3):
            m = int(round(n * s / iso))
            if m % 2:
                m -= 1
            targets.append(m)
            new_sp.append(iso)
        else:
            targets.append(n)
            new_sp.append(float(s))
    targets = tuple(targets)
    new_sp = np.array(new_sp)

    log('input : %s' % os.path.basename(in_file))
    log('  matrix %s  spacing %s' %
        ('x'.join(map(str, arr.shape)), ','.join('%.4f' % v for v in sp)))
    log('  isotropic target = %.4f mm (coarsest, i.e. native, axis)' % iso)
    for ax, (n, m, s) in enumerate(zip(arr.shape, targets, sp)):
        tag = 'crop (zero-filled)' if m != n else 'keep (native)'
        log('  axis %d: %3d @ %.4f  ->  %3d @ %.4f   %s'
            % (ax, n, s, m, (iso if m != n else s), tag))

    # JSON cross-check only -- never overrides the geometry-derived crop.
    acq = rec = None
    try:
        with open(json_file) as fh:
            j = _json.load(fh)
        acq, rec = j.get('AcquisitionMatrixPE'), j.get('ReconMatrixPE')
    except Exception as e:                                        # noqa: BLE001
        log('  (could not read JSON for cross-check: %s)' % e)
    if acq and rec:
        log('  JSON: AcquisitionMatrixPE=%s ReconMatrixPE=%s' % (acq, rec))
        if rec <= acq:
            log('  !! ReconMatrixPE <= AcquisitionMatrixPE: this does NOT look '
                'like zero-fill. Check before trusting the crop.')
        cropped = [m for n, m in zip(arr.shape, targets) if m != n]
        if cropped and not any(abs(m - acq) <= 1 for m in cropped):
            log('  !! geometry-derived crop %s does not match AcquisitionMatrixPE '
                '%s. The in-plane phase-encode axis should; if the SLICE axis '
                'does not, its ZIP factor differs -- expected, and handled from '
                'the header, but worth confirming.' % (cropped, acq))

    if targets == arr.shape:
        log('nothing to do: already isotropic. Copying through.')
        out_arr = arr
    else:
        out_arr = kspace_crop(arr, targets, apod=apod)

    new_affine = regrid_affine(img.affine, arr.shape, out_arr.shape, new_sp)
    out_img = nib.Nifti1Image(out_arr.astype(np.float32), new_affine)
    # Stamp BOTH forms with the same affine and the same code. Passing an affine
    # to Nifti1Image sets only the sform (code 2) and leaves qform_code=0 with a
    # placeholder qform -- which then disagrees with the sform and makes FSLeyes
    # (and other tools that read the qform) treat the file as a different space
    # than a flirt output whose qform and sform both read "scanner anat" (code
    # 1). Setting qform == sform == new_affine with code 1 keeps every tool in
    # agreement and matches the rest of the pipeline.
    out_img.set_qform(new_affine, code=1)
    out_img.set_sform(new_affine, code=1)
    out_img.header.set_zooms(tuple(float(v) for v in new_sp))
    nib.save(out_img, str(out_file))
    log('wrote %s   (matrix %s, spacing %s)'
        % (out_file, 'x'.join(map(str, out_arr.shape)),
           ','.join('%.4f' % v for v in new_sp)))

    # Reorient the written file in place (FSL edits the file, not the array).
    if radiological:
        to_radiological(str(out_file), log)

    # Optionally paste a legacy grid on top, for backward compatibility. Done last,
    # after the radiological flip, so both volumes share the same convention.
    if match_ref:
        match_reference(str(out_file), str(match_ref), log)

    return out_arr, targets


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('in_file', help='Untampered GE recon (the zero-filled volume)')
    p.add_argument('json_file', help='Sidecar JSON (for the acq/recon cross-check)')
    p.add_argument('out_file', help='Output isotropic NIfTI (a .png is written too)')
    p.add_argument('--apod', type=float, default=0.0,
                   help='Tukey taper on the retained k-space, 0=brick-wall (exact '
                        'ZIP inverse, some Gibbs), up to 1=Hann (no ringing, mild '
                        'blur). Default 0.')
    p.add_argument('--no_radiological', dest='radiological', action='store_false',
                   help='Skip the FSL neurological->radiological (LAS) reorient. '
                        'On by default to match the raw pipeline; needs FSL on '
                        'PATH.')
    p.add_argument('--match_ref', default=None,
                   help='An existing anatomical (e.g. your legacy flirt output) '
                        'whose grid the result should adopt EXACTLY, so it stays '
                        'aligned with masks/segmentations/registrations built on '
                        'that grid. Copies the reference sform/qform/codes verbatim, '
                        'absorbing the ~0.15 mm origin-convention offset. Aborts if '
                        'the matrix or orientation differ (i.e. if the two are not '
                        'the same grid up to a translation).')
    a = p.parse_args(argv)
    dezip(a.in_file, a.json_file, a.out_file, apod=a.apod,
          radiological=a.radiological, match_ref=a.match_ref)


if __name__ == '__main__':
    main()
