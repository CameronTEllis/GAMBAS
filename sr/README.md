# Retraining GAMBAS for 2 mm → 1 mm anatomical super-resolution

Infrastructure for retraining this repo's GAMBAS generator on pairs of
(simulated 2 mm isotropic input, real 1 mm isotropic target) anatomical volumes,
plus a simulator that approximates what a native 2 mm acquisition would actually
have produced.

Nothing here modifies the original training entry points. `train.py`, `test.py`
and `inference.py` still work as before. Four bug fixes were made outside `sr/` —
they are listed at the bottom.

---

## Data layout

Inputs are expected as `<id>_<session>_<age>_<weighting>.nii.gz`, e.g.
`12345_02_2.2_t1w.nii.gz`. **Age is in months, with a decimal point** — `2.2`
means 2.2 months. Underscore is a reserved delimiter, so filenames are
parsed **positionally** rather than by pattern matching — change the layout with
`--name_schema` (a comma-separated field list) instead of editing regexes. The
names `id`, `session`, `age` and `weighting` are the ones the pipeline
understands; extra fields are parsed and ignored.

| field | from | on `12345_02_2.2_t1w` |
|---|---|---|
| subject | `id` | `12345` |
| session | `id` + `session` | `12345_02` |
| weighting | `weighting`, lowercased | `t1w` |
| age | `age`, verbatim + numeric parse | `2.2`, `2.2` (months) |

Positional parsing is used because it cannot fail the way regexes do: a subject
id containing the substring `t1w`, or an extra field appearing mid-schema, both
silently corrupt a regex parse and neither corrupts this one.

Because the age field contains a `.`, `strip_ext` only removes a **recognised**
image extension and never falls back to `os.path.splitext` — that fallback splits
at the last dot and would turn `12345_02_2.2_t1w` into `12345_02_2`, silently
dropping the weighting and changing the grouping key. It is idempotent, so it is
safe to apply to a path or an already-stripped stem.

The numeric age parse stays unit-agnostic even though the units are known here:
`2.2`, `06mo` and `0.5y` all round-trip as tokens and parse to 2.2 / 6.0 / 0.5.

**Names that do not tokenise to the schema are a hard error, not a warning.**
A mis-parsed name yields a wrong grouping key, which yields a subject leak that
nothing downstream can detect. `make_folds.py` refuses to build a split and lists
the offenders; `--allow_bad_names` downgrades them to singleton subjects (still
leak-safe, but unstratified). It also prints a parse preview every run:

```
stem                           subject    session      group        weighting age
12345_01_2.2_t2w               12345      12345_01     12345        t2w      2.2
12345_02_11.4_t1w              12345      12345_02     12345        t1w      11.4
```

Set `SUBJECT_REGEX` / `SUBGROUP_REGEX` in `config.sh` only for a cohort the
positional scheme cannot express (mixed-in BIDS names, say); leaving them empty
derives everything from `NAME_SCHEMA`.

### Simulator output

Products are written as siblings of `originals/`, one directory per simulation
method, preserving filenames exactly:

```
DATA/
  originals/                     your 1 mm volumes            (--in_dir)
  lowres-kspace/                 2 mm on the 1 mm grid  <-- network INPUT
  lowres-kspace-native/          the true 2 mm volumes
  hr/                            reoriented 1 mm targets  <-- pair against THIS
  qc-kspace/                     per-volume provenance JSON
  qc-kspace-png/                 QC figures
```

`--out_dir` is the parent (`DATA/`), so `SIM_DIR` in `config.sh` is normally the
**parent** of `HR_DIR`. One directory per method lets you generate several and
compare without re-running anything; `--method_tag` separates variants of the same
mode (`kspace-snr20` vs `kspace-snr40`).

**Pair against `hr/`, not `originals/`.** `--reorient` (on by default) puts
volumes in canonical RAI, and `--conform` can change the grid. Input and target
must share a grid exactly, so the simulator writes the reoriented target
alongside its output. Pointing the builder at untouched `originals/` risks a
silent direction-cosine mismatch — `check_dataset.py` catches it, but only after
you have built the dataset.

### Why the native 2 mm volume isn't always exactly half-size

`lowres-<method>/` — the training input — is **always exactly your input grid**;
it is cropped back with the same offsets it was padded with. `lowres-<method>-native/`
can be one voxel larger per axis than `n/2`, and that is expected.

The requirement is divisibility by **4**, not by 2:

```python
rem = (-n) % (2 * f)        # f = 2, so modulo 4
```

| original dim | pad | native 2 mm | exactly n/2? |
|---|---|---|---|
| 256, 176, 192 (≡0 mod 4) | 0 | 128, 88, 96 | yes |
| 182, 154, 90 (≡2 mod 4) | 1 + 1 | 92, 78, 46 | no, one voxel larger |

An even dimension is therefore not sufficient. The reason for 4 rather than 2 is
that the coarse matrix is forced even, so the k-space crop stays symmetric about
DC — an odd coarse dimension puts DC off-centre and introduces a half-voxel
spatial shift. Guaranteeing `padded / 2` is even needs `padded` divisible by 4.

Consequences: the native volume is centre-aligned with the original, so nothing is
shifted; its FOV just extends 1 mm past each edge, which is the edge-replicate
padding showing through. It is used only for the record and for the BSpline
baseline in `evaluate_sr.py`, where SimpleITK resamples it against the HR image as
reference, so the extra FOV is handled correctly.

Per-subject numbers are recorded in `qc-<method>/*.json` as `input_size`,
`pad_before`, `padded_shape` and `lowres_shape`.

## Quick start

```bash
# 0. one-time, on a GPU node (mamba_ssm compiles CUDA kernels)
bash sr/cluster/setup_env.sh

# 1. point the pipeline at your data
$EDITOR sr/cluster/config.local.sh      # uncomment HR_DIR, SIM_DIR, CKPT_DIR, ...

# 2. prove the whole thing works on synthetic phantoms (~2 min on a GPU)
python -m sr.smoke_test --work_dir /scratch/$USER/smoke --patch_size 128

# 3. submit the real chain
cd sr/cluster && ./submit_all.sh
```

`submit_all.sh` chains four SLURM jobs with `afterok` dependencies:
simulate (CPU array) → build dataset → train (GPU) → evaluate (GPU).
`DRY_RUN=1 ./submit_all.sh` prints the `sbatch` commands without submitting.

---

## How the 2 mm simulation works

**The short version:** we truncate k-space, we do not blur-and-decimate.

A scanner samples a finite region of k-space. The voxel you get is the object
convolved with a point spread function set by that sampling window, then sampled
on the coarse grid. Trilinear or BSpline downsampling of a fine image is a
*different* operation with a different PSF, and it produces none of the Gibbs
ringing that real MRI shows at tissue boundaries. A network trained on
blur-and-decimate inputs learns to invert a blur kernel that no scanner applies,
and will underperform on real 2 mm data.

`sr/simulate_lowres.py` therefore:

1. FFTs the 1 mm volume (edge-replicate padded to an even, factor-divisible grid,
   so truncation does not ring off the FOV boundary).
2. Multiplies by a k-space window (`--mode`):
   - `kspace` **(default)** — hard rectangular truncation to the 2 mm Nyquist.
     Sinc PSF, realistic Gibbs ringing. The most defensible model for a 3D
     acquisition.
   - `kspace_hann` — truncation plus raised-cosine apodisation, mimicking vendor
     filtering. Less ringing, slightly blurrier.
   - `slab` — rectangular in-plane, Gaussian slice profile along `--slice_axis`
     with FWHM = `--slab_fwhm_factor` × slice thickness. Use this if your target
     protocol is **2D multi-slice** rather than 3D, since the slice-select RF
     pulse gives a much broader, non-sinc PSF through-plane.
   - `gaussian` — blur-then-decimate. Included **only** as an ablation. It is not
     a good acquisition model.
3. Crops k-space onto the coarse grid, so the output is a genuine 2 mm volume
   with no interpolation applied after the fact.
4. Adds Rician magnitude noise at `--target_snr`. With the default
   `--snr_mode voxel_gain`, the SNR you request is interpreted at the *source*
   resolution and the coarse volume is given the SNR benefit its 8× larger voxels
   would really have. `--snr_mode fixed` applies the number literally.

### What it cannot model

Stated plainly, because it bounds what your trained model will generalise to:

- **Partial Fourier, GRAPPA/SENSE g-factor noise amplification, elliptical
  k-space shutters.** Not modelled. Real accelerated 2 mm data has spatially
  varying, correlated noise that this does not reproduce.
- **Contrast differences.** The coarse protocol would likely run a different
  TR/TE/flip angle. Tissue contrast here is inherited unchanged from the 1 mm
  source.
- **Motion.** A different acquisition duration means different motion artefact.
  Not modelled.
- **Residual 1 mm PSF.** The source is already PSF-blurred at 1 mm, so the
  simulated 2 mm volume is marginally blurrier than a true 2 mm acquisition.
  Unavoidable without raw k-space.
- **Gradient nonlinearity / B0 distortion** differences between protocols.

If you have even a handful of subjects scanned at both resolutions, the highest
value next step is to compare their real 2 mm volume against the simulation
(`sr/qc_figure.py` gives you the per-axis spectra to do it) and tune `--mode`,
`--slab_fwhm_factor` and `--target_snr` to match.

### Verifying the simulation

```bash
python -m sr.qc_figure --sim_dir $DATA --method kspace \
    --out_dir $DATA/qc-kspace-png --n 5
```

Look at the **per-axis power spectrum** panel (bottom left). The simulated curve
must fall off a cliff exactly at 0.25 cycles/voxel (the 2 mm Nyquist on a 1 mm
grid). On the phantoms this drop is about two orders of magnitude.

Do **not** judge this from the radial spectrum (bottom right). A rectangular
k-space window retains energy out to the cube corner at |k| = Nyquist·√3 ≈ 0.43,
so the radial profile has no clean cliff even when the simulation is perfectly
correct. Both panels are plotted so this cannot be misread.

---

## Why the input is on the 1 mm grid

`sr/build_sr_dataset.py` feeds the network `lowres-<method>/`, not
`lowres-<method>-native/`. This is forced by two facts about the repo:

1. `utils/NiftiDataset.RandomCrop` crops the image and the label with a **single**
   `RegionOfInterest` index and size. If the two volumes have different array
   shapes, it crops different anatomy from each.
2. The GAMBAS generator is shape-preserving: two stride-2 encoder convs, two
   transposed-conv decoders, output size == input size. It has no upsampling
   head.

So the network's job is to restore the high-frequency content the 2 mm
acquisition never measured, on a grid it already shares with the target. The
`lowres_on_hr_grid` volume is produced by **k-space zero-filling**
(`--upsample sinc`), which is the canonical MRI way to view a low-resolution
acquisition on a fine grid and adds no new frequency content. The true 2 mm
volumes are kept in `lowres_native/` for the record and for the BSpline baseline
in evaluation.

If you would rather the network learn the upsampling itself, you need to add a
decoder upsampling stage and replace `RandomCrop` with a paired crop that scales
indices between the two grids. That is a larger change than this infrastructure
makes.

---

## Augmentation: why the repo default is wrong here

`utils/NiftiDataset.Augmentation` picks one of 8 branches per sample. Several are
actively harmful when the target's high-frequency content is the entire learning
signal. Measured on a phantom:

| branch | what it does to the **target** | measured effect |
|---|---|---|
| 1, additive Gaussian noise | noises the label | asks the network to predict a specific noise realisation — unlearnable, caps PSNR |
| 2, RecursiveGaussian | **blurs the label** | removes **71%** of the label's high-frequency energy |
| 3, 4, rotation / BSpline | linear+BSpline resamples both | systematic low-pass of the target on ~2/8 of samples |
| 5, flip | nothing | `flipit()` builds `img` then returns `image` — it has never flipped anything |
| 6, 7, brightness / contrast | independent `np.random.randint` for image and label | offsets differed by **21 intensity units** in a single draw, breaking the paired relationship |

`sr/sr_transforms.SRAugmentation` replaces it with:

- axis flips and 90° rotations — exact voxel permutations, zero interpolation
  blur, applied identically to input and target;
- gamma and gain jitter with **identical parameters** for both, so the mapping
  the network must learn is unchanged (verified: label HF energy stays within
  0.93–1.07× of untouched);
- Rician noise jitter on the **input only**, for SNR robustness across scanners
  without corrupting the target.

`--legacy_augment` restores the original behaviour if you want to measure the
difference; `--no_augment` disables augmentation entirely.

`PairedRandomCrop` also fixes two latent problems in `RandomCrop`: its
`np.random.randint(10, size_old[0] - size_new[0])` lower bound raises when a
volume is only slightly larger than the patch, and its retry loop is unbounded,
so a mostly-empty volume can hang a dataloader worker forever.

---

## Training

```bash
python -m sr.train_sr \
  --data_path $DATASET_DIR/train --val_path $DATASET_DIR/val \
  --checkpoints_dir $CKPT_DIR --name sr_2mm_to_1mm \
  --patch_size 128 --batch_size 1 \
  --lambda_A 100 --lambda_adv 0.0 \
  --val_freq 5 --val_metric psnr
```

What `train_sr.py` adds over `train.py`:

- **Real validation.** Full-volume Gaussian-blended sliding-window inference on
  held-out subjects every `--val_freq` epochs, reporting PSNR / SSIM / L1 against
  the 1 mm truth **and the same metrics for the sinc-interpolated input**. You
  always see whether the network is beating the free baseline. Written to
  `val_metrics.csv` and TensorBoard.
- **`best_net_G.pth`** selected on the validation metric, not just the last epoch.
- **Preemption-safe resume.** `--continue_train` restores generator,
  discriminator, optimiser state, LR-scheduler state and epoch counter. The repo
  saved none of the last three, so a restart previously reset Adam's moments and
  the LR schedule. `03_train.sbatch` traps `SIGUSR1` and requeues itself.
- **Deterministic seeding** of Python, NumPy, Torch and dataloader workers.
- Optional AMP (`--amp`), gradient accumulation, and `--iters_per_epoch` for a
  fixed number of steps per epoch when you have few volumes.

### Warm start: `--init_from`

At ~50 training volumes this is likely the largest single win available, so it is
worth doing first. Download the released generator weights from the
[GAMBAS v1.0 release](https://github.com/levente-1/GAMBAS/releases/tag/v1.0) and
set `INIT_FROM` in `config.local.sh`. The published model is paediatric ULF T2w,
so the domain differs from yours, but the low-level operation — deconvolve the
PSF, restore edges — transfers well.

The repo's own `load_networks` derives its path from
`<checkpoints_dir>/<name>/<epoch>_net_G.pth` and loads G and D together, so it
cannot be pointed at a downloaded file. `--init_from` can, and it reconciles the
three things that otherwise break or silently half-succeed:

- a uniform `module.` prefix (present or not depending on whether the checkpoint
  was saved from a `DataParallel` wrapper);
- `InstanceNorm` running buffers, which this model does not define;
- per-tensor shape mismatches from a different `--ngf` / `--input_nc`.

Coverage is **reported and enforced**. A warm start that silently loads 4% of the
generator is worse than none, because it looks like it worked, so training
refuses to begin below `--init_min_coverage` (default 0.5) unless you pass
`--init_allow_partial`. The guard uses the stricter of tensor-count and
parameter-count coverage — they diverge exactly in the dangerous case where a few
large tensors fail to match, which reads as 80% of tensors but only 44% of the
actual weights.

`--continue_train` takes priority: when a run is resuming, `--init_from` is
ignored rather than overwriting partly-trained weights.

### Choosing `--lambda_adv`

**Default is now 0.0** — a pure L1 regressor. At ~50 volumes a discriminator can
effectively memorise the training set, GAN instability is hardest to diagnose at
small n, and L1 maximises PSNR, so this is the honest baseline any adversarial
variant has to beat. It also skips the discriminator's forward/backward entirely,
freeing the activation memory a 128³ patch needs.

Then try 1.0 and compare `hf_energy_pred` against `hf_energy_target` in the
evaluation output: the adversarial term's job is to close that gap, and if it
overshoots the target's value it is inventing texture rather than restoring it.

PSNR alone will always favour the L1 model. Judge sharpness with `hf_energy` and
your eyes, not PSNR.

### Degradation randomisation: `--randomize_degradation`

Off by default. This re-simulates the 2 mm input from that volume's own 1 mm
target on every training draw, with a freshly sampled apodisation and SNR,
instead of using the precomputed `lowres-<method>/` file.

```bash
RANDOMIZE_DEGRADATION=1 APOD_RANGE="0.0 0.3" SNR_RANGE="20 40" ./submit_cv.sh dev
```

**Apodisation is now a continuous axis.** `apod` is the Tukey taper fraction over
the retained k-space band, and the two original discrete modes are exactly its
endpoints — `apod=0` *is* `--mode kspace` (rectangular, full Gibbs ringing) and
`apod=1` *is* `--mode kspace_hann`. Verified: the windows match to float32
rounding, and a sweep reproduces both endpoint sharpness values.

The default range `[0, 0.3]` is deliberately narrow. Measured on a phantom, the
full interval spans an **87% change in mean gradient magnitude** and 3.9 dB of
baseline PSNR — much wider than plausible vendor variation. Randomising over all
of it would make the model hedge across a span reality does not contain. `[0, 0.3]`
covers about 7%.

**What not to include.** `--degradation_modes` defaults to `kspace` alone, which
with continuous apodisation already spans `kspace_hann`. Do **not** add
`gaussian`: it is a deliberately unphysical blur-then-decimate model kept for
ablation, and training on it teaches the network to invert a kernel no scanner
produces (the transform warns if you try). Add `slab` only if your protocol is
genuinely 2D multi-slice.

**Two implementation points that matter:**

1. The degradation runs on the **whole volume, before cropping**. k-space
   truncation is global — truncating a 128³ patch is a different operator, with a
   PSF set by the patch's own Nyquist and ringing at the patch edges. It also runs
   before `SRAugmentation`, since it replaces the input wholesale, and it
   auto-disables `--input_noise_std` so you don't get a second, full-bandwidth
   noise term on the fine grid.
2. Noise is added on the **coarse grid**, before interpolation back to 1 mm. Real
   2 mm noise is band-limited to the acquired band; adding it on the fine grid
   would hand the network high-frequency content no 2 mm scan contains — a
   corrupting cue given that synthesising high frequencies is the entire task.
   Rician combination is nonlinear, so it does not commute with interpolation.

**Training only.** Validation and test always use the precomputed fixed
condition, so best-checkpoint selection and your reported number stay on one
interpretable operator.

**Why on the fly rather than pre-generating variants.** Besides the continuous
axis and no disk multiplication, it avoids a real hazard: pre-generated variants
share both anatomy *and* filename across `lowres-*` directories, so a dataset
built by unioning them must force every variant of a subject into the same fold.
Get that wrong and variant-of-X trains while another variant-of-X tests — a total
anatomy leak that `check_dataset.py` cannot detect, because each pair is
individually valid.

`sr/kspace.py` holds the one implementation of the forward model, used by both the
offline simulator and this transform. If they drifted apart you would train on one
operator and evaluate against another, and no metric would reveal it — so
`smoke_test.py` step 4c asserts the online and offline results agree.

### Subgroup balancing: `--balance_subgroups`

Your split is 38 t1w / 14 t2w. Left alone, the minority weighting quietly
underperforms and it is invisible inside a pooled metric. `--balance_subgroups`
reweights patch sampling; `--balance_power` sets how far:

| power | t1w / t2w sampled share | each t2w volume revisited |
|---|---|---|
| 0.0 | 73% / 27% (natural) | 1.00× |
| **0.5** (default) | 62% / 38% | 1.65× |
| 1.0 | 50% / 50% | 2.71× |

Full balance is available but sqrt is the default: because training crops random
patches, each extra visit to a volume is a *different* patch, so oversampling
costs far less than it would with whole-image training — but it is still the same
14 individuals' anatomy, and 2.71× is a real overfitting pressure on them. The
realised shares above were verified empirically over 200k draws.

Labels come from the original filenames via `manifest.json`, so this depends on
`--name_schema` being right. The printed table shows natural vs sampled share
every run, so you can confirm you got what you asked for.

### Memory

At 128³, batch 1, `ngf 64`, expect roughly 30–40 GB of GPU memory with the
discriminator active. If you OOM, in order of effectiveness:

1. `--patch_size 96` (or 64) — biggest single win, memory scales as the cube
2. `--lambda_adv 0.0` — skips the discriminator entirely (see the fix below)
3. `--amp` — bf16 autocast; verify `mamba_ssm` behaves on your GPU first
4. `--grad_accum 2` with a smaller patch

Patch dimensions must be divisible by 4 (two stride-2 encoders); `sr_options.py`
enforces this with a clear error rather than a shape mismatch deep in the decoder.

---

## Evaluation

```bash
python -m sr.evaluate_sr \
  --test_path $DATASET_DIR/test \
  --checkpoints_dir $CKPT_DIR --name sr_2mm_to_1mm --which_epoch best \
  --out_dir $EVAL_DIR --lowres_native_dir $SIM_DIR/lowres_native \
  --save_predictions
```

Reports, per volume and aggregated: PSNR, SSIM (Gaussian-windowed 3D, optionally
brain-masked), MAE, RMSE, gradient sharpness, and high-frequency energy — for the
prediction, for the sinc input, and for BSpline resampling of the true 2 mm
volume. Also writes radial power spectra so you can plot exactly which band the
model restored.

**Read `hf_energy_pred` against `hf_energy_target`.** A model with good PSNR whose
`hf_energy` sits near the sinc baseline has learned deblurring, not
super-resolution.

---

## Splits and cross-validation

This cohort is 38 T1w + 14 T2w infants and toddlers, and the **anatomical
weighting is confounded with age** — the T2w scans are the younger subjects. So
there is one protocol-and-age subgroup axis, not two separable factors. Nothing
in this pipeline can disentangle them; separating them would need subjects
scanned with both weightings at matched ages. The tools label the axis honestly
rather than implying otherwise.

Two things follow. First, splits must be **stratified** on that axis, or a fold
can end up with almost no T2w. Second, metrics must be reported **per subgroup**:
on matched phantom anatomy with an identical forward model I measured a **2.05 dB**
baseline PSNR gap between weightings, so a pooled mean describes neither.

### Recommended two-phase workflow

```bash
# Phase 1 -- development split. Tune patch size, epochs, lambda_adv HERE.
python -m sr.make_folds --sim_dir $SIM_DIR --out $SIM_DIR/folds_dev.json \
    --mode single --test_counts t1w=8,t2w=3 --val_counts t1w=5,t2w=2
cd sr/cluster && ./submit_cv.sh dev

# Phase 2 -- FREEZE the hyperparameters, then 5-fold CV for reported numbers.
cd sr/cluster && K=5 ./submit_cv.sh
```

Tuning on the CV folds and then quoting CV numbers is selection bias — the folds
stop being held out. Freeze first.

On this cohort the fold generator produces:

```
fold 0    8 T1w   3 T2w    11        train/val/test sizes
fold 1    8 T1w   2 T2w    10        [(31,10,11), (31,11,10), (31,10,11),
fold 2    8 T1w   3 T2w    11         (32,10,10), (31,11,10)]
fold 3    7 T1w   3 T2w    10
fold 4    7 T1w   3 T2w    10
```

2 or 3 T2w per fold is optimal given 14 volumes over 5 folds and whole-subject
atomicity. Every volume is tested exactly once and (in the default
`--val_mode rotate`) validated exactly once.

### Grouping: `--group_by subject | session | volume`

Much of this cohort is longitudinal, so the same ID appears at several ages. What
must stay together is a choice:

| level | keeps together | allows to cross a split |
|---|---|---|
| `subject` (default) | everything from one `id` | nothing |
| `session` | the t1w and t2w of one `id_session` visit | different visits of one id |
| `volume` | nothing | everything |

**The stricter setting does not cost you training data.** This is the part that
is easy to get backwards. In K-fold CV, grouping changes *which* volumes land in
the test fold, not *how many* are available for training. Measured on a mock of
this cohort (40 volumes, 20 IDs, 10 of them scanned at 3 ages, 5-fold):

```
group_by=subject   train volumes/fold [24,24,24,24,24]   test subjects seen in training: [0,0,0,0,0]
group_by=session   train volumes/fold [24,24,24,24,24]   test subjects seen in training: [5,4,4,5,4]
group_by=volume    train volumes/fold [24,24,24,24,24]   test subjects seen in training: [5,4,4,5,4]
```

Identical training-set sizes. The only thing relaxing the grouping buys is that
4–5 of each fold's ~7 test subjects were already seen during training. For
super-resolution the target *is* the input's own anatomy at higher frequency, and
individual cortical folding is established near term and largely stable
afterwards, so a model that saw an individual at 6 months has a real advantage on
that same individual at 12 months. The reported gain becomes partly memorisation,
and the bias scales with how much of the cohort is longitudinal.

If you want to relax it anyway, `session` is the defensible middle: two
weightings from one session are the same brain at the same moment, which is the
most severe leak available, and it costs nothing to prevent. And because
`--group_by` only changes the fold spec, running both is cheap — the difference
between them **is** the size of the leak, which is a number worth having rather
than an assumption. `make_folds.py` prints a loud warning and records
`meta.leakage_risk` in `folds.json` for anything other than `subject`, so the
caveat travels with the results.

### Other guarantees

- **Stratification is the objective subject to grouping.** Exact per-fold counts
  are often unreachable when groups are atomic, so it uses the greedy
  largest-group-first heuristic from sklearn's `StratifiedGroupKFold`.
- Asserted after generation: no group spans two splits in any fold (at whatever
  level you chose), every fold covers every volume, every volume is tested
  exactly once, and every fold's training set contains both subgroups.

### Reading the results

`build_sr_dataset.py` renumbers volumes to `0.nii.gz` so the dataloader's
positional pairing cannot mispair anything — which erases the weighting from the
filename. `evaluate_sr.py` recovers it through `manifest.json`, so
`per_volume.csv` carries `stem`, `subgroup` and `fold` columns.

`sr/aggregate_cv.py` pools the folds. It reports per-subgroup means with
intervals, and — the number to actually quote — **within-volume paired deltas
against the sinc baseline**, which cancel most of the shared-model dependence
between folds. Illustrative output shape:

```
subgroup    n   delta PSNR (dB)
ALL        40   +1.516 [+1.273, +1.759]
t1w        26   +1.927 [+1.711, +2.143]
t2w        14   +0.754 [+0.481, +1.026]   <- much weaker
```

There is also a secondary **age breakdown**, since age parses cleanly from the
filename. Ages are decimal months, so you can bin on real developmental
boundaries rather than on your own sampling:

```bash
python -m sr.aggregate_cv --eval_root $EVAL_DIR_cv --out $EVAL_DIR_cv/summary \
    --age_bin_edges 0 6 12 24 36
```

Because weighting is confounded with age here, the youngest bins are all-t2w and
the oldest all-t1w — the table makes that visible rather than hiding it. The
informative comparison is *within* a weighting: if the t1w delta rises across age
bins, that is an age effect not attributable to protocol.

The pooled `ALL` row is dominated by the majority weighting and would have hidden
that. Note the honest caveat the tool prints: per-volume scores are **not**
independent across folds, since the models share training data, so the intervals
understate true uncertainty. They summarise spread; they are not valid
frequentist CIs for a new cohort.

## Files

| file | purpose |
|---|---|
| `simulate_lowres.py` | k-space forward model: 1 mm → native 2 mm + 2 mm-on-1 mm-grid |
| `make_folds.py` | subject-grouped, subgroup-stratified dev split and CV folds → `folds.json` |
| `kspace.py` | the one forward model: Tukey-apodised truncation, coarse-grid noise, stable seeding |
| `naming.py` | positional filename schema: subject / session / weighting / age |
| `checkpoint_utils.py` | warm-start key/shape reconciliation and balance weights. Stdlib only |
| `build_sr_dataset.py` | GAMBAS `images/`+`labels/` layout; materialises a fold, or splits randomly |
| `check_dataset.py` | pre-flight: pairing, geometry, swapped folders, degenerate volumes. No torch needed |
| `aggregate_cv.py` | pools folds into per-subgroup estimates and paired deltas |
| `qc_figure.py` | per-subject QC PNG including the per-axis spectrum proof |
| `sr_transforms.py` | SR-safe augmentation, paired crop, non-blurring pad |
| `sr_options.py` | option set with SR-appropriate defaults and validation flags |
| `sr_metrics.py` | PSNR, 3D SSIM, sharpness, HF energy, spectra. NumPy/SciPy only |
| `train_sr.py` | training loop with validation, best-checkpoint, resume |
| `evaluate_sr.py` | test-set metrics with baselines |
| `smoke_test.py` | 8-step end-to-end shakedown on synthetic phantoms |
| `cluster/` | `config.sh`, `setup_env.sh`, four `.sbatch` scripts, `submit_all.sh`, `submit_cv.sh` |

### Recommended order on a new cluster

1. `bash sr/cluster/setup_env.sh` (GPU node, interactive)
2. `python -m sr.smoke_test --work_dir /scratch/$USER/smoke --patch_size 128`
   — note the reported peak GPU memory before committing to a patch size
3. `cd sr/cluster && DRY_RUN=1 ./submit_all.sh` to inspect the submissions
4. `./submit_all.sh simulate build` — then look at `qc_figure.py` output before
   spending GPU hours
5. `./submit_all.sh train eval`

---

## Changes made outside `sr/`

Three bugs that this pipeline depends on:

1. **`models/gambas_model.py`** — `--lambda_adv` was declared in
   `modify_commandline_options` but never applied to the loss, so setting it did
   nothing (`resvit_model.py` does apply it). Now applied, and when it is exactly
   0 the discriminator forward/backward is skipped entirely, saving its
   activations and roughly a third of the step time.

2. **`options/base_options.py`** — `--gpu_ids` was parsed character by character
   (`list(opt.gpu_ids)` then `int()` per character), so `'0'` worked but `'-1'`
   crashed on `int('-')` and `'0,1'` crashed on `int(',')`. Multi-GPU and CPU were
   both unreachable. Now splits on commas/whitespace, and raises a clear error if
   GPUs are requested without CUDA.

3. **`models/gambas_model.py`** — `initialize()` called `self.load_network(...)`
   (singular), which is not defined on `BaseModel` or anywhere in the repo, so
   **any** use of `--continue_train` with `--model gambas` raised `AttributeError`
   before training could start. It was also redundant: `BaseModel.setup()` runs
   immediately afterwards and calls `load_networks()` (plural, correct) under the
   same condition. `ea_gan_model.py` and `resvit_model.py` have these exact lines
   commented out for this reason; `gambas_model.py` was missed. This one matters
   here because the preemption/requeue design depends on `--continue_train`.

Plus one fixed inside `sr/` itself, worth recording because it was invisible:
`simulate_lowres.py` derived its per-subject noise seed from `hash((seed, name))`.
Python salts string hashing per process (PEP 456), so `--seed` had **no effect** —
the noise realisation in every training input differed on each run while the flag
implied otherwise. Now derived from `blake2b` via `kspace.stable_seed`, and
verified: same seed twice is bit-identical, different seeds differ.

4. Not changed, but worth knowing: `utils/NiftiDataset.resample_sitk_image` and
   `CropBackground` use `np.int`, removed in NumPy ≥ 1.24. They are only reached
   when `--resample True`, which this pipeline leaves off. If you turn resampling
   on, replace `np.int` with `int`.
