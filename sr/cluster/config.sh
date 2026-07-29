#!/bin/bash
# ============================================================================
# config.sh -- single place to set every path and resource request.
# All the sbatch scripts in this folder source this file.
#
#   cp config.sh config.local.sh   # then edit config.local.sh
#
# The scripts prefer config.local.sh if it exists, so your edits survive a
# `git pull`.
# ============================================================================

# ---- paths -----------------------------------------------------------------
# Repo checkout (the directory containing train.py, models/, sr/)
export GAMBAS_ROOT="${GAMBAS_ROOT:-$HOME/code/GAMBAS}"

# Folder of your original 1 mm isotropic anatomicals, named
#   <id>_<session>_<age>_<t1w|t2w>.nii.gz   e.g. 12345_02_2.2_t1w.nii.gz
# (age in months, decimal)
export HR_DIR="${HR_DIR:-/scratch/$USER/data/originals}"

# Parent for the simulation products. With the default --layout method these are
# siblings of your originals/, so SIM_DIR is normally the PARENT of HR_DIR:
#   $SIM_DIR/originals/              your input
#   $SIM_DIR/lowres-<method>/        2 mm on the 1 mm grid  <- training input
#   $SIM_DIR/lowres-<method>-native/ the true 2 mm volumes
#   $SIM_DIR/hr/                     reoriented 1 mm targets (pair against THIS)
#   $SIM_DIR/qc-<method>/            per-volume provenance JSON
export SIM_DIR="${SIM_DIR:-$(dirname "$HR_DIR")}"

# Where build_sr_dataset.py writes train/ val/ test/
export DATASET_DIR="${DATASET_DIR:-/scratch/$USER/data/sr_dataset}"

# Checkpoints, logs, tensorboard
export CKPT_DIR="${CKPT_DIR:-/scratch/$USER/checkpoints}"
export EXP_NAME="${EXP_NAME:-sr_2mm_to_1mm}"

# Evaluation output
export EVAL_DIR="${EVAL_DIR:-/scratch/$USER/eval/$EXP_NAME}"

# SLURM logs
export LOG_DIR="${LOG_DIR:-/scratch/$USER/slurm_logs}"

# ---- simulation parameters -------------------------------------------------
export TARGET_SPACING="${TARGET_SPACING:-2.0 2.0 2.0}"
export SIM_MODE="${SIM_MODE:-kspace}"        # kspace|kspace_hann|slab|gaussian
export TARGET_SNR="${TARGET_SNR:-30}"        # 0 to disable noise
export UPSAMPLE="${UPSAMPLE:-sinc}"
# Directory tag for this simulation config; defaults to the mode. Override to
# keep variants apart, e.g. METHOD_TAG=kspace-snr20.
export METHOD_TAG="${METHOD_TAG:-$SIM_MODE}"

# ---- dataset split ---------------------------------------------------------
export VAL_FRAC="${VAL_FRAC:-0.10}"
export TEST_FRAC="${TEST_FRAC:-0.10}"
# Filenames are parsed POSITIONALLY on underscores, which is reserved as the
# delimiter:  <id>_<session>_<age>_<weighting>.nii.gz
# Reorder or rename fields here if the convention changes; the names id, session,
# age and weighting are the ones the pipeline understands.
export NAME_SCHEMA="${NAME_SCHEMA:-id,session,age,weighting}"

# Regex escape hatches. Leave EMPTY to derive everything from NAME_SCHEMA, which
# is what you want. Set one only for a cohort the positional scheme cannot
# express (e.g. mixed-in BIDS names: SUBJECT_REGEX='sub-[A-Za-z0-9]+').
export SUBJECT_REGEX="${SUBJECT_REGEX:-}"
export SUBGROUP_REGEX="${SUBGROUP_REGEX:-}"
# subject | session | volume. 'subject' keeps all of one individual's scans on
# one side of every split. 'session' lets longitudinal timepoints cross while
# keeping same-session t1w/t2w together. See sr/README.md -- in K-fold CV the
# stricter setting costs zero training volumes.
export GROUP_BY="${GROUP_BY:-subject}"

# ---- training --------------------------------------------------------------
export PATCH_SIZE="${PATCH_SIZE:-128 128 128}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export LAMBDA_L1="${LAMBDA_L1:-100.0}"       # --lambda_A
# 0.0 = pure L1 regressor. This is the DEFAULT and the right place to start: at
# ~50 training volumes a discriminator can effectively memorise the set, GAN
# instability is hardest to diagnose at small n, and L1 maximises PSNR, so it is
# the honest baseline any adversarial variant must beat. It also skips the
# discriminator's forward/backward, freeing the activation memory a 128^3 patch
# needs. Raise it only once you have the L1 number and can compare hf_energy
# against the target's.
export LAMBDA_ADV="${LAMBDA_ADV:-0.0}"

# ---- warm start ------------------------------------------------------------
# Path to the released GAMBAS generator weights, used to initialise training.
# At n~50 this is likely the biggest single win available. Download from
#   https://github.com/levente-1/GAMBAS/releases/tag/v1.0
# Leave empty to train from random initialisation.
export INIT_FROM="${INIT_FROM:-}"
export INIT_MIN_COVERAGE="${INIT_MIN_COVERAGE:-0.5}"

# ---- on-the-fly degradation randomisation ----------------------------------
# Re-simulate the 2 mm input from the 1 mm target per TRAINING sample with a
# randomly drawn apodisation and SNR, instead of using the precomputed
# lowres-<method>/ volume. Validation and test always stay on the precomputed
# fixed condition, so the metric you select and report on stays interpretable.
#
# Off by default: it is a domain-randomisation choice, not a free win. Train
# matched first, then turn this on and compare on the dev split -- at n~50 the
# difference may sit inside the noise, in which case prefer the simpler model.
export RANDOMIZE_DEGRADATION="${RANDOMIZE_DEGRADATION:-0}"
# Tukey taper fraction range. 0 = rectangular truncation, 1 = full Hann. The
# endpoints differ by ~87% in mean gradient magnitude -- far wider than plausible
# vendor variation -- so keep this narrow.
export APOD_RANGE="${APOD_RANGE:-0.0 0.3}"
export SNR_RANGE="${SNR_RANGE:-20 40}"
# Families to sample. Leave as 'kspace': continuous apodisation already spans
# kspace_hann via APOD_RANGE. Do not add 'gaussian' (unphysical, ablation only);
# add 'slab' only for genuinely 2D multi-slice protocols.
export DEGRADATION_MODES="${DEGRADATION_MODES:-kspace}"
export DEGRADATION_P="${DEGRADATION_P:-1.0}"

# ---- subgroup balancing ----------------------------------------------------
# Equalise the t1w/t2w sampling imbalance. 1 = on.
export BALANCE_SUBGROUPS="${BALANCE_SUBGROUPS:-1}"
# 0 = natural frequency, 1 = full balance, 0.5 = sqrt (recommended). On a 38/14
# split, full balance revisits each t2w volume 2.71x as often; sqrt gives 1.65x.
export BALANCE_POWER="${BALANCE_POWER:-0.5}"
export NITER="${NITER:-100}"
export NITER_DECAY="${NITER_DECAY:-100}"
export LR="${LR:-0.0002}"
export WORKERS="${WORKERS:-8}"
export ITERS_PER_EPOCH="${ITERS_PER_EPOCH:-0}"  # >0 = fixed steps/epoch

# ---- SLURM resource requests ----------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"          # e.g. myproject
export CPU_PARTITION="${CPU_PARTITION:-cpu}"
export GPU_PARTITION="${GPU_PARTITION:-gpu}"
export GPU_TYPE="${GPU_TYPE:-}"                    # e.g. a100 -> gres=gpu:a100:1

export PREP_TIME="${PREP_TIME:-02:00:00}"
export PREP_MEM="${PREP_MEM:-16G}"
export PREP_CPUS="${PREP_CPUS:-2}"
export PREP_ARRAY="${PREP_ARRAY:-0-19}"            # 20 shards; %8 to throttle

export TRAIN_TIME="${TRAIN_TIME:-48:00:00}"
export TRAIN_MEM="${TRAIN_MEM:-64G}"
export TRAIN_CPUS="${TRAIN_CPUS:-8}"

export EVAL_TIME="${EVAL_TIME:-04:00:00}"
export EVAL_MEM="${EVAL_MEM:-32G}"

# ---- environment -----------------------------------------------------------
# How to get a working Python with torch + mamba_ssm + SimpleITK.
# Pick ONE of the three styles below by setting ENV_STYLE.
export ENV_STYLE="${ENV_STYLE:-conda}"             # conda | venv | container

export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-gambas}"

export VENV_PATH="${VENV_PATH:-$HOME/venvs/gambas}"

# Apptainer/Singularity image built from the repo Dockerfile
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/scratch/$USER/gambas.sif}"

# Modules to load before anything else (space separated, blank to skip)
export MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"      # e.g. "cuda/12.1 gcc/11.3"

# ---- helper: activate the python environment -------------------------------
activate_env() {
  if [ -n "$MODULES_TO_LOAD" ]; then
    # shellcheck disable=SC2086
    module load $MODULES_TO_LOAD
  fi
  case "$ENV_STYLE" in
    conda)
      # shellcheck disable=SC1090
      source "$CONDA_SH"
      conda activate "$CONDA_ENV"
      ;;
    venv)
      # shellcheck disable=SC1091
      source "$VENV_PATH/bin/activate"
      ;;
    container)
      : # handled by the PY wrapper below
      ;;
    *)
      echo "unknown ENV_STYLE=$ENV_STYLE" >&2; exit 1 ;;
  esac
}

# Run python, transparently going through the container when ENV_STYLE=container.
PY() {
  if [ "$ENV_STYLE" = "container" ]; then
    apptainer exec --nv \
      --bind "$GAMBAS_ROOT:$GAMBAS_ROOT" \
      --bind "$(dirname "$SIM_DIR"):$(dirname "$SIM_DIR")" \
      --bind "$(dirname "$CKPT_DIR"):$(dirname "$CKPT_DIR")" \
      "$CONTAINER_IMAGE" python3 "$@"
  else
    python3 "$@"
  fi
}
export -f activate_env
export -f PY

# ---- helper: assemble sbatch extras ---------------------------------------
sbatch_common() {
  local extras=""
  [ -n "$SLURM_ACCOUNT" ] && extras="$extras --account=$SLURM_ACCOUNT"
  echo "$extras"
}
export -f sbatch_common
