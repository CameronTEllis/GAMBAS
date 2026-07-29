#!/bin/bash
# ============================================================================
# config.local.sh -- YOUR machine-specific settings.
#
# Every script in this folder prefers this file over config.sh, so your edits
# survive a `git pull`. It sources config.sh first and then lets you override
# only what differs, which means you never have to keep a full copy of the
# template in sync.
#
# Nothing below is active yet -- every override is commented out, so right now
# this file behaves exactly like config.sh. Uncomment and edit what you need.
#
# Sanity-check your settings without submitting anything:
#     DRY_RUN=1 ./submit_all.sh
#     DRY_RUN=1 ./submit_cv.sh
# ============================================================================

# Load the template defaults first. Overrides go AFTER this line.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

# ---- paths -----------------------------------------------------------------
# export GAMBAS_ROOT="$HOME/code/GAMBAS"
# export HR_DIR="/scratch/$USER/data/hr_1mm"          # your 38 T1w + 14 T2w
# export SIM_DIR="/scratch/$USER/data/sr_sim"
# export DATASET_DIR="/scratch/$USER/data/sr_dataset"
# export CKPT_DIR="/scratch/$USER/checkpoints"
# export EVAL_DIR="/scratch/$USER/eval"
# export LOG_DIR="/scratch/$USER/slurm_logs"
# export EXP_NAME="sr_2mm_to_1mm"

# ---- environment -----------------------------------------------------------
# export ENV_STYLE="conda"                            # conda | venv | container
# export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
# export CONDA_ENV="gambas"
# export MODULES_TO_LOAD="cuda/12.1 gcc/11.3"

# ---- SLURM -----------------------------------------------------------------
# export SLURM_ACCOUNT="myproject"
# export CPU_PARTITION="cpu"
# export GPU_PARTITION="gpu"
# export GPU_TYPE="a100"                              # -> --gres=gpu:a100:1
# export TRAIN_TIME="48:00:00"
# export TRAIN_MEM="64G"

# ---- simulation ------------------------------------------------------------
# Both weightings are 1 mm isotropic, so both are 3D acquisitions and `kspace`
# is correct for both. Switch to `slab` only if one of them is really a 2D TSE
# reconstructed to 1 mm slices.
# export SIM_MODE="kspace"
# export TARGET_SNR="30"

# ---- training --------------------------------------------------------------
# Start at LAMBDA_ADV=0.0 (pure L1). At n=52 a discriminator can effectively
# memorise the training set, and switching it off also frees the memory for a
# 128^3 patch.
# export PATCH_SIZE="128 128 128"
# export LAMBDA_ADV="0.0"
# export NITER="100"
# export NITER_DECAY="100"

# ---- splits (used by submit_cv.sh) ----------------------------------------
# export SUBGROUP_REGEX='(T1w|T2w)'
# export DEV_TEST_COUNTS="T1w=8,T2w=3"
# export DEV_VAL_COUNTS="T1w=5,T2w=2"
# export VAL_MODE="rotate"                            # rotate | carve
