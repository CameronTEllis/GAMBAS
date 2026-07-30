#!/bin/bash
#SBATCH --job-name=sr_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
# ============================================================================
# Step 3 -- train.
#
# Requeue-on-timeout: the job traps SIGUSR1 (sent 180 s before the wall clock
# expires by --signal above), saves nothing extra (the trainer already
# checkpoints `latest` + optimiser state every --save_latest_freq iterations)
# and requeues itself. On restart it picks up with --continue_train.
#
#   sbatch --requeue 03_train.sbatch
#
# Memory guidance for a 128^3 patch, batch 1, ngf 64: expect ~30-40 GB of GPU
# memory for the GAN (G + D + pool). If you OOM:
#   PATCH_SIZE="96 96 96"      most effective single change
#   LAMBDA_ADV=0.0            drops the discriminator entirely
#   --amp                     bf16 autocast (verify mamba_ssm tolerates it first)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi
activate_env

# ---- requeue on timeout ----------------------------------------------------
requeue_handler() {
  echo "$(date) caught USR1 -- requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_handler USR1

# Warm start and balancing are optional; build the flags conditionally so an
# empty INIT_FROM does not become --init_from ''.
TRAIN_EXTRA=""
# --global_residual defaults ON in sr_options, so only the opt-out needs passing.
[ "${GLOBAL_RESIDUAL:-1}" = "0" ] && TRAIN_EXTRA="$TRAIN_EXTRA --no_global_residual"
[ -n "${INIT_FROM:-}" ] && TRAIN_EXTRA="$TRAIN_EXTRA --init_from $INIT_FROM --init_min_coverage $INIT_MIN_COVERAGE"
[ "${BALANCE_SUBGROUPS:-0}" = "1" ] && TRAIN_EXTRA="$TRAIN_EXTRA --balance_subgroups --balance_power $BALANCE_POWER --name_schema $NAME_SCHEMA"
# Needs --name_schema too, which the line above supplies only when balancing is on.
[ "${CAP_SUBJECT_SHARE:-0}" != "0" ] && TRAIN_EXTRA="$TRAIN_EXTRA --cap_subject_share $CAP_SUBJECT_SHARE --name_schema $NAME_SCHEMA"
[ "${RANDOMIZE_DEGRADATION:-0}" = "1" ] && TRAIN_EXTRA="$TRAIN_EXTRA --randomize_degradation --apod_range $APOD_RANGE --snr_range $SNR_RANGE --degradation_modes $DEGRADATION_MODES --degradation_p $DEGRADATION_P --target_spacing $TARGET_SPACING"

# ---- resume automatically if a checkpoint already exists -------------------
RESUME_FLAG=""
if [ -f "$CKPT_DIR/$EXP_NAME/latest_net_G.pth" ]; then
  RESUME_FLAG="--continue_train --which_epoch latest"
  echo "found existing checkpoint -> resuming"
fi

echo "host        : $(hostname)"
echo "gpu         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "dataset     : $DATASET_DIR"
echo "checkpoints : $CKPT_DIR/$EXP_NAME"
echo "patch       : $PATCH_SIZE   batch $BATCH_SIZE"
echo "lambda_A    : $LAMBDA_L1    lambda_adv $LAMBDA_ADV"
echo "init_from   : ${INIT_FROM:-<random init>}"
echo "balance     : ${BALANCE_SUBGROUPS:-0} (power $BALANCE_POWER)"
echo "rand degrad : ${RANDOMIZE_DEGRADATION:-0}  apod [$APOD_RANGE]  snr [$SNR_RANGE]  modes $DEGRADATION_MODES"
echo "start       : $(date)"

mkdir -p "$CKPT_DIR/$EXP_NAME"
cd "$GAMBAS_ROOT"

# Run in the background so the USR1 trap above can fire while training is in
# flight (a foreground command would block signal handling until it returned).
#
# Note: this deliberately does NOT use `srun`. PY is a shell function (it wraps
# apptainer when ENV_STYLE=container), and srun execs a binary rather than
# interpreting a shell, so `srun PY ...` fails with "PY: command not found".
# This is a single-task, single-node job, so running directly is equivalent.
export PYTHONUNBUFFERED=1
# shellcheck disable=SC2086
PY -m sr.train_sr \
  --data_path "$DATASET_DIR/train" \
  --val_path  "$DATASET_DIR/val" \
  --checkpoints_dir "$CKPT_DIR" \
  --name "$EXP_NAME" \
  --model gambas --netG gambas \
  --patch_size $PATCH_SIZE \
  --batch_size "$BATCH_SIZE" \
  --lambda_A "$LAMBDA_L1" \
  --lambda_adv "$LAMBDA_ADV" \
  --niter "$NITER" --niter_decay "$NITER_DECAY" \
  --lr "$LR" \
  --workers "$WORKERS" \
  --iters_per_epoch "$ITERS_PER_EPOCH" \
  --val_freq 5 --val_metric psnr \
  --sr_augment \
  --tensorboard \
  $TRAIN_EXTRA \
  $RESUME_FLAG &

wait
echo "end         : $(date)"
