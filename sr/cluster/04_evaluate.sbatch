#!/bin/bash
#SBATCH --job-name=sr_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# ============================================================================
# Step 4 -- evaluate on the held-out test split, with baselines.
# ============================================================================
set -euo pipefail

# SLURM runs a COPY of this script from a spool dir; prefer the exported config
# dir or the sbatch submit dir, falling back to BASH_SOURCE for direct runs.
HERE="${SR_CLUSTER_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi
activate_env

WHICH="${WHICH_EPOCH:-best}"

echo "checkpoint  : $CKPT_DIR/$EXP_NAME/${WHICH}_net_G.pth"
echo "test        : $DATASET_DIR/test"
echo "out         : $EVAL_DIR"
cd "$GAMBAS_ROOT"
mkdir -p "$EVAL_DIR"

# shellcheck disable=SC2086
PY -m sr.evaluate_sr \
  --test_path "$DATASET_DIR/test" \
  --checkpoints_dir "$CKPT_DIR" \
  --name "$EXP_NAME" \
  --which_epoch "$WHICH" \
  --out_dir "$EVAL_DIR" \
  --lowres_native_dir "$SIM_DIR/lowres-$METHOD_TAG-native" \
  --name_schema "$NAME_SCHEMA" \
  --patch_size $PATCH_SIZE \
  --pred_tag "${PRED_TAG:-}" \
  ${EXCLUDE_LIST:+$([ -f "$EXCLUDE_LIST" ] && echo "--exclude_list $EXCLUDE_LIST")} \
  --save_predictions

echo "done: $(date)"
