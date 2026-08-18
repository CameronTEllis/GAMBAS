#!/bin/bash
#SBATCH --job-name=sr_sim
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# cpus/mem/time/partition/array are supplied by submit_all.sh via the CLI so
# they can come from config.sh. Sensible fallbacks if you sbatch this directly:
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
# ============================================================================
# Step 1 -- simulate 2 mm acquisitions from the 1 mm volumes.
#
# CPU-only, embarrassingly parallel. Each array task takes every Nth volume,
# so you can resubmit with a different array size without redoing work
# (already-written outputs are skipped unless --overwrite).
#
#   sbatch --array=0-19%8 01_simulate.sbatch      # 20 shards, 8 at a time
# ============================================================================
set -euo pipefail

# SLURM runs a COPY of this script from a spool dir, so BASH_SOURCE does not point
# at sr/cluster. Prefer SR_CLUSTER_DIR (exported by submit_all.sh) or
# SLURM_SUBMIT_DIR (the dir you ran sbatch from); fall back to BASH_SOURCE for a
# direct non-SLURM run.
HERE="${SR_CLUSTER_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi
activate_env

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host        : $(hostname)"
echo "shard       : $SHARD / $NSHARDS"
echo "HR_DIR      : $HR_DIR"
echo "SIM_DIR     : $SIM_DIR"
echo "spacing     : $TARGET_SPACING"
echo "mode        : $SIM_MODE   snr=$TARGET_SNR   upsample=$UPSAMPLE"
echo "method tag  : $METHOD_TAG  -> $SIM_DIR/lowres-$METHOD_TAG/"
echo "start       : $(date)"

mkdir -p "$SIM_DIR"
cd "$GAMBAS_ROOT"

# shellcheck disable=SC2086
PY -m sr.simulate_lowres \
  --in_dir "$HR_DIR" \
  --out_dir "$SIM_DIR" \
  --target_spacing $TARGET_SPACING \
  --mode "$SIM_MODE" \
  --target_snr "$TARGET_SNR" \
  --upsample "$UPSAMPLE" \
  --layout method \
  --method_tag "$METHOD_TAG" \
  --copy_hr \
  --reorient \
  --shard_index "$SHARD" \
  --num_shards "$NSHARDS"

echo "end         : $(date)"
