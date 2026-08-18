#!/bin/bash
#SBATCH --job-name=sr_build
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# ============================================================================
# Step 2 -- assemble the GAMBAS-format dataset with a subject-level split.
# Cheap: symlinks only. Runs in seconds.
# ============================================================================
set -euo pipefail

# SLURM runs a COPY of this script from a spool dir; prefer the exported config
# dir or the sbatch submit dir, falling back to BASH_SOURCE for direct runs.
HERE="${SR_CLUSTER_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi
activate_env

echo "SIM_DIR     : $SIM_DIR"
echo "DATASET_DIR : $DATASET_DIR"
cd "$GAMBAS_ROOT"

PY -m sr.build_sr_dataset \
  --sim_dir "$SIM_DIR" \
  --out_root "$DATASET_DIR" \
  --method "$METHOD_TAG" \
  --val_frac "$VAL_FRAC" \
  --test_frac "$TEST_FRAC" \
  --link

echo "--- sanity check ---"
PY -m sr.check_dataset --root "$DATASET_DIR" --n 4
