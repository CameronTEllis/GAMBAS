#!/bin/bash
# ============================================================================
# submit_all.sh -- chain the four stages with SLURM dependencies.
#
#   ./submit_all.sh              # everything: simulate -> build -> train -> eval
#   ./submit_all.sh train eval   # only those stages (no dependency on earlier ones)
#   DRY_RUN=1 ./submit_all.sh    # print the sbatch commands and exit
#
# Resource requests come from config.sh (or config.local.sh) and are passed on
# the sbatch command line, which overrides the #SBATCH fallbacks in each script.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi

mkdir -p "$LOG_DIR"
COMMON="$(sbatch_common)"
DRY_RUN="${DRY_RUN:-0}"

run_sbatch() {
  local desc="$1"; shift
  # Informational output MUST go to stderr. This function's stdout is captured
  # by `JID=$(run_sbatch ...)` to extract the job id, so anything printed to
  # stdout other than the id ends up inside the id -- which then lands in
  # `--dependency=afterok:<garbage>` and the chain silently never runs.
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] $desc" >&2
    echo "          sbatch --export=ALL,SR_CLUSTER_DIR=$HERE $*" >&2
    echo "PENDING"
    return
  fi
  local out
  # Pass the real config dir to the job. SLURM copies the batch script to a spool
  # dir before running it, so the script's own BASH_SOURCE points there, not to
  # sr/cluster -- which is why sourcing config.sh via BASH_SOURCE fails. Exporting
  # SR_CLUSTER_DIR (and the scripts preferring it) makes config.sh findable no
  # matter where this was launched from.
  out="$(sbatch --export="ALL,SR_CLUSTER_DIR=$HERE" "$@")"
  echo "$desc -> $out" >&2
  echo "$out" | awk '{print $NF}'
}

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(simulate build train eval)
fi
contains() { local n="$1"; shift; for s in "$@"; do [ "$s" = "$n" ] && return 0; done; return 1; }

DEP=""

# ---- 1. simulate -----------------------------------------------------------
if contains simulate "${STAGES[@]}"; then
  # shellcheck disable=SC2086
  JID_SIM=$(run_sbatch "simulate" $COMMON \
      --job-name=sr_sim \
      --partition="$CPU_PARTITION" \
      --array="$PREP_ARRAY" \
      --cpus-per-task="$PREP_CPUS" \
      --mem="$PREP_MEM" \
      --time="$PREP_TIME" \
      --output="$LOG_DIR/sr_sim_%A_%a.out" \
      --error="$LOG_DIR/sr_sim_%A_%a.err" \
      "$HERE/01_simulate.sbatch")
  DEP="--dependency=afterok:$JID_SIM"
fi

# ---- 2. build dataset ------------------------------------------------------
if contains build "${STAGES[@]}"; then
  # shellcheck disable=SC2086
  JID_BUILD=$(run_sbatch "build_dataset" $COMMON $DEP \
      --job-name=sr_build \
      --partition="$CPU_PARTITION" \
      --cpus-per-task=1 --mem=4G --time=00:30:00 \
      --output="$LOG_DIR/sr_build_%j.out" \
      --error="$LOG_DIR/sr_build_%j.err" \
      "$HERE/02_build_dataset.sbatch")
  DEP="--dependency=afterok:$JID_BUILD"
fi

# ---- 3. train --------------------------------------------------------------
if contains train "${STAGES[@]}"; then
  GRES="gpu:1"
  [ -n "$GPU_TYPE" ] && GRES="gpu:${GPU_TYPE}:1"
  # shellcheck disable=SC2086
  CONSTRAINT_OPT=""; [ -n "${GPU_CONSTRAINT:-}" ] && CONSTRAINT_OPT="--constraint=${GPU_CONSTRAINT}"
  JID_TRAIN=$(run_sbatch "train" $COMMON $DEP \
      --job-name="$EXP_NAME" \
      --partition="$GPU_PARTITION" \
      --gres="$GRES" $CONSTRAINT_OPT \
      --cpus-per-task="$TRAIN_CPUS" \
      --mem="$TRAIN_MEM" \
      --time="$TRAIN_TIME" \
      --requeue \
      --output="$LOG_DIR/${EXP_NAME}_%j.out" \
      --error="$LOG_DIR/${EXP_NAME}_%j.err" \
      "$HERE/03_train.sbatch")
  DEP="--dependency=afterany:$JID_TRAIN"
fi

# ---- 4. evaluate -----------------------------------------------------------
if contains eval "${STAGES[@]}"; then
  GRES="gpu:1"
  [ -n "$GPU_TYPE" ] && GRES="gpu:${GPU_TYPE}:1"
  # shellcheck disable=SC2086
  CONSTRAINT_OPT=""; [ -n "${GPU_CONSTRAINT:-}" ] && CONSTRAINT_OPT="--constraint=${GPU_CONSTRAINT}"
  JID_EVAL=$(run_sbatch "evaluate" $COMMON $DEP \
      --job-name=sr_eval \
      --partition="$GPU_PARTITION" \
      --gres="$GRES" $CONSTRAINT_OPT \
      --cpus-per-task=4 \
      --mem="$EVAL_MEM" \
      --time="$EVAL_TIME" \
      --output="$LOG_DIR/sr_eval_%j.out" \
      --error="$LOG_DIR/sr_eval_%j.err" \
      "$HERE/04_evaluate.sbatch")
fi

echo
echo "submitted. watch with:  squeue -u $USER"
echo "logs in:                $LOG_DIR"
