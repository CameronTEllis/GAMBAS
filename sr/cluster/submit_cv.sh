#!/bin/bash
# ============================================================================
# submit_cv.sh -- run the whole K-fold cross-validation as one dependency chain.
#
#   ./submit_cv.sh                 # 5-fold CV, folds_cv.json
#   K=5 ./submit_cv.sh
#   DRY_RUN=1 ./submit_cv.sh       # print the sbatch commands only
#   ./submit_cv.sh dev             # the single development split instead
#
# Layout produced:
#   $DATASET_DIR_cv/fold0/{train,val,test}      materialised split (symlinks)
#   $CKPT_DIR/${EXP_NAME}_fold0/                checkpoints, tb, val_metrics.csv
#   $EVAL_DIR_cv/fold0/per_volume.csv           per-fold metrics
#   $EVAL_DIR_cv/summary/cv_summary.csv         pooled, after the aggregate job
#
# The folds are independent, so all K training jobs are submitted at once and
# the scheduler runs as many as your allocation allows. A final job depends on
# `afterok` of every eval job and pools the results.
#
# IMPORTANT: run the development split first and freeze your hyperparameters
# (patch size, epochs, lambda_adv) before you run CV. Tuning on CV folds and then
# reporting CV numbers is selection bias -- the folds stop being held out.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi

MODE="${1:-cv}"
K="${K:-5}"
FOLDS_JSON="${FOLDS_JSON:-$SIM_DIR/folds_${MODE}.json}"
DRY_RUN="${DRY_RUN:-0}"
COMMON="$(sbatch_common)"
mkdir -p "$LOG_DIR"

GRES="gpu:1"; [ -n "$GPU_TYPE" ] && GRES="gpu:${GPU_TYPE}:1"
# Restrict GPU architectures (see GPU_CONSTRAINT in config.sh). Only the GPU jobs
# (train/eval) take it; build/aggregate are CPU-only.
CONSTRAINT_OPT=""; [ -n "${GPU_CONSTRAINT:-}" ] && CONSTRAINT_OPT="--constraint=${GPU_CONSTRAINT}"

# Regex overrides are optional: an empty value means "derive from NAME_SCHEMA",
# and passing --subject_regex '' would be parsed as a literal empty pattern.
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

# Only pass --exclude_list when the file actually exists: evaluate_sr treats a
# missing path as a hard error, and the list is optional.
EVAL_EXTRA=""
[ -n "${EXCLUDE_LIST:-}" ] && [ -f "${EXCLUDE_LIST}" ] && EVAL_EXTRA="--exclude_list $EXCLUDE_LIST"

NAME_OPTS="--name_schema $NAME_SCHEMA"
[ -n "${SUBJECT_REGEX:-}" ]  && NAME_OPTS="$NAME_OPTS --subject_regex $SUBJECT_REGEX"
[ -n "${SUBGROUP_REGEX:-}" ] && NAME_OPTS="$NAME_OPTS --subgroup_regex $SUBGROUP_REGEX"

run_sbatch() {
  local desc="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] $desc" >&2
    echo "          sbatch $*" >&2
    echo "PENDING"
    return
  fi
  local out; out="$(sbatch "$@")"
  echo "$desc -> $out" >&2
  echo "$out" | awk '{print $NF}'
}

# ---------------------------------------------------------------------------
# 0. generate the fold specification (cheap, do it inline)
# ---------------------------------------------------------------------------
if [ ! -f "$FOLDS_JSON" ]; then
  echo "=== generating $FOLDS_JSON ==="
  activate_env
  cd "$GAMBAS_ROOT"
  if [ "$MODE" = "dev" ]; then
    PY -m sr.make_folds --sim_dir "$SIM_DIR" --out "$FOLDS_JSON" \
      --mode single \
      --test_counts "${DEV_TEST_COUNTS:-t1w=8,t2w=3}" \
      --val_counts  "${DEV_VAL_COUNTS:-t1w=5,t2w=2}" \
      $NAME_OPTS \
      --group_by "$GROUP_BY" --method "$METHOD_TAG"
  else
    PY -m sr.make_folds --sim_dir "$SIM_DIR" --out "$FOLDS_JSON" \
      --mode cv --k "$K" --val_mode "${VAL_MODE:-rotate}" \
      $NAME_OPTS \
      --group_by "$GROUP_BY" --method "$METHOD_TAG"
  fi
else
  echo "reusing existing $FOLDS_JSON (delete it to regenerate)"
fi

# Read the fold ids straight out of the json so this loop always matches the file.
FOLD_IDS=$(python3 -c "
import json,sys
print(' '.join(str(f['fold']) for f in json.load(open('$FOLDS_JSON'))['folds']))")
echo "folds: $FOLD_IDS"

DS_ROOT="${DATASET_DIR}_${MODE}"
EV_ROOT="${EVAL_DIR}_${MODE}"
EVAL_DEPS=()

for FOLD in $FOLD_IDS; do
  TAG="fold${FOLD}"
  DS="$DS_ROOT/$TAG"
  NAME="${EXP_NAME}_${TAG}"
  EV="$EV_ROOT/$TAG"

  # ---- build ---------------------------------------------------------------
  # shellcheck disable=SC2086
  JB=$(run_sbatch "build $TAG" $COMMON \
      --job-name="srb_$TAG" --partition="$CPU_PARTITION" \
      --cpus-per-task=1 --mem=4G --time=00:30:00 \
      --output="$LOG_DIR/srb_${TAG}_%j.out" --error="$LOG_DIR/srb_${TAG}_%j.err" \
      --wrap="set -euo pipefail
source '$HERE/config.local.sh' 2>/dev/null || source '$HERE/config.sh'
activate_env; cd '$GAMBAS_ROOT'
PY -m sr.build_sr_dataset --sim_dir '$SIM_DIR' --out_root '$DS' \
   --method '$METHOD_TAG' --folds_json '$FOLDS_JSON' --fold '$FOLD' --link
PY -m sr.check_dataset --root '$DS' --n 0 --patch_size $PATCH_SIZE")

  # ---- train --------------------------------------------------------------
  # shellcheck disable=SC2086
  JT=$(run_sbatch "train $TAG" $COMMON --dependency=afterok:$JB \
      --job-name="$NAME" --partition="$GPU_PARTITION" --gres="$GRES" $CONSTRAINT_OPT \
      --cpus-per-task="$TRAIN_CPUS" --mem="$TRAIN_MEM" --time="$TRAIN_TIME" \
      --requeue --signal=B:USR1@180 \
      --output="$LOG_DIR/${NAME}_%j.out" --error="$LOG_DIR/${NAME}_%j.err" \
      --wrap="set -euo pipefail
source '$HERE/config.local.sh' 2>/dev/null || source '$HERE/config.sh'
activate_env; cd '$GAMBAS_ROOT'
requeue_handler(){ echo requeueing; scontrol requeue \$SLURM_JOB_ID; exit 0; }
trap requeue_handler USR1
RESUME=''
[ -f '$CKPT_DIR/$NAME/latest_net_G.pth' ] && RESUME='--continue_train --which_epoch latest'
export PYTHONUNBUFFERED=1
PY -m sr.train_sr --data_path '$DS/train' --val_path '$DS/val' \
  --checkpoints_dir '$CKPT_DIR' --name '$NAME' \
  --model gambas --netG gambas --patch_size $PATCH_SIZE \
  --batch_size $BATCH_SIZE --lambda_A $LAMBDA_L1 --lambda_adv $LAMBDA_ADV \
  --l1_edge_weight ${L1_EDGE_WEIGHT:-0.0} \
  --niter $NITER --niter_decay $NITER_DECAY --lr $LR --workers $WORKERS \
  --iters_per_epoch $ITERS_PER_EPOCH --val_freq 5 --val_metric psnr \
  --sr_augment --tensorboard $TRAIN_EXTRA \$RESUME &
wait")

  # ---- evaluate -----------------------------------------------------------
  # shellcheck disable=SC2086
  JE=$(run_sbatch "eval $TAG" $COMMON --dependency=afterok:$JT \
      --job-name="sre_$TAG" --partition="$GPU_PARTITION" --gres="$GRES" $CONSTRAINT_OPT \
      --cpus-per-task=4 --mem="$EVAL_MEM" --time="$EVAL_TIME" \
      --output="$LOG_DIR/sre_${TAG}_%j.out" --error="$LOG_DIR/sre_${TAG}_%j.err" \
      --wrap="set -euo pipefail
source '$HERE/config.local.sh' 2>/dev/null || source '$HERE/config.sh'
activate_env; cd '$GAMBAS_ROOT'
PY -m sr.evaluate_sr --test_path '$DS/test' --checkpoints_dir '$CKPT_DIR' \
  --name '$NAME' --which_epoch '${WHICH_EPOCH:-best}' --out_dir '$EV' \
  --lowres_native_dir '$SIM_DIR/lowres-$METHOD_TAG-native' --patch_size $PATCH_SIZE \
  --fold '$FOLD' --name_schema '$NAME_SCHEMA' --pred_tag '${PRED_TAG:-}' $EVAL_EXTRA \
  --save_predictions")
  EVAL_DEPS+=("$JE")
done

# ---------------------------------------------------------------------------
# final: pool the folds
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" != "1" ]; then
  DEP="afterok:$(IFS=:; echo "${EVAL_DEPS[*]}")"
else
  DEP="afterok:PENDING"
fi
# shellcheck disable=SC2086
run_sbatch "aggregate" $COMMON --dependency="$DEP" \
    --job-name="sr_cvagg" --partition="$CPU_PARTITION" \
    --cpus-per-task=1 --mem=4G --time=00:20:00 \
    --output="$LOG_DIR/sr_cvagg_%j.out" --error="$LOG_DIR/sr_cvagg_%j.err" \
    --wrap="set -euo pipefail
source '$HERE/config.local.sh' 2>/dev/null || source '$HERE/config.sh'
activate_env; cd '$GAMBAS_ROOT'
PY -m sr.aggregate_cv --eval_root '$EV_ROOT' --out '$EV_ROOT/summary'" >/dev/null

echo
echo "submitted $MODE: $(echo "$FOLD_IDS" | wc -w) fold(s)"
echo "  folds spec : $FOLDS_JSON"
echo "  datasets   : $DS_ROOT/fold*"
echo "  checkpoints: $CKPT_DIR/${EXP_NAME}_fold*"
echo "  pooled      : $EV_ROOT/summary/cv_summary.csv"
echo
echo "watch: squeue -u $USER"
