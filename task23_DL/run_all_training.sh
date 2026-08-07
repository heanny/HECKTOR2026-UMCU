#!/bin/bash
# run_all_training.sh — Train all input combinations for Task 2, 3, and Multitask
#
# Usage:
#   bash run_all_training.sh                   # default GPU=3
#   GPU=0 bash run_all_training.sh             # override GPU
#
# To run scripts in parallel across GPUs, e.g.:
#   GPU=1 SCRIPTS="train_task2.py"     bash run_all_training.sh &
#   GPU=2 SCRIPTS="train_task3.py"     bash run_all_training.sh &
#   GPU=3 SCRIPTS="train_multitask.py" bash run_all_training.sh &

cd /autofs/arch11/DATA/HOMES/HECKTOR_2026/task2_3_prediction

GPU=${GPU:-3}

# ── Scripts to run (override with env var SCRIPTS) ────────────────────────────
SCRIPTS=(
    "train_task2.py"
    "train_task3.py"
    "train_multitask.py"
)

# ── Input combinations (ordered roughly by complexity) ────────────────────────
# Each entry = space-separated --input tokens; bash word-splits them for argparse
COMBOS=(
    "clinical"
    "CT PET"
    "clinical CT PET"
    "prob_T prob_N"
    "clinical prob_T prob_N"
    "CT prob_T prob_N"
    "clinical CT prob_T prob_N"
    "PET prob_T prob_N"
    "CT PET prob_T prob_N"
    "clinical CT PET prob_T prob_N"
    "CT PET hard_T hard_N"
    "clinical CT PET hard_T hard_N"
    "CT PET prob_T prob_N hard_T hard_N"
    "clinical CT PET prob_T prob_N hard_T hard_N"
)

# ─────────────────────────────────────────────────────────────────────────────

TOTAL=$(( ${#SCRIPTS[@]} * ${#COMBOS[@]} ))
COUNT=0
FAILED=()
START_ALL=$(date +%s)

for SCRIPT in "${SCRIPTS[@]}"; do
    # Derive task name for checkpoint path lookup
    case "$SCRIPT" in
        "train_task2.py")     TASK_NAME="task2" ;;
        "train_task3.py")     TASK_NAME="task3" ;;
        "train_multitask.py") TASK_NAME="multitask" ;;
        *)                    TASK_NAME="unknown" ;;
    esac

    for COMBO in "${COMBOS[@]}"; do
        COUNT=$(( COUNT + 1 ))

        # Skip if all 5 folds already finished (summary.json is written after fold 4)
        RUN_TAG="${COMBO// /+}"
        SUMMARY="results/checkpoints/${TASK_NAME}/${RUN_TAG}/summary.json"
        if [ -f "$SUMMARY" ]; then
            echo ""
            printf "  [%2d/%d]  ↷  SKIP (done)  python %s --input %s\n" \
                   "$COUNT" "$TOTAL" "$SCRIPT" "$COMBO"
            continue
        fi

        START=$(date +%s)

        echo ""
        echo "══════════════════════════════════════════════════════════════════"
        printf "  [%2d/%d]  %s  python %s --input %s\n" \
               "$COUNT" "$TOTAL" "$(date '+%H:%M:%S')" "$SCRIPT" "$COMBO"
        echo "══════════════════════════════════════════════════════════════════"

        # $COMBO intentionally unquoted so bash splits tokens → argparse sees them separately
        if CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT --input $COMBO --predict_test --predict_train; then
            ELAPSED=$(( $(date +%s) - START ))
            echo "  ✓  done in ${ELAPSED}s"
        else
            ELAPSED=$(( $(date +%s) - START ))
            echo "  ✗  FAILED after ${ELAPSED}s  →  $SCRIPT --input $COMBO"
            FAILED+=("$SCRIPT --input $COMBO")
        fi
    done
done

TOTAL_ELAPSED=$(( $(date +%s) - START_ALL ))
HOURS=$(( TOTAL_ELAPSED / 3600 ))
MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "══════════════════════════════════════════════════════════════════"
printf "  Finished %d runs in %dh %02dm\n" "$TOTAL" "$HOURS" "$MINS"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All runs succeeded."
else
    echo "  ${#FAILED[@]} run(s) FAILED:"
    for F in "${FAILED[@]}"; do
        echo "    ✗  $F"
    done
fi
echo "══════════════════════════════════════════════════════════════════"
