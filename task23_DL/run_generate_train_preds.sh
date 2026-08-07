#!/bin/bash
# run_generate_train_preds.sh — Generate train_ensemble_predictions.csv for all runs.
#
# Loads existing 5-fold checkpoints (no retraining) and runs all 5 models on
# every training patient, averaging probabilities/risk scores exactly as for
# test predictions.  Output: predictions/<task>/<run_tag>/train_ensemble_predictions.csv
#
# Usage:
#   bash run_generate_train_preds.sh           # default GPU=3
#   GPU=0 bash run_generate_train_preds.sh     # override GPU

cd /autofs/arch11/DATA/HOMES/HECKTOR_2026/task2_3_prediction

GPU=${GPU:-4}

SCRIPTS=(
    "train_task2.py"
    "train_task3.py"
    "train_multitask.py"
)

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

TOTAL=$(( ${#SCRIPTS[@]} * ${#COMBOS[@]} ))
COUNT=0
FAILED=()
START_ALL=$(date +%s)

for SCRIPT in "${SCRIPTS[@]}"; do
    case "$SCRIPT" in
        "train_task2.py")     TASK_NAME="task2" ;;
        "train_task3.py")     TASK_NAME="task3" ;;
        "train_multitask.py") TASK_NAME="multitask" ;;
        *)                    TASK_NAME="unknown" ;;
    esac

    for COMBO in "${COMBOS[@]}"; do
        COUNT=$(( COUNT + 1 ))

        RUN_TAG="${COMBO// /+}"
        OUT="results/predictions/${TASK_NAME}/${RUN_TAG}/train_ensemble_predictions.csv"
        if [ -f "$OUT" ]; then
            printf "  [%2d/%d]  ↷  SKIP (exists)  %s  --input %s\n" \
                   "$COUNT" "$TOTAL" "$SCRIPT" "$COMBO"
            continue
        fi

        START=$(date +%s)
        echo ""
        echo "══════════════════════════════════════════════════════════════════"
        printf "  [%2d/%d]  %s  python %s --input %s\n" \
               "$COUNT" "$TOTAL" "$(date '+%H:%M:%S')" "$SCRIPT" "$COMBO"
        echo "══════════════════════════════════════════════════════════════════"

        if CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT --input $COMBO --predict_only --predict_train; then
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
