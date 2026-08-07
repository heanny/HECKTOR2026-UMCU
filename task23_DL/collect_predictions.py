"""
collect_predictions.py — Combine all ensembled predictions into one Excel file.

Output: results/predictions/all_predictions.xlsx

Layout
------
  Rows    : one per patient (train + test)
  Columns : PatientID, split, then one or more columns per (task, run_tag):

    task2     -> {prefix}_T_pred   (int 0-3: T1=0 T2=1 T3=2 T4=3)
                {prefix}_N_pred   (int 0-3: N0=0 N1=1 N2=2 N3=3)
    task3     -> {prefix}_risk
    multitask -> {prefix}_T_pred, {prefix}_N_pred, {prefix}_risk

  where prefix = "{task}__{run_tag}"  (e.g. "task2__clinical+CT+PET")

  Split is determined by which file the patient came from:
    train_ensemble_predictions.csv -> "train"
    test_predictions.csv           -> "test"

Usage
-----
  python collect_predictions.py
  python collect_predictions.py --out results/predictions/my_preds.xlsx
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

PRED_ROOT = Path("results/predictions")

# Which output columns to include per task (always included)
TASK_COLS = {
    "task2":     ["T_pred", "N_pred"],
    "task3":     ["risk_score"],
    "multitask": ["T_pred", "N_pred", "risk_score"],
}

# Probability columns (only added with --prob)
TASK_PROB_COLS = {
    "task2":     [f"T_prob_T{j+1}" for j in range(4)] + [f"N_prob_N{j}" for j in range(4)],
    "task3":     [],
    "multitask": [f"T_prob_T{j+1}" for j in range(4)] + [f"N_prob_N{j}" for j in range(4)],
}

# Friendly rename for output column suffix
COL_RENAME = {"risk_score": "risk"}


def _short(col: str) -> str:
    return COL_RENAME.get(col, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PRED_ROOT / "all_predictions.xlsx"),
                    help="Output Excel path.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # {PatientID: {col_name: value, ..., "split": "train"/"test"}}
    records: dict[str, dict] = {}

    for task, base_cols in TASK_COLS.items():
        task_dir = PRED_ROOT / task
        if not task_dir.exists():
            print(f"  [skip] {task_dir} not found")
            continue

        want_cols = base_cols + TASK_PROB_COLS[task]

        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_tag = run_dir.name
            prefix  = f"{task}__{run_tag}"

            for fname, split_label in [
                ("train_ensemble_predictions.csv", "train"),
                ("test_predictions.csv",           "test"),
            ]:
                fpath = run_dir / fname
                if not fpath.exists():
                    continue

                df = pd.read_csv(fpath).set_index("PatientID")

                for pid in df.index:
                    rec = records.setdefault(str(pid), {"split": split_label})
                    for col in want_cols:
                        if col in df.columns:
                            rec[f"{prefix}__{_short(col)}"] = df.at[pid, col]

    if not records:
        print("No prediction files found. Run run_generate_train_preds.sh first.")
        sys.exit(1)

    df_out = pd.DataFrame.from_dict(records, orient="index")
    df_out.index.name = "PatientID"
    df_out = df_out.reset_index()

    # Sort: train first, then test; alphabetical within each group
    df_out = (df_out
              .assign(_sort=df_out["split"].map({"train": 0, "test": 1}))
              .sort_values(["_sort", "PatientID"])
              .drop(columns="_sort")
              .reset_index(drop=True))

    # Reorder: PatientID, split, then prediction columns in task/run_tag order
    pred_cols = [c for c in df_out.columns if c not in ("PatientID", "split")]
    df_out = df_out[["PatientID", "split"] + pred_cols]

    df_out.to_excel(out_path, index=False)

    n_train = (df_out["split"] == "train").sum()
    n_test  = (df_out["split"] == "test").sum()
    print(f"Saved  {len(df_out)} patients  x  {len(pred_cols)} prediction columns  ->  {out_path}")
    print(f"  train : {n_train} patients")
    print(f"  test  : {n_test}  patients")
    print(f"  tasks found: {sorted({c.split('__')[0] for c in pred_cols})}")


if __name__ == "__main__":
    main()
