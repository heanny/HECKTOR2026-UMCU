"""
train_baselines.py — Clinical-only Cox PH baseline for HECKTOR 2026 Task 3.

Task 3 : Cox Proportional Hazards → RFS prediction
         Metric : Concordance Index (C-index)

HPV encoding (3-group, HPV-POSITIVE as reference)
--------------------------------------------------
  HPV_neg = 1 if HPV negative, 0 otherwise  → coeff = "HPV (negative : positive)"
  HPV_unk = 1 if HPV unknown,  0 otherwise  → coeff = "HPV (unknown  : positive)"
  (both = 0) → HPV positive patient  (reference group)

  Positive coefficients mean higher hazard (shorter RFS) relative to HPV-positive.
  Expected: HPV_neg > 0  (HPV-negative has worse prognosis than HPV-positive).

  Derived: "HPV (negative : unknown)" = coef(HPV_neg) − coef(HPV_unk).

Two models compared
-------------------
  Full     : Age + Gender + HPV_neg + HPV_unk + Treatment  (5 features)
  Reduced  : HPV_neg + HPV_unk + Treatment                 (3 features, significant only)

Per-fold reports
----------------
  • Train C-index  (in-sample)
  • Val   C-index  (held-out CV fold)
  • Test  C-index  (CHUV patients with RFS/Relapse labels)
  • Coefficient table: log-HR, HR, 95% CI, z, p-value, significance

Summary reports
---------------
  • Per-model pooled coefficient table (mean ± std across 5 folds)
  • OOF C-index   (concatenated out-of-fold predictions)
  • Side-by-side C-index comparison: Full vs Reduced

Outputs
-------
  results/predictions/task3/clinical_cox_full/
  results/predictions/task3/clinical_cox_reduced/
    oof_predictions.csv
    test_predictions.csv          (with --predict_test)
    coefficients_fold{0..4}.csv
    coefficients_pooled.csv
    summary.json

Usage
-----
python train_baselines.py
python train_baselines.py --predict_test
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLINICAL_CSV, SPLITS_JSON, SEED, TEST_CENTER, PRED_DIR,
)
from utils import (
    load_splits, get_test_patient_ids,
    compute_cindex,
)


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ─── Feature column names ─────────────────────────────────────────────────────
# Full set: all 5 clinical features
COX_FEAT_COLS_FULL    = ["Age", "Gender", "HPV_neg", "HPV_unk", "Treatment"]
# Reduced: only the statistically significant predictors
COX_FEAT_COLS_REDUCED = ["HPV_neg", "HPV_unk", "Treatment"]

FEAT_LABELS = {
    "Age":        "Age (z-score)",
    "Gender":     "Gender (M=1, F=0)",
    "HPV_neg":    "HPV (negative : positive)",   # HPV-positive is reference (=0)
    "HPV_unk":    "HPV (unknown  : positive)",   # HPV-positive is reference (=0)
    "Treatment":  "Treatment (CRT=1, RT=0)",
}

MODEL_NAMES = {
    "full":    "Full model  (Age + Gender + HPV_neg + HPV_unk + Treatment)",
    "reduced": "Reduced model  (HPV_neg + HPV_unk + Treatment)",
}


# ─── Clinical feature preparation ────────────────────────────────────────────

def _prepare_rfs_3group(df: pd.DataFrame, fit_stats: dict = None):
    """
    Prepare RFS clinical features with 3-group HPV dummy encoding.
    HPV-positive is the reference group (coded 0 in both dummy variables).

    Features returned (5 total, order = COX_FEAT_COLS_FULL):
      0  Age         — z-scored (fit on training)
      1  Gender      — M=1, F=0
      2  HPV_neg     — 1 if HPV negative,     else 0   (ref: HPV positive)
      3  HPV_unk     — 1 if HPV unknown/NaN,  else 0   (ref: HPV positive)
      4  Treatment   — 1=CRT, 0=RT  (median-imputed; 19/782 missing)

    Parameters
    ----------
    df        : DataFrame indexed by PatientID.
    fit_stats : None → fit statistics from df (training set).
                dict → apply pre-fitted statistics (val / test sets).

    Returns
    -------
    features  : np.ndarray [N, 5]
    fit_stats : dict
    """
    df = df.copy()

    # ── Gender ────────────────────────────────────────────────────────────────
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0}).fillna(0.0)
    df["Gender"] = df["Gender"].astype(np.float32)

    # ── HPV 3-group dummies (HPV-positive is reference) ──────────────────────
    df["HPV_neg"] = (df["HPV Status"] == 0).astype(np.float32)
    df["HPV_unk"] = df["HPV Status"].isna().astype(np.float32)

    # ── Treatment median imputation (19/782 missing) ──────────────────────────
    if fit_stats is None:
        trt_median = float(df["Treatment"].median())
    else:
        trt_median = fit_stats["trt_median"]
    df["Treatment"] = df["Treatment"].fillna(trt_median).astype(np.float32)

    # ── Age z-score ───────────────────────────────────────────────────────────
    if fit_stats is None:
        age_mean = float(df["Age"].mean())
        age_std  = float(df["Age"].std()) + 1e-6
    else:
        age_mean = fit_stats["age_mean"]
        age_std  = fit_stats["age_std"]
    df["Age"] = ((df["Age"] - age_mean) / age_std).astype(np.float32)

    features = df[COX_FEAT_COLS_FULL].values.astype(np.float32)
    stats    = {
        "age_mean":   age_mean,
        "age_std":    age_std,
        "trt_median": trt_median,
    }
    return features, stats


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if np.isnan(p):  return "  "
    if p < 0.001:    return "***"
    if p < 0.01:     return "** "
    if p < 0.05:     return "*  "
    if p < 0.1:      return ".  "
    return "ns "


def _make_cox_df(X_full: np.ndarray, df_ref: pd.DataFrame,
                 feat_cols: list,
                 duration_col: str = None,
                 event_col: str    = None) -> pd.DataFrame:
    """
    Build a CoxPHFitter DataFrame from the full feature array, selecting
    only the columns in feat_cols.
    X_full column order must match COX_FEAT_COLS_FULL.
    """
    cox = pd.DataFrame(X_full, columns=COX_FEAT_COLS_FULL, index=df_ref.index)
    cox = cox[feat_cols]
    if duration_col is not None:
        cox["duration"] = df_ref[duration_col].values
    if event_col is not None:
        cox["event"] = df_ref[event_col].astype(float).values
    return cox


def _format_coef_table(summary_df: pd.DataFrame, header: str = "") -> str:
    """Render a CoxPH summary DataFrame as a readable table string."""
    SEP = "  " + "-" * 80
    col_hdr = (
        f"  {'Feature':<32}  {'coef':>7}  {'HR':>6}  "
        f"{'HR 95% CI':>18}  {'z':>6}  {'p':>8}  sig"
    )
    lines = [header, col_hdr, SEP] if header else [col_hdr, SEP]

    for feat in summary_df.index:
        row   = summary_df.loc[feat]
        label = FEAT_LABELS.get(feat, feat)
        coef  = row["coef"]
        hr    = row["exp(coef)"]
        lo    = row["exp(coef) lower 95%"]
        hi    = row["exp(coef) upper 95%"]
        z     = row["z"]
        p     = row["p"]
        sig   = _sig_stars(p)
        ci    = f"[{lo:.3f}, {hi:.3f}]"
        lines.append(
            f"  {label:<32}  {coef:>7.4f}  {hr:>6.3f}  "
            f"{ci:>18}  {z:>6.3f}  {p:>8.4f}  {sig}"
        )

    lines.append(SEP)
    lines.append("  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1  ns p≥0.1")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Task 3 — Cox Proportional Hazards (per-fold)
# ════════════════════════════════════════════════════════════════════════════

def _task3_fold(fold_idx: int, train_ids: list, val_ids: list,
                df_full: pd.DataFrame, df_test: pd.DataFrame,
                feat_cols: list):
    """
    Fit Cox model on one fold.

    Parameters
    ----------
    feat_cols : list of column names to include (subset of COX_FEAT_COLS_FULL)

    Returns
    -------
    oof_df    : OOF predictions for val patients
    train_ci  : C-index on training data (in-sample)
    val_ci    : C-index on validation fold
    test_ci   : C-index on CHUV test patients (NaN if too few events)
    stats     : preprocessing fit_stats for this fold
    cph       : fitted CoxPHFitter
    """
    df_tr  = df_full.loc[df_full.index.isin(train_ids)]
    df_val = df_full.loc[df_full.index.isin(val_ids)]

    X_tr,  stats = _prepare_rfs_3group(df_tr)
    X_val, _     = _prepare_rfs_3group(df_val, stats)

    # ── Build and clean training DataFrame ───────────────────────────────────
    cox_tr = _make_cox_df(X_tr, df_tr, feat_cols, "RFS", "Relapse")
    cox_tr = cox_tr.dropna(subset=["duration", "event"])
    cox_tr = cox_tr[cox_tr["duration"] > 0]

    # ── Fit ──────────────────────────────────────────────────────────────────
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_tr, duration_col="duration", event_col="event",
            show_progress=False)

    # ── Train C-index (in-sample, optimistic) ────────────────────────────────
    risk_tr  = cph.predict_log_partial_hazard(cox_tr).values
    train_ci = compute_cindex(
        risk_tr, cox_tr["duration"].values, cox_tr["event"].values
    )

    # ── Validation risk + C-index ─────────────────────────────────────────────
    cox_val  = _make_cox_df(X_val, df_val, feat_cols)
    risk_val = cph.predict_log_partial_hazard(cox_val).values

    val_surv = (
        pd.DataFrame({"PatientID": df_val.index.tolist(), "risk_score": risk_val})
        .merge(df_full[["RFS", "Relapse"]].reset_index(), on="PatientID")
        .dropna(subset=["RFS", "Relapse"])
    )
    val_ci = compute_cindex(
        val_surv["risk_score"].values,
        val_surv["RFS"].values,
        val_surv["Relapse"].astype(float).values,
    )

    # ── Test C-index (CHUV patients with known labels) ────────────────────────
    test_ci = float("nan")
    n_test_used = 0
    n_test_ev   = 0
    if df_test is not None and len(df_test) > 0:
        X_test, _ = _prepare_rfs_3group(df_test, stats)
        cox_test  = _make_cox_df(X_test, df_test, feat_cols, "RFS", "Relapse")
        cox_test  = cox_test.dropna(subset=["duration", "event"])
        cox_test  = cox_test[cox_test["duration"] > 0]
        n_test_used = len(cox_test)
        n_test_ev   = int(cox_test["event"].sum())
        if n_test_used >= 5 and n_test_ev >= 1:
            risk_test = cph.predict_log_partial_hazard(cox_test).values
            test_ci   = compute_cindex(
                risk_test,
                cox_test["duration"].values,
                cox_test["event"].values,
            )

    # ── Console output ────────────────────────────────────────────────────────
    test_str = (
        f"{test_ci:.4f}  (N={n_test_used}, events={n_test_ev})"
        if not np.isnan(test_ci) else "n/a"
    )
    logging.info(
        f"  Train C-index = {train_ci:.4f}  |  "
        f"Val = {val_ci:.4f}  |  Test = {test_str}"
    )
    logging.info(
        "\n" + _format_coef_table(
            cph.summary,
            header=f"  Coefficients — Fold {fold_idx}  [{'+'.join(feat_cols)}]"
        )
    )

    # ── Derived contrast: HPV (negative : unknown) ────────────────────────────
    if "HPV_neg" in feat_cols and "HPV_unk" in feat_cols:
        b_neg = cph.params_["HPV_neg"]
        b_unk = cph.params_["HPV_unk"]
        logging.info(
            f"  Derived  HPV (negative : unknown) = "
            f"coef(HPV_neg) − coef(HPV_unk) = {b_neg:.4f} − {b_unk:.4f} = "
            f"{b_neg - b_unk:.4f}   HR = {np.exp(b_neg - b_unk):.3f}"
        )

    # ── OOF rows ──────────────────────────────────────────────────────────────
    rows = [
        {"PatientID": pid, "fold": fold_idx, "risk_score": float(r)}
        for pid, r in zip(df_val.index.tolist(), risk_val)
    ]
    return pd.DataFrame(rows), train_ci, val_ci, test_ci, stats, cph


# ════════════════════════════════════════════════════════════════════════════
#  run_task3 — runs BOTH models and prints side-by-side comparison
# ════════════════════════════════════════════════════════════════════════════

def _run_one_model(model_key: str, feat_cols: list,
                   df_full: pd.DataFrame, folds: list,
                   test_ids: list, pred_dir: Path,
                   predict_test: bool) -> dict:
    """
    Run 5-fold CV for one Cox model (full or reduced).
    Returns a dict with all C-index results and pooled coefficients.
    """
    df_test = df_full.loc[df_full.index.isin(test_ids)]

    all_oof        = []
    fold_train_ci  = []
    fold_val_ci    = []
    fold_test_ci   = []
    fold_models    = []
    fold_summaries = []

    logging.info(
        f"\n{'█'*64}\n"
        f"  {MODEL_NAMES[model_key]}\n"
        f"  Features: {feat_cols}\n"
        f"{'█'*64}"
    )

    for fold_idx, fold in enumerate(folds):
        train_ids_f = fold["train"]
        val_ids_f   = fold["val"]
        logging.info(
            f"\n  ── Fold {fold_idx}  "
            f"train={len(train_ids_f)}  val={len(val_ids_f)} ──"
        )
        oof_df, tr_ci, val_ci, te_ci, stats, cph = _task3_fold(
            fold_idx, train_ids_f, val_ids_f, df_full, df_test, feat_cols
        )
        all_oof.append(oof_df)
        fold_train_ci.append(tr_ci)
        fold_val_ci.append(val_ci)
        fold_test_ci.append(te_ci)
        fold_models.append((stats, cph))
        fold_summaries.append(cph.summary.copy())

        # Save per-fold coefficient CSV
        coef_out = cph.summary[
            ["coef", "exp(coef)", "exp(coef) lower 95%",
             "exp(coef) upper 95%", "z", "p"]
        ].copy()
        coef_out.index = [FEAT_LABELS.get(c, c) for c in coef_out.index]
        coef_out.to_csv(pred_dir / f"coefficients_fold{fold_idx}.csv")

    # ── OOF C-index ───────────────────────────────────────────────────────────
    oof = pd.concat(all_oof, ignore_index=True)
    oof.to_csv(pred_dir / "oof_predictions.csv", index=False)

    oof_lbl = (
        oof
        .join(df_full[["RFS", "Relapse"]], on="PatientID")
        .dropna(subset=["RFS", "Relapse"])
    )
    oof_ci = compute_cindex(
        oof_lbl["risk_score"].values,
        oof_lbl["RFS"].values,
        oof_lbl["Relapse"].astype(float).values,
    )

    # ── Pooled coefficient table (mean ± std across 5 folds) ─────────────────
    coef_keys = ["coef", "exp(coef)",
                 "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p"]
    pooled_rows = []
    for feat in feat_cols:
        row = {"feature": FEAT_LABELS.get(feat, feat)}
        for col in coef_keys:
            vals = [s.loc[feat, col] for s in fold_summaries]
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"]  = float(np.std(vals))
        pooled_rows.append(row)

    pooled_df = pd.DataFrame(pooled_rows).set_index("feature")
    pooled_df.to_csv(pred_dir / "coefficients_pooled.csv")

    pool_disp = pd.DataFrame(
        {col: [r[f"{col}_mean"] for r in pooled_rows] for col in coef_keys},
        index=feat_cols,
    )

    # ── Derived contrast: HPV (negative : unknown) ────────────────────────────
    derived = []
    if "HPV_neg" in feat_cols and "HPV_unk" in feat_cols:
        neg_coefs = [s.loc["HPV_neg", "coef"] for s in fold_summaries]
        unk_coefs = [s.loc["HPV_unk", "coef"] for s in fold_summaries]
        derived   = [n - u for n, u in zip(neg_coefs, unk_coefs)]

    # ── C-index table for this model ──────────────────────────────────────────
    ci_lines = [
        f"\n── {MODEL_NAMES[model_key]} — C-index per fold ──",
        f"  {'Fold':<6}  {'Train':>7}  {'Val':>7}  {'Test(CHUV)':>12}",
        "  " + "-" * 38,
    ]
    for i, (tr, va, te) in enumerate(
            zip(fold_train_ci, fold_val_ci, fold_test_ci)):
        te_s = f"{te:.4f}" if not np.isnan(te) else "    n/a"
        ci_lines.append(f"  {i:<6}  {tr:>7.4f}  {va:>7.4f}  {te_s:>12}")
    ci_lines += [
        "  " + "-" * 38,
        f"  {'mean':<6}  {np.mean(fold_train_ci):>7.4f}  "
        f"{np.mean(fold_val_ci):>7.4f}  "
        f"{np.nanmean(fold_test_ci):>12.4f}",
        f"  {'±std':<6}  {np.std(fold_train_ci):>7.4f}  "
        f"{np.std(fold_val_ci):>7.4f}  "
        f"{np.nanstd(fold_test_ci):>12.4f}",
        f"\n  OOF C-index (all {len(oof_lbl)} labeled val patients) = {oof_ci:.4f}",
    ]
    logging.info("\n".join(ci_lines))

    # ── Pooled coefficient table ──────────────────────────────────────────────
    logging.info(
        "\n" + _format_coef_table(
            pool_disp,
            header=f"── Pooled Coefficients — {MODEL_NAMES[model_key]}"
        )
    )

    std_lines = ["  coef ± std per feature across folds:"]
    for i, feat in enumerate(feat_cols):
        std_lines.append(
            f"    {FEAT_LABELS[feat]:<32}  "
            f"{pooled_rows[i]['coef_mean']:>+7.4f} ± {pooled_rows[i]['coef_std']:.4f}"
        )
    if derived:
        std_lines += [
            "",
            f"  Derived  HPV (negative : unknown)  =  "
            f"coef(HPV_neg) − coef(HPV_unk)",
            f"    mean = {np.mean(derived):+.4f} ± {np.std(derived):.4f}   "
            f"HR = {np.exp(np.mean(derived)):.3f}",
        ]
    logging.info("\n".join(std_lines))

    # ── Save summary JSON ─────────────────────────────────────────────────────
    valid_te = [c for c in fold_test_ci if not np.isnan(c)]
    summary  = {
        "model":          f"CoxPH_{model_key}_3groupHPV",
        "features":       feat_cols,
        "HPV_reference":  "HPV positive",
        "penalizer":      0.1,
        "fold_train_ci":  fold_train_ci,
        "fold_val_ci":    fold_val_ci,
        "fold_test_ci":   fold_test_ci,
        "mean_train_ci":  float(np.mean(fold_train_ci)),
        "mean_val_ci":    float(np.mean(fold_val_ci)),
        "std_val_ci":     float(np.std(fold_val_ci)),
        "mean_test_ci":   float(np.nanmean(fold_test_ci)) if valid_te else None,
        "oof_ci":         float(oof_ci),
        "pooled_coef": {
            FEAT_LABELS.get(feat, feat): {
                "coef_mean": pooled_rows[i]["coef_mean"],
                "coef_std":  pooled_rows[i]["coef_std"],
                "HR_mean":   pooled_rows[i]["exp(coef)_mean"],
                "p_mean":    pooled_rows[i]["p_mean"],
            }
            for i, feat in enumerate(feat_cols)
        },
    }
    if derived:
        summary["derived_HPV_neg_vs_unk"] = {
            "coef_mean":  float(np.mean(derived)),
            "coef_std":   float(np.std(derived)),
            "HR_mean":    float(np.exp(np.mean(derived))),
        }
    with open(pred_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"  Outputs saved → {pred_dir}")

    # ── Test predictions: ensemble of all 5 Cox models ────────────────────────
    if predict_test:
        risk_all = []
        for fold_stats, cph in fold_models:
            X_test, _ = _prepare_rfs_3group(df_test, fold_stats)
            cox_t     = _make_cox_df(X_test, df_test, feat_cols)
            risk_all.append(cph.predict_log_partial_hazard(cox_t).values)
        risk_avg = np.mean(risk_all, axis=0)
        out = pred_dir / "test_predictions.csv"
        pd.DataFrame({
            "PatientID":  df_test.index.tolist(),
            "risk_score": risk_avg,
        }).to_csv(out, index=False)
        logging.info(f"  Test predictions (5-fold ensemble) → {out}")

    return {
        "fold_train_ci": fold_train_ci,
        "fold_val_ci":   fold_val_ci,
        "fold_test_ci":  fold_test_ci,
        "oof_ci":        float(oof_ci),
    }


def run_task3(df_full: pd.DataFrame, folds: list, test_ids: list,
              base_pred_dir: Path, predict_test: bool = False):
    """Run full and reduced models, then print side-by-side comparison."""

    results = {}
    for model_key, feat_cols in [
        ("full",    COX_FEAT_COLS_FULL),
        ("reduced", COX_FEAT_COLS_REDUCED),
    ]:
        pred_dir = base_pred_dir / f"clinical_cox_{model_key}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        results[model_key] = _run_one_model(
            model_key, feat_cols, df_full, folds, test_ids, pred_dir, predict_test
        )

    # ── Side-by-side comparison ───────────────────────────────────────────────
    full_r    = results["full"]
    reduced_r = results["reduced"]

    cmp_lines = [
        "\n" + "═" * 72,
        "  MODEL COMPARISON — Full vs Reduced (HPV + Treatment only)",
        "═" * 72,
        f"  {'Model':<30}  {'OOF C-idx':>9}  {'Val mean':>8}  "
        f"{'Val ±std':>8}  {'Test mean':>9}",
        "  " + "-" * 66,
    ]
    for _, name, r in [
        ("full",    "Full  (Age+Gender+HPV+Trt)", full_r),
        ("reduced", "Reduced  (HPV+Trt only)",    reduced_r),
    ]:
        val_ci = r["fold_val_ci"]
        cmp_lines.append(
            f"  {name:<30}  {r['oof_ci']:>9.4f}  "
            f"{np.mean(val_ci):>8.4f}  "
            f"{np.std(val_ci):>8.4f}  "
            f"{np.nanmean(r['fold_test_ci']):>9.4f}"
        )

    cmp_lines += [
        "  " + "-" * 66,
        "",
        "  Per-fold Val C-index:",
        f"  {'Fold':<6}  {'Full':>7}  {'Reduced':>9}  {'Δ (Full−Red)':>13}",
        "  " + "-" * 38,
    ]
    for i, (f_ci, r_ci) in enumerate(
            zip(full_r["fold_val_ci"], reduced_r["fold_val_ci"])):
        cmp_lines.append(
            f"  {i:<6}  {f_ci:>7.4f}  {r_ci:>9.4f}  {f_ci - r_ci:>+13.4f}"
        )
    cmp_lines += [
        "  " + "-" * 38,
        f"  {'mean':<6}  "
        f"{np.mean(full_r['fold_val_ci']):>7.4f}  "
        f"{np.mean(reduced_r['fold_val_ci']):>9.4f}  "
        f"{np.mean(full_r['fold_val_ci']) - np.mean(reduced_r['fold_val_ci']):>+13.4f}",
        "═" * 72,
    ]
    logging.info("\n".join(cmp_lines))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Clinical-only Cox PH baseline for HECKTOR 2026 Task 3"
    )
    p.add_argument(
        "--predict_test", action="store_true",
        help="Save averaged CHUV risk scores after all folds.",
    )
    args = p.parse_args()

    set_seed(SEED)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler()],
    )

    df_full  = pd.read_csv(CLINICAL_CSV).set_index("PatientID")
    folds    = load_splits(SPLITS_JSON)
    test_ids = get_test_patient_ids(CLINICAL_CSV, TEST_CENTER)

    logging.info(
        f"Patients={len(df_full)}  Folds={len(folds)}  "
        f"Test(CHUV)={len(test_ids)}"
    )

    base_pred_dir = PRED_DIR / "task3"
    run_task3(df_full, folds, test_ids, base_pred_dir, args.predict_test)


if __name__ == "__main__":
    main()
