"""
clinical_utils.py — Clinical feature encoding, copied from the training repo
(task2_3_prediction/utils.py) so the container reproduces training bit-for-bit.

Usage in the container
----------------------
1. At startup, fit stats on all non-CHUV training patients (matches the
   `predict_test` path in train_multitask.py):

       df = pd.read_csv(TRAIN_CSV).set_index("PatientID")
       df_alltrain = df.loc[~df.index.str.startswith("CHUV")]
       _, stg_stats = prepare_staging_clinical(df_alltrain)
       _, rfs_stats = prepare_rfs_clinical(df_alltrain)

2. Per test patient, build a 1-row DataFrame from the ehr.json dict (same
   column names as the training CSV) and apply the frozen stats:

       stg_vec, _ = prepare_staging_clinical(df_one, stg_stats)   # [1, 4]
       rfs_vec, _ = prepare_rfs_clinical(df_one, rfs_stats)       # [1, 5]
"""

import numpy as np
import pandas as pd

from constants import STAGING_CLINICAL_COLS, RFS_CLINICAL_COLS


def _prepare_clinical_subset(df, base_cols, missing_cols=(), fit_stats=None):
    df = df.copy()

    # HPV 3-group dummies (HPV-positive is the reference group)
    if "HPV Status" in df.columns:
        hpv = df["HPV Status"]
        df["HPV_neg"] = (hpv == 0).astype(float)
        df["HPV_unk"] = hpv.isna().astype(float)

    # Gender encoding
    if "Gender" in df.columns:
        if df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
        df["Gender"] = df["Gender"].fillna(0.0)

    # Silent median imputation for non-Age/Gender base_cols (e.g. Treatment)
    silent_cols = [c for c in base_cols if c not in ("Age", "Gender")]
    if fit_stats is None:
        silent_medians = {col: float(df[col].median()) for col in silent_cols}
    else:
        silent_medians = fit_stats.get("silent_medians", {})
    for col in silent_cols:
        df[col] = df[col].fillna(silent_medians.get(col, 0.0))

    # Age: impute with training median, then z-score normalise
    if "Age" in base_cols:
        if fit_stats is None:
            age_median = float(df["Age"].median())
            age_mean   = float(df["Age"].mean())
            age_std    = float(df["Age"].std()) + 1e-6
        else:
            age_median = fit_stats["age_median"]
            age_mean   = fit_stats["age_mean"]
            age_std    = fit_stats["age_std"]
        df["Age"] = df["Age"].fillna(age_median)
        df["Age"] = (df["Age"] - age_mean) / age_std
    else:
        age_median = age_mean = age_std = None

    features = df[base_cols].values.astype(np.float32)

    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)

    new_stats = {
        "silent_medians": silent_medians,
        "age_median":     age_median,
        "age_mean":       age_mean,
        "age_std":        age_std,
    }
    return features, new_stats


def prepare_staging_clinical(df, fit_stats=None):
    return _prepare_clinical_subset(df, STAGING_CLINICAL_COLS, [], fit_stats)


def prepare_rfs_clinical(df, fit_stats=None):
    return _prepare_clinical_subset(df, RFS_CLINICAL_COLS, [], fit_stats)


# Column order expected in the 1-row DataFrame built from ehr.json.
EHR_COLUMNS = [
    "Age", "Gender", "CenterID",
    "Tobacco Consumption", "Alcohol Consumption",
    "Performance Status", "Treatment", "HPV Status",
]


def ehr_to_dataframe(ehr: dict) -> pd.DataFrame:
    """Build a 1-row DataFrame from an ehr.json dict, with numeric coercion.

    Missing keys / null values become NaN so the imputation paths above fire
    exactly as they did in training.
    """
    row = {}
    for col in EHR_COLUMNS:
        val = ehr.get(col, None)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            row[col] = np.nan
        else:
            row[col] = val
    df = pd.DataFrame([row])
    # Coerce everything except Gender (may legitimately be "M"/"F") to numeric
    for col in EHR_COLUMNS:
        if col == "Gender":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
