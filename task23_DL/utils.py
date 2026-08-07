"""
utils.py — Helper functions shared by Task 2, Task 3, and multi-task training.

Clinical feature preparation
─────────────────────────────
Two task-specific entry points:

  prepare_staging_clinical(df, fit_stats=None)
      → [N, N_STAGING_CLINICAL]  (Age, Gender, HPV_neg, HPV_unk)
      Used for: Task 2 standalone, staging branch in multi-task model.

  prepare_rfs_clinical(df, fit_stats=None)
      → [N, N_RFS_CLINICAL]  (Age, Gender, HPV_neg, HPV_unk, Treatment)
      Used for: Task 3 standalone, RFS branch in multi-task model.

HPV encoding (3-group, matches train_baselines.py):
  HPV_neg = 1 if HPV Status == 0 (negative),  0 otherwise
  HPV_unk = 1 if HPV Status is NaN (unknown),  0 otherwise
  (both = 0) → HPV-positive patient  (reference group)

Both return (features: np.ndarray, fit_stats: dict).
Pass the returned fit_stats to val/test calls to avoid data leakage.

Input argument parsing
──────────────────────
  parse_input_arg(input_tokens)
      Split the --input token list into (use_clinical, image_channels).
      "clinical" is the special token; everything else must be an image channel.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines.utils import concordance_index

from config import (
    STAGING_CLINICAL_COLS, STAGING_MISSING_COLS,
    RFS_CLINICAL_COLS,     RFS_MISSING_COLS,
    ALL_IMAGE_CHANNELS,    ALL_INPUT_TOKENS,
)


# ─── Input argument parsing ───────────────────────────────────────────────────

def parse_input_arg(input_tokens: list) -> tuple:
    """
    Parse the --input argument list into (use_clinical, image_channels).

    "clinical" is the special token for the clinical feature branch.
    All other tokens must be valid image channel names from ALL_IMAGE_CHANNELS.

    Parameters
    ----------
    input_tokens : list of str
        e.g. ["clinical", "CT", "PET", "prob_T", "prob_N"]

    Returns
    -------
    use_clinical   : bool  — True if "clinical" was in input_tokens
    image_channels : list  — ordered list of image channel names (may be [])

    Examples
    --------
    parse_input_arg(["clinical", "CT", "PET", "prob_T", "prob_N"])
        → (True,  ["CT", "PET", "prob_T", "prob_N"])
    parse_input_arg(["CT", "PET", "prob_T", "prob_N"])
        → (False, ["CT", "PET", "prob_T", "prob_N"])
    parse_input_arg(["clinical"])
        → (True,  [])

    Raises
    ------
    ValueError  if any token is unknown, or if the list is effectively empty.
    """
    unknown = [t for t in input_tokens if t not in ALL_INPUT_TOKENS]
    if unknown:
        raise ValueError(
            f"Unknown --input tokens: {unknown}. "
            f"Valid tokens: {ALL_INPUT_TOKENS}"
        )
    use_clinical   = "clinical" in input_tokens
    image_channels = [t for t in input_tokens if t in ALL_IMAGE_CHANNELS]
    if not use_clinical and not image_channels:
        raise ValueError(
            "--input must contain at least 'clinical' or one image channel. "
            f"Valid: {ALL_INPUT_TOKENS}"
        )
    return use_clinical, image_channels


# ─── Split helpers ────────────────────────────────────────────────────────────

def load_splits(splits_json: Path) -> list:
    """
    Load splits_final.json produced by STU-Net / nnUNet.
    IDs are already in PatientID format (e.g. 'CHUM-002').

    Returns list of 5 dicts: [{"train": [...], "val": [...]}, ...]
    """
    with open(splits_json) as f:
        raw = json.load(f)
    return [{"train": list(fd["train"]), "val": list(fd["val"])} for fd in raw]


def get_test_patient_ids(clinical_csv: Path, test_center: str = "CHUV") -> list:
    df = pd.read_csv(clinical_csv)
    return df.loc[df["PatientID"].str.startswith(test_center), "PatientID"].tolist()


# ─── Survival metric ─────────────────────────────────────────────────────────

def compute_cindex(
    risk_scores: np.ndarray,
    times:       np.ndarray,
    events:      np.ndarray,
) -> float:
    """
    Harrell's C-index.
    risk_scores : higher = higher predicted risk (worse prognosis).
    lifelines.concordance_index expects higher = longer survival, so we negate.
    """
    return concordance_index(times, -risk_scores, events)


# ─── Generic clinical feature encoding ───────────────────────────────────────

def _prepare_clinical_subset(
    df:           pd.DataFrame,
    base_cols:    list,
    missing_cols: list,
    fit_stats:    dict = None,
) -> tuple:
    """
    Generic encoder for a clinical feature subset.

    HPV encoding (3-group, matches train_baselines.py)
    ---------------------------------------------------
    If "HPV Status" is present in df, two dummy columns are derived BEFORE
    any other processing:
      HPV_neg = 1 if HPV Status == 0 (negative),  0 otherwise
      HPV_unk = 1 if HPV Status is NaN (unknown),  0 otherwise
      (both = 0) → HPV positive = reference group
    base_cols should already reference "HPV_neg" / "HPV_unk", not "HPV Status".
    missing_cols should be [] when using the 3-group HPV encoding.

    Steps
    -----
    1. HPV 3-group dummies (HPV_neg, HPV_unk) from "HPV Status" column.
    2. Gender: map {'M':1,'F':0} if string; fill numeric NaN with 0.
    3. Silent median imputation for base_cols that are not Age/Gender
       (e.g. Treatment, HPV_neg, HPV_unk).  No missingness indicator added.
    4. Age: median-impute NaN, then z-score normalise (tracked in fit_stats).
    5. Safety: replace any residual NaN with 0 and warn.

    Parameters
    ----------
    df           : DataFrame (one row per patient).  Must contain columns
                   needed to derive base_cols (including "HPV Status" if
                   "HPV_neg"/"HPV_unk" appear in base_cols).
    base_cols    : final feature columns to assemble (must exist in df after
                   preprocessing, e.g. ["Age","Gender","HPV_neg","HPV_unk"]).
    missing_cols : kept for API compatibility; pass [] for the new HPV encoding.
    fit_stats    : None → fit from df (training).  dict → apply (val / test).

    Returns
    -------
    features  : np.ndarray  [N, len(base_cols)]
    fit_stats : dict
    """
    df = df.copy()

    # -- HPV 3-group dummies (HPV-positive is the reference group) ------------
    # HPV_neg = 1 if confirmed negative (HPV Status == 0), 0 otherwise
    # HPV_unk = 1 if unknown/missing (NaN),                0 otherwise
    # (HPV_neg=0, HPV_unk=0) → patient is HPV positive (reference)
    if "HPV Status" in df.columns:
        hpv = df["HPV Status"]
        df["HPV_neg"] = (hpv == 0).astype(float)
        df["HPV_unk"] = hpv.isna().astype(float)

    # -- Gender encoding -------------------------------------------------------
    if "Gender" in df.columns:
        if df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
        df["Gender"] = df["Gender"].fillna(0.0)   # NaN → 0 (unknown → female)

    # -- Silent median imputation for non-Age/Gender base_cols ----------------
    # (e.g. Treatment; HPV_neg and HPV_unk are already NaN-free from above)
    # No missingness indicator is added — just fill so there are no NaN.
    silent_cols = [c for c in base_cols if c not in ("Age", "Gender")]
    if fit_stats is None:
        silent_medians = {col: float(df[col].median()) for col in silent_cols}
    else:
        silent_medians = fit_stats.get("silent_medians", {})
    for col in silent_cols:
        df[col] = df[col].fillna(silent_medians.get(col, 0.0))

    # -- Age: impute with training median, then z-score normalise -------------
    if "Age" in base_cols:
        if fit_stats is None:
            age_median = float(df["Age"].median())
            age_mean   = float(df["Age"].mean())
            age_std    = float(df["Age"].std()) + 1e-6
        else:
            age_median = fit_stats["age_median"]
            age_mean   = fit_stats["age_mean"]
            age_std    = fit_stats["age_std"]
        df["Age"] = df["Age"].fillna(age_median)          # impute before normalising
        df["Age"] = (df["Age"] - age_mean) / age_std
    else:
        age_median = age_mean = age_std = None

    # -- Assemble final columns ------------------------------------------------
    features = df[base_cols].values.astype(np.float32)

    # -- Safety net: warn and zero-fill any residual NaN ----------------------
    if np.isnan(features).any():
        nan_cols = [base_cols[i] for i in range(features.shape[1])
                    if np.isnan(features[:, i]).any()]
        print(f"  [Warning] NaN in clinical features after encoding "
              f"({nan_cols}); replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)

    new_stats = {
        "silent_medians": silent_medians,
        "age_median":     age_median,
        "age_mean":       age_mean,
        "age_std":        age_std,
    }
    return features, new_stats


# ─── Task-specific wrappers ───────────────────────────────────────────────────

def prepare_staging_clinical(
    df:        pd.DataFrame,
    fit_stats: dict = None,
) -> tuple:
    """
    Features for Task 2 / staging branch:
      Age (z-score), Gender (0/1), HPV Status, HPV Status_missing
      → shape [N, N_STAGING_CLINICAL]  (N_STAGING_CLINICAL = 4)
    """
    return _prepare_clinical_subset(
        df, STAGING_CLINICAL_COLS, STAGING_MISSING_COLS, fit_stats
    )


def prepare_rfs_clinical(
    df:        pd.DataFrame,
    fit_stats: dict = None,
) -> tuple:
    """
    Features for Task 3 / RFS branch:
      Age (z-score), Gender (0/1), HPV Status, Treatment, HPV Status_missing
      → shape [N, N_RFS_CLINICAL]  (N_RFS_CLINICAL = 5)
    """
    return _prepare_clinical_subset(
        df, RFS_CLINICAL_COLS, RFS_MISSING_COLS, fit_stats
    )


# ─── Run-folder helpers (shared by all training scripts) ─────────────────────

from config import CKPT_DIR, LOG_DIR, PRED_DIR   # imported here for convenience


def run_tag(input_tokens: list) -> str:
    """
    Build a run directory tag from --input tokens.

    Examples
    --------
    run_tag(["clinical","CT","PET","prob_T","prob_N"]) → "clinical+CT+PET+prob_T+prob_N"
    run_tag(["CT","PET","prob_T","prob_N"])             → "CT+PET+prob_T+prob_N"
    run_tag(["clinical"])                               → "clinical"
    """
    return "+".join(input_tokens)


def run_dirs(input_tokens: list, task_name: str):
    """
    Return (ckpt_dir, log_dir) for this task + input configuration, creating them.

    Layout:
      results/checkpoints/<task_name>/<run_tag>/
      results/logs/<task_name>/<run_tag>/
    """
    tag      = run_tag(input_tokens)
    ckpt_dir = CKPT_DIR / task_name / tag
    log_dir  = LOG_DIR  / task_name / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)
    return ckpt_dir, log_dir


def pred_dir_for(input_tokens: list, task_name: str) -> Path:
    """results/predictions/<task_name>/<run_tag>/  (created on demand)."""
    d = PRED_DIR / task_name / run_tag(input_tokens)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_config(ckpt_dir: Path, cfg: dict):
    """Write config.json to ckpt_dir."""
    path = ckpt_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Run config -> {path}")


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def save_stats(stats: dict, path: Path):
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_stats(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
