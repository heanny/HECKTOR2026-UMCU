"""
config.py — Paths and hyperparameters for HECKTOR 2026 Task 2 & 3.
"""
from pathlib import Path

# ─── ROOT PATHS ──────────────────────────────────────────────────────────────
BASE_DIR     = Path("/autofs/arch11/DATA/HOMES/HECKTOR_2026")
TASK_DIR     = BASE_DIR / "task2_3_prediction"

# Clinical data
CLINICAL_CSV = TASK_DIR / "HECKTOR_2026_training_data.csv"

# Task-1 split file (5-fold CV, same as segmentation task)
SPLITS_JSON  = (
    BASE_DIR
    / "task1_segmentation/stu-net_2025/nnUNet_preprocessed"
    / "Dataset001_HECKTOR2026/splits_final.json"
)

# Raw CT / PET images  (_0000 = CT, _0001 = PET)
IMAGES_TR_DIR = (
    BASE_DIR
    / "task1_segmentation/stu-net_2025/nnUNet_raw"
    / "Dataset001_HECKTOR2026/imagesTr"
)
IMAGES_TS_DIR = (
    BASE_DIR
    / "task1_segmentation/stu-net_2025/nnUNet_raw"
    / "Dataset001_HECKTOR2026/imagesTs"
)

# Segmentation probability maps
# Training set: out-of-fold (OOF) predictions from STU-Net
OOF_PROBMAPS_DIR = (
    BASE_DIR / "task1_segmentation/stu-net_2025/predictions/oof_probmaps"
)
# Test set (CHUV): 5-fold ensemble
ENSEMBLE_DIR = (
    BASE_DIR / "task1_segmentation/stu-net_2025/predictions/ensemble_5fold"
)

# Outputs
RESULTS_DIR = TASK_DIR / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
PRED_DIR    = RESULTS_DIR / "predictions"
LOG_DIR     = RESULTS_DIR / "logs"
CACHE_DIR   = RESULTS_DIR / "cache"   # pre-processed volumes: one .npy per channel per patient

# ─── IMAGE RESAMPLING SETTINGS ───────────────────────────────────────────────
# Input images are 1×1×1 mm³, nominally 200×200×310 voxels (X×Y×Z).
# Some patients have Z = 298; pad Z to 310 symmetrically before resampling.
NATIVE_SPACING   = (1.0, 1.0, 1.0)   # mm
TARGET_SPACING   = (2.0, 2.0, 2.0)   # mm – resample to this
PAD_TO_SHAPE     = (200, 200, 310)    # pad to this at native spacing first
# After resampling 200→100, 200→100, 310→155:
ROI_SIZE         = (100, 100, 155)    # model input shape (H×W×D)

# ─── LABEL MAPS ──────────────────────────────────────────────────────────────
T_STAGE_MAP = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
N_STAGE_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
N_T_CLASSES = 4
N_N_CLASSES = 4

# ─── IMAGE CHANNEL SELECTION ─────────────────────────────────────────────────
# All supported channel names (order here defines the canonical order when
# multiple channels are selected).
ALL_IMAGE_CHANNELS = ["CT", "PET", "prob_T", "prob_N", "hard_T", "hard_N"]

# Default selection used when nothing is specified.
DEFAULT_IMAGE_CHANNELS = ["CT", "PET", "prob_T", "prob_N"]

# ─── INPUT TOKEN SETS ─────────────────────────────────────────────────────────
# "clinical" is a special token for the clinical feature branch.
# Used by --input in all three training scripts.
ALL_INPUT_TOKENS = ["clinical"] + ALL_IMAGE_CHANNELS   # all valid --input tokens
DEFAULT_INPUT    = ["clinical"] + DEFAULT_IMAGE_CHANNELS

# ─── CLINICAL FEATURE COLUMNS ────────────────────────────────────────────────
# Full set (kept for reference / future use)
CLINICAL_COLS = [
    "Age", "Gender", "CenterID",
    "Tobacco Consumption", "Alcohol Consumption",
    "Performance Status", "Treatment", "HPV Status",
]
MISSING_COLS = [
    "Tobacco Consumption", "Alcohol Consumption",
    "Performance Status", "HPV Status",
]
N_CLINICAL_FEATURES = len(CLINICAL_COLS) + len(MISSING_COLS)   # 12 (legacy)

# ── Task-specific subsets ─────────────────────────────────────────────────────
# HPV 3-group dummy encoding (matches train_baselines.py):
#   HPV_neg = 1 if HPV Status == 0 (negative),   0 otherwise
#   HPV_unk = 1 if HPV Status is NaN (unknown),   0 otherwise
#   HPV positive (HPV Status == 1) is the reference group (both dummies = 0)

# Task 2  /  staging branch in multi-task: Age, Gender, HPV_neg, HPV_unk
STAGING_CLINICAL_COLS = ["Age", "Gender", "HPV_neg", "HPV_unk"]
STAGING_MISSING_COLS  = []                        # HPV dummies carry no separate indicator
N_STAGING_CLINICAL    = len(STAGING_CLINICAL_COLS)          # 4

# Task 3 standalone  /  RFS clinical branch in multi-task:
#   Age, Gender, HPV_neg, HPV_unk, Treatment
RFS_CLINICAL_COLS     = ["Age", "Gender", "HPV_neg", "HPV_unk", "Treatment"]
RFS_MISSING_COLS      = []                        # HPV dummies carry no separate indicator
N_RFS_CLINICAL        = len(RFS_CLINICAL_COLS)              # 5

# ─── TRAINING HYPERPARAMETERS ─────────────────────────────────────────────────
# Batch size 2 recommended; full-res 4-ch 100×100×155 is ~25 MB/sample.
BATCH_SIZE          = 12
NUM_WORKERS         = 8
MAX_EPOCHS = N_EPOCHS = 200
EARLY_STOP_PATIENCE = 25
LR                  = 2e-4
WEIGHT_DECAY        = 1e-4
LR_MILESTONES       = [100, 150]
LR_GAMMA            = 0.1
SEED                = 42

# ─── MISC ────────────────────────────────────────────────────────────────────
TEST_CENTER = "CHUV"   # held-out test center
