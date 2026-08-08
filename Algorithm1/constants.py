"""
constants.py — Path-free constants for the HECKTOR 2026 inference container.

This mirrors the constants used during training (task2_3_prediction/config.py)
but deliberately contains NO filesystem paths, so it is safe to import inside
the Grand Challenge container. Values here MUST match the training config, or
the packaged weights will be applied to mismatched inputs.
"""

# Image resampling (must match training/config.py)
# preprocess_jin.py crops raw CT/PET to 200x200x310 @ 1 mm (X,Y,Z).
# The multitask DenseNet then pads to PAD_TO_SHAPE and resamples 1 mm -> 2 mm.
NATIVE_SPACING = (1.0, 1.0, 1.0)     # mm  (cropped nnUNet image spacing)
TARGET_SPACING = (2.0, 2.0, 2.0)     # mm  (DenseNet model spacing)
PAD_TO_SHAPE   = (200, 200, 310)     # pad cropped image to this (X,Y,Z) first
ROI_SIZE       = (100, 100, 155)     # DenseNet input spatial shape

# preprocess_jin.py fixed crop box (X, Y, Z), centred on PET hot-spot
CROP_BOX_SIZE  = (200, 200, 310)

# Label maps (must match training) 
T_STAGE_MAP = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
N_STAGE_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
INV_T_STAGE = {v: k for k, v in T_STAGE_MAP.items()}
INV_N_STAGE = {v: k for k, v in N_STAGE_MAP.items()}
N_T_CLASSES = 4
N_N_CLASSES = 4

# Image channels for the chosen model: clinical + CT + PET + hard_T + hard_N
IMAGE_CHANNELS = ["CT", "PET", "hard_T", "hard_N"]   # in_channels = 4
IN_CHANNELS    = len(IMAGE_CHANNELS)

# Clinical feature columns (must match utils.py) 
# Staging branch  (Task 2): Age, Gender, HPV_neg, HPV_unk           -> dim 4
# RFS     branch  (Task 3): Age, Gender, HPV_neg, HPV_unk, Treatment -> dim 5
STAGING_CLINICAL_COLS = ["Age", "Gender", "HPV_neg", "HPV_unk"]
RFS_CLINICAL_COLS     = ["Age", "Gender", "HPV_neg", "HPV_unk", "Treatment"]
N_STAGING_CLINICAL    = len(STAGING_CLINICAL_COLS)   # 4
N_RFS_CLINICAL        = len(RFS_CLINICAL_COLS)        # 5

# Segmentation labels emitted by STU-Net / expected in output.mha
LABEL_BACKGROUND = 0
LABEL_GTVP       = 1   # primary tumour  -> hard_T
LABEL_GTVN       = 2   # lymph node      -> hard_N

TEST_CENTER = "CHUV"   # held-out center used to freeze clinical stats
