"""
dataset.py — PyTorch Dataset for HECKTOR 2026 Task 2 (T/N staging) and
             Task 3 (RFS prediction).

Configurable image channels
────────────────────────────
Any subset of:
  "CT"     — HU clipped and normalised to [0, 1]
  "PET"    — normalised by 99th-percentile SUV
  "prob_T" — GTV-T soft probability  (nnUNet .npz, channel 1)
  "prob_N" — GTV-N soft probability  (nnUNet .npz, channel 2)
  "hard_T" — GTV-T binary hard mask  (label == 1 in .nii.gz)
  "hard_N" — GTV-N binary hard mask  (label == 2 in .nii.gz)

Default: ["CT", "PET", "prob_T", "prob_N"]  →  in_channels = 4

The channel list is passed to HECKTOR2026Dataset(channels=[...]) and
determines both the stacking order and model in_channels.

Preprocessing pipeline
  1. Load only the files required by the selected channels.
  2. Pad spatial dims to PAD_TO_SHAPE (handles Z < 310).
  3. Resample to TARGET_SPACING (2×2×2 mm³) → ~(100, 100, 155).
  4. Augmentation (training only):
        Spatial (all channels) : flip one random axis, RandAffine p=0.5
        Intensity (CT, PET only): RandGaussianNoise / Scale / Shift / Smooth
"""

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import zoom

from monai.transforms import (
    RandAffine,
    RandGaussianNoise,
    RandScaleIntensity,
    RandShiftIntensity,
    RandGaussianSmooth,
)

from config import (
    CLINICAL_CSV,
    IMAGES_TR_DIR, IMAGES_TS_DIR,
    OOF_PROBMAPS_DIR, ENSEMBLE_DIR,
    PAD_TO_SHAPE, NATIVE_SPACING, TARGET_SPACING, ROI_SIZE,
    T_STAGE_MAP, N_STAGE_MAP,
    ALL_IMAGE_CHANNELS, DEFAULT_IMAGE_CHANNELS,
    CACHE_DIR,
)


# ─── Low-level image utilities ────────────────────────────────────────────────

def _load_nifti(path: Path) -> tuple:
    """Return (data: np.float32 [H,W,D], header) from a .nii/.nii.gz."""
    img  = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.header


def _load_probmap(path: Path) -> np.ndarray:
    """
    Load nnUNet probability .npz → [2, X, Y, Z]  (GTV-T, GTV-N probabilities).

    nnUNet saves probabilities in its internal axis order [C, Z, Y, X]
    (key = 'probabilities', dtype float32).
    NIfTI / nibabel uses [X, Y, Z], so we transpose to align with CT/PET.

    class 0 = background, class 1 = GTV-T, class 2 = GTV-N
    """
    npz = np.load(str(path))
    # Try known keys; fall back to whatever key exists
    if "probabilities" in npz:
        probs = npz["probabilities"].astype(np.float32)
    elif "softmax" in npz:
        probs = npz["softmax"].astype(np.float32)
    elif "arr_0" in npz:
        probs = npz["arr_0"].astype(np.float32)
    else:
        probs = npz[list(npz.keys())[0]].astype(np.float32)

    # probs: [C, Z, Y, X]  →  [C, X, Y, Z]  (matches NIfTI nibabel convention)
    probs = probs.transpose(0, 3, 2, 1)   # [C, X, Y, Z]

    # Return GTV-T (ch 1) and GTV-N (ch 2) only
    return probs[1:3]   # [2, X, Y, Z]


def _pad_to(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Symmetrically zero-pad volume to at least target_shape.
    Works for both 3-D [H,W,D] and 4-D [C,H,W,D] arrays
    (spatial dims are always the last 3).
    """
    spatial = volume.shape[-3:]
    pad_width = []
    for cur, tgt in zip(spatial, target_shape):
        diff  = max(0, tgt - cur)
        before = diff // 2
        after  = diff - before
        pad_width.append((before, after))

    if volume.ndim == 4:
        full_pad = [(0, 0)] + pad_width
    else:
        full_pad = pad_width

    return np.pad(volume, full_pad, mode="constant", constant_values=0)


def _resample(volume: np.ndarray,
              native_spacing: tuple,
              target_spacing: tuple,
              order: int = 1) -> np.ndarray:
    """
    Resample volume from native_spacing to target_spacing using scipy zoom.
    Operates on the last 3 spatial dimensions (supports [H,W,D] and [C,H,W,D]).
    """
    zoom_factors_spatial = tuple(
        n / t for n, t in zip(native_spacing, target_spacing)
    )
    if volume.ndim == 4:
        zoom_factors = (1.0,) + zoom_factors_spatial
    else:
        zoom_factors = zoom_factors_spatial

    return zoom(volume, zoom_factors, order=order).astype(np.float32)


def _normalize_ct(ct: np.ndarray) -> np.ndarray:
    """Clip to [-200, 300] HU, then scale to [0, 1]."""
    ct = np.clip(ct, -200.0, 300.0)
    return (ct + 200.0) / 500.0


def _normalize_pet(pet: np.ndarray) -> np.ndarray:
    """Normalise PET SUV by the 99th percentile, clip to [0, 1]."""
    p99 = np.percentile(pet, 99)
    if p99 > 1e-6:
        pet = pet / p99
    return np.clip(pet, 0.0, 1.0)


def preprocess_patient(
    ct:             np.ndarray | None,   # [H, W, D] raw HU  — or None
    pet:            np.ndarray | None,   # [H, W, D] raw SUV — or None
    probs:          np.ndarray | None,   # [2, H, W, D]  (prob_T ch0, prob_N ch1) — or None
    hard_seg:       np.ndarray | None,   # [H, W, D]  integer labels 0/1/2 — or None
    channels:       list,                # ordered list from ALL_IMAGE_CHANNELS
    native_spacing: tuple = NATIVE_SPACING,
    target_spacing: tuple = TARGET_SPACING,
    pad_to_shape:   tuple = PAD_TO_SHAPE,
) -> np.ndarray:
    """
    Build a [C, H, W, D] array for the requested channel subset, then pad
    and resample to target_spacing.

    Channel names and their sources
    --------------------------------
    "CT"     : _normalize_ct(ct)
    "PET"    : _normalize_pet(pet)
    "prob_T" : probs[0]                         (GTV-T soft probability)
    "prob_N" : probs[1]                         (GTV-N soft probability)
    "hard_T" : (hard_seg == 1).astype(float32)  (GTV-T binary mask)
    "hard_N" : (hard_seg == 2).astype(float32)  (GTV-N binary mask)

    Raises ValueError for unknown channel names or missing source data.
    """
    # Validate channel names
    unknown = [c for c in channels if c not in ALL_IMAGE_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}. "
                         f"Valid: {ALL_IMAGE_CHANNELS}")

    # Clinical-only mode: no image channels requested → return a dummy zero volume.
    # The model will have use_image=False and ignores this tensor entirely.
    if not channels:
        return np.zeros((1,) + ROI_SIZE, dtype=np.float32)

    # Normalise what we have
    ct_norm  = _normalize_ct(ct)   if ct       is not None else None
    pet_norm = _normalize_pet(pet) if pet      is not None else None
    hard_t   = ((hard_seg == 1).astype(np.float32)
                if hard_seg is not None else None)
    hard_n   = ((hard_seg == 2).astype(np.float32)
                if hard_seg is not None else None)

    _src = {
        "CT":     ct_norm,
        "PET":    pet_norm,
        "prob_T": probs[0] if probs is not None else None,
        "prob_N": probs[1] if probs is not None else None,
        "hard_T": hard_t,
        "hard_N": hard_n,
    }

    arrays = []
    for ch in channels:
        arr = _src[ch]
        if arr is None:
            raise ValueError(
                f"Channel '{ch}' was requested but the corresponding "
                f"source data was not loaded."
            )
        arrays.append(arr)

    img = np.stack(arrays, axis=0)   # [C, H, W, D]

    # Kill NaN/Inf from corrupted source files before resampling —
    # scipy.ndimage.zoom propagates NaN across the entire volume if any exist.
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    img = _pad_to(img, pad_to_shape)
    img = _resample(img, native_spacing, target_spacing, order=1)

    # Second pass: zoom can still introduce NaN at borders from NaN inputs.
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    return img.astype(np.float32)


# ─── Dataset class ────────────────────────────────────────────────────────────

class HECKTOR2026Dataset(Dataset):
    """
    Parameters
    ----------
    patient_ids        : list of PatientID strings to include
    clinical_staging   : np.ndarray [N, N_STAGING_CLINICAL]
                         Features for staging branch / Task 2.
                         (Age, Gender, HPV Status, HPV_missing)
                         Pass None when task == 'rfs'.
    clinical_rfs       : np.ndarray [N, N_RFS_CLINICAL]
                         Features for RFS branch / Task 3.
                         (Age, Gender, HPV Status, Treatment, HPV_missing)
                         Pass None when task == 'stage'.
    pid_index          : dict {PatientID: row_index_in_clinical arrays}
    task               : 'stage' | 'rfs' | 'both'
    channels           : ordered list of image channel names to load.
                         Any subset of ALL_IMAGE_CHANNELS.
                         Default: DEFAULT_IMAGE_CHANNELS = ["CT","PET","prob_T","prob_N"]
                         Examples:
                           ["CT", "PET", "hard_T", "hard_N"]   — hard seg ablation
                           ["CT", "PET", "prob_T", "prob_N", "hard_T", "hard_N"]
                           ["CT", "PET"]
    is_test            : True for CHUV test patients (no labels returned)
    augment            : augmentation (training only)

    Returned batch keys
    -------------------
    task='stage' : image [C,H,W,D], clinical_staging, t_label, n_label, pid
    task='rfs'   : image [C,H,W,D], clinical_rfs, time, event, pid
    task='both'  : image [C,H,W,D], clinical_staging, clinical_rfs,
                   t_label, n_label, time, event, pid

    image.shape[0] == len(channels)
    """

    def __init__(
        self,
        patient_ids:      list,
        clinical_staging: np.ndarray,          # [N, N_STAGING_CLINICAL]  or None
        clinical_rfs:     np.ndarray,          # [N, N_RFS_CLINICAL]      or None
        pid_index:        dict,
        task:             str        = "rfs",
        channels:         list       = None,   # None → DEFAULT_IMAGE_CHANNELS
        is_test:          bool       = False,
        augment:          bool       = False,
    ):
        self.clinical_staging = clinical_staging
        self.clinical_rfs     = clinical_rfs
        self.pid_index        = pid_index
        self.task             = task
        # None → use default; explicit [] → clinical-only mode (no image channels)
        self.channels         = list(channels) if channels is not None else DEFAULT_IMAGE_CHANNELS
        self.is_test          = is_test
        self.augment          = augment

        # Validate channel names up front
        unknown = [c for c in self.channels if c not in ALL_IMAGE_CHANNELS]
        if unknown:
            raise ValueError(f"Unknown channels: {unknown}. Valid: {ALL_IMAGE_CHANNELS}")

        # Load clinical CSV once (used for labels)
        self.df = pd.read_csv(CLINICAL_CSV).set_index("PatientID")

        # Keep only patients that have the required labels (skip otherwise)
        self.valid_ids = self._filter_valid(patient_ids)

        # In-memory cache: populated on first access, served from RAM thereafter.
        # Each worker process has its own copy, so memory cost per worker is
        # len(valid_ids) * C * H * W * D * 4 bytes (e.g. ~21 GB for 6-channel run).
        self._ram_cache: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filter_valid(self, patient_ids: list) -> list:
        """
        Return the subset of patient_ids that have all required labels.

        Exclusion reasons (counted and printed if any patients are dropped):
          not-in-CSV       : PatientID not found in clinical DataFrame at all
          missing-label    : NaN for a required label column
          unmapped-stage   : T-stage / N-stage string not in T_STAGE_MAP / N_STAGE_MAP
                             (e.g. "TX", "T0", "Tis" — valid strings but not modelled)
        """
        valid        = []
        n_no_row     = 0
        n_miss_label = 0
        n_bad_stage  = 0

        for pid in patient_ids:
            if not self.is_test:
                if pid not in self.df.index:
                    n_no_row += 1
                    continue
                row = self.df.loc[pid]

                if self.task == "rfs":
                    if pd.isna(row.get("Relapse")) or pd.isna(row.get("RFS")):
                        n_miss_label += 1
                        continue

                elif self.task == "stage":
                    t_val = row.get("T-stage")
                    n_val = row.get("N-stage")
                    if pd.isna(t_val) or pd.isna(n_val):
                        n_miss_label += 1
                        continue
                    if (str(t_val).strip() not in T_STAGE_MAP
                            or str(n_val).strip() not in N_STAGE_MAP):
                        n_bad_stage += 1
                        continue

                elif self.task == "both":
                    t_val = row.get("T-stage")
                    n_val = row.get("N-stage")
                    if (pd.isna(t_val) or pd.isna(n_val)
                            or pd.isna(row.get("Relapse")) or pd.isna(row.get("RFS"))):
                        n_miss_label += 1
                        continue
                    if (str(t_val).strip() not in T_STAGE_MAP
                            or str(n_val).strip() not in N_STAGE_MAP):
                        n_bad_stage += 1
                        continue

            valid.append(pid)

        if not self.is_test:
            n_excluded = n_no_row + n_miss_label + n_bad_stage
            if n_excluded > 0:
                print(
                    f"  [Dataset task={self.task!r}]  {len(patient_ids)} patients → "
                    f"{len(valid)} valid  "
                    f"(excluded: {n_no_row} not-in-CSV, "
                    f"{n_miss_label} missing-label, "
                    f"{n_bad_stage} unmapped-stage-value)"
                )

        return valid

    def _image_dirs(self, pid: str):
        """Return (images_dir, probmaps_dir) for a patient."""
        if pid.startswith("CHUV"):
            return IMAGES_TS_DIR, ENSEMBLE_DIR
        return IMAGES_TR_DIR, OOF_PROBMAPS_DIR

    def _load_patient_volume(self, pid: str) -> np.ndarray:
        """
        Load only the files required by self.channels, then preprocess.

        Fast path  : load pre-processed .npy files from CACHE_DIR if available.
                     Each file is [100, 100, 155] float32 — just stack and return.
        Slow path  : read raw NIfTI/NPZ, normalise, pad, resample (run once to
                     build the cache via preprocess_cache.py).

        When self.channels is empty (clinical-only mode), returns a dummy
        zero volume of shape (1, *ROI_SIZE) without touching the file system.
        """
        ch = self.channels

        # Clinical-only mode — no images to load
        if not ch:
            return np.zeros((1,) + ROI_SIZE, dtype=np.float32)

        # ── RAM cache: serve from memory after first load ─────────────────────
        if pid in self._ram_cache:
            return self._ram_cache[pid]

        # ── Fast path: load from pre-computed disk cache ──────────────────────
        cache_pid = CACHE_DIR / pid
        if cache_pid.exists():
            arrays = []
            for c in ch:
                p = cache_pid / f"{c}.npy"
                if p.exists():
                    arrays.append(np.load(str(p)))
                else:
                    break   # cache incomplete — fall through to slow path
            else:
                # All channels found in cache
                vol = np.stack(arrays, axis=0)   # [C, H, W, D]
                self._ram_cache[pid] = vol
                return vol

        images_dir, probmaps_dir = self._image_dirs(pid)

        ct_path  = images_dir   / f"{pid}_0000.nii.gz"
        pet_path = images_dir   / f"{pid}_0001.nii.gz"
        seg_path = probmaps_dir / f"{pid}.nii.gz"
        npz_path = probmaps_dir / f"{pid}.npz"

        # -- Native spacing: read from CT header (nibabel lazy, ~zero cost) ---
        try:
            native_sp = tuple(
                float(v) for v in nib.load(str(ct_path)).header.get_zooms()[:3]
            )
        except Exception:
            native_sp = NATIVE_SPACING

        # -- Load only what is required ----------------------------------------
        ct       = _load_nifti(ct_path)[0]  if "CT"  in ch else None
        pet      = _load_nifti(pet_path)[0] if "PET" in ch else None

        # Soft probability maps (prob_T / prob_N)
        probs = None
        if "prob_T" in ch or "prob_N" in ch:
            if npz_path.exists():
                probs = _load_probmap(npz_path)          # [2, H, W, D]
            else:
                # Fallback: binarise hard seg as 0/1 prob maps
                seg, _ = _load_nifti(seg_path)
                probs  = np.stack(
                    [(seg == 1).astype(np.float32),
                     (seg == 2).astype(np.float32)],
                    axis=0,
                )

        # Hard segmentation masks (hard_T / hard_N)
        hard_seg = None
        if "hard_T" in ch or "hard_N" in ch:
            hard_seg = _load_nifti(seg_path)[0].astype(np.uint8)

        vol = preprocess_patient(
            ct, pet, probs, hard_seg, ch,
            native_sp, TARGET_SPACING, PAD_TO_SHAPE,
        )
        if not np.isfinite(vol).all():
            print(f"  [Warning] NaN/Inf in volume for {pid} after preprocessing — replacing with 0.")
            vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0)
        self._ram_cache[pid] = vol
        return vol

    # ── MONAI augmentation transforms (built once, reused each call) ──────────

    _SPATIAL_TRANSFORMS = [
        RandAffine(
            prob=0.3,
            rotate_range=[(-0.26, 0.26)] * 3,   # ±15 deg
            scale_range=[(-0.10, 0.10)] * 3,     # ±10 %
            translate_range=[(-10, 10)] * 3,     # ±10 voxels
            spatial_size=(100, 100, 155),
            mode="bilinear",
            padding_mode="zeros",
        ),
    ]

    _INTENSITY_TRANSFORMS = [
        RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
        RandScaleIntensity(factors=0.10, prob=0.3),
        RandShiftIntensity(offsets=0.05, prob=0.3),
        RandGaussianSmooth(
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
            prob=0.2,
        ),
    ]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        Two-stage augmentation:
          1. Spatial (all 4 channels together):
             RandFlip x3  +  RandAffine
          2. Intensity (CT ch0 and PET ch1 independently):
             RandGaussianNoise, RandScaleIntensity,
             RandShiftIntensity, RandGaussianSmooth
          Prob maps (ch2, ch3) receive spatial augmentation only.
          All channels are clipped to [0, 1] after augmentation.
        """
        if not self.channels:
            return img   # dummy zero tensor — no augmentation needed

        t = torch.as_tensor(img, dtype=torch.float32)   # [C, H, W, D]

        # -- Stage 1a: flip one randomly chosen axis (or none) -----------------
        # choices: 0=no flip, 1=flip-X, 2=flip-Y, 3=flip-Z  (equal probability)
        flip_choice = np.random.randint(0, 4)
        if flip_choice > 0:
            t = torch.flip(t, dims=[flip_choice])   # dims 1/2/3 → X/Y/Z

        # -- Stage 1b: affine (all channels) -----------------------------------
        for T in self._SPATIAL_TRANSFORMS:
            t = T(t)

        # MONAI may return a MetaTensor; convert to plain torch.Tensor
        t = torch.as_tensor(t.as_tensor() if hasattr(t, "as_tensor") else t,
                             dtype=torch.float32)

        # -- Stage 2: intensity (CT and PET channels only) --------------------
        # Find the actual indices of CT and PET in the selected channel list
        for ch_name in ("CT", "PET"):
            if ch_name not in self.channels:
                continue
            ci = self.channels.index(ch_name)           # dynamic index
            ch_t = t[ci : ci + 1]                       # [1, H, W, D]
            for T in self._INTENSITY_TRANSFORMS:
                ch_t = T(ch_t)
                if hasattr(ch_t, "as_tensor"):
                    ch_t = ch_t.as_tensor()
            t[ci : ci + 1] = ch_t.to(dtype=torch.float32)

        # -- Clip all channels to [0, 1] ---------------------------------------
        t = t.clamp(0.0, 1.0)

        return t.numpy().astype(np.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.valid_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.valid_ids[idx]

        # Image
        img = self._load_patient_volume(pid)              # [4, H, W, D]
        if self.augment:
            img = self._augment(img)
        image_tensor = torch.tensor(img, dtype=torch.float32)

        row_idx = self.pid_index.get(pid, 0)
        sample  = {"image": image_tensor, "pid": pid}

        # Clinical features — emit only what this task needs
        if self.task in ("stage", "both") and self.clinical_staging is not None:
            sample["clinical_staging"] = torch.tensor(
                self.clinical_staging[row_idx], dtype=torch.float32
            )
        if self.task in ("rfs", "both") and self.clinical_rfs is not None:
            sample["clinical_rfs"] = torch.tensor(
                self.clinical_rfs[row_idx], dtype=torch.float32
            )

        if self.is_test:
            return sample

        row = self.df.loc[pid]

        if self.task in ("stage", "both"):
            t_label = T_STAGE_MAP.get(str(row["T-stage"]).strip(), -1)
            n_label = N_STAGE_MAP.get(str(row["N-stage"]).strip(), -1)
            sample["t_label"] = torch.tensor(t_label, dtype=torch.long)
            sample["n_label"] = torch.tensor(n_label, dtype=torch.long)

        if self.task in ("rfs", "both"):
            sample["time"]  = torch.tensor(float(row["RFS"]),    dtype=torch.float32)
            sample["event"] = torch.tensor(int(float(row["Relapse"])), dtype=torch.float32)

        return sample
