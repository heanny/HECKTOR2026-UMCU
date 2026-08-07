"""
preprocess_cache.py — Pre-process and cache all patient volumes to disk.

Runs the full pipeline (normalise → NaN-fix → pad → resample) once for every
patient and saves one float32 .npy file per channel:

  results/cache/<PatientID>/<channel>.npy   shape [100, 100, 155]

After this script finishes, the dataset loads cached arrays instead of raw
NIfTI files, reducing per-sample load time from ~6 s to ~0.05 s.

Channels cached
---------------
  CT      — HU normalised to [0, 1]
  PET     — SUV normalised by 99th percentile
  prob_T  — GTV-T soft probability (nnUNet .npz)
  prob_N  — GTV-N soft probability (nnUNet .npz)
  hard_T  — GTV-T binary mask from hard segmentation
  hard_N  — GTV-N binary mask from hard segmentation

Usage
-----
  python preprocess_cache.py                   # 8 parallel workers
  python preprocess_cache.py --workers 16
  python preprocess_cache.py --overwrite       # re-generate existing files
  python preprocess_cache.py --pid CHUM-001    # single patient (debug)
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLINICAL_CSV,
    IMAGES_TR_DIR, IMAGES_TS_DIR,
    OOF_PROBMAPS_DIR, ENSEMBLE_DIR,
    NATIVE_SPACING, TARGET_SPACING, PAD_TO_SHAPE,
    ALL_IMAGE_CHANNELS, CACHE_DIR,
)
from dataset import (
    _load_nifti, _load_probmap,
    _normalize_ct, _normalize_pet,
    _pad_to, _resample,
)


# ─── Per-patient worker ───────────────────────────────────────────────────────

def _image_dirs(pid: str):
    if pid.startswith("CHUV"):
        return IMAGES_TS_DIR, ENSEMBLE_DIR
    return IMAGES_TR_DIR, OOF_PROBMAPS_DIR


def preprocess_one(pid: str, overwrite: bool = False) -> tuple:
    """
    Preprocess all 6 channels for one patient and save to cache.

    Returns
    -------
    (pid, channels_saved: list, error_msg: str | None)
    """
    out_dir = CACHE_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir, probmaps_dir = _image_dirs(pid)

    ct_path  = images_dir   / f"{pid}_0000.nii.gz"
    pet_path = images_dir   / f"{pid}_0001.nii.gz"
    seg_path = probmaps_dir / f"{pid}.nii.gz"
    npz_path = probmaps_dir / f"{pid}.npz"

    # Native voxel spacing from CT header (needed for resampling)
    try:
        native_sp = tuple(
            float(v) for v in nib.load(str(ct_path)).header.get_zooms()[:3]
        )
    except Exception:
        native_sp = NATIVE_SPACING

    channels_saved = []
    errors         = []

    # Load prob maps once if any prob channel is requested (avoid double load)
    _probs_cache = {}

    for ch in ALL_IMAGE_CHANNELS:
        out_path = out_dir / f"{ch}.npy"
        if out_path.exists() and not overwrite:
            channels_saved.append(ch)
            continue

        try:
            # ── Load raw array ────────────────────────────────────────────────
            if ch == "CT":
                arr, _ = _load_nifti(ct_path)
                arr    = _normalize_ct(arr)

            elif ch == "PET":
                arr, _ = _load_nifti(pet_path)
                arr    = _normalize_pet(arr)

            elif ch in ("prob_T", "prob_N"):
                if "probs" not in _probs_cache:
                    if npz_path.exists():
                        _probs_cache["probs"] = _load_probmap(npz_path)
                    else:
                        seg, _ = _load_nifti(seg_path)
                        _probs_cache["probs"] = np.stack([
                            (seg == 1).astype(np.float32),
                            (seg == 2).astype(np.float32),
                        ], axis=0)
                arr = _probs_cache["probs"][0 if ch == "prob_T" else 1]

            elif ch in ("hard_T", "hard_N"):
                if "seg" not in _probs_cache:
                    _probs_cache["seg"], _ = _load_nifti(seg_path)
                label = 1 if ch == "hard_T" else 2
                arr   = (_probs_cache["seg"] == label).astype(np.float32)

            # ── NaN/Inf fix before resampling ─────────────────────────────────
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

            # ── Pad → resample ────────────────────────────────────────────────
            arr = _pad_to(arr, PAD_TO_SHAPE)
            arr = _resample(arr, native_sp, TARGET_SPACING, order=1)

            # ── NaN/Inf fix after resampling (zoom can propagate edge NaN) ────
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

            np.save(str(out_path), arr.astype(np.float32))
            channels_saved.append(ch)

        except FileNotFoundError:
            errors.append(f"{ch}: file not found")
        except Exception as e:
            errors.append(f"{ch}: {e}")

    err_str = "; ".join(errors) if errors else None
    return pid, channels_saved, err_str


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Pre-process all patient volumes and cache to disk."
    )
    ap.add_argument("--workers",   type=int,  default=8,
                    help="Parallel workers (default 8).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-generate files that already exist.")
    ap.add_argument("--pid",       type=str,  default=None,
                    help="Process a single patient ID (for debugging).")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df       = pd.read_csv(CLINICAL_CSV)
    all_pids = df["PatientID"].tolist()

    if args.pid:
        all_pids = [args.pid]

    print(f"Patients   : {len(all_pids)}")
    print(f"Output dir : {CACHE_DIR}")
    print(f"Workers    : {args.workers}  |  Overwrite: {args.overwrite}")
    print()

    done   = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(preprocess_one, pid, args.overwrite): pid
            for pid in all_pids
        }
        for future in as_completed(futures):
            pid, saved, err = future.result()
            done += 1
            tag = f"[{done:3d}/{len(all_pids)}]  {pid:<20}  saved={saved}"
            if err:
                tag += f"   !! {err}"
                errors.append(f"{pid}: {err}")
            print(tag)

    print(f"\n{'='*60}")
    print(f"Done.  {len(all_pids) - len(errors)} / {len(all_pids)} patients OK.")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
