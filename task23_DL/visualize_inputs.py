"""
visualize_inputs.py
───────────────────
Sanity-check and visualise the 4-channel model input for one or several patients.

Figure layout  (6 columns x 3 rows):
  col 0  CT (grey)
  col 1  PET (hot)
  col 2  CT + GT label overlay      (GTV-T red, GTV-N cyan)  -- blank for test
  col 3  CT + predicted seg overlay (GTV-T red, GTV-N cyan)  -- OOF or ensemble
  col 4  GTV-T probability  (Reds colourmap, white = 1.0)
  col 5  GTV-N probability  (Blues colourmap, dark = 1.0)

Rows: Axial / Coronal / Sagittal through the GTV centre-of-mass (predicted seg).

For training patients (non-CHUV), Dice_T and Dice_N are printed in the title.
CHUV test patients have no GT so columns 2 shows "No GT available".

Usage
-----
python visualize_inputs.py --pid CHUM-001
python visualize_inputs.py --pid CHUV-003
python visualize_inputs.py --n 3          # first 3 train + 3 test patients
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import center_of_mass

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"Z:\HOMES\HECKTOR_2026")
IMAGES_TR_DIR = BASE_DIR / "task1_segmentation/stu-net_2025/nnUNet_raw/Dataset001_HECKTOR2026/imagesTr"
IMAGES_TS_DIR = BASE_DIR / "task1_segmentation/stu-net_2025/nnUNet_raw/Dataset001_HECKTOR2026/imagesTs"
LABELS_TR_DIR = BASE_DIR / "task1_segmentation/stu-net_2025/nnUNet_raw/Dataset001_HECKTOR2026/labelsTr"
OOF_DIR       = BASE_DIR / "task1_segmentation/stu-net_2025/predictions/oof_probmaps"
ENS_DIR       = BASE_DIR / "task1_segmentation/stu-net_2025/predictions/ensemble_5fold"
SPLITS_JSON   = BASE_DIR / "task1_segmentation/stu-net_2025/nnUNet_preprocessed/Dataset001_HECKTOR2026/splits_final.json"
OUT_DIR       = BASE_DIR / "task2_3_prediction/results/visualize"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _img_dirs(pid):
    if pid.startswith("CHUV"):
        return IMAGES_TS_DIR, ENS_DIR
    return IMAGES_TR_DIR, OOF_DIR


def _load_nii(path):
    img  = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img


def _load_probmap(npz_path):
    """NPZ [C,Z,Y,X]  ->  [C,X,Y,Z]  (aligns with NIfTI nibabel convention)."""
    npz = np.load(str(npz_path))
    key = ("probabilities" if "probabilities" in npz
           else "softmax"  if "softmax"       in npz
           else list(npz.keys())[0])
    p = npz[key].astype(np.float32)     # [C, Z, Y, X]
    return p.transpose(0, 3, 2, 1)      # [C, X, Y, Z]


def _affines_match(imgs):
    ref = imgs[0].affine
    return all(np.allclose(img.affine, ref, atol=1e-3) for img in imgs[1:])


def _norm_ct(ct):
    return (np.clip(ct, -200., 300.) + 200.) / 500.

def _norm_pet(pet):
    p = np.percentile(pet, 99)
    return np.clip(pet / p, 0., 1.) if p > 1e-6 else pet

def _dice(pred, gt, label):
    p = (pred == label).astype(np.float32)
    g = (gt   == label).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return 2 * inter / denom if denom > 0 else float("nan")

def _seg_rgba(s2d):
    h, w = s2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[s2d == 1] = [1.0, 0.15, 0.15, 0.6]  # GTV-T red
    rgba[s2d == 2] = [0.0, 0.90, 0.90, 0.6]  # GTV-N cyan
    return rgba


# ── registration report ───────────────────────────────────────────────────────

def registration_report(pid):
    img_dir, prob_dir = _img_dirs(pid)

    ct_path   = img_dir  / f"{pid}_0000.nii.gz"
    pet_path  = img_dir  / f"{pid}_0001.nii.gz"
    seg_path  = prob_dir / f"{pid}.nii.gz"          # predicted hard seg
    npz_path  = prob_dir / f"{pid}.npz"
    gt_path   = LABELS_TR_DIR / f"{pid}.nii.gz"     # GT label (train only)

    missing = [p for p in [ct_path, pet_path, seg_path, npz_path] if not p.exists()]
    if missing:
        print(f"[{pid}]  MISSING: {[p.name for p in missing]}")
        return {}

    ct_data,  ct_img  = _load_nii(ct_path)
    pet_data, pet_img = _load_nii(pet_path)
    seg_data, seg_img = _load_nii(seg_path)
    probs             = _load_probmap(npz_path)

    has_gt   = gt_path.exists()
    gt_data  = None
    dice_t   = dice_n = float("nan")
    if has_gt:
        gt_data, _ = _load_nii(gt_path)
        gt_data    = gt_data.astype(np.uint8)
        dice_t     = _dice(seg_data, gt_data, 1)
        dice_n     = _dice(seg_data, gt_data, 2)

    print(f"\n{'='*64}")
    print(f"  Patient  : {pid}  {'(test - no GT)' if not has_gt else '(train)'}")
    print(f"{'='*64}")
    for label, img, data in [("CT", ct_img, ct_data), ("PET", pet_img, pet_data),
                              ("Pred seg", seg_img, seg_data)]:
        zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
        orig  = tuple(round(float(v), 2) for v in img.affine[:3, 3])
        print(f"  {label:<10} shape={img.shape}  zooms={zooms}  origin={orig}")
    print(f"  NPZ key : 'probabilities'  raw=(3,Z,Y,X)  "
          f"transposed [C,X,Y,Z]={probs.shape}")
    aligned = _affines_match([ct_img, pet_img, seg_img])
    print(f"  Affines (CT/PET/pred seg): {'OK' if aligned else 'MISMATCH'}")
    nii_sp  = ct_data.shape
    npz_sp  = probs.shape[1:]
    print(f"  NIfTI {nii_sp} vs NPZ {npz_sp}: {'MATCH' if nii_sp==npz_sp else 'MISMATCH'}")
    n_t = int((seg_data == 1).sum())
    n_n = int((seg_data == 2).sum())
    print(f"  Pred seg voxels: GTV-T={n_t}  GTV-N={n_n}")
    if has_gt:
        g_t = int((gt_data == 1).sum())
        g_n = int((gt_data == 2).sum())
        print(f"  GT   seg voxels: GTV-T={g_t}  GTV-N={g_n}")
        print(f"  Dice            : GTV-T={dice_t:.3f}  GTV-N={dice_n:.3f}")

    return {
        "ct": ct_data, "pet": pet_data,
        "seg": seg_data,                # predicted hard seg [X,Y,Z] labels 0/1/2
        "gt":  gt_data,                 # GT labels [X,Y,Z] or None
        "probs": probs,                 # [3, X, Y, Z]  ch0=bg ch1=GTV-T ch2=GTV-N
        "has_gt": has_gt,
        "dice_t": dice_t, "dice_n": dice_n,
    }


# ── visualisation ─────────────────────────────────────────────────────────────

def visualise(pid, save=True, show=False):
    data = registration_report(pid)
    if not data:
        return

    ct     = _norm_ct(data["ct"])
    pet    = _norm_pet(data["pet"])
    seg    = data["seg"]             # predicted hard seg
    gt     = data["gt"]              # GT or None
    probs  = data["probs"]           # [3, X, Y, Z]
    prob_t = probs[1]                # GTV-T probability  [X, Y, Z]
    prob_n = probs[2]                # GTV-N probability  [X, Y, Z]

    # slice position: centre-of-mass of predicted GTV
    mask = (seg > 0).astype(np.float32)
    if mask.sum() > 0:
        com = center_of_mass(mask)
        cx, cy, cz = int(com[0]), int(com[1]), int(com[2])
    else:
        cx, cy, cz = [s // 2 for s in ct.shape]
    print(f"  GTV CoM  (X,Y,Z): ({cx}, {cy}, {cz})")

    # ── Build (ct, pet, gt_seg, pred_seg, prob_t, prob_n) per view ───────────
    def _slices(fixed_axis):
        """Return per-view slices for all 6 channels, fixing one spatial axis."""
        idx = {0: cx, 1: cy, 2: cz}
        def s(arr):
            sl = [slice(None)] * 3
            sl[fixed_axis] = idx[fixed_axis]
            return np.rot90(arr[tuple(sl)])
        return (s(ct), s(pet),
                s(gt) if gt is not None else None,
                s(seg),
                s(prob_t), s(prob_n))

    views = [
        ("Axial   z=" + str(cz),  _slices(2)),
        ("Coronal y=" + str(cy),  _slices(1)),
        ("Sagittal x=" + str(cx), _slices(0)),
    ]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=3, ncols=6, figsize=(24, 12),
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )

    col_titles = ["CT", "PET",
                  "CT + GT label\n(ground truth)",
                  "CT + predicted seg\n(OOF / ensemble)",
                  "GTV-T probability", "GTV-N probability"]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, pad=5)

    for row, (view_name, (ct_s, pet_s, gt_s, seg_s, pt_s, pn_s)) in enumerate(views):
        axs = axes[row]

        axs[0].imshow(ct_s,  cmap="gray", aspect="auto", vmin=0, vmax=1)
        axs[0].set_ylabel(view_name, fontsize=8)

        axs[1].imshow(pet_s, cmap="hot",  aspect="auto", vmin=0, vmax=1)

        # GT overlay
        axs[2].imshow(ct_s, cmap="gray", aspect="auto", vmin=0, vmax=1)
        if gt_s is not None:
            axs[2].imshow(_seg_rgba(gt_s), aspect="auto")
        else:
            axs[2].text(0.5, 0.5, "No GT\navailable",
                        ha="center", va="center", transform=axs[2].transAxes,
                        color="white", fontsize=10)
            axs[2].set_facecolor("black")

        # Predicted seg overlay
        axs[3].imshow(ct_s, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axs[3].imshow(_seg_rgba(seg_s), aspect="auto")

        # GTV-T prob map
        axs[4].imshow(pt_s, cmap="Reds",  aspect="auto", vmin=0, vmax=1)
        if pt_s.max() > 0.5:
            try:
                axs[4].contour(pt_s, levels=[0.5], colors="red",  linewidths=0.8)
            except Exception:
                pass

        # GTV-N prob map
        axs[5].imshow(pn_s, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        if pn_s.max() > 0.5:
            try:
                axs[5].contour(pn_s, levels=[0.5], colors="cyan", linewidths=0.8)
            except Exception:
                pass

        for ax in axs:
            ax.axis("off")

    # colorbars
    fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"),
                 ax=axes[:, 4], shrink=0.55, label="GTV-T prob")
    fig.colorbar(plt.cm.ScalarMappable(cmap="Blues"),
                 ax=axes[:, 5], shrink=0.55, label="GTV-N prob")

    legend = [
        mpatches.Patch(color=(1.0, 0.15, 0.15, 0.85), label="GTV-T"),
        mpatches.Patch(color=(0.0, 0.90, 0.90, 0.85), label="GTV-N"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.35, 0.0))

    dice_str = (f"  |  Dice GTV-T={data['dice_t']:.3f}  GTV-N={data['dice_n']:.3f}"
                if data["has_gt"] else "  |  test set (no GT)")
    fig.suptitle(f"{pid}{dice_str}", fontsize=12, y=1.01)

    if save:
        out = OUT_DIR / f"{pid}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  Figure saved -> {out}")
    if show:
        plt.show()
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def _collect_pids(n):
    with open(SPLITS_JSON) as f:
        folds = json.load(f)
    train_pids = folds[0]["val"][:n]
    chuv_files = sorted(ENS_DIR.glob("CHUV-*.nii.gz"))
    test_pids  = [p.name.replace(".nii.gz", "") for p in chuv_files[:n]]
    return train_pids + test_pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid",  default=None, help="Single PatientID")
    ap.add_argument("--n",    type=int, default=2,
                    help="Visualise first N from train-fold0-val + N from CHUV test")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pids = [args.pid] if args.pid else _collect_pids(args.n)
    print(f"\nChecking {len(pids)} patient(s): {pids}")
    for pid in pids:
        try:
            visualise(pid, save=True, show=args.show)
        except Exception as e:
            print(f"  ERROR for {pid}: {e}", file=sys.stderr)
    print(f"\nDone. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
