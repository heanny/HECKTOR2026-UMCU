"""
train_multitask.py - 5-fold cross-validation for DenseNet121MultiTask.

Combines Task 2 (T/N staging) and Task 3 (RFS) in a single model and
training loop using a two-phase loss schedule:

  Phase 1  epochs  1 .. warmup_epochs  :  loss = L_T + L_N
  Phase 2  epochs  warmup+1 .. max     :  loss = L_T + L_N + rfs_weight * L_Cox

Primary metrics
---------------
  Task 2 : mean balanced accuracy  = (bacc_T + bacc_N) / 2   (official)
  Task 3 : C-index                                             (official)
  Joint  : mean_bacc + cindex  (used for early-stopping & model selection)

Output folder layout (per input configuration)
-----------------------------------------------
  results/
    checkpoints/multitask/<run_tag>/
      config.json           ← all args, input config, dims — saved once
      fold0.pt              ← best checkpoint (includes config inside)
      fold0_metrics.json    ← best val metrics, easy to read without loading .pt
      fold1.pt
      ...
    logs/multitask/<run_tag>/
      fold0.csv             ← per-epoch train/val metrics
      fold1.csv
      ...
    predictions/multitask/<run_tag>/
      oof_predictions.csv   ← T/N preds + risk scores for all val patients
      test_predictions.csv  ← ensemble test predictions (with --predict_test)

  <run_tag> = input tokens joined by "+"
              e.g. "clinical+CT+PET+prob_T+prob_N"

Usage
-----
python train_multitask.py
python train_multitask.py --predict_test
python train_multitask.py --input clinical CT PET prob_T prob_N   (default)
python train_multitask.py --input CT PET prob_T prob_N            (imaging only)
python train_multitask.py --input clinical                        (clinical only)
python train_multitask.py --input clinical CT PET hard_T hard_N --fold 0
python train_multitask.py --input clinical CT PET prob_T prob_N hard_T hard_N --rfs_weight 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SPLITS_JSON, CLINICAL_CSV, CKPT_DIR, LOG_DIR,
    BATCH_SIZE, NUM_WORKERS, LR, WEIGHT_DECAY, N_EPOCHS,
    T_STAGE_MAP, N_STAGE_MAP,
    N_STAGING_CLINICAL, N_RFS_CLINICAL,
    ALL_INPUT_TOKENS, DEFAULT_INPUT,
    TEST_CENTER,
)
from dataset   import HECKTOR2026Dataset
from models    import DenseNet121MultiTask
from losses    import NegativeLogLikelihood, OrdinalCrossEntropyLoss
from utils     import (
    load_splits, compute_cindex,
    prepare_staging_clinical, prepare_rfs_clinical,
    get_test_patient_ids,
    run_dirs, pred_dir_for, save_run_config,
    parse_input_arg,
)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion_stage, criterion_cox, device, rfs_weight):
    model.eval()
    t_true, t_preds = [], []
    n_true, n_preds = [], []
    all_risks, all_times, all_events = [], [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            images   = batch["image"].to(device)
            clc_stg  = batch["clinical_staging"].to(device)
            clc_rfs  = batch["clinical_rfs"].to(device)
            t_labels = batch["t_label"].to(device)
            n_labels = batch["n_label"].to(device)
            times    = batch["time"].to(device)
            events   = batch["event"].to(device)

            t_logits, n_logits, risk = model(images, clc_stg, clc_rfs)

            loss_t = criterion_stage(t_logits, t_labels)
            loss_n = criterion_stage(n_logits, n_labels)
            loss_cox, _, _ = criterion_cox(risk, times, events)
            loss = loss_t + loss_n + rfs_weight * loss_cox

            total_loss += loss.item()
            n_batches  += 1

            t_preds.extend(t_logits.argmax(1).cpu().numpy())
            n_preds.extend(n_logits.argmax(1).cpu().numpy())
            t_true.extend(t_labels.cpu().numpy())
            n_true.extend(n_labels.cpu().numpy())

            all_risks.extend(risk.cpu().numpy())
            all_times.extend(times.cpu().numpy())
            all_events.extend(events.cpu().numpy())

    t_bacc    = balanced_accuracy_score(t_true, t_preds)
    n_bacc    = balanced_accuracy_score(n_true, n_preds)
    mean_bacc = (t_bacc + n_bacc) / 2.0

    cindex    = compute_cindex(
        np.array(all_risks), np.array(all_times), np.array(all_events)
    )

    # Joint score used for early stopping (equal weight by default)
    joint_score = mean_bacc + cindex

    return {
        "loss":       total_loss / max(n_batches, 1),
        "t_bacc":     t_bacc,
        "n_bacc":     n_bacc,
        "mean_bacc":  mean_bacc,
        "cindex":     cindex,
        "joint":      joint_score,
    }


# ─── One fold ─────────────────────────────────────────────────────────────────

def train_fold(fold_idx, fold, df_full, args, device):
    """
    Train one fold.

    Returns
    -------
    best_score : float
    stg_stats  : dict  — staging clinical fit statistics (for OOF / test inference)
    rfs_stats  : dict  — RFS clinical fit statistics     (for OOF / test inference)
    """
    train_pids = fold["train"]
    val_pids   = fold["val"]

    print(f"\n{'='*64}")
    print(f"  Fold {fold_idx}  |  train={len(train_pids)}  val={len(val_pids)}")
    print(f"{'='*64}")

    # Clinical features — fit on training set only, apply same stats to val
    df_train = df_full.loc[df_full.index.isin(train_pids)]
    df_val   = df_full.loc[df_full.index.isin(val_pids)]

    tr_stg_clc, stg_stats = prepare_staging_clinical(df_train)
    val_stg_clc, _        = prepare_staging_clinical(df_val, stg_stats)

    tr_rfs_clc, rfs_stats = prepare_rfs_clinical(df_train)
    val_rfs_clc, _        = prepare_rfs_clinical(df_val, rfs_stats)

    # pid_index must match the clinical array row order, which follows
    # df_full's order (from .loc[...isin(...)]), NOT the splits list order.
    train_ds = HECKTOR2026Dataset(
        train_pids,
        clinical_staging=tr_stg_clc,
        clinical_rfs=tr_rfs_clc,
        pid_index={pid: i for i, pid in enumerate(df_train.index)},
        task="both", channels=args.image_channels,
        is_test=False, augment=True,
    )
    val_ds = HECKTOR2026Dataset(
        val_pids,
        clinical_staging=val_stg_clc,
        clinical_rfs=val_rfs_clc,
        pid_index={pid: i for i, pid in enumerate(df_val.index)},
        task="both", channels=args.image_channels,
        is_test=False, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = DenseNet121MultiTask(
        in_channels  = len(args.image_channels) if args.use_image else 1,
        use_image    = args.use_image,
        use_clinical = args.use_clinical,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion_stage = OrdinalCrossEntropyLoss(
        n_classes=max(len(T_STAGE_MAP), len(N_STAGE_MAP))
    ).to(device)
    criterion_cox   = NegativeLogLikelihood(l2_reg=args.l2_reg)

    ckpt_dir, log_dir = run_dirs(args.input, "multitask")
    ckpt_path         = ckpt_dir / f"fold{fold_idx}.pt"
    metrics_path      = ckpt_dir / f"fold{fold_idx}_metrics.json"
    log_path          = log_dir  / f"fold{fold_idx}.csv"

    best_score   = -1.0
    patience_cnt = 0
    log_rows     = []

    for epoch in range(1, args.epochs + 1):
        # Two-phase loss weight
        rfs_w = 0.0 if epoch <= args.warmup else args.rfs_weight

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        t_epoch_start = time.time()
        t_batch_end   = time.time()

        for batch_idx, batch in enumerate(train_loader):
            t_load = time.time() - t_batch_end

            images   = batch["image"].to(device)
            clc_stg  = batch["clinical_staging"].to(device)
            clc_rfs  = batch["clinical_rfs"].to(device)
            t_labels = batch["t_label"].to(device)
            n_labels = batch["n_label"].to(device)
            times    = batch["time"].to(device)
            events   = batch["event"].to(device)
            t_gpu_start = time.time()

            optimizer.zero_grad()
            t_logits, n_logits, risk = model(images, clc_stg, clc_rfs)

            loss_t = criterion_stage(t_logits, t_labels)
            loss_n = criterion_stage(n_logits, n_labels)
            loss_cox, _, _ = criterion_cox(risk, times, events)
            loss = loss_t + loss_n + rfs_w * loss_cox

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

            t_compute = time.time() - t_gpu_start
            loss_val  = loss.item()
            if epoch == 1:   # per-batch detail only on first epoch
                print(
                    f"  ep{epoch:03d} batch {batch_idx:3d}/{len(train_loader)}:  "
                    f"load={t_load:.2f}s  compute={t_compute:.2f}s  "
                    f"loss={loss_val:.4f}"
                )
            if not np.isfinite(loss_val):
                print(
                    f"  *** NaN/Inf loss at epoch {epoch} batch {batch_idx} ***\n"
                    f"  PIDs in batch : {list(batch['pid'])}\n"
                    f"  image  has NaN: {torch.isnan(images).any().item()}  "
                    f"has Inf: {torch.isinf(images).any().item()}\n"
                    f"  clc_stg has NaN: {torch.isnan(clc_stg).any().item()}  "
                    f"clc_rfs has NaN: {torch.isnan(clc_rfs).any().item()}\n"
                    f"  t_logits NaN: {torch.isnan(t_logits).any().item()}  "
                    f"n_logits NaN: {torch.isnan(n_logits).any().item()}  "
                    f"risk NaN: {torch.isnan(risk).any().item()}\n"
                    f"  t_label: {t_labels.tolist()}  n_label: {n_labels.tolist()}"
                )

            t_batch_end = time.time()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        epoch_time = time.time() - t_epoch_start
        print(f"  [timing] epoch {epoch:3d} total: {epoch_time:.1f}s")

        # ── Validate ──────────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, criterion_stage, criterion_cox,
                           device, rfs_w)

        phase = "warm-up" if epoch <= args.warmup else "joint  "
        print(
            f"  [{phase}] ep {epoch:3d}/{args.epochs}"
            f"  trn_loss={train_loss:.4f}"
            f"  val_loss={metrics['loss']:.4f}"
            f"  T-bacc={metrics['t_bacc']:.3f}"
            f"  N-bacc={metrics['n_bacc']:.3f}"
            f"  cindex={metrics['cindex']:.3f}"
            f"  joint={metrics['joint']:.3f}"
            + (" [new best]" if metrics["joint"] > best_score else "")
        )

        log_rows.append({
            "epoch": epoch, "phase": phase.strip(),
            "train_loss": train_loss, **metrics,
        })

        # ── Checkpoint payload (always includes run config for reproducibility) ─
        ckpt_payload = {
            "epoch":       epoch,
            "fold":        fold_idx,
            "state_dict":  model.state_dict(),
            "metrics":     metrics,
            # Input config — needed to reconstruct model at inference time
            "input":              args.input,
            "image_channels":     args.image_channels,
            "use_image":          args.use_image,
            "use_clinical":       args.use_clinical,
            "in_channels":        len(args.image_channels) if args.use_image else 1,
            "n_staging_clinical": N_STAGING_CLINICAL,
            "n_rfs_clinical":     N_RFS_CLINICAL,
            "warmup":             args.warmup,
            "rfs_weight":         args.rfs_weight,
            "l2_reg":             args.l2_reg,
        }

        # ── Early stopping (only after warm-up) ───────────────────────────────
        if epoch > args.warmup:
            if metrics["joint"] > best_score:
                best_score   = metrics["joint"]
                patience_cnt = 0
                torch.save(ckpt_payload, ckpt_path)
                with open(metrics_path, "w") as f:
                    json.dump({"epoch": epoch, **metrics,
                               "input": args.input}, f, indent=2)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                    break
        else:
            # Warm-up: save latest as fallback
            torch.save(ckpt_payload, ckpt_path)

    # ── Save epoch log ────────────────────────────────────────────────────────
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"  Best joint score : {best_score:.4f}")
    print(f"  Checkpoint       : {ckpt_path}")
    print(f"  Best metrics JSON: {metrics_path}")
    print(f"  Epoch log CSV    : {log_path}")
    return best_score, stg_stats, rfs_stats


# ─── Inference helpers ────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device: torch.device) -> DenseNet121MultiTask:
    """Reconstruct model from checkpoint; all config is stored inside the .pt file."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DenseNet121MultiTask(
        in_channels  = ckpt.get("in_channels",          4),
        n_stg_clc    = ckpt.get("n_staging_clinical",   N_STAGING_CLINICAL),
        n_rfs_clc    = ckpt.get("n_rfs_clinical",        N_RFS_CLINICAL),
        use_image    = ckpt.get("use_image",             True),
        use_clinical = ckpt.get("use_clinical",          True),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _run_inference(model, loader, fold_idx: int, device) -> pd.DataFrame:
    """
    Run the multitask model over a loader.

    Returns a DataFrame with one row per patient containing:
      T/N stage predictions (argmax + per-class softmax) and risk score.
    """
    inv_t = {v: k for k, v in T_STAGE_MAP.items()}
    inv_n = {v: k for k, v in N_STAGE_MAP.items()}
    rows  = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs     = batch["image"].to(device)
            clc_stg  = batch["clinical_staging"].to(device)
            clc_rfs  = batch["clinical_rfs"].to(device)
            t_logits, n_logits, risk = model(imgs, clc_stg, clc_rfs)
            t_prob  = t_logits.softmax(-1).cpu().numpy()
            n_prob  = n_logits.softmax(-1).cpu().numpy()
            risk_np = risk.cpu().numpy()
            for i, pid in enumerate(batch["pid"]):
                tp  = int(t_prob[i].argmax())
                np_ = int(n_prob[i].argmax())
                rows.append({
                    "PatientID":    pid,
                    "fold":         fold_idx,
                    "T_pred":       tp,
                    "N_pred":       np_,
                    "T_stage_pred": inv_t[tp],
                    "N_stage_pred": inv_n[np_],
                    **{f"T_prob_T{j+1}": float(t_prob[i][j]) for j in range(4)},
                    **{f"N_prob_N{j}":   float(n_prob[i][j]) for j in range(4)},
                    "risk_score":   float(risk_np[i]),
                })
    return pd.DataFrame(rows)


def predict_oof(fold_idx, val_ids, df_val,
                stg_clc, rfs_clc, ckpt_path, args) -> pd.DataFrame:
    """OOF predictions for one fold's validation patients."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model  = _load_model(ckpt_path, device)
    ds = HECKTOR2026Dataset(
        val_ids,
        clinical_staging=stg_clc,
        clinical_rfs=rfs_clc,
        pid_index={pid: i for i, pid in enumerate(df_val.index)},
        task="both", channels=args.image_channels,
        is_test=False, augment=False,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    return _run_inference(model, loader, fold_idx, device)


def predict_test(test_ids, df_test, stg_clc, rfs_clc,
                 all_ckpts, args) -> pd.DataFrame:
    """
    Ensemble test predictions: average T/N softmax and risk score across
    all 5 fold checkpoints, then argmax for the final stage labels.
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ds = HECKTOR2026Dataset(
        test_ids,
        clinical_staging=stg_clc,
        clinical_rfs=rfs_clc,
        pid_index={pid: i for i, pid in enumerate(df_test.index)},
        task="both", channels=args.image_channels,
        is_test=True, augment=False,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)

    fold_dfs = [
        _run_inference(_load_model(ck, device), loader, fi, device)
        for fi, ck in enumerate(all_ckpts)
    ]

    # Average per-patient probabilities and risk scores across folds
    t_cols   = [f"T_prob_T{j+1}" for j in range(4)]
    n_cols   = [f"N_prob_N{j}"   for j in range(4)]
    avg_cols = t_cols + n_cols + ["risk_score"]
    avg = (pd.concat(fold_dfs)
             .groupby("PatientID")[avg_cols]
             .mean()
             .reset_index())
    inv_t = {v: k for k, v in T_STAGE_MAP.items()}
    inv_n = {v: k for k, v in N_STAGE_MAP.items()}
    avg["T_pred"]       = avg[t_cols].values.argmax(axis=1)
    avg["N_pred"]       = avg[n_cols].values.argmax(axis=1)
    avg["T_stage_pred"] = avg["T_pred"].map(inv_t)
    avg["N_stage_pred"] = avg["N_pred"].map(inv_n)
    return avg


def predict_train(train_ids, df_train, stg_clc, rfs_clc, all_ckpts, args) -> pd.DataFrame:
    """Average T/N softmax and risk score across all 5 fold models for all training patients."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ds = HECKTOR2026Dataset(
        train_ids,
        clinical_staging=stg_clc,
        clinical_rfs=rfs_clc,
        pid_index={pid: i for i, pid in enumerate(df_train.index)},
        task="both", channels=args.image_channels,
        is_test=False, augment=False,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    fold_dfs = [
        _run_inference(_load_model(ck, device), loader, fi, device)
        for fi, ck in enumerate(all_ckpts)
    ]
    t_cols   = [f"T_prob_T{j+1}" for j in range(4)]
    n_cols   = [f"N_prob_N{j}"   for j in range(4)]
    avg_cols = t_cols + n_cols + ["risk_score"]
    avg = (pd.concat(fold_dfs)
             .groupby("PatientID")[avg_cols]
             .mean()
             .reset_index())
    inv_t = {v: k for k, v in T_STAGE_MAP.items()}
    inv_n = {v: k for k, v in N_STAGE_MAP.items()}
    avg["T_pred"]       = avg[t_cols].values.argmax(axis=1)
    avg["N_pred"]       = avg[n_cols].values.argmax(axis=1)
    avg["T_stage_pred"] = avg["T_pred"].map(inv_t)
    avg["N_stage_pred"] = avg["N_pred"].map(inv_n)
    return avg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",         type=int,   default=None,
                    help="Single fold index (0-4). Default: run all 5 folds.")
    ap.add_argument("--epochs",       type=int,   default=N_EPOCHS)
    ap.add_argument("--warmup",       type=int,   default=15,
                    help="Epochs of staging-only warm-up before Cox loss is added.")
    ap.add_argument("--rfs_weight",   type=float, default=1.0,
                    help="Weight for Cox loss in phase 2 (default 1.0).")
    ap.add_argument("--patience",     type=int,   default=25)
    ap.add_argument("--l2_reg",       type=float, default=1e-4)
    ap.add_argument("--gpu",          type=int,   default=0)
    ap.add_argument("--predict_test",  action="store_true",
                    help="Generate CHUV test predictions after all folds finish.")
    ap.add_argument("--predict_train", action="store_true",
                    help="Generate train ensemble predictions (all 5 models on all training patients).")
    ap.add_argument("--predict_only",  action="store_true",
                    help="Skip training; load existing checkpoints and run requested predictions.")
    ap.add_argument(
        "--input", nargs="+",
        default=DEFAULT_INPUT,
        metavar="TOKEN",
        help=(
            "Model inputs (space-separated). Include 'clinical' for clinical "
            "features; add image channel names for imaging. "
            f"Available tokens: {ALL_INPUT_TOKENS}. "
            "Examples:\n"
            "  --input clinical CT PET prob_T prob_N  (default)\n"
            "  --input CT PET prob_T prob_N           (imaging only)\n"
            "  --input clinical                       (clinical only)"
        ),
    )
    args = ap.parse_args()

    # ── Parse --input into use flags ─────────────────────────────────────────
    args.use_clinical, args.image_channels = parse_input_arg(args.input)
    args.use_image = len(args.image_channels) > 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Input tokens  : {args.input}")
    print(f"use_image     : {args.use_image}  channels={args.image_channels}")
    print(f"use_clinical  : {args.use_clinical}")
    print(f"Warmup        : {args.warmup} epochs  |  rfs_weight: {args.rfs_weight}")

    # Save config.json once for this run (before folds start)
    ckpt_dir, _ = run_dirs(args.input, "multitask")
    save_run_config(ckpt_dir, {
        "task":               "multitask",
        "input":              args.input,
        "image_channels":     args.image_channels,
        "use_image":          args.use_image,
        "use_clinical":       args.use_clinical,
        "in_channels":        len(args.image_channels) if args.use_image else 1,
        "n_staging_clinical": N_STAGING_CLINICAL,
        "n_rfs_clinical":     N_RFS_CLINICAL,
        "epochs":             args.epochs,
        "warmup":             args.warmup,
        "rfs_weight":         args.rfs_weight,
        "patience":           args.patience,
        "l2_reg":             args.l2_reg,
        "batch_size":         BATCH_SIZE,
        "lr":                 LR,
    })

    splits   = load_splits(SPLITS_JSON)
    df_full  = pd.read_csv(CLINICAL_CSV).set_index("PatientID")
    test_ids = get_test_patient_ids(CLINICAL_CSV, TEST_CENTER)

    folds_to_run = [args.fold] if args.fold is not None else list(range(len(splits)))
    pred_dir     = pred_dir_for(args.input, "multitask")

    if args.predict_only:
        ckpt_paths = [ckpt_dir / f"fold{fi}.pt" for fi in range(5)]
        missing = [str(p) for p in ckpt_paths if not p.exists()]
        if missing:
            print("--predict_only: missing checkpoints:\n" + "\n".join(missing))
            sys.exit(1)
        print("--predict_only: skipping training.")
    else:
        scores     = []
        ckpt_paths = []
        all_oof    = []

        for fi in folds_to_run:
            val_ids = splits[fi]["val"]

            score, stg_stats_f, rfs_stats_f = train_fold(
                fi, splits[fi], df_full, args, device
            )
            scores.append(score)

            ckpt_path = ckpt_dir / f"fold{fi}.pt"
            ckpt_paths.append(ckpt_path)

            # ── OOF predictions for this fold ─────────────────────────────────
            df_val_f    = df_full.loc[df_full.index.isin(val_ids)]
            val_stg_f,_ = prepare_staging_clinical(df_val_f, stg_stats_f)
            val_rfs_f,_ = prepare_rfs_clinical(df_val_f, rfs_stats_f)
            all_oof.append(
                predict_oof(fi, val_ids, df_val_f,
                            val_stg_f, val_rfs_f, ckpt_path, args)
            )

        # ── OOF summary ───────────────────────────────────────────────────────
        if all_oof:
            oof = pd.concat(all_oof, ignore_index=True)
            oof.to_csv(pred_dir / "oof_predictions.csv", index=False)

            oof_lbl = oof.join(
                df_full[["T-stage", "N-stage", "RFS", "Relapse"]], on="PatientID"
            ).dropna(subset=["T-stage", "N-stage", "RFS", "Relapse"])

            if len(oof_lbl) > 0:
                oof_lbl["T_true"] = oof_lbl["T-stage"].map(T_STAGE_MAP)
                oof_lbl["N_true"] = oof_lbl["N-stage"].map(N_STAGE_MAP)
                t_bacc = balanced_accuracy_score(oof_lbl["T_true"], oof_lbl["T_pred"])
                n_bacc = balanced_accuracy_score(oof_lbl["N_true"], oof_lbl["N_pred"])
                oof_ci = compute_cindex(
                    oof_lbl["risk_score"].values,
                    oof_lbl["RFS"].values,
                    oof_lbl["Relapse"].astype(float).values,
                )
                print(
                    f"\n── OOF Summary ──────────────────────────────────────\n"
                    f"  T-stage BAcc  = {t_bacc:.4f}\n"
                    f"  N-stage BAcc  = {n_bacc:.4f}\n"
                    f"  Mean    BAcc  = {(t_bacc + n_bacc) / 2:.4f}\n"
                    f"  OOF C-index   = {oof_ci:.4f}\n"
                    f"  (over {len(oof_lbl)} patients with all 4 labels)\n"
                    f"─────────────────────────────────────────────────────"
                )

        # ── 5-fold summary ────────────────────────────────────────────────────
        if len(scores) > 1:
            print(f"\n5-fold mean joint score: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
            summary = {
                "input":       args.input,
                "fold_scores": scores,
                "mean_joint":  float(np.mean(scores)),
                "std_joint":   float(np.std(scores)),
            }
            with open(ckpt_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved -> {ckpt_dir / 'summary.json'}")

    # ── Test predictions (ensemble over all 5 folds) ──────────────────────────
    if args.predict_test and len(ckpt_paths) == 5:
        print("\nGenerating CHUV test predictions ...")
        df_alltrain = df_full.loc[~df_full.index.str.startswith(TEST_CENTER)]
        _, all_stg_stats = prepare_staging_clinical(df_alltrain)
        _, all_rfs_stats = prepare_rfs_clinical(df_alltrain)
        df_test    = df_full.loc[df_full.index.isin(test_ids)]
        test_stg,_ = prepare_staging_clinical(df_test, all_stg_stats)
        test_rfs,_ = prepare_rfs_clinical(df_test,     all_rfs_stats)
        test_df = predict_test(
            test_ids, df_test, test_stg, test_rfs, ckpt_paths, args
        )
        out = pred_dir / "test_predictions.csv"
        test_df.to_csv(out, index=False)
        print(f"Test predictions saved -> {out}")
    elif args.predict_test:
        print(
            f"\n[Warning] --predict_test requires all 5 folds; "
            f"only {len(ckpt_paths)} checkpoint(s) available. Skipping."
        )

    # ── Train ensemble predictions ────────────────────────────────────────────
    if args.predict_train and len(ckpt_paths) == 5:
        print("\nGenerating train ensemble predictions ...")
        df_alltrain  = df_full.loc[~df_full.index.str.startswith(TEST_CENTER)]
        train_stg, _ = prepare_staging_clinical(df_alltrain)
        train_rfs, _ = prepare_rfs_clinical(df_alltrain)
        train_df = predict_train(
            df_alltrain.index.tolist(), df_alltrain,
            train_stg, train_rfs, ckpt_paths, args
        )
        out = pred_dir / "train_ensemble_predictions.csv"
        train_df.to_csv(out, index=False)
        print(f"Train ensemble predictions saved -> {out}")
    elif args.predict_train:
        print(
            f"\n[Warning] --predict_train requires all 5 folds; "
            f"only {len(ckpt_paths)} checkpoint(s) available. Skipping."
        )


if __name__ == "__main__":
    main()
