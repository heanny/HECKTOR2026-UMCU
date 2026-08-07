"""
train_task3.py — 5-fold CV for HECKTOR 2026 Task 3 (RFS survival prediction).

Official metric  : Concordance Index (C-index) — higher is better.
Model selection  : best validation C-index.
Early stopping   : patience epochs with no improvement (default 25).

Output folder layout
--------------------
  results/
    checkpoints/task3/<run_tag>/
      config.json               all args + dims, saved once per run
      fold0.pt                  best checkpoint (state_dict + config inside)
      fold0_metrics.json        best val metrics (human-readable)
      ...
      summary.json              mean+std C-index across folds
    logs/task3/<run_tag>/
      fold0.csv                 per-epoch train/val metrics
      ...
    predictions/task3/<run_tag>/
      oof_predictions.csv
      test_predictions.csv      (only with --predict_test)

  <run_tag> = input tokens joined by "+"
              e.g. "clinical+CT+PET+prob_T+prob_N"

Usage
-----
python train_task3.py
python train_task3.py --input clinical CT PET prob_T prob_N   (default)
python train_task3.py --input CT PET prob_T prob_N            (imaging only)
python train_task3.py --input clinical                        (clinical only)
python train_task3.py --fold 2 --predict_test
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLINICAL_CSV, SPLITS_JSON,
    BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    LR, WEIGHT_DECAY, LR_MILESTONES, LR_GAMMA, SEED, TEST_CENTER,
    N_RFS_CLINICAL,
    ALL_INPUT_TOKENS, DEFAULT_INPUT,
)
from dataset import HECKTOR2026Dataset
from models  import DenseNet121Task3
from losses  import NegativeLogLikelihood
from utils   import (
    load_splits, get_test_patient_ids,
    prepare_rfs_clinical,
    compute_cindex,
    run_dirs, pred_dir_for, save_run_config,
    parse_input_arg,
)

TASK = "task3"


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss              = 0.0
    all_risks, all_t, all_e = [], [], []

    with torch.no_grad():
        for batch in loader:
            image   = batch["image"].to(device)
            clc_rfs = batch["clinical_rfs"].to(device)
            times   = batch["time"].to(device)
            events  = batch["event"].to(device)

            risk = model(image, clc_rfs)
            loss, _, _ = criterion(risk, times, events)
            total_loss += loss.item()

            all_risks.extend(risk.cpu().numpy())
            all_t.extend(times.cpu().numpy())
            all_e.extend(events.cpu().numpy())

    cindex = compute_cindex(
        np.array(all_risks), np.array(all_t), np.array(all_e)
    )
    return {"loss": total_loss / max(1, len(loader)), "cindex": cindex}


# ─── Single fold ─────────────────────────────────────────────────────────────

def train_one_fold(fold_idx, train_ids, val_ids, df_full, args, device) -> tuple:
    """Train one fold. Returns (ckpt_path, history_list, fit_stats)."""

    ckpt_dir, log_dir = run_dirs(args.input, TASK)

    # ── Clinical features — fit on train, apply to val ────────────────────────
    df_tr  = df_full.loc[df_full.index.isin(train_ids)]
    df_val = df_full.loc[df_full.index.isin(val_ids)]

    tr_clc,  stats = prepare_rfs_clinical(df_tr)
    val_clc, _     = prepare_rfs_clinical(df_val, stats)

    # pid_index must match the clinical array row order, which follows
    # df_full's order (from .loc[...isin(...)]), NOT the splits list order.
    train_ds = HECKTOR2026Dataset(
        train_ids,
        clinical_staging=None, clinical_rfs=tr_clc,
        pid_index={pid: i for i, pid in enumerate(df_tr.index)},
        task="rfs", channels=args.image_channels,
        is_test=False, augment=True,
    )
    val_ds = HECKTOR2026Dataset(
        val_ids,
        clinical_staging=None, clinical_rfs=val_clc,
        pid_index={pid: i for i, pid in enumerate(df_val.index)},
        task="rfs", channels=args.image_channels,
        is_test=False, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = DenseNet121Task3(
        in_channels  = len(args.image_channels) if args.use_image else 1,
        n_clinical   = N_RFS_CLINICAL,
        use_image    = args.use_image,
        use_clinical = args.use_clinical,
    ).to(device)
    criterion = NegativeLogLikelihood(l2_reg=args.l2_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    ckpt_path    = ckpt_dir / f"fold{fold_idx}.pt"
    metrics_path = ckpt_dir / f"fold{fold_idx}_metrics.json"
    log_path     = log_dir  / f"fold{fold_idx}.csv"

    best_cindex  = 0.0
    patience_cnt = 0
    history      = []

    for epoch in range(1, args.epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        t_epoch_start = time.time()
        t_batch_end   = time.time()

        for batch_idx, batch in enumerate(train_loader):
            t_load = time.time() - t_batch_end

            image   = batch["image"].to(device)
            clc_rfs = batch["clinical_rfs"].to(device)
            times   = batch["time"].to(device)
            events  = batch["event"].to(device)
            t_gpu_start = time.time()

            optimizer.zero_grad()
            risk          = model(image, clc_rfs)
            loss, nll, l2 = criterion(risk, times, events)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            t_compute = time.time() - t_gpu_start
            loss_val  = loss.item()
            if epoch == 1:   # per-batch detail only on first epoch
                logging.info(
                    f"  ep{epoch:03d} batch {batch_idx:3d}/{len(train_loader)}:  "
                    f"load={t_load:.2f}s  compute={t_compute:.2f}s  "
                    f"loss={loss_val:.4f}"
                )
            if not np.isfinite(loss_val):
                logging.warning(
                    f"  *** NaN/Inf loss at epoch {epoch} batch {batch_idx} ***\n"
                    f"  PIDs in batch : {list(batch['pid'])}\n"
                    f"  image  has NaN: {torch.isnan(image).any().item()}  "
                    f"has Inf: {torch.isinf(image).any().item()}\n"
                    f"  clc    has NaN: {torch.isnan(clc_rfs).any().item()}  "
                    f"has Inf: {torch.isinf(clc_rfs).any().item()}\n"
                    f"  risk   has NaN: {torch.isnan(risk).any().item()}\n"
                    f"  times: {times.tolist()}  events: {events.tolist()}"
                )

            t_batch_end = time.time()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        epoch_time = time.time() - t_epoch_start
        logging.info(f"  [timing] epoch {epoch:3d} total: {epoch_time:.1f}s")

        # ── Validate ──────────────────────────────────────────────────────────
        m      = evaluate(model, val_loader, criterion, device)
        val_ci = m["cindex"]

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": m["loss"], "val_cindex": val_ci})

        logging.info(
            f"Fold {fold_idx} | Ep {epoch:03d} | "
            f"train={train_loss:.4f} | val_loss={m['loss']:.4f} | "
            f"val_cindex={val_ci:.4f} | pat={patience_cnt}/{args.patience}"
            + ("  [best]" if val_ci > best_cindex else "")
        )

        # ── Checkpoint + early stopping ───────────────────────────────────────
        ckpt_payload = {
            "epoch": epoch, "fold": fold_idx,
            "state_dict": model.state_dict(),
            "metrics": m,
            # Input config — needed to reconstruct model at inference time
            "input":          args.input,
            "image_channels": args.image_channels,
            "use_image":      args.use_image,
            "use_clinical":   args.use_clinical,
            "in_channels":    len(args.image_channels) if args.use_image else 1,
            "n_rfs_clinical": N_RFS_CLINICAL,
        }
        if val_ci > best_cindex:
            best_cindex  = val_ci
            patience_cnt = 0
            torch.save(ckpt_payload, ckpt_path)
            with open(metrics_path, "w") as f:
                json.dump({"epoch": epoch, **m, "input": args.input}, f, indent=2)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logging.info(f"  Early stop at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(log_path, index=False)
    logging.info(f"Fold {fold_idx} done — best C-index = {best_cindex:.4f}")
    logging.info(f"  checkpoint : {ckpt_path}")
    logging.info(f"  epoch log  : {log_path}")
    return ckpt_path, history, stats


# ─── Inference helpers ────────────────────────────────────────────────────────

def _load_model(ckpt_path, device):
    """Load model from checkpoint; input config is read from the checkpoint itself."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = DenseNet121Task3(
        in_channels  = ckpt.get("in_channels", 4),
        n_clinical   = ckpt.get("n_rfs_clinical", N_RFS_CLINICAL),
        use_image    = ckpt.get("use_image", True),
        use_clinical = ckpt.get("use_clinical", True),
    ).to(device)
    m.load_state_dict(ckpt["state_dict"])
    return m


def _run_inference(model, loader, fold_idx, device) -> pd.DataFrame:
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            image   = batch["image"].to(device)
            clc_rfs = batch["clinical_rfs"].to(device)
            risk    = model(image, clc_rfs).cpu().numpy()
            for i, pid in enumerate(batch["pid"]):
                rows.append({"PatientID": pid, "fold": fold_idx,
                             "risk_score": float(risk[i])})
    return pd.DataFrame(rows)


def predict_oof(fold_idx, val_ids, clc_rfs, pid_index, ckpt_path, args):
    device = torch.device(args.device)
    model  = _load_model(ckpt_path, device)
    ds     = HECKTOR2026Dataset(
        val_ids, clinical_staging=None, clinical_rfs=clc_rfs,
        pid_index=pid_index, task="rfs", channels=args.image_channels,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=NUM_WORKERS)
    return _run_inference(model, loader, fold_idx, device)


def predict_test(test_ids, clc_rfs, pid_index, all_ckpts, args):
    """Average risk scores from all 5 fold models."""
    device = torch.device(args.device)
    ds     = HECKTOR2026Dataset(
        test_ids, clinical_staging=None, clinical_rfs=clc_rfs,
        pid_index=pid_index, task="rfs", channels=args.image_channels, is_test=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=NUM_WORKERS)
    fold_dfs = [
        _run_inference(_load_model(ck, device), loader, fi, device)
        for fi, ck in enumerate(all_ckpts)
    ]
    return (pd.concat(fold_dfs).groupby("PatientID")["risk_score"]
              .mean().reset_index())


def predict_train(train_ids, clc_rfs, pid_index, all_ckpts, args):
    """Average risk scores from all 5 fold models over all training patients."""
    device = torch.device(args.device)
    ds     = HECKTOR2026Dataset(
        train_ids, clinical_staging=None, clinical_rfs=clc_rfs,
        pid_index=pid_index, task="rfs", channels=args.image_channels,
        is_test=False, augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=NUM_WORKERS)
    fold_dfs = [
        _run_inference(_load_model(ck, device), loader, fi, device)
        for fi, ck in enumerate(all_ckpts)
    ]
    return (pd.concat(fold_dfs).groupby("PatientID")["risk_score"]
              .mean().reset_index())


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="HECKTOR 2026 Task 3 — RFS prediction")
    p.add_argument("--fold",         type=int,   default=-1,
                   help="Fold 0-4; -1 trains all folds.")
    p.add_argument("--device",       type=str,   default="cuda:0")
    p.add_argument("--epochs",       type=int,   default=MAX_EPOCHS)
    p.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--l2_reg",       type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--predict_test",  action="store_true",
                   help="Generate CHUV test predictions after training.")
    p.add_argument("--predict_train", action="store_true",
                   help="Generate train ensemble predictions (all 5 models on all training patients).")
    p.add_argument("--predict_only",  action="store_true",
                   help="Skip training; load existing checkpoints and run requested predictions.")
    p.add_argument(
        "--input", nargs="+", default=DEFAULT_INPUT,
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
    args = p.parse_args()

    # ── Parse --input into use flags ─────────────────────────────────────────
    args.use_clinical, args.image_channels = parse_input_arg(args.input)
    args.use_image = len(args.image_channels) > 0

    set_seed(SEED)
    device = torch.device(args.device)

    ckpt_dir, _ = run_dirs(args.input, TASK)
    save_run_config(ckpt_dir, {
        "task":           TASK,
        "input":          args.input,
        "image_channels": args.image_channels,
        "use_image":      args.use_image,
        "use_clinical":   args.use_clinical,
        "in_channels":    len(args.image_channels) if args.use_image else 1,
        "n_rfs_clinical": N_RFS_CLINICAL,
        "epochs":         args.epochs,
        "patience":       args.patience,
        "lr":             args.lr,
        "l2_reg":         args.l2_reg,
        "batch_size":     args.batch_size,
        "weight_decay":   WEIGHT_DECAY,
    })

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ckpt_dir / "train.log")],
    )
    logging.info(f"Input tokens      : {args.input}")
    logging.info(f"use_image         : {args.use_image}  channels={args.image_channels}")
    logging.info(f"use_clinical      : {args.use_clinical}")
    logging.info(f"RFS clinical features : {N_RFS_CLINICAL}")

    df_full  = pd.read_csv(CLINICAL_CSV).set_index("PatientID")
    folds    = load_splits(SPLITS_JSON)
    test_ids = get_test_patient_ids(CLINICAL_CSV, TEST_CENTER)

    fold_range = range(len(folds)) if args.fold == -1 else [args.fold]
    pred_dir   = pred_dir_for(args.input, TASK)

    if args.predict_only:
        ckpt_paths = [ckpt_dir / f"fold{fi}.pt" for fi in range(5)]
        missing = [str(p) for p in ckpt_paths if not p.exists()]
        if missing:
            logging.error("--predict_only: missing checkpoints:\n" + "\n".join(missing))
            sys.exit(1)
        logging.info("--predict_only: skipping training.")
    else:
        all_oof     = []
        ckpt_paths  = []
        fold_scores = []

        for fold_idx in fold_range:
            train_ids = folds[fold_idx]["train"]
            val_ids   = folds[fold_idx]["val"]
            logging.info(
                f"\n{'='*64}\nFold {fold_idx} | train={len(train_ids)} val={len(val_ids)}\n{'='*64}"
            )

            ckpt, hist, fold_stats = train_one_fold(
                fold_idx, train_ids, val_ids, df_full, args, device)
            ckpt_paths.append(ckpt)

            df_val    = df_full.loc[df_full.index.isin(val_ids)]
            val_clc,_ = prepare_rfs_clinical(df_val, fold_stats)
            pidx      = {pid: i for i, pid in enumerate(df_val.index)}
            all_oof.append(predict_oof(fold_idx, val_ids, val_clc, pidx, ckpt, args))

            m_path = ckpt_dir / f"fold{fold_idx}_metrics.json"
            if m_path.exists():
                with open(m_path) as f:
                    fold_scores.append(json.load(f).get("cindex", 0.0))

        # ── OOF C-index ──────────────────────────────────────────────────────
        if all_oof:
            oof = pd.concat(all_oof, ignore_index=True)
            oof.to_csv(pred_dir / "oof_predictions.csv", index=False)

            oof_lbl = (oof.join(df_full[["RFS", "Relapse"]], on="PatientID")
                          .dropna(subset=["RFS", "Relapse"]))
            oof_ci  = compute_cindex(
                oof_lbl["risk_score"].values,
                oof_lbl["RFS"].values,
                oof_lbl["Relapse"].astype(float).values,
            )
            logging.info(
                f"\n── OOF Summary ──────────────────────────────────────\n"
                f"  OOF C-index = {oof_ci:.4f}\n"
                f"─────────────────────────────────────────────────────"
            )

        # ── 5-fold summary ────────────────────────────────────────────────────
        if len(fold_scores) > 1:
            summary = {
                "input":         args.input,
                "fold_cindices": fold_scores,
                "mean":          float(np.mean(fold_scores)),
                "std":           float(np.std(fold_scores)),
            }
            with open(ckpt_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logging.info(
                f"5-fold mean C-index: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}"
            )

    # ── Test prediction ───────────────────────────────────────────────────────
    if args.predict_test and len(ckpt_paths) == 5:
        logging.info("Generating CHUV test predictions ...")
        df_test     = df_full.loc[df_full.index.isin(test_ids)]
        df_alltrain = df_full.loc[~df_full.index.str.startswith(TEST_CENTER)]
        _, all_stats = prepare_rfs_clinical(df_alltrain)
        test_clc, _  = prepare_rfs_clinical(df_test, all_stats)
        pidx_test    = {pid: i for i, pid in enumerate(df_test.index)}
        test_df = predict_test(test_ids, test_clc, pidx_test, ckpt_paths, args)
        out = pred_dir / "test_predictions.csv"
        test_df.to_csv(out, index=False)
        logging.info(f"Test predictions saved -> {out}")

    # ── Train ensemble prediction ─────────────────────────────────────────────
    if args.predict_train and len(ckpt_paths) == 5:
        logging.info("Generating train ensemble predictions ...")
        df_alltrain   = df_full.loc[~df_full.index.str.startswith(TEST_CENTER)]
        train_ids_all = df_alltrain.index.tolist()
        train_clc, _  = prepare_rfs_clinical(df_alltrain)
        pidx_train    = {pid: i for i, pid in enumerate(df_alltrain.index)}
        train_df = predict_train(train_ids_all, train_clc, pidx_train, ckpt_paths, args)
        out = pred_dir / "train_ensemble_predictions.csv"
        train_df.to_csv(out, index=False)
        logging.info(f"Train ensemble predictions saved -> {out}")


if __name__ == "__main__":
    main()
