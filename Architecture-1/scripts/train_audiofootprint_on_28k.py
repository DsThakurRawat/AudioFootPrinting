#!/usr/bin/env python3
"""
Train AudioFootPrint (MusicScope-CL) on the 28k Dataset (22,583 Train Split)

This script provides the complete training pipeline to train AudioFootPrint's
SupCon projection head and Calibrated FusionMLP on the exact same 22,583-track
training split used by the mirex2predet detectors (using md5 hash partitioning).

Outputs checkpoints to:
    Architecture-1/model/checkpoints/28k_trained/
      ├── supcon_28k_best.pt
      └── fusion_28k_best.pt
"""
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ARCH1_DIR = _SCRIPTS_DIR.parent
_MODEL_DIR = _ARCH1_DIR / "model"
for p in (str(_SCRIPTS_DIR), str(_MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from fusion_model import FusionMLP, TemperatureScaler, label_smoothing_bce
from new_inference import FastMusicScopeScorer, fast_extract_narrative, fast_load_audio_chunk
from track_b_style import StyleEncoder
from train_supcon import SupConProjectionHead, supcon_loss


def _bucket(path: str) -> int:
    return int(hashlib.md5(path.encode()).hexdigest()[:8], 16) % 100


def get_28k_train_val_split():
    """Returns train (bucket >= 20: 22,583) and val (bucket 10..19: 2,747) DataFrames."""
    parquet_path = config.PROCESSED_DATA_DIR / "eval_28k_checkpoint.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"28k dataset not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    
    # Resolve physical file paths if needed
    paths = []
    for fp in df["file_path"]:
        p_str = str(fp)
        if os.path.exists(p_str):
            paths.append(p_str)
        else:
            alt = p_str.replace(r"D:\mirex\data\raw", r"D:\AudioFootPrint\data\raw")
            if os.path.exists(alt):
                paths.append(alt)
            else:
                paths.append(p_str)
    df["resolved_path"] = paths

    # Deterministic partitioning matching mirex2predet
    buckets = [ _bucket(p) for p in df["resolved_path"] ]
    df["bucket"] = buckets

    df_train = df[df["bucket"] >= 20].copy()
    df_val = df[(df["bucket"] >= 10) & (df["bucket"] < 20)].copy()
    df_test = df[df["bucket"] < 10].copy()

    return df_train, df_val, df_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  AudioFootPrint (MusicScope-CL) Training on 28k Dataset Split")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    df_train, df_val, df_test = get_28k_train_val_split()
    print(f"Dataset split (matching mirex2predet deterministically):")
    print(f"  Train : {len(df_train):,d} tracks (bucket >= 20)")
    print(f"  Val   : {len(df_val):,d} tracks (bucket 10..19)")
    print(f"  Test  : {len(df_test):,d} tracks (bucket < 10)")
    print(f"  Total : {len(df_train) + len(df_val) + len(df_test):,d} tracks\n")

    out_ckpt_dir = config.CHECKPOINT_DIR / "28k_trained"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = config.PROCESSED_DATA_DIR / "cache_28k_training"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fused_train_path = cache_dir / "fused_train_22k.npy"
    labels_train_path = cache_dir / "labels_train_22k.npy"
    fused_val_path = cache_dir / "fused_val_2k.npy"
    labels_val_path = cache_dir / "labels_val_2k.npy"

    scorer = FastMusicScopeScorer(mode="balanced")

    def extract_or_load(df_split, feat_path, lab_path, name):
        if feat_path.exists() and lab_path.exists():
            print(f"[{name}] Loading cached features from {feat_path.name}...")
            return np.load(feat_path), np.load(lab_path)

        print(f"[{name}] Multi-process extracting 704-D features for {len(df_split):,d} tracks...")
        paths = df_split["resolved_path"].tolist()
        labels_map = dict(zip(df_split["resolved_path"], df_split["is_ai"]))

        X_list, y_list = [], []
        t0 = time.perf_counter()
        total = len(paths)
        batch_sz = 128
        workers = 12

        from new_inference import _extract_task
        import multiprocessing as mp

        with mp.Pool(processes=workers) as pool:
            for chunk_start in range(0, total, batch_sz):
                chunk_paths = paths[chunk_start:chunk_start + batch_sz]
                batch_data = list(pool.imap(_extract_task, chunk_paths, chunksize=8))

                waves = []
                narrs = []
                valid_labels = []

                for p, y_int16, feats, err in batch_data:
                    if err:
                        continue
                    waves.append(y_int16.astype(np.float32) / 32767.0)
                    narrs.append(scorer._prepare_narr_vector(feats))
                    valid_labels.append(int(labels_map.get(p, 0)))

                if waves:
                    specs_t = scorer._compute_specs_on_gpu(np.array(waves, dtype=np.float32))
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                        surfs = scorer.unified_model.encoder(specs_t).cpu().numpy()

                    for s_emb, n_vec, lab in zip(surfs, narrs, valid_labels):
                        fused = np.concatenate([s_emb, n_vec], axis=0).astype(np.float32)
                        X_list.append(fused)
                        y_list.append(lab)

                done = min(chunk_start + batch_sz, total)
                rate = done / max(time.perf_counter() - t0, 0.001)
                if done % 1024 == 0 or done == total:
                    print(f"  [{name}] {done:,d}/{total:,d} ({done/total*100:.1f}%) | Speed: {rate:.1f} trk/s")

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        np.save(feat_path, X_arr)
        np.save(lab_path, y_arr)
        print(f"  [{name}] Extraction complete in {(time.perf_counter()-t0)/60:.1f} min! Matrix: {X_arr.shape}")
        return X_arr, y_arr

    X_train, y_train = extract_or_load(df_train, fused_train_path, labels_train_path, "TRAIN")
    X_val, y_val = extract_or_load(df_val, fused_val_path, labels_val_path, "VAL")

    print(f"\nFeature matrices ready:")
    print(f"  Train X: {X_train.shape}, y: {y_train.shape}")
    print(f"  Val   X: {X_val.shape}, y: {y_val.shape}")

    # Move to GPU
    X_tr_gpu = torch.from_numpy(X_train).to(device)
    y_tr_gpu = torch.from_numpy(y_train).to(device)
    X_va_gpu = torch.from_numpy(X_val).to(device)
    y_va_gpu = torch.from_numpy(y_val).to(device)

    # 1. Train SupCon Projection Head on 28k subset
    print("\n[Step 1/2] Training SupCon Projection Head (704 -> 128)...")
    supcon_head = SupConProjectionHead(in_dim=704, out_dim=128).to(device)
    optimizer_sup = torch.optim.AdamW(supcon_head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    batch_sz = 1024
    n_tr = len(X_tr_gpu)

    for epoch in range(1, 21):
        supcon_head.train()
        perm = torch.randperm(n_tr, device=device)
        total_loss, n_batches = 0.0, 0
        for b in range(0, n_tr, batch_sz):
            idx = perm[b:b + batch_sz]
            bx, by = X_tr_gpu[idx], y_tr_gpu[idx]
            optimizer_sup.zero_grad(set_to_none=True)
            z = supcon_head(bx)
            loss = supcon_loss(z, by, temperature=config.SUPCON_TEMPERATURE)
            if not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(supcon_head.parameters(), max_norm=1.0)
                optimizer_sup.step()
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "proj_head_state": supcon_head.state_dict(),
                "in_dim": 704,
                "out_dim": 128,
                "loss": best_loss,
                "epoch": epoch
            }, out_ckpt_dir / "supcon_28k_best.pt")
        if epoch % 5 == 0 or epoch == 20:
            print(f"  Epoch {epoch:2d}/20 | SupCon Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")

    # 2. Train Calibrated FusionMLP on projected features
    print("\n[Step 2/2] Training Calibrated FusionMLP + Temperature Scaler...")
    supcon_head.eval()
    with torch.no_grad():
        Z_train = supcon_head(X_tr_gpu)
        Z_val = supcon_head(X_va_gpu)

    fusion_mlp = FusionMLP(in_dim=128).to(device)
    optimizer_mlp = torch.optim.AdamW(fusion_mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    best_val_auc = 0.0

    for epoch in range(1, 26):
        fusion_mlp.train()
        perm = torch.randperm(n_tr, device=device)
        for b in range(0, n_tr, batch_sz):
            idx = perm[b:b + batch_sz]
            bz, by = Z_train[idx], y_tr_gpu[idx]
            optimizer_mlp.zero_grad(set_to_none=True)
            logits = fusion_mlp(bz).squeeze(-1)
            loss = label_smoothing_bce(logits, by, smoothing=0.05)
            loss.backward()
            optimizer_mlp.step()

        fusion_mlp.eval()
        with torch.no_grad():
            val_logits = fusion_mlp(Z_val).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            y_val_np = y_val
            val_auc = roc_auc_score(y_val_np, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            scaler = TemperatureScaler().to(device)
            optimizer_scale = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)
            frozen_logits = val_logits.detach()

            def closure():
                optimizer_scale.zero_grad()
                loss_cal = nn.functional.binary_cross_entropy_with_logits(scaler(frozen_logits), y_va_gpu)
                loss_cal.backward()
                return loss_cal

            try:
                optimizer_scale.step(closure)
            except Exception:
                pass

            torch.save({
                "model_state": fusion_mlp.state_dict(),
                "scaler_state": scaler.state_dict(),
                "in_dim": 128,
                "val_auc": best_val_auc,
                "temperature": float(scaler.temperature.item()),
                "feature_cols": scorer.feature_cols
            }, out_ckpt_dir / "fusion_28k_best.pt")

        if epoch % 5 == 0 or epoch == 25:
            print(f"  Epoch {epoch:2d}/25 | Val AUROC: {val_auc:.4f} (Best: {best_val_auc:.4f})")

    print("\n" + "=" * 80)
    print(f"  TRAINING COMPLETE: 28k Model Checkpoints Saved to:")
    print(f"    -> {out_ckpt_dir / 'supcon_28k_best.pt'}")
    print(f"    -> {out_ckpt_dir / 'fusion_28k_best.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
