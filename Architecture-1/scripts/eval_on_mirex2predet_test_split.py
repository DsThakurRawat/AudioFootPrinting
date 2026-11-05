#!/usr/bin/env python3
"""
Evaluate AudioFootPrint (MusicScope-CL) on the exact 2,804 test set used by mirex2predet.

Features:
- Multi-process (12 workers) 704-D feature extraction with persistent caching to cache_28k_training/
- High-speed scoring using both:
    1) 28k-trained model (supcon_28k_best.pt + fusion_28k_best.pt)
    2) 200k-trained production model (supcon_200k_best.pt + fusion_200k_best.pt)
- Merges pre-computed comparator scores from mirex2predet zoo:
    - Fourier Baseline
    - ArtifactNet
    - Spectttra
    - MusicDet
- Computes comprehensive AUROC, EER, Accuracy, and Per-Family breakdown
- Outputs results to Architecture-1/results/audiofootprint_scores_on_2804_test.csv
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ARCH1_DIR = _SCRIPTS_DIR.parent
_MODEL_DIR = _ARCH1_DIR / "model"
for p in (str(_SCRIPTS_DIR), str(_MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from fusion_model import FusionMLP, TemperatureScaler
from new_inference import FastMusicScopeScorer, _extract_task
from train_audiofootprint_on_28k import get_28k_train_val_split
from train_supcon import SupConProjectionHead


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def load_model_pipeline(supcon_path: Path, fusion_path: Path, device: torch.device):
    supcon = SupConProjectionHead(in_dim=704, out_dim=128).to(device)
    sup_ckpt = torch.load(supcon_path, map_location=device, weights_only=False)
    supcon.load_state_dict(sup_ckpt["proj_head_state"])
    supcon.eval()

    fus_ckpt = torch.load(fusion_path, map_location=device, weights_only=False)
    fusion = FusionMLP(in_dim=fus_ckpt.get("in_dim", 128)).to(device)
    fusion.load_state_dict(fus_ckpt["model_state"])
    fusion.eval()

    scaler = TemperatureScaler().to(device)
    if "scaler_state" in fus_ckpt:
        scaler.load_state_dict(fus_ckpt["scaler_state"])
    scaler.eval()

    return supcon, fusion, scaler


def main():
    parser = argparse.ArgumentParser(description="Evaluate AudioFootPrint on the 2,804 test split of mirex2predet")
    parser.add_argument("--output_csv", type=str,
                        default=str(_ARCH1_DIR / "results" / "audiofootprint_scores_on_2804_test.csv"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  AUDIOFOOTPRINT: BENCHMARKING ON 2,804 HELD-OUT TEST SPLIT (28k BENCHMARK)")
    print("=" * 85)

    _, _, df_test = get_28k_train_val_split()
    total = len(df_test)
    print(f"Loaded {total:,d} test tracks (bucket < 10 matching mirex2predet).\n")

    # Cache directory for features
    cache_dir = config.PROCESSED_DATA_DIR / "cache_28k_training"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fused_test_path = cache_dir / "fused_test_2804.npy"
    labels_test_path = cache_dir / "labels_test_2804.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = FastMusicScopeScorer(mode="balanced")

    if fused_test_path.exists() and labels_test_path.exists():
        print(f"[TEST] Loading cached 704-D test features from {fused_test_path.name}...")
        X_test = np.load(fused_test_path)
        y_test = np.load(labels_test_path)
    else:
        print(f"[TEST] Multi-process extracting 704-D features for {total:,d} tracks across {args.workers} workers...")
        paths = df_test["resolved_path"].tolist()
        labels_map = dict(zip(df_test["resolved_path"], df_test["is_ai"]))

        X_list = []
        y_list = []
        t0 = time.perf_counter()

        with mp.Pool(processes=args.workers) as pool:
            for chunk_start in range(0, total, args.batch_size):
                chunk_paths = paths[chunk_start:chunk_start + args.batch_size]
                batch_data = list(pool.imap(_extract_task, chunk_paths, chunksize=8))

                waves = []
                narrs = []
                valid_labels = []

                for p, y_int16, feats, err in batch_data:
                    if err:
                        # Fallback for decoding failure: silence
                        y_int16 = np.zeros(int(config.SAMPLE_RATE * config.CHUNK_SECONDS), dtype=np.int16)
                        feats = {}
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

                done = min(chunk_start + args.batch_size, total)
                elapsed = time.perf_counter() - t0
                speed = done / max(elapsed, 0.001)
                print(f"  [TEST] {done:,d}/{total:,d} ({done/total*100:5.1f}%) | Speed: {speed:4.1f} trk/s | Elapsed: {elapsed/60:4.1f} min")

        X_test = np.array(X_list, dtype=np.float32)
        y_test = np.array(y_list, dtype=np.float32)
        np.save(fused_test_path, X_test)
        np.save(labels_test_path, y_test)
        print(f"[TEST] Feature extraction complete in {(time.perf_counter()-t0)/60:.2f} min! Matrix: {X_test.shape}\n")

    # Score with 28k-trained model
    ckpt_28k_dir = config.CHECKPOINT_DIR / "28k_trained"
    supcon_28k, fusion_28k, scaler_28k = load_model_pipeline(
        ckpt_28k_dir / "supcon_28k_best.pt",
        ckpt_28k_dir / "fusion_28k_best.pt",
        device
    )

    # Score with 200k-trained production model
    ckpt_200k_dir = config.CHECKPOINT_DIR
    supcon_200k, fusion_200k, scaler_200k = load_model_pipeline(
        ckpt_200k_dir / "supcon_200k_best.pt",
        ckpt_200k_dir / "fusion_200k_best.pt",
        device
    )

    print("[Scoring] Executing neural forward passes for 28k and 200k models...")
    X_test_gpu = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        # 28k
        z_28k = supcon_28k(X_test_gpu)
        l_28k = fusion_28k(z_28k).squeeze(-1)
        probs_28k = torch.sigmoid(scaler_28k(l_28k)).cpu().numpy()

        # 200k
        z_200k = supcon_200k(X_test_gpu)
        l_200k = fusion_200k(z_200k).squeeze(-1)
        probs_200k = torch.sigmoid(scaler_200k(l_200k)).cpu().numpy()

    # Load baseline scores from mirex2predet zoo
    zoo_dir = Path(r"D:\mirex2predet\mirex_detectors\checkpoints\zoo")
    zoo_scores = {}
    for m in ["fourier", "artifactnet", "spectttra", "musicdet"]:
        js = zoo_dir / m / "full" / "test_scores.jsonl"
        if js.exists():
            mapping = {}
            with open(js, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        p_norm = os.path.normpath(r["path"]).lower()
                        mapping[p_norm] = float(r.get("score", 0.0))
            zoo_scores[m] = mapping

    # Assemble comprehensive results DataFrame
    records = []
    for idx, (_, row) in enumerate(df_test.iterrows()):
        p_res = row["resolved_path"]
        p_norm = os.path.normpath(p_res).lower()
        p_orig_norm = os.path.normpath(str(row["file_path"])).lower()

        # baseline logits
        rec = {
            "path": p_res,
            "label": int(row["is_ai"]),
            "family": str(row.get("generator_family", row.get("family", "real" if int(row["is_ai"]) == 0 else "unknown"))),
            "audiofootprint_score": float(probs_28k[idx]),
            "audiofootprint_28k_score": float(probs_28k[idx]),
            "audiofootprint_200k_score": float(probs_200k[idx]),
        }
        for m in ["fourier", "artifactnet", "spectttra", "musicdet"]:
            m_dict = zoo_scores.get(m, {})
            # Look up by resolved or original path
            sc = m_dict.get(p_norm, m_dict.get(p_orig_norm, np.nan))
            rec[f"{m}_baseline_score"] = sc
            # sigmoid probability for comparison
            rec[f"{m}_prob"] = 1.0 / (1.0 + np.exp(-sc)) if not np.isnan(sc) else np.nan

        records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved comprehensive test scores ({len(df_out):,d} tracks) to: {out_csv}")

    # Metrics computation
    y_true = df_out["label"].values
    p_28k = df_out["audiofootprint_28k_score"].values
    p_200k = df_out["audiofootprint_200k_score"].values

    auc_28k = roc_auc_score(y_true, p_28k)
    eer_28k = compute_eer(y_true, p_28k)
    acc_28k = accuracy_score(y_true, (p_28k >= scorer.threshold).astype(int))

    auc_200k = roc_auc_score(y_true, p_200k)
    eer_200k = compute_eer(y_true, p_200k)
    acc_200k = accuracy_score(y_true, (p_200k >= scorer.threshold).astype(int))

    print("\n" + "=" * 85)
    print("  AUDIOFOOTPRINT TEST EVALUATION SUMMARY (2,804 TRACKS)")
    print("=" * 85)
    print(f"  AudioFootPrint (28k Retrained)  : AUROC = {auc_28k:.4f} ({auc_28k*100:.2f}%) | EER = {eer_28k:.4f} | Acc = {acc_28k*100:.2f}%")
    print(f"  AudioFootPrint (200k Production): AUROC = {auc_200k:.4f} ({auc_200k*100:.2f}%) | EER = {eer_200k:.4f} | Acc = {acc_200k*100:.2f}%")

    for m in ["fourier", "artifactnet", "spectttra", "musicdet"]:
        col = f"{m}_baseline_score"
        valid = df_out.dropna(subset=[col])
        if len(valid) > 0:
            m_auc = roc_auc_score(valid["label"], valid[col])
            m_eer = compute_eer(valid["label"], valid[col])
            print(f"  {m.capitalize():<32}: AUROC = {m_auc:.4f} ({m_auc*100:.2f}%) | EER = {m_eer:.4f}")

    print("-" * 85)
    print(f"  Per-Family Detection Rate (28k Retrained | Threshold = {scorer.threshold:.4f}):")
    for fam in sorted(df_out["family"].unique()):
        sub = df_out[df_out["family"] == fam]
        if fam in ("real", "human"):
            rec = (sub["audiofootprint_28k_score"] < scorer.threshold).mean()
            print(f"    {fam:<16}: {rec*100:6.2f}% Human Accuracy ({len(sub):,d} tracks)")
        else:
            rec = (sub["audiofootprint_28k_score"] >= scorer.threshold).mean()
            print(f"    {fam:<16}: {rec*100:6.2f}% AI Caught     ({len(sub):,d} tracks)")
    print("=" * 85)


if __name__ == "__main__":
    main()
