#!/usr/bin/env python3
"""
Dedicated Evaluator: Evaluate mirex2predet 28k Models on the AIME Benchmark (6,500 Tracks).

This script leaves evaluate_aime_full.py completely untouched.
It loads the 28k-trained checkpoints from mirex2predet:
  - Fourier Baseline (`checkpoints/zoo/fourier/full/best.ckpt`)
  - ArtifactNet (`checkpoints/zoo/artifactnet/full/best.ckpt`)
  - Spectttra (`checkpoints/zoo/spectttra/full/best.ckpt`)

Streams AIME shards and outputs predictions to:
  Architecture-1/results/aime_mirex2predet_results.csv
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import soxr
import torch
from sklearn.metrics import roc_auc_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ARCH1_DIR = _SCRIPTS_DIR.parent
_MIREX_DIR = Path(r"D:\mirex2predet\mirex_detectors")

for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detectors import build as build_mirex_detector


def load_mirex_model(name: str, device: str):
    cp = _MIREX_DIR / "checkpoints" / "zoo" / name / "full" / "best.ckpt"
    if not cp.exists():
        print(f"  [Warning] Checkpoint for {name} not found at {cp}")
        return None
    det = build_mirex_detector(name)
    sd = torch.load(cp, map_location=device)
    det.load_state_dict(sd.get("model", sd), strict=False)
    det.to(device).eval()
    return det


def evaluate_mirex_aime(
    data_dir: Path = Path(r"D:\AudioFootPrint\data\raw\aime\data"),
    output_csv: Path = _ARCH1_DIR / "results" / "aime_mirex2predet_results.csv",
    models: list[str] | None = None,
    max_shards: int | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_run = models or ["fourier", "artifactnet", "spectttra"]

    print("=" * 85)
    print("  MIREX2PREDET 28k MODELS: EVALUATING ON UNSEEN AIME BENCHMARK (6,500 TRACKS)")
    print(f"  Device: {device} | Models: {models_to_run}")
    print("=" * 85)

    if not data_dir.exists():
        raise FileNotFoundError(f"AIME data directory not found at: {data_dir}")

    all_shards = sorted(data_dir.glob("*.parquet"))
    if max_shards:
        all_shards = all_shards[:max_shards]

    print(f"Total AIME shards selected: {len(all_shards)}\n")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        try:
            df_existing = pd.read_csv(output_csv)
            if len(df_existing) >= 6500:
                print(f"  [Found Cached Results] {output_csv} already contains {len(df_existing):,d} scored tracks!")
                print("=" * 85)
                print("  MIREX2PREDET 28k AIME SCORECARD SUMMARY")
                print("=" * 85)
                y_true = df_existing["is_ai"].values
                for name in models_to_run:
                    if f"{name}_score" in df_existing.columns:
                        sc = df_existing[f"{name}_score"].values
                        auc_val = roc_auc_score(y_true, sc)
                        ai_caught = (df_existing[df_existing["is_ai"] == 1][f"{name}_pred"] == "AI").mean() * 100
                        human_acc = (df_existing[df_existing["is_ai"] == 0][f"{name}_pred"] == "Human").mean() * 100
                        print(f"  {name.upper():<14}: AUROC = {auc_val:.4f} ({auc_val*100:.2f}%) | AI Caught: {ai_caught:.1f}% | Human Acc: {human_acc:.1f}%")
                print("=" * 85)
                return
        except Exception as e:
            print(f"  [Notice] Could not load cached file: {e}. Proceeding with fresh evaluation.")

    loaded = {}
    for name in models_to_run:
        m = load_mirex_model(name, device)
        if m:
            loaded[name] = m
            print(f"  [Loaded] {name:<14} from checkpoints/zoo/{name}/full/best.ckpt")

    if not loaded:
        sys.exit("No mirex models could be loaded.")

    results = []
    t_start = time.perf_counter()
    total_tracks = 0

    for s_idx, shard_path in enumerate(all_shards, 1):
        t_shard = pq.read_table(str(shard_path))
        df = t_shard.to_pandas()

        for _, row in df.iterrows():
            tid = str(row["id"])
            model_name = str(row["model"])
            desc = str(row["description"])
            audio_bytes = row["audio"]["bytes"]
            is_human = ("Human" in model_name or "Jamendo" in model_name)
            is_ai = 0 if is_human else 1

            try:
                with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                    orig_sr = f.samplerate
                    data = f.read(dtype="float32", always_2d=False)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
            except Exception:
                continue

            record = {
                "id": tid,
                "model_family": model_name,
                "is_human": is_human,
                "ground_truth": "Human" if is_human else "AI",
                "is_ai": is_ai,
                "prompt": desc,
            }

            with torch.inference_mode():
                for name, det in loaded.items():
                    target_sr = det.sr
                    target_len = int(target_sr * det.seg_seconds)
                    res = soxr.resample(data, orig_sr, target_sr) if orig_sr != target_sr else data
                    if len(res) < target_len:
                        res = np.pad(res, (0, target_len - len(res)))
                    else:
                        start = (len(res) - target_len) // 2
                        res = res[start:start + target_len]

                    x_t = torch.from_numpy(res).unsqueeze(0).to(device)
                    score = float(det.score(x_t).cpu().item())
                    record[f"{name}_score"] = score
                    record[f"{name}_pred"] = "AI" if score >= 0.0 else "Human"

            results.append(record)
            total_tracks += 1

        if s_idx % 10 == 0 or s_idx == len(all_shards):
            elapsed_min = (time.perf_counter() - t_start) / 60
            speed = total_tracks / max((time.perf_counter() - t_start), 0.001)
            print(f"  Shard [{s_idx:3d}/{len(all_shards):3d}] ({s_idx/len(all_shards)*100:5.1f}%) | "
                  f"Tracks: {total_tracks:5,d} | Speed: {speed:4.1f} trk/s | Elapsed: {elapsed_min:4.1f} min", flush=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print("\n" + "=" * 85)
    print(f"      EVALUATION COMPLETE: {len(df_out):,d} TRACKS SCORED IN {(time.perf_counter() - t_start)/60:.2f} MINUTES")
    print(f"      Predictions saved to -> {output_csv}")
    print("=" * 85)

    # Scorecard
    print("\n" + "=" * 85)
    print("  MIREX2PREDET 28k AIME SCORECARD SUMMARY")
    print("=" * 85)
    y_true = df_out["is_ai"].values

    for name in loaded:
        sc = df_out[f"{name}_score"].values
        auc_val = roc_auc_score(y_true, sc)
        ai_caught = (df_out[df_out["is_ai"] == 1][f"{name}_pred"] == "AI").mean() * 100
        human_acc = (df_out[df_out["is_ai"] == 0][f"{name}_pred"] == "Human").mean() * 100
        print(f"  {name.upper():<14}: AUROC = {auc_val:.4f} ({auc_val*100:.2f}%) | AI Caught: {ai_caught:.1f}% | Human Acc: {human_acc:.1f}%")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mirex2predet models on AIME benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\AudioFootPrint\data\raw\aime\data"))
    parser.add_argument("--output-csv", type=Path, default=_ARCH1_DIR / "results" / "aime_mirex2predet_results.csv")
    parser.add_argument("--models", nargs="+", default=["fourier", "artifactnet", "spectttra"])
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    evaluate_mirex_aime(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        models=args.models,
        max_shards=args.max_shards
    )
