#!/usr/bin/env python3
"""
Dedicated Evaluator: Evaluate AudioFootPrint 28k-Trained Weights on Unseen AIME Benchmark (6,500 Tracks).

Pure Apples-to-Apples Out-of-Domain Assessment:
Loads:
  - SupCon Head: `checkpoints/28k_trained/supcon_28k_best.pt`
  - FusionMLP:   `checkpoints/28k_trained/fusion_28k_best.pt`

Outputs:
  - Architecture-1/results/aime_audiofootprint_28k_results.csv
"""
import argparse
import io
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ARCH1_DIR = _SCRIPTS_DIR.parent
_MODEL_DIR = _ARCH1_DIR / "model"
for p in (str(_SCRIPTS_DIR), str(_MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer, fast_extract_narrative


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def _worker_process_file(args):
    track_id, model, desc, file_path = args
    try:
        with sf.SoundFile(file_path) as f:
            data = f.read(dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)

            target_len = int(config.SAMPLE_RATE * config.CHUNK_SECONDS)
            if len(data) < target_len:
                padded = np.pad(data, (0, target_len - len(data)))
            else:
                padded = data[:target_len]

            y_int16 = (padded * 32767.0).astype(np.int16)
            feats = fast_extract_narrative(data, config.SAMPLE_RATE)
            return track_id, model, desc, y_int16, feats, None
    except Exception as e:
        return track_id, model, desc, None, None, str(e)


def evaluate_aime_28k(
    data_dir: Path = Path(r"D:\AudioFootPrint\data\raw\aime\data"),
    output_csv: Path = _ARCH1_DIR / "results" / "aime_audiofootprint_28k_results.csv",
    workers: int = 12,
):
    print("=" * 85)
    print("  AUDIOFOOTPRINT (28k WEIGHTS): BENCHMARKING ON 6,500 UNSEEN AIME TRACKS")
    print("=" * 85)

    if not data_dir.exists():
        raise FileNotFoundError(f"AIME data directory not found at: {data_dir}")

    all_shards = sorted(data_dir.glob("*.parquet"))
    print(f"Total AIME shards: {len(all_shards)} (6,500 tracks)")

    ckpt_28k_dir = config.CHECKPOINT_DIR / "28k_trained"
    supcon_ckpt = str(ckpt_28k_dir / "supcon_28k_best.pt")
    fusion_ckpt = str(ckpt_28k_dir / "fusion_28k_best.pt")

    scorer = FastMusicScopeScorer(
        supcon_ckpt=supcon_ckpt,
        fusion_ckpt=fusion_ckpt,
        mode="balanced"
    )

    results = []
    t_start = time.perf_counter()
    total_tracks_processed = 0

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        with mp.Pool(processes=workers) as pool:
            for s_idx, shard_path in enumerate(all_shards, 1):
                t_shard = pq.read_table(str(shard_path))
                df = t_shard.to_pandas()

                tasks = []
                temp_files = []
                for _, row in df.iterrows():
                    tid = str(row["id"])
                    model_name = str(row["model"])
                    desc = str(row["description"])
                    audio_bytes = row["audio"]["bytes"]

                    tmp_file = temp_dir / f"{tid}.wav"
                    with open(tmp_file, "wb") as f:
                        f.write(audio_bytes)

                    tasks.append((tid, model_name, desc, str(tmp_file)))
                    temp_files.append(tmp_file)

                extracted = list(pool.imap(_worker_process_file, tasks, chunksize=4))

                for tf in temp_files:
                    try:
                        tf.unlink()
                    except OSError:
                        pass

                waves = []
                narrs = []
                meta = []

                for tid, model_name, desc, y_int16, feats, err in extracted:
                    if err:
                        continue
                    waves.append(y_int16.astype(np.float32) / 32767.0)
                    narrs.append(scorer._prepare_narr_vector(feats))
                    is_human = ("Human" in model_name or "Jamendo" in model_name)
                    meta.append({
                        "id": tid,
                        "model": model_name,
                        "is_human": is_human,
                        "ground_truth": "Human" if is_human else "AI",
                        "prompt": desc,
                    })

                if waves:
                    specs_t = scorer._compute_specs_on_gpu(np.array(waves, dtype=np.float32))
                    narr_t = torch.from_numpy(np.array(narrs, dtype=np.float32)).to(scorer.device, non_blocking=True)

                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                        probs = scorer.unified_model(specs_t, narr_t).cpu().numpy()

                    for m, p in zip(meta, probs):
                        prob_val = float(p)
                        pred = "AI" if prob_val >= scorer.threshold else "Human"
                        correct = (pred == m["ground_truth"])
                        m["score"] = prob_val
                        m["prediction"] = pred
                        m["correct"] = correct
                        results.append(m)

                total_tracks_processed += len(meta)
                pct = (s_idx / len(all_shards)) * 100
                elapsed_min = (time.perf_counter() - t_start) / 60
                speed = total_tracks_processed / max((time.perf_counter() - t_start), 0.001)

                if s_idx % 20 == 0 or s_idx == len(all_shards):
                    print(f"  Shard [{s_idx:3d}/{len(all_shards):3d}] ({pct:5.1f}%) | Tracks: {total_tracks_processed:5,d} | Speed: {speed:4.1f} trk/s | Elapsed: {elapsed_min:4.1f} min", flush=True)

    df_out = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print("\n" + "=" * 85)
    print(f"      EVALUATION COMPLETE: {len(df_out):,d} TRACKS SCORED IN {(time.perf_counter() - t_start) / 60:.2f} MINUTES")
    print(f"      Predictions saved to -> {output_csv}")
    print("=" * 85)

    # Scorecard computation
    y_true = (df_out["ground_truth"] == "AI").astype(int).values
    scores = df_out["score"].values
    auc_val = roc_auc_score(y_true, scores)
    eer_val = compute_eer(y_true, scores)
    ai_caught = (df_out[df_out["ground_truth"] == "AI"]["prediction"] == "AI").mean() * 100
    human_acc = (df_out[df_out["ground_truth"] == "Human"]["prediction"] == "Human").mean() * 100

    print("\n" + "=" * 85)
    print("  AUDIOFOOTPRINT 28k AIME SCORECARD SUMMARY")
    print("=" * 85)
    print(f"  AUROC           : {auc_val:.4f} ({auc_val*100:.2f}%)")
    print(f"  Equal Error Rate: {eer_val:.4f} ({eer_val*100:.2f}%)")
    print(f"  AI Caught (τ=0.18): {ai_caught:.2f}%")
    print(f"  Human Control Acc : {human_acc:.2f}%")
    print("-" * 85)
    print("  Per-Generator Breakdown:")
    for mod, grp in df_out.groupby("model"):
        if grp["is_human"].iloc[0]:
            acc = (grp["prediction"] == "Human").mean() * 100
            print(f"    {mod:<20}: {acc:6.2f}% Human Accuracy ({len(grp)} tracks)")
        else:
            rec = (grp["prediction"] == "AI").mean() * 100
            print(f"    {mod:<20}: {rec:6.2f}% AI Caught     ({len(grp)} tracks)")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 28k AudioFootPrint on AIME")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    evaluate_aime_28k(workers=args.workers)
