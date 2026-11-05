#!/usr/bin/env python3
"""
Comprehensive Evaluation of MusicScope-CL on the Unseen AIME Benchmark (6,500 Tracks)
Architecture-1 (Dual-Branch Contrastive Acoustic & Narrative Manifold)

Evaluates all 12 AI generative models + MTG-Jamendo human control:
  1. Udio
  2. Suno v3
  3. Suno v3.5
  4. MusicGen Small
  5. MusicGen Medium
  6. MusicGen Large
  7. AudioLDM 2 Large
  8. AudioLDM 2 Music
  9. Mustango
  10. Riffusion
  11. Stable Audio v1
  12. Stable Audio v2
  13. MTG-Jamendo (Human)
"""

import argparse
import csv
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import soxr
import torch

# Ensure model directory is in sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ARCH1_DIR = _SCRIPTS_DIR.parent
_MODEL_DIR = _ARCH1_DIR / "model"
for p in (str(_SCRIPTS_DIR), str(_MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer, fast_extract_narrative


def _worker_process_file(args):
    """Worker function: reads audio from temporary file, extracts 128-D narrative features,
    and returns 30s int16 PCM for GPU spectrogram computation."""
    track_id, model, desc, file_path = args
    try:
        with sf.SoundFile(file_path) as f:
            orig_sr = f.samplerate
            data = f.read(dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1)

        if orig_sr != config.SAMPLE_RATE:
            y_22k = soxr.resample(data, orig_sr, config.SAMPLE_RATE)
        else:
            y_22k = data

        feats = fast_extract_narrative(y_22k, sr=config.SAMPLE_RATE)

        target_len = int(config.SAMPLE_RATE * config.CHUNK_SECONDS)
        if len(y_22k) < target_len:
            y_pad = np.pad(y_22k, (0, target_len - len(y_22k)))
        else:
            y_pad = y_22k[:target_len]

        y_int16 = (np.clip(y_pad, -1.0, 1.0) * 32767.0).astype(np.int16)
        return track_id, model, desc, y_int16, feats, None
    except Exception as e:
        zero_buf = np.zeros(int(config.SAMPLE_RATE * config.CHUNK_SECONDS), dtype=np.int16)
        return track_id, model, desc, zero_buf, {}, str(e)


def evaluate_aime(
    data_dir: Path = Path(r"D:\AudioFootPrint\data\raw\aime\data"),
    output_csv: Path = _ARCH1_DIR / "results" / "aime_full_evaluation_results.csv",
    batch_size: int = 64,
    workers: int = 12,
    max_shards: int | None = None,
    shard_step: int = 1,
):
    print("=" * 85)
    print("      MIREX 2026: UNSEEN AIME BENCHMARK FULL EVALUATION (6,500 TRACKS)")
    print("=" * 85)

    if not data_dir.exists():
        raise FileNotFoundError(f"AIME data directory not found at: {data_dir}")

    all_shards = sorted(data_dir.glob("*.parquet"))
    if shard_step > 1:
        shards = all_shards[::shard_step]
    else:
        shards = all_shards

    if max_shards:
        shards = shards[:max_shards]

    print(f"Found {len(all_shards)} total AIME shards. Selected {len(shards)} shards for evaluation.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scorer = FastMusicScopeScorer()

    results = []
    t_start = time.perf_counter()
    total_tracks_processed = 0

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        with mp.Pool(processes=workers) as pool:
            for s_idx, shard_path in enumerate(shards, 1):
                t_shard = pq.read_table(str(shard_path))
                df = t_shard.to_pandas()

                # 1. Unpack shard audio into temp files for fast C-level reading
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

                # 2. Parallel CPU extraction
                extracted = list(pool.imap(_worker_process_file, tasks, chunksize=4))

                # Clean up temp audio files
                for tf in temp_files:
                    try:
                        tf.unlink()
                    except OSError:
                        pass

                # 3. GPU Forward Pass in batch
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
                pct = (s_idx / len(shards)) * 100
                elapsed_min = (time.perf_counter() - t_start) / 60
                speed = total_tracks_processed / max((time.perf_counter() - t_start), 0.001)

                if s_idx % 10 == 0 or s_idx == len(shards):
                    print(f"  Shard [{s_idx:3d}/{len(shards):3d}] ({pct:5.1f}%) | Tracks: {total_tracks_processed:5,d} | Speed: {speed:4.1f} trk/s | Elapsed: {elapsed_min:4.1f} min", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print("\n" + "=" * 85)
    print(f"      EVALUATION COMPLETE: {len(res_df):,d} TRACKS SCORED IN {(time.perf_counter() - t_start) / 60:.2f} MINUTES")
    print(f"      Average Throughput: {total_tracks_processed / (time.perf_counter() - t_start):.1f} tracks/second")
    print("=" * 85)
    print(f"Full per-track predictions saved to:\n  -> {output_csv}\n")

    # Generate summary scorecard
    summary_rows = []
    print("=" * 85)
    print(f"{'Model Family':<22} | {'Tracks':<6} | {'Mean Score':<10} | {'Median Score':<12} | {'Detection / Acc'}")
    print("-" * 85)

    for m, grp in res_df.groupby("model"):
        mean_s = grp["score"].mean()
        med_s = grp["score"].median()
        is_h = grp["is_human"].iloc[0]
        if is_h:
            acc = (grp["prediction"] == "Human").mean() * 100
            metric_str = f"{acc:5.1f}% Human Acc"
        else:
            det = (grp["prediction"] == "AI").mean() * 100
            metric_str = f"{det:5.1f}% Caught"
        print(f"{m:<22} | {len(grp):<6} | {mean_s:<10.4f} | {med_s:<12.4f} | {metric_str}")
        summary_rows.append({
            "model": m,
            "tracks": len(grp),
            "mean_score": mean_s,
            "median_score": med_s,
            "performance": metric_str,
        })

    print("=" * 85)

    ai_df = res_df[~res_df["is_human"]]
    human_df = res_df[res_df["is_human"]]
    if len(ai_df) > 0:
        overall_ai_det = (ai_df["prediction"] == "AI").mean() * 100
        print(f"OVERALL AI DETECTION RATE : {overall_ai_det:6.2f}% ({ai_df['prediction'].value_counts().get('AI', 0):,d}/{len(ai_df):,d})")
    if len(human_df) > 0:
        human_acc = (human_df["prediction"] == "Human").mean() * 100
        print(f"OVERALL HUMAN ACCURACY    : {human_acc:6.2f}% ({human_df['prediction'].value_counts().get('Human', 0):,d}/{len(human_df):,d})")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MusicScope-CL on Unseen AIME Benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\AudioFootPrint\data\raw\aime\data"), help="Path to AIME parquet data")
    parser.add_argument("--output-csv", type=Path, default=_ARCH1_DIR / "results" / "aime_full_evaluation_results.csv", help="Output results CSV")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU forward pass batch size")
    parser.add_argument("--workers", type=int, default=12, help="Number of CPU worker threads")
    parser.add_argument("--max-shards", type=int, default=None, help="Limit number of shards to evaluate (default: all)")
    parser.add_argument("--shard-step", type=int, default=1, help="Step between shards (e.g. 1 for all, 2 for half)")
    args = parser.parse_args()

    evaluate_aime(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        workers=args.workers,
        max_shards=args.max_shards,
        shard_step=args.shard_step,
    )
