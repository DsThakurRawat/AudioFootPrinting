#!/usr/bin/env python3
"""
Master Execution Pipeline: Automated Training, Evaluation, and Comparison Suite.

Executes all 5 stages in sequence:
  Stage 1: Train AudioFootPrint on 28k Dataset (22,583 Train Split)
  Stage 2: Score AudioFootPrint on 2,804 Held-Out Test Split
  Stage 3: Score mirex2predet Comparator Models on 6,500 AIME Benchmark
  Stage 4: Generate Publication Graphical Suite (ROC/PR, Radar, KDE, ECE)
  Stage 5: Generate Unified Head-to-Head Comparison Report
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_stage(title: str, cmd: list[str], cwd: Path):
    print("\n" + "=" * 80)
    print(f"  >>> STARTING STAGE: {title}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working Dir: {cwd}")
    print("=" * 80 + "\n", flush=True)

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(cwd))
    elapsed_min = (time.perf_counter() - t0) / 60.0

    if res.returncode != 0:
        print(f"\n[ERROR] Stage '{title}' failed with exit code {res.returncode}!", flush=True)
        sys.exit(res.returncode)

    print(f"\n[DONE] Stage '{title}' completed successfully in {elapsed_min:.2f} minutes.\n", flush=True)


def main():
    t_global_start = time.perf_counter()
    print("=" * 85)
    print("      MASTER EXECUTION PIPELINE: TRAINING, EVALUATION & GRAPHICAL SUITE")
    print("=" * 85, flush=True)

    # Stage 1: Train AudioFootPrint on 28k dataset
    run_stage(
        "1. Train AudioFootPrint on 28k Split (22,583 Tracks)",
        [PY, str(ROOT / "scripts" / "train_audiofootprint_on_28k.py")],
        cwd=ROOT
    )

    # Stage 2: Evaluate AudioFootPrint on the 2,804 Test Split
    run_stage(
        "2. Evaluate AudioFootPrint on 2,804 Test Split",
        [PY, str(ROOT / "scripts" / "eval_on_mirex2predet_test_split.py")],
        cwd=ROOT
    )

    # Stage 3: Evaluate mirex2predet 28k Models on AIME (6,500 Tracks)
    run_stage(
        "3. Evaluate mirex2predet Models on 6,500 AIME Benchmark",
        [PY, str(ROOT / "scripts" / "eval_mirex_models_on_aime.py")],
        cwd=ROOT
    )

    # Stage 4: Generate Publication Graphical Suite
    run_stage(
        "4. Generate Graphical Figures (ROC, Radar, Violin, ECE)",
        [PY, str(ROOT / "scripts" / "generate_graphical_comparison.py")],
        cwd=ROOT
    )

    # Stage 5: Generate Master Comparison Report
    run_stage(
        "5. Compile Master Head-to-Head Comparison Report",
        [PY, str(ROOT / "results" / "generate_master_comparison.py")],
        cwd=ROOT
    )

    total_elapsed = (time.perf_counter() - t_global_start) / 60.0
    print("\n" + "=" * 85)
    print(f"      ALL PIPELINES FULLY EXECUTED & COMPLETED IN {total_elapsed:.2f} MINUTES!")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
