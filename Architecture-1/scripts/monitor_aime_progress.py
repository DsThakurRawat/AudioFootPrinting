#!/usr/bin/env python3
"""
Live Progress Monitor for AIME 6,500-Track Benchmark Evaluation
Architecture-1 (MusicScope-CL)

Run this in your terminal:
    python Architecture-1/scripts/monitor_aime_progress.py
"""

import os
import re
import sys
import time
from pathlib import Path
import pandas as pd

LOG_PATH = Path(r"C:\Users\RF AND SIMULATION\.gemini\antigravity-ide\brain\cc8c118f-2a4c-43a2-8848-ede7c99ef9af\.system_generated\tasks\task-948.log")
CSV_PATH = Path(r"D:\AudioFootPrint\Architecture-1\results\aime_full_evaluation_results.csv")

SHARD_PATTERN = re.compile(
    r"Shard\s*\[\s*(\d+)/(\d+)\]\s*\(\s*([\d\.]+)%\)\s*\|\s*Tracks:\s*(\d+)\s*\|\s*Speed:\s*([\d\.]+)\s*trk/s\s*\|\s*Elapsed:\s*([\d\.]+)\s*min"
)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_progress_bar(pct: float, width: int = 40) -> str:
    filled = int(width * (pct / 100.0))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def main():
    print("Connecting to live AIME evaluation monitor...")
    time.sleep(1)

    while True:
        clear_screen()
        print("=" * 80)
        print("     LIVE MONITOR: MIREX 2026 UNSEEN AIME 6,500-TRACK EVALUATION")
        print("=" * 80)

        if not LOG_PATH.exists():
            print(f"Waiting for log file to appear at: {LOG_PATH}")
            time.sleep(2)
            continue

        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            lines = []

        latest_shard = 0
        total_shards = 210
        pct = 0.0
        tracks_done = 0
        speed = 0.0
        elapsed_min = 0.0
        is_completed = False

        for line in reversed(lines):
            if "EVALUATION COMPLETE" in line:
                is_completed = True
            m = SHARD_PATTERN.search(line)
            if m:
                latest_shard = int(m.group(1))
                total_shards = int(m.group(2))
                pct = float(m.group(3))
                tracks_done = int(m.group(4))
                speed = float(m.group(5))
                elapsed_min = float(m.group(6))
                break

        # Calculate ETA
        total_expected_tracks = 6500
        rem_tracks = max(0, total_expected_tracks - tracks_done)
        eta_min = (rem_tracks / max(speed, 0.01)) / 60.0 if not is_completed else 0.0

        print(f"\nOverall Progress : {render_progress_bar(pct)}")
        print(f"Shards Processed : {latest_shard:,d} / {total_shards:,d} shards")
        print(f"Tracks Scored    : {tracks_done:,d} / {total_expected_tracks:,d} tracks")
        print(f"Inference Speed  : {speed:.1f} tracks/second (12 CPU workers + GPU)")
        print(f"Elapsed Time     : {elapsed_min:.1f} minutes")
        print(f"Estimated ETA    : {eta_min:.1f} minutes remaining" if not is_completed else "Estimated ETA    : COMPLETED!")

        # Show live model score breakdown if CSV has been saved
        if CSV_PATH.exists():
            try:
                df = pd.read_csv(CSV_PATH)
                if len(df) > 0 and "model" in df.columns:
                    print("\n" + "-" * 80)
                    print(f"Live Scorecard (Checkpoint: {len(df):,d} tracks saved to CSV):")
                    print("-" * 80)
                    print(f"{'Model Family':<22} | {'Tracks':<6} | {'Mean P(AI)':<12} | {'Performance'}")
                    print("-" * 80)
                    for m_name, grp in df.groupby("model"):
                        mean_p = grp["score"].mean()
                        is_human = ("Human" in m_name or "Jamendo" in m_name)
                        if is_human:
                            acc = (grp["prediction"] == "Human").mean() * 100
                            perf = f"{acc:5.1f}% Human Acc"
                        else:
                            det = (grp["prediction"] == "AI").mean() * 100
                            perf = f"{det:5.1f}% Caught"
                        print(f"{m_name:<22} | {len(grp):<6} | {mean_p:<12.4f} | {perf}")
                    print("-" * 80)
            except Exception:
                pass

        print("\n" + "=" * 80)
        if is_completed:
            print("  >>> STATUS: [COMPLETE] All 6,500 tracks successfully evaluated!")
            print(f"  Results saved in: {CSV_PATH}")
            print("=" * 80)
            break
        else:
            print("  [Auto-refreshing every 3 seconds. Press Ctrl+C to stop monitor]")
            print("=" * 80)

        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


if __name__ == "__main__":
    main()
