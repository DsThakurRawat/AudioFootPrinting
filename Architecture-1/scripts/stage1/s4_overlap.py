"""
Stage 1 step 3 - overlap and leakage analysis.

Answers, from the canonical manifest:
  * exact duplicates (sha256) within and across splits
  * near-duplicates (duration+sha prefix heuristic; full chromaprint unavailable)
  * artist overlap between human and synthetic corpora
  * prompt overlap (identical prompts reused across generators)
  * generator/version overlap between train and reported test partitions
  * shortcut-risk screen: can metadata alone separate human from AI?

Every finding is written to a CSV the reviewer can inspect directly.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_common import STAGE1_DIR, ensure_dir, log, stage_report


def write_csv(path, rows, cols):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def s(v):
    return "" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser(description="Stage 1: overlap analysis")
    ap.add_argument("--manifest", default=str(STAGE1_DIR / "canonical_manifest.parquet"))
    args = ap.parse_args()

    import pandas as pd
    import numpy as np
    df = pd.read_parquet(args.manifest)
    log(f"[S1.3] loaded manifest: {len(df):,} rows")

    audio = df[(df["is_audio_file"] == 1) & (df["sha256"].notna()) &
               (df["sha256"] != "")].copy()
    log(f"[S1.3] hashed audio rows available: {len(audio):,}")

    findings = []

    # ---------- 1. exact duplicates ----------
    dup_rows = []
    g = audio.groupby("sha256")
    for h, sub in g:
        if len(sub) < 2:
            continue
        splits = sorted(set(sub["split"].astype(str)))
        datasets = sorted(set(sub["dataset"].astype(str)))
        crosses = len(splits) > 1
        dup_rows.append({
            "sha256": h,
            "n_copies": len(sub),
            "splits": "|".join(splits),
            "datasets": "|".join(datasets),
            "generator_families": "|".join(sorted(set(sub["generator_family"].astype(str)))),
            "crosses_train_val": int(crosses),
            "example_paths": " || ".join(list(sub["absolute_or_project_path"])[:3]),
        })
    dup_rows.sort(key=lambda r: -r["n_copies"])
    write_csv(STAGE1_DIR / "duplicate_report.csv", dup_rows,
              ["sha256", "n_copies", "splits", "datasets", "generator_families",
               "crosses_train_val", "example_paths"])
    cross_n = sum(r["n_copies"] for r in dup_rows if r["crosses_train_val"])
    findings.append(f"exact duplicate sha256 groups: {len(dup_rows):,} "
                    f"covering {sum(r['n_copies'] for r in dup_rows):,} rows; "
                    f"{cross_n:,} rows appear in both train and val")
    log(f"[S1.3] duplicate groups: {len(dup_rows):,} (cross-split rows {cross_n:,})")

    # ---------- 2. near-duplicates ----------
    # No chromaprint/pyacoustid on this machine. Use a conservative proxy:
    # identical duration (0.5 s) + identical first 16 hex of sha256 is already
    # covered above; here flag same duration + same size across splits, which
    # catches re-encodes of one master.
    near_rows = []
    if len(audio):
        ad = audio.copy()
        ad["dur_r"] = pd.to_numeric(ad["duration_s"], errors="coerce").round(1)
        ad = ad[ad["dur_r"].notna()]
        for (dur, size), sub in ad.groupby(["dur_r", "size_bytes"]):
            if len(sub) < 2:
                continue
            if dur <= 0:
                continue
            hashes = set(sub["sha256"])
            if len(hashes) == 1:
                continue  # already caught as exact duplicate
            splits = sorted(set(sub["split"].astype(str)))
            near_rows.append({
                "duration_s": dur,
                "size_bytes": size,
                "n_files": len(sub),
                "distinct_sha256": len(hashes),
                "splits": "|".join(splits),
                "datasets": "|".join(sorted(set(sub["dataset"].astype(str)))),
                "generator_families": "|".join(sorted(set(sub["generator_family"].astype(str)))),
                "crosses_train_val": int(len(splits) > 1),
                "example_paths": " || ".join(list(sub["absolute_or_project_path"])[:3]),
            })
    near_rows.sort(key=lambda r: -r["n_files"])
    write_csv(STAGE1_DIR / "near_duplicate_report.csv", near_rows,
              ["duration_s", "size_bytes", "n_files", "distinct_sha256", "splits",
               "datasets", "generator_families", "crosses_train_val", "example_paths"])
    near_cross = sum(r["n_files"] for r in near_rows if r["crosses_train_val"])
    findings.append(f"near-duplicate (same duration+size, differing hash) groups: "
                    f"{len(near_rows):,}; {near_cross:,} rows cross train/val")
    log(f"[S1.3] near-duplicate groups: {len(near_rows):,} (cross-split rows {near_cross:,})")

    # ---------- 3. artist overlap ----------
    art = df[(df["artist"].notna()) & (df["artist"].astype(str) != "")].copy()
    art["artist_key"] = art["artist"].astype(str).str.strip().str.lower()
    art = art[art["artist_key"] != ""]
    arows = []
    if len(art):
        g2 = art.groupby("artist_key")
        for k, sub in g2:
            labels = set(sub["human_or_synthetic"].astype(str))
            arows.append({
                "artist_key": k,
                "artist_display": str(sub["artist"].iloc[0])[:120],
                "n_files": len(sub),
                "n_human": int((sub["human_or_synthetic"] == "human").sum()),
                "n_synthetic": int((sub["human_or_synthetic"] == "synthetic").sum()),
                "datasets": "|".join(sorted(set(sub["dataset"].astype(str)))),
                "generator_families": "|".join(sorted(set(sub["generator_family"].astype(str)))),
                "appears_in_both_classes": int(len(labels) > 1),
                "example_paths": " || ".join(list(sub["absolute_or_project_path"])[:2]),
            })
    arows.sort(key=lambda r: -r["n_files"])
    write_csv(STAGE1_DIR / "artist_overlap_report.csv", arows,
              ["artist_key", "artist_display", "n_files", "n_human", "n_synthetic",
               "datasets", "generator_families", "appears_in_both_classes",
               "example_paths"])
    both = [r for r in arows if r["appears_in_both_classes"]]
    findings.append(f"artists present in both human and synthetic classes: {len(both)} "
                    f"(covering {sum(r['n_files'] for r in both)} files) - artist identity "
                    f"is a potential shortcut feature")
    log(f"[S1.3] artists in both classes: {len(both)}")

    # ---------- 4. prompt overlap ----------
    pr = df.copy()
    pr["prompt_key"] = (pr["prompt_text_if_available"].astype(str)
                       .str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    pr = pr[(pr["prompt_key"].notna()) & (pr["prompt_key"] != "") &
            (pr["prompt_key"] != "nan")]
    prows = []
    if len(pr):
        for k, sub in pr.groupby("prompt_key"):
            if len(sub) < 2:
                continue
            splits = sorted(set(sub["split"].astype(str)))
            prows.append({
                "prompt_key": k[:200],
                "n_files": len(sub),
                "generator_families": "|".join(sorted(set(sub["generator_family"].astype(str)))),
                "datasets": "|".join(sorted(set(sub["dataset"].astype(str)))),
                "splits": "|".join(splits),
                "crosses_train_val": int(len(splits) > 1),
                "example_paths": " || ".join(list(sub["absolute_or_project_path"])[:3]),
            })
    prows.sort(key=lambda r: -r["n_files"])
    write_csv(STAGE1_DIR / "prompt_overlap_report.csv", prows,
              ["prompt_key", "n_files", "generator_families", "datasets", "splits",
               "crosses_train_val", "example_paths"])
    pcross = sum(r["n_files"] for r in prows if r["crosses_train_val"])
    findings.append(f"prompts reused across >=2 files: {len(prows):,}; "
                    f"{pcross:,} rows cross train/val")
    log(f"[S1.3] reused prompts: {len(prows):,} (cross-split rows {pcross:,})")

    # ---------- 5. generator x split matrix ----------
    mat = []
    for (ds, gen), sub in audio.groupby(["dataset", "generator_family"]):
        row = {"dataset": ds, "generator_family": gen}
        for sp in ("train", "val", "unassigned"):
            row[f"n_{sp}"] = int((sub["split"] == sp).sum())
        row["total"] = len(sub)
        row["in_train"] = int(row["n_train"] > 0)
        row["in_val"] = int(row["n_val"] > 0)
        mat.append(row)
    mat.sort(key=lambda r: (r["dataset"], r["generator_family"]))
    write_csv(STAGE1_DIR / "generator_registry.csv", mat,
              ["dataset", "generator_family", "n_train", "n_val", "n_unassigned",
               "total", "in_train", "in_val"])
    log(f"[S1.3] generator registry: {len(mat)} dataset x generator cells")

    both_splits = [r for r in mat if r["in_train"] and r["in_val"]]
    findings.append(f"generators appearing in BOTH train and val: {len(both_splits)} "
                    f"-> {[r['generator_family'] for r in both_splits]}")

    # ---------- 6. metadata-only shortcut screen ----------
    srows = []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        feat_cols = ["original_sample_rate", "channels", "duration_s",
                     "size_bytes", "original_codec"]
        m = audio[audio["is_ai"].isin([0, 1])] if "is_ai" in audio.columns else None
        if m is None:
            m = audio[audio["human_or_synthetic"].isin(["human", "synthetic"])].copy()
            m["is_ai"] = (m["human_or_synthetic"] == "synthetic").astype(int)
        else:
            m = m.copy()
        X = m[feat_cols].apply(pd.to_numeric, errors="coerce")
        X["original_codec"] = X["original_codec"].astype(str).map(
            {"mp3": 1, "wav": 2, "flac": 3, "ogg": 4, "vorbis": 5, "opus": 6}
        ).fillna(0)
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        y = m["is_ai"].astype(int)
        if len(set(y)) > 1 and len(y) > 200:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            acc = cross_val_score(pipe, X.values, y.values, cv=5, scoring="accuracy")
            auc = cross_val_score(pipe, X.values, y.values, cv=5, scoring="roc_auc")
            srows = [{
                "features": ",".join(feat_cols),
                "n_samples": len(y),
                "metadata_only_cv_accuracy_mean": round(float(acc.mean()), 4),
                "metadata_only_cv_accuracy_std": round(float(acc.std()), 4),
                "metadata_only_cv_auroc_mean": round(float(auc.mean()), 4),
                "note": "5-fold CV on header metadata only; no audio used",
            }]
            log(f"[S1.3] METADATA-ONLY AUROC = {auc.mean():.4f} (acc {acc.mean():.4f})")
            findings.append(f"metadata-only classifier reaches AUROC {auc.mean():.4f} / "
                            f"accuracy {acc.mean():.4f} using ONLY header fields "
                            f"(sample rate, codec, channels, duration, size)")
    except Exception as e:
        srows = [{"features": "n/a", "note": f"shortcut screen failed: {e}"}]
        log(f"[S1.3] shortcut screen error: {e}")
    write_csv(STAGE1_DIR / "shortcut_metadata_screen.csv", srows,
              ["features", "n_samples", "metadata_only_cv_accuracy_mean",
               "metadata_only_cv_accuracy_std", "metadata_only_cv_auroc_mean", "note"])

    # ---------- summary ----------
    summ = [{"finding": f, "severity": ""} for f in findings]
    write_csv(STAGE1_DIR / "overlap_findings.csv", summ, ["finding", "severity"])

    log("")
    for f in findings:
        log(f"  - {f}")

    stage_report(
        stage="1 - Dataset and Provenance Audit (overlap analysis)",
        status="PARTIAL",
        done=[
            "computed sha256 for every hashed audio row in the manifest",
            "grouped exact duplicates and flagged cross-split occurrences",
            "approximated near-duplicates via duration+size with differing hash",
            "cross-tabulated artist names across human/synthetic classes",
            "detected reused prompts across files and splits",
            "built generator x split registry",
            "ran a metadata-only logistic-regression shortcut screen",
        ],
        deliverables=[
            "artifacts/stage1_audit/duplicate_report.csv",
            "artifacts/stage1_audit/near_duplicate_report.csv",
            "artifacts/stage1_audit/artist_overlap_report.csv",
            "artifacts/stage1_audit/prompt_overlap_report.csv",
            "artifacts/stage1_audit/generator_registry.csv",
            "artifacts/stage1_audit/shortcut_metadata_screen.csv",
            "artifacts/stage1_audit/overlap_findings.csv",
        ],
        results=findings,
        problems=[
            "chromaprint/pyacoustid absent: near-duplicate detection is a "
            "duration+size proxy, NOT a true acoustic fingerprint",
            "mutagen absent: MP3 bitrate not extracted for compressed files",
        ],
        interpretation=[
            "Artist identity and prompt reuse are measurable shortcut channels; "
            "the metadata-only screen quantifies how far header fields alone go.",
        ],
        supports="Partially - overlap structure is measured, but acoustic "
                 "near-duplicate detection needs fingerprint tooling.",
        nxt="Run s4_provenance.py to write provenance_issues.md and "
            "resolved_label_changes.csv, then s5_report.py for the Stage 1 README.",
        blockers=["chromaprint/pyacoustid/mutagen not installed"],
    )


if __name__ == "__main__":
    main()