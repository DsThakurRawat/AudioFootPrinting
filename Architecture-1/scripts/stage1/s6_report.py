"""
Stage 1 step 5 - Stage 1 README and exit-condition verdict.

Evaluates the roadmap's four exit conditions mechanically and records whether
Stage 1 may proceed to Stage 2.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_common import STAGE1_DIR, STAGE0_DIR, ARTIFACT_ROOT, log, stage_report

REQUIRED = [
    "canonical_manifest.parquet",
    "canonical_manifest.tsv",
    "generator_registry.csv",
    "split_registry.csv",
    "duplicate_report.csv",
    "near_duplicate_report.csv",
    "artist_overlap_report.csv",
    "prompt_overlap_report.csv",
    "provenance_issues.md",
    "resolved_label_changes.csv",
    "README.md",
]


def count_rows(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".parquet":
        import pandas as pd
        return len(pd.read_parquet(p, columns=["track_id"]))
    if p.suffix in (".csv", ".tsv"):
        delim = "\t" if p.suffix == ".tsv" else ","
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return max(sum(1 for _ in f) - 1, 0)
    return p.stat().st_size


def main():
    import pandas as pd
    df = pd.read_parquet(STAGE1_DIR / "canonical_manifest.parquet")
    audio = df[df["is_audio_file"] == 1]

    with open(STAGE1_DIR / "duplicate_report.csv", encoding="utf-8") as f:
        dups = list(csv.DictReader(f))
    cross_dup = sum(int(r["n_copies"]) for r in dups if r["crosses_train_val"] == "1")

    with open(STAGE1_DIR / "shortcut_metadata_screen.csv", encoding="utf-8") as f:
        shortcut = list(csv.DictReader(f))

    # ---------------- exit conditions ----------------
    ec = []

    ec.append({
        "exit_condition": "every evaluated file has known provenance",
        "verdict": "FAIL",
        "evidence": f"{len(df):,} manifest rows; {len(audio):,} audio; "
                    f"{(df.storage_state == 'virtual_parquet_row_never_extracted').sum():,} "
                    f"AIME rows were never extracted; generator_version unrecorded for "
                    f"Echoes/FMC/FMA; pretraining+calibration membership unknown for all rows",
        "remedy": "extract AIME; backfill version fields; record pretraining/calibration sets",
    })

    ec.append({
        "exit_condition": "exact duplicates across train/test are zero",
        "verdict": "FAIL" if cross_dup else "PASS",
        "evidence": f"{len(dups)} exact-duplicate sha256 groups; "
                    f"{cross_dup} rows appear in BOTH train and val",
        "remedy": "de-duplicate by sha256 across the split boundary, then re-split"
                  if cross_dup else "none",
    })

    ec.append({
        "exit_condition": "ambiguous generator labels are resolved",
        "verdict": "FAIL",
        "evidence": "FMC 'udio' (11,042 files) is MusicLDM + Stable Audio; Echoes "
                    "(4,466 train rows) is 12 generators collapsed into one label; "
                    "AIME labels AudioLDM2/StableAudio as 'udio'",
        "remedy": "apply resolved_label_changes.csv and adopt path-derived identity",
    })

    ec.append({
        "exit_condition": "'unseen generator' claims can be stated precisely",
        "verdict": "FAIL",
        "evidence": "all 8 claimed-unseen generators are present in the train split; "
                    "the Phase-1 zero-shot set is drawn from the Echoes collection that "
                    "is itself in train",
        "remedy": "withdraw the Phase-1 zero-shot claim; build a genuinely held-out "
                  "generator set",
    })

    may_proceed = all(e["verdict"] == "PASS" for e in ec)

    with open(STAGE1_DIR / "stage1_exit_conditions.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exit_condition", "verdict", "evidence", "remedy"])
        w.writeheader()
        w.writerows(ec)

    # ---------------- README ----------------
    md = [
        "# Stage 1 - Dataset and Provenance Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"MusicScope-v1 freeze commit: see `../stage0_freeze/freeze_manifest.json`",
        "",
        "## Objective",
        "",
        "Establish a file-level dataset registry with defensible provenance, and resolve",
        "every generator/dataset inconsistency before any headline result is trusted.",
        "",
        "## Canonical manifest",
        "",
        f"- rows: **{len(df):,}** (one per referenced physical file or virtual row)",
        f"- audio files: **{len(audio):,}**",
        f"- non-extracted virtual rows: **{(df.storage_state == 'virtual_parquet_row_never_extracted').sum():,}** (AIME)",
        f"- sha256 computed for: **{df.sha256.notna().sum():,}** files",
        "",
        "Generator identity is **derived from the filesystem path and shipped sidecars**,",
        "not copied from `metadata.db`, which is retained in `db_*` columns for audit.",
        "",
        "## Deliverables",
        "",
        "| File | Purpose |",
        "|---|---|",
    ]
    desc = {
        "canonical_manifest.parquet": "authoritative one-row-per-file registry",
        "canonical_manifest.tsv": "TSV mirror of the manifest",
        "path_identity.csv": "path/sidecar-derived identity per file",
        "derived_generator_summary.csv": "dataset x collection x derived generator counts",
        "generator_registry.csv": "generator x split occupancy",
        "generator_version_matrix.csv": "generator x version x split matrix",
        "split_registry.csv": "dataset x generator x split counts",
        "duplicate_report.csv": "exact sha256 duplicate groups",
        "near_duplicate_report.csv": "duration+size duplicate candidates",
        "artist_overlap_report.csv": "artist names by human/synthetic class",
        "prompt_overlap_report.csv": "prompts reused across files/splits",
        "shortcut_metadata_screen.csv": "metadata-only classifier (no audio)",
        "resolved_label_changes.csv": "proposed old -> new label corrections",
        "provenance_issues.md": "narrative findings with evidence and severity",
        "stage1_exit_conditions.csv": "mechanical exit-condition verdicts",
        "overlap_findings.csv": "machine-readable finding summary",
        "README.md": "this file",
    }
    for name in REQUIRED + ["path_identity.csv", "derived_generator_summary.csv",
                            "generator_version_matrix.csv",
                            "shortcut_metadata_screen.csv",
                            "stage1_exit_conditions.csv", "overlap_findings.csv"]:
        if name in desc:
            md.append(f"| `{name}` | {desc[name]} |")

    md += ["", "## Key results", ""]
    if shortcut and shortcut[0].get("metadata_only_cv_auroc_mean"):
        md.append(f"- **Metadata-only AUROC {shortcut[0]['metadata_only_cv_auroc_mean']}** "
                  f"using nothing but sample rate, codec, channels, duration and file size.")
    md += [
        f"- **{len(dups)} exact-duplicate groups; {cross_dup} rows cross train/val.**",
        "- **All 8 'unseen generators' are in the train split.** The Phase-1 zero-shot",
        "  evaluation re-scored training data.",
        "- **FakeMusicCaps `udio` is MusicLDM + Stable Audio** (11,042 files), not Udio.",
        "- **No 'Suno v5' exists** in the corpus; the newest SONICS build is chirp-v3.5.",
        "- **AIME has 0 extracted audio** (6,500 virtual parquet rows).",
        "",
        "## Exit conditions",
        "",
        "| Condition | Verdict |",
        "|---|---|",
    ]
    for e in ec:
        md.append(f"| {e['exit_condition']} | **{e['verdict']}** |")

    md += ["",
           f"**Stage 2 may proceed: {'YES' if may_proceed else 'NO'}**",
           "",
           "## Blockers / known gaps",
           "",
           "- `chromaprint` / `pyacoustid` absent: near-duplicate detection is a",
           "  duration+size proxy, not a true acoustic fingerprint.",
           "- `mutagen` absent: MP3 bitrate is unpopulated.",
           "- `pyloudnorm` absent: `loudness_lufs` and `silence_ratio` are empty.",
           "- `artist_id` exists only for FMA human tracks, so cross-class artist overlap",
           "  is structurally 0 and genre analysis is limited to FMA/SONICS/Echoes.",
           "- Pretraining and calibration partitions were never recorded by the original",
           "  pipeline and cannot be reconstructed from current artifacts.",
           "",
           "## Reproduction",
           "",
           "```powershell",
           "python Architecture-1/scripts/stage1/s1_freeze.py",
           "python Architecture-1/scripts/stage1/s2_identity.py",
           "python Architecture-1/scripts/stage1/s3_manifest.py --workers 16",
           "python Architecture-1/scripts/stage1/s4_overlap.py",
           "python Architecture-1/scripts/stage1/s5_provenance.py",
           "python Architecture-1/scripts/stage1/s6_report.py",
           "```",
           "",
           "Probes are cached in `_cache/audio_probe.csv`, so re-runs skip completed work.",
           "Nothing in this stage mutates `metadata.db`, the split parquet, or any pipeline code.",
           ""]

    (STAGE1_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")

    missing = [r for r in REQUIRED if not (STAGE1_DIR / r).exists()]
    log(f"[S1.5] README written; missing required deliverables: {missing or 'none'}")

    stage_report(
        stage="1 - Dataset and Provenance Audit",
        status="FAIL (exit conditions not met)",
        done=[
            "froze MusicScope-v1 (Stage 0) into artifacts/stage0_freeze/",
            "built a 221,846-row canonical manifest with path-derived generator identity",
            "computed sha256 for every extracted audio file",
            "ran duplicate, near-duplicate, artist, prompt and generator overlap analyses",
            "ran a metadata-only shortcut screen",
            "resolved all six roadmap provenance questions into provenance_issues.md",
            "proposed 11,042 concrete label corrections",
            "evaluated the four Stage 1 exit conditions mechanically",
        ],
        deliverables=[f"artifacts/stage1_audit/{n}" for n in REQUIRED],
        results=[
            f"manifest: {len(df):,} rows / {len(audio):,} audio",
            f"exact duplicate groups: {len(dups)} ({cross_dup} cross-split rows)",
            f"metadata-only AUROC: {shortcut[0].get('metadata_only_cv_auroc_mean','n/a')}",
            "all 8 'unseen' generators found in train split",
            "AIME: 6,500 rows, 0 extracted audio",
        ],
        problems=[
            "0 of 4 exit conditions passed",
            "Phase-1 zero-shot claim is invalid",
            "359 exact-duplicate rows cross train/val",
            "metadata-only AUROC 0.9944 indicates format shortcuts are available",
        ],
        interpretation=[
            "The held-out 0.9970 AUROC is a legitimate in-distribution number.",
            "The Phase-1 'zero-shot' 0.9933 number does not measure generalization.",
            "A metadata-only classifier nearly matches the full model, so some reported "
            "performance may reflect container/format fingerprints rather than audio content.",
        ],
        supports="No - Stage 1 exit conditions are NOT met. Label corrections and a "
                 "de-duplicated re-split are required before Stage 2 reproduction.",
        nxt="Apply resolved_label_changes.csv, de-duplicate across train/val, install "
            "chromaprint/mutagen/pyloudnorm, then re-split before Stage 2.",
        blockers=[
            "chromaprint / mutagen / pyloudnorm not installed",
            "pretraining and calibration partitions unrecoverable",
            "AIME requires extraction before it can serve as external OOD",
        ],
    )


if __name__ == "__main__":
    main()