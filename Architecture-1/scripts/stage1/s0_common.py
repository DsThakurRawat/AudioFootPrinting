"""
Stage 0/1 shared utilities for the MusicScope Transactions provenance audit.

Centralises path handling, canonical field names, and IO helpers so every
stage script agrees on identifiers and never silently rewrites prior output.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = Path(os.environ.get("MIREX_RAW_DIR", REPO_ROOT / "data" / "raw"))
PROCESSED = Path(os.environ.get("MIREX_PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
METADATA_DB = PROCESSED / "metadata.db"
SPLIT_PARQUET = PROCESSED / "train_val_split_200k.parquet"

ARTIFACT_ROOT = Path(os.environ.get(
    "MIREX_STAGE1_DIR", REPO_ROOT / "artifacts"))
STAGE0_DIR = ARTIFACT_ROOT / "stage0_freeze"
STAGE1_DIR = ARTIFACT_ROOT / "stage1_audit"
CACHE_DIR = STAGE1_DIR / "_cache"

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".wma"}
SIDECAR_EXTS = {".json", ".csv", ".jsonl", ".txt", ".tsv", ".parquet", ".arrow",
                ".metadata", ".done", ".md", ".pickle", ".zip", ".tag"}

# Audio container subtypes reported by soundfile, mapped to human codec names.
SUBTYPE_CODEC = {
    "MPEG": "mp3", "MPEG Layer 3": "mp3", "WAV": "wav", "WAVEX": "wav",
    "FLAC": "flac", "OGG": "ogg", "VORBIS": "vorbis", "OPUS": "opus",
    "AIFF": "aiff", "PCM": "pcm", "FLOAT": "pcm_f32", "DOUBLE": "pcm_f64",
    "ALAW": "alaw", "ULAW": "ulaw", "MULAW": "ulaw", "GSM610": "gsm610",
}

CANONICAL_FIELDS = [
    "track_id",
    "absolute_or_project_path",
    "project_relative_path",
    "exists_on_disk",
    "size_bytes",
    "sha256",
    "acoustic_fingerprint",
    "dataset",
    "dataset_version",
    "source_platform",
    "generator_family",
    "generator_name",
    "generator_version",
    "generator_evidence",
    "checkpoint_if_known",
    "human_or_synthetic",
    "fully_or_partially_synthetic",
    "artist",
    "genre",
    "prompt_id",
    "prompt_text_if_available",
    "original_audio_reference",
    "duration_s",
    "original_sample_rate",
    "original_codec",
    "original_bitrate_bps",
    "channels",
    "loudness_lufs",
    "silence_ratio",
    "is_audio_file",
    "split",
    "used_for_pretraining",
    "used_for_training",
    "used_for_validation",
    "used_for_calibration",
    "used_for_test",
    "strat_class",
    "db_source_dataset",
    "db_generator_family",
    "db_generator_version",
    "db_split",
    "db_artist_id",
    "db_quarantined",
    "label_confidence",
    "notes",
]


def norm_path(p) -> str:
    """Canonical form for joining across sqlite / parquet / filesystem.

    Lowercases, normalises separators, and strips redundant prefixes so the
    same physical file joins regardless of which source recorded it.
    """
    s = str(p).replace("\\", "/").strip()
    while "//" in s:
        s = s.replace("//", "/")
    return s.lower()


def project_relative(p) -> str:
    """Path relative to the repository root, forward-slashed."""
    s = str(p).replace("\\", "/")
    root = str(REPO_ROOT).replace("\\", "/")
    low = s.lower()
    if low.startswith(root.lower()):
        return s[len(root):].lstrip("/")
    return s


def ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def guard_overwrite(path: Path, overwrite: bool = False) -> Path:
    """Refuse to clobber an existing artifact unless explicitly allowed.

    Stage rules forbid overwriting previous results; this makes accidental
    destruction impossible rather than merely discouraged.
    """
    if path.exists() and not overwrite:
        raise SystemExit(
            f"[guard] refusing to overwrite existing artifact: {path}\n"
            f"        pass --overwrite to replace it intentionally.")
    ensure_dir(path.parent)
    return path


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def classify_file(path: Path):
    """Return (is_audio, kind) for one filesystem entry."""
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        return True, "audio"
    if ext in SIDECAR_EXTS:
        return False, "sidecar"
    return False, "other"


def log(msg: str) -> None:
    print(msg, flush=True)


def stage_report(stage: str, status: str, done, deliverables, results,
                 problems, interpretation, supports, nxt, blockers):
    """Emit the roadmap's mandatory reporting format."""
    bar = "=" * 78
    print(bar)
    print(f"STAGE: {stage}")
    print(f"STATUS: {status}")
    print(bar)
    print("\nWHAT WAS DONE:")
    for d in done:
        print(f"- {d}")
    print("\nDELIVERABLES CREATED:")
    for d in deliverables:
        print(f"- {d}")
    print("\nKEY RESULTS:")
    for r in results:
        print(f"- {r}")
    print("\nPROBLEMS FOUND:")
    for p in problems or ["- none"]:
        print(f"- {p}")
    print("\nSCIENTIFIC INTERPRETATION:")
    for i in interpretation:
        print(f"- {i}")
    print(f"\nDOES THIS SUPPORT THE CURRENT HYPOTHESIS? {supports}")
    print(f"\nNEXT RECOMMENDED ACTION: {nxt}")
    print("\nBLOCKERS:")
    for b in blockers or ["- none"]:
        print(f"- {b}")
    print(bar)