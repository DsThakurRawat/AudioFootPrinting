"""
Stage 1 step 2 - canonical manifest.

Joins every provenance source into one row per physical audio file:
  metadata.db            -> legacy labels (retained, not trusted)
  train_val_split parquet-> authoritative split + strat_class
  path_identity.csv      -> path-derived generator truth (s2_identity.py)
  filesystem             -> existence, size, sha256
  soundfile header       -> duration, sample rate, channels, codec, bitrate

Design rules:
  * Never mutates metadata.db or the split parquet.
  * Absence is recorded explicitly (exists_on_disk=False), not silently dropped.
  * Cached per-file probes so re-runs are cheap and resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_common import (DATA_RAW, METADATA_DB, SPLIT_PARQUET, STAGE1_DIR,
                       CACHE_DIR, SUBTYPE_CODEC, ensure_dir, log,
                       project_relative, REPO_ROOT)

PROBE_CACHE = CACHE_DIR / "audio_probe.csv"
HASH_CACHE = CACHE_DIR / "sha256_cache.csv"


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def load_cache(path: Path, key: str):
    d = {}
    for r in load_csv(path):
        d[r[key]] = r
    return d


def probe_one(rel: str):
    """Header-only audio probe + sha256 for one project-relative path.

    metadata.db stores AIME as virtual references like
    ``aime/data/train-00000-of-00210.parquet#row=7`` - those rows were never
    extracted to individual audio files. They are recorded as explicitly
    unextracted rather than treated as missing files.
    """
    p = Path(rel)
    rec = {"project_relative_path": rel}

    if "#row=" in rel:
        rec.update({"size_bytes": -2, "exists_on_disk": 0, "sha256": "",
                    "storage_state": "virtual_parquet_row_never_extracted"})
        return rec

    if not p.is_absolute():
        p = REPO_ROOT / rel
    try:
        st = p.stat()
        rec["size_bytes"] = st.st_size
        rec["exists_on_disk"] = 1
        rec["storage_state"] = "extracted_file"
    except OSError:
        rec["size_bytes"] = -1
        rec["exists_on_disk"] = 0
        rec["storage_state"] = "referenced_but_absent"
        return rec

    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        rec["sha256"] = h.hexdigest()
    except OSError:
        rec["sha256"] = ""

    try:
        import soundfile as sf
        info = sf.info(str(p))
        rec["duration_s"] = round(float(info.frames) / float(info.samplerate), 4) if info.samplerate else ""
        rec["original_sample_rate"] = info.samplerate
        rec["channels"] = info.channels
        rec["original_codec"] = SUBTYPE_CODEC.get(str(info.format), str(info.format))
        rec["subtype"] = str(info.subtype)
        if info.format in ("WAV", "AIFF", "FLAC") and info.subtype == "PCM":
            rec["original_bitrate_bps"] = int(info.samplerate * info.channels * 16)
        elif info.format == "WAV":
            rec["original_bitrate_bps"] = ""
        else:
            rec["original_bitrate_bps"] = ""
    except Exception as e:
        rec["duration_s"] = ""
        rec["original_sample_rate"] = ""
        rec["channels"] = ""
        rec["original_codec"] = ""
        rec["probe_error"] = f"{type(e).__name__}: {e}"[:120]
    return rec


def run_pool(paths, workers, label, chunk=256):
    out = []
    total = len(paths)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(probe_one, paths, chunksize=chunk), 1):
            out.append(rec)
            if i % 5000 == 0 or i == total:
                log(f"       {label}: {i:,}/{total:,}")
    return out


def write_csv(path: Path, rows, cols):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def main():
    ap = argparse.ArgumentParser(description="Stage 1: build canonical manifest")
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap file count")
    args = ap.parse_args()
    ensure_dir(STAGE1_DIR)
    ensure_dir(CACHE_DIR)

    # ---------- sources ----------
    log("[S1.2] loading identity table...")
    ident_rows = load_csv(STAGE1_DIR / "path_identity.csv")
    log(f"       path_identity rows: {len(ident_rows):,}")

    ident = {}
    for r in ident_rows:
        rel = r["project_relative_path"]
        rec = dict(r)
        rec.pop("norm_path", None)
        ident[rel] = rec

    log("[S1.2] loading metadata.db...")
    db = {}
    if METADATA_DB.exists():
        con = sqlite3.connect(f"file:{METADATA_DB.as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        for row in con.execute("select * from tracks"):
            rp = project_relative(row["file_path"]) if row["file_path"] else ""
            db[rp] = dict(row)
        con.close()
    log(f"       metadata.db rows: {len(db):,}")

    log("[S1.2] loading split parquet...")
    split = {}
    try:
        import pandas as pd
        df = pd.read_parquet(SPLIT_PARQUET)
        for r in df.to_dict("records"):
            split[project_relative(r["file_path"])] = r
        log(f"       split rows: {len(split):,}")
    except Exception as e:
        log(f"       WARNING: could not read split parquet: {e}")

    all_rel = sorted(set(ident) | set(db) | set(split))
    if args.limit:
        all_rel = all_rel[:args.limit]
    log(f"[S1.2] union of all sources: {len(all_rel):,} paths")

    # ---------- probe + hash (resumable) ----------
    cached = load_cache(PROBE_CACHE, "project_relative_path")
    todo = [r for r in all_rel if r not in cached]
    log(f"[S1.2] probing {len(todo):,} files ({len(cached):,} cached)")
    if todo:
        recs = run_pool(todo, args.workers, "probe+hash")
        cached.update({r["project_relative_path"]: r for r in recs})
        cols = ["project_relative_path", "exists_on_disk", "size_bytes", "sha256",
                "storage_state", "duration_s", "original_sample_rate", "channels",
                "original_codec", "subtype", "original_bitrate_bps", "probe_error"]
        write_csv(PROBE_CACHE, cached.values(), cols)
    log(f"[S1.2] probe cache: {len(cached):,} entries")

    # ---------- assemble ----------
    log("[S1.2] assembling manifest...")
    rows = []
    import pandas as pd
    cols_numeric = ["exists_on_disk", "size_bytes", "is_audio_file",
                    "used_for_training", "used_for_validation",
                    "duration_s", "original_sample_rate", "channels",
                    "original_bitrate_bps", "db_quarantined"]
    for rel in all_rel:
        i = ident.get(rel, {})
        d = db.get(rel, {})
        s = split.get(rel, {})
        p = cached.get(rel, {})

        ds = i.get("source_dataset") or d.get("source_dataset") or "unknown"
        gen_derived = i.get("derived_generator") or i.get("fmc_generator_family") \
            or i.get("echoes_generator") or i.get("sonics_generator") or ""
        coll = i.get("echoes_collection") or i.get("fmc_dir") or ""

        is_audio = 1 if (p.get("original_codec") or p.get("ext") in
                         (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a")) else 0
        if i.get("fmc_generator_family") == "__macosx":
            is_audio = 0

        is_ai = s.get("is_ai", d.get("is_ai"))
        if is_ai is None:
            is_ai = 1 if gen_derived not in ("human", "") else 0

        rec = {
            "track_id": d.get("track_id") or s.get("track_id") or rel,
            "absolute_or_project_path": rel,
            "exists_on_disk": p.get("exists_on_disk", 0),
            "storage_state": p.get("storage_state", "unknown"),
            "size_bytes": p.get("size_bytes", -1),
            "sha256": p.get("sha256", ""),
            "is_audio_file": is_audio,
            "dataset": ds,
            "dataset_version": i.get("sonics_version", "") or d.get("source_dataset", ""),
            "source_platform": {"sonics": "huggingface", "aime": "huggingface",
                                "suno_audio": "huggingface", "echoes": "huggingface",
                                "fakemusiccaps": "github", "fma": "zenodo"}.get(ds, ""),
            "generator_family": gen_derived or d.get("generator_family", ""),
            "generator_name": i.get("fmc_generator_name") or i.get("echoes_manifest_generator", ""),
            "generator_version": d.get("generator_version") or "",
            "generator_evidence": "path_derived",
            "human_or_synthetic": "human" if not is_ai else "synthetic",
            "fully_or_partially_synthetic": "unknown",
            "artist": i.get("sonics_artist", ""),
            "genre": i.get("sonics_genre") or i.get("echoes_manifest_genre", ""),
            "prompt_id": i.get("sonics_song_id", ""),
            "prompt_text_if_available": (i.get("sonics_lyrics", "") or
                                         i.get("sonics_style", "") or
                                         i.get("sonics_topic", "") or
                                         i.get("echoes_manifest_description", "")),
            "original_audio_reference": i.get("echoes_manifest_original_audio", ""),
            "duration_s": p.get("duration_s", ""),
            "original_sample_rate": p.get("original_sample_rate", ""),
            "original_codec": p.get("original_codec", ""),
            "original_bitrate_bps": p.get("original_bitrate_bps", ""),
            "channels": p.get("channels", ""),
            "loudness_lufs": "",
            "silence_ratio": "",
            "split": s.get("split", d.get("split", "unassigned")),
            "used_for_pretraining": "",
            "used_for_training": 1 if s.get("split") == "train" else 0,
            "used_for_validation": 1 if s.get("split") == "val" else 0,
            "used_for_calibration": "",
            "used_for_test": "",
            "strat_class": s.get("strat_class", ""),
            "db_source_dataset": d.get("source_dataset", ""),
            "db_generator_family": d.get("generator_family", ""),
            "db_generator_version": d.get("generator_version", "") or "",
            "db_split": d.get("split", ""),
            "db_artist_id": d.get("artist_id", "") or "",
            "db_quarantined": d.get("quarantined", ""),
            "label_confidence": "high" if is_audio else "none",
            "echoes_collection": coll,
            "sonics_algorithm": i.get("sonics_algorithm", ""),
            "probe_error": p.get("probe_error", ""),
            "notes": "",
        }
        # Normalise column dtypes so pyarrow can serialise the frame:
        # mixed str/int probe fields otherwise raise "Expected bytes, got int".
        for k in cols_numeric:
            rec[k] = pd.to_numeric(rec.get(k), errors="coerce")
        rows.append(rec)

    cols = ["track_id", "absolute_or_project_path", "exists_on_disk", "storage_state",
            "size_bytes", "sha256", "is_audio_file", "dataset", "dataset_version",
            "source_platform", "generator_family", "generator_name",
            "generator_version", "generator_evidence", "human_or_synthetic",
            "fully_or_partially_synthetic", "artist", "genre", "prompt_id",
            "prompt_text_if_available", "original_audio_reference", "duration_s",
            "original_sample_rate", "original_codec", "original_bitrate_bps",
            "channels", "loudness_lufs", "silence_ratio", "split",
            "used_for_pretraining", "used_for_training", "used_for_validation",
            "used_for_calibration", "used_for_test", "strat_class",
            "echoes_collection", "sonics_algorithm",
            "db_source_dataset", "db_generator_family", "db_generator_version",
            "db_split", "db_artist_id", "db_quarantined", "label_confidence",
            "probe_error", "notes"]

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        pq = STAGE1_DIR / "canonical_manifest.parquet"
        ensure_dir(pq.parent)
        df.to_parquet(pq, index=False)
        log(f"[S1.2] wrote {pq}")
        df.to_csv(STAGE1_DIR / "canonical_manifest.tsv", sep="\t", index=False)
        log(f"[S1.2] wrote {STAGE1_DIR / 'canonical_manifest.tsv'}")
    except Exception as e:
        log(f"       parquet/tsv write failed ({e}); CSV fallback")
        write_csv(STAGE1_DIR / "canonical_manifest.csv", rows, cols)

    # ---------- split registry ----------
    reg = Counter()
    for r in rows:
        reg[(r["dataset"], r["generator_family"], r["split"])] += 1
    write_csv(STAGE1_DIR / "split_registry.csv",
              [{"dataset": k[0], "generator_family": k[1], "split": k[2],
                "n_files": v} for k, v in sorted(reg.items())],
              ["dataset", "generator_family", "split", "n_files"])
    log(f"[S1.2] wrote {STAGE1_DIR / 'split_registry.csv'}")

    # ---------- quick integrity summary ----------
    log("")
    log("[S1.2] integrity summary")
    log(f"       present on disk : {sum(1 for r in rows if r['exists_on_disk'])}")
    log(f"       audio files     : {sum(1 for r in rows if r['is_audio_file'])}")
    log(f"       hashed          : {sum(1 for r in rows if r['sha256'])}")
    dup = Counter(r["sha256"] for r in rows if r["sha256"])
    ndup = sum(v for v in dup.values() if v > 1)
    log(f"       duplicate sha256 groups: {sum(1 for v in dup.values() if v > 1)} "
        f"covering {ndup} rows")


if __name__ == "__main__":
    main()