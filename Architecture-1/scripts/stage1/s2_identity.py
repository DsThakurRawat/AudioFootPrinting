"""
Stage 1 step 1 - canonical path-derived identity resolution.

Builds the authoritative generator/dataset identity for every audio file by
parsing the filesystem layout and the shipped sidecar metadata, rather than
trusting the pre-existing (partly incorrect) metadata.db labels.

Sub-generators are recovered from directory structure:
  echoes/Echoes/{TTA,ATA}/<generator>/*.mp3
  fakemusiccaps/<dir>/*                 (dir -> generator mapping)
  sonics/fake_songs/fake_<id>_<gen>_<k>  (filename -> generator)

Outputs a path-keyed identity table consumed by s2_manifest.py.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_common import (DATA_RAW, STAGE1_DIR, ensure_dir, norm_path,
                       project_relative, log)

# FakeMusicCaps ships these directories; the old labels collapsed two of them
# into "udio", which is a demonstrable labelling error.
FMC_DIR_MAP = {
    "audioldm2": ("audioldm", "AudioLDM"),
    "MusicGen_medium": ("musicgen", "MusicGen"),
    "musicldm": ("musicldm", "MusicLDM"),
    "mustango": ("mustango", "Mustango"),
    "stable_audio_open": ("stableaudio", "Stable Audio"),
    "__MACOSX": ("__macosx", "AppleDouble resource fork (NOT AUDIO)"),
    ".extracted": ("extracted_marker", "extraction marker"),
}

ECHOES_COLLECTION = {"TTA": "text_to_audio", "ATA": "audio_to_audio"}

SONICS_FILE_RE = re.compile(r"^fake_(\d+)_(suno|udio)_(\d+)$", re.I)


def parse_echoes(path: Path):
    parts = path.parts
    try:
        i = list(parts).index("Echoes")
    except ValueError:
        return {}
    if i + 2 >= len(parts):
        return {}
    coll = parts[i + 1].upper()
    gen = parts[i + 2]
    return {
        "echoes_collection": coll,
        "echoes_collection_type": ECHOES_COLLECTION.get(coll, "unknown"),
        "echoes_generator": gen.lower(),
    }


def parse_fmc(path: Path):
    parts = path.parts
    try:
        i = list(parts).index("fakemusiccaps")
    except ValueError:
        return {}
    d = parts[i + 1] if i + 1 < len(parts) else ""
    fam, name = FMC_DIR_MAP.get(d, ("unknown", d))
    return {"fmc_dir": d, "fmc_generator_family": fam, "fmc_generator_name": name}


def parse_sonics(path: Path):
    stem = path.stem
    m = SONICS_FILE_RE.match(stem)
    out = {}
    if m:
        out.update({"sonics_song_id": m.group(1),
                    "sonics_generator": m.group(2).lower(),
                    "sonics_variant": m.group(3)})
    return out


def load_echoes_sidecar():
    """Echoes ships a manifest with true generator + original-audio reference."""
    p = DATA_RAW / "echoes" / "Echoes" / "dataset_manifest.csv"
    if not p.exists():
        return {}
    idx = {}
    with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("path_in_dataset") or "").strip()
            if not pid:
                continue
            idx[norm_path(pid)] = {
                "echoes_manifest_generator": (row.get("generator") or "").strip().lower(),
                "echoes_manifest_type": (row.get("type") or "").strip(),
                "echoes_manifest_original_audio": (row.get("original_audio") or "").strip(),
                "echoes_manifest_genre": (row.get("genre") or "").strip(),
                "echoes_manifest_description": (row.get("description") or "").strip(),
            }
    return idx


def load_sonics_sidecar():
    """SONICS CSVs carry the generating algorithm and the prompt text."""
    out = {}
    for fn, key in (("fake_songs.csv", "fake"), ("real_songs.csv", "real")):
        p = DATA_RAW / "sonics" / fn
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                fn_ = (row.get("filename") or "").strip()
                if not fn_:
                    continue
                out[norm_path(fn_)] = {
                    "sonics_sidecar_kind": key,
                    "sonics_algorithm": (row.get("algorithm") or "").strip(),
                    "sonics_style": (row.get("style") or "").strip(),
                    "sonics_topic": (row.get("topic") or "").strip(),
                    "sonics_genre": (row.get("genre") or "").strip(),
                    "sonics_mood": (row.get("mood") or "").strip(),
                    "sonics_lyrics": (row.get("lyrics") or "").strip(),
                    "sonics_artist": (row.get("artist") or "").strip(),
                    "sonics_title": (row.get("title") or "").strip(),
                    "sonics_year": (row.get("year") or "").strip(),
                    "sonics_split": (row.get("split") or "").strip(),
                    "sonics_label": (row.get("label") or "").strip(),
                    "sonics_bit_rate": (row.get("bit_rate") or "").strip(),
                }
    return out


def main():
    ensure_dir(STAGE1_DIR)
    log("[S1.1] loading sidecar metadata...")
    echoes_idx = load_echoes_sidecar()
    sonics_idx = load_sonics_sidecar()
    log(f"       echoes manifest rows: {len(echoes_idx):,}")
    log(f"       sonics sidecar rows : {len(sonics_idx):,}")

    log("[S1.1] walking audio tree...")
    AUDIO = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}
    rows = []
    counters = Counter()
    for root, dirs, files in os.walk(DATA_RAW):
        rp = Path(root)
        for fn in files:
            p = rp / fn
            if p.suffix.lower() not in AUDIO:
                continue
            rel = project_relative(p)
            rec = {"project_relative_path": rel,
                   "norm_path": norm_path(p),
                   "filename": fn,
                   "ext": p.suffix.lower()}

            top = None
            for part in p.parts:
                if part in ("aime", "echoes", "fakemusiccaps", "fma",
                            "mtg_jamendo", "sdd", "sonics", "suno_audio"):
                    top = part
                    break
            rec["source_dataset"] = top or "unknown"

            rec.update(parse_echoes(p))
            rec.update(parse_fmc(p))
            rec.update(parse_sonics(p))

            rel_key = norm_path(rel)
            if rel_key in echoes_idx:
                rec.update(echoes_idx[rel_key])
            fn_key = norm_path(fn)
            if fn_key in sonics_idx:
                rec.update(sonics_idx[fn_key])
            else:
                base = norm_path(p.name)
                if base in sonics_idx:
                    rec.update(sonics_idx[base])

            rows.append(rec)
            counters[top or "unknown"] += 1

    log(f"[S1.1] resolved {len(rows):,} audio files")
    for k, v in counters.most_common():
        log(f"       {k:<16} {v:,}")

    # ---- generator cross-tab: old db label vs path-derived truth ----
    log("[S1.1] cross-tabulating path-derived generators...")
    xrows = []
    for r in rows:
        ds = r["source_dataset"]
        if ds == "echoes":
            gen = r.get("echoes_generator") or r.get("echoes_manifest_generator") or "unknown"
            coll = r.get("echoes_collection", "?")
        elif ds == "fakemusiccaps":
            gen = r.get("fmc_generator_family", "unknown")
            coll = r.get("fmc_dir", "?")
        elif ds == "sonics":
            gen = r.get("sonics_generator") or r.get("sonics_algorithm") or "unknown"
            coll = "sonics"
        elif ds == "suno_audio":
            gen = "suno"
            coll = "huggingface"
        elif ds == "aime":
            gen = "unknown_aime"
            coll = "huggingface"
        else:
            gen = "human"
            coll = "library"
        xrows.append({"source_dataset": ds, "collection": coll, "derived_generator": gen,
                      "norm_path": r["norm_path"]})

    out = STAGE1_DIR / "path_identity.csv"
    cols = sorted({k for r in rows for k in r})
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    log(f"[S1.1] wrote {out} ({len(rows):,} rows, {len(cols)} cols)")

    # generator x collection summary (feeds generator_registry.csv in s2)
    summ = Counter((r["source_dataset"], r["collection"], r["derived_generator"])
                   for r in xrows)
    out2 = STAGE1_DIR / "derived_generator_summary.csv"
    with open(out2, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "collection", "derived_generator", "file_count"])
        for (ds, coll, gen), n in sorted(summ.items()):
            w.writerow([ds, coll, gen, n])
    log(f"[S1.1] wrote {out2} ({len(summ)} distinct generators)")


if __name__ == "__main__":
    main()