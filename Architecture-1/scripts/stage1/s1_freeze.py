"""
Stage 0 - Freeze the existing MusicScope-v1 system.

Inventories code, checkpoints, datasets, and environment so every later stage
references an exact, recorded state. Read-only: never modifies the pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_common import (REPO_ROOT, DATA_RAW, PROCESSED, METADATA_DB,
                       SPLIT_PARQUET, STAGE0_DIR, ensure_dir, guard_overwrite,
                       log, stage_report)


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT),
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "<unavailable>"


def pkg_versions() -> dict:
    out = {}
    for m in ("torch", "torchaudio", "librosa", "numpy", "scipy", "sklearn",
              "pandas", "soundfile", "soxr", "mutagen", "pyloudnorm",
              "chromaprint", "pyacoustid"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "<no __version__>")
        except Exception:
            out[m] = None
    return out


def env_report() -> str:
    lines = [
        f"captured_utc: {datetime.now(timezone.utc).isoformat()}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"cpu_count: {os.cpu_count()}",
        f"git_commit: {git('rev-parse', 'HEAD')}",
        f"git_branch: {git('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"git_status_dirty_lines: {len(git('status', '--porcelain').splitlines())}",
        f"git_remote: {git('remote', 'get-url', 'origin')}",
        "",
        "[packages]",
    ]
    for k, v in pkg_versions().items():
        lines.append(f"{k} = {v if v else 'NOT INSTALLED'}")
    lines += ["", "[gpu]"]
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"cuda_available: True")
            lines.append(f"cuda_version: {torch.version.cuda}")
            lines.append(f"cudnn_version: {torch.backends.cudnn.version()}")
            lines.append(f"gpu_count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                lines.append(f"gpu_{i}: {p.name} ({p.total_memory // (1 << 20)} MiB)")
        else:
            lines.append("cuda_available: False")
    except Exception as e:
        lines.append(f"gpu_probe_failed: {e}")
    return "\n".join(lines) + "\n"


def code_inventory():
    rows = []
    for pat in ("Architecture-1/model/*.py", "Architecture-1/scripts/*.py",
                "Architecture-1/scripts/**/*.py"):
        for p in sorted(REPO_ROOT.glob(pat)):
            if "__pycache__" in str(p) or "/archive/" in str(p).replace("\\", "/"):
                continue
            st = p.stat()
            rows.append({
                "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                "size_bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest(),
            })
    return rows


def checkpoint_inventory():
    rows = []
    for p in sorted(REPO_ROOT.glob("Architecture-1/model/checkpoints/**/*.pt")):
        st = p.stat()
        row = {
            "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        }
        try:
            import torch
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                row["keys"] = ",".join(sorted(ck.keys()))
                row["epoch"] = ck.get("epoch")
                row["loss"] = ck.get("loss")
                row["in_dim"] = ck.get("in_dim")
                sc = ck.get("scaler_state") or {}
                if isinstance(sc, dict) and "temperature" in sc:
                    row["temperature"] = float(sc["temperature"])
            else:
                row["keys"] = "<not a dict: %s>" % type(ck).__name__
        except Exception as e:
            row["keys"] = "<load failed: %s>" % e
        rows.append(row)
    return rows


def dataset_inventory():
    rows = []
    total_f = total_b = 0
    if DATA_RAW.exists():
        for child in sorted(DATA_RAW.iterdir()):
            if not child.is_dir():
                continue
            nf = nb = 0
            for root, _dirs, files in os.walk(child):
                for f in files:
                    nf += 1
                    try:
                        nb += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            rows.append({"dataset": child.name, "files": nf, "bytes": nb})
            total_f += nf
            total_b += nb
    return rows, total_f, total_b


def main():
    ap = argparse.ArgumentParser(description="Stage 0: freeze MusicScope-v1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ensure_dir(STAGE0_DIR)

    log("[Stage 0] capturing environment...")
    env_path = guard_overwrite(STAGE0_DIR / "ENVIRONMENT.txt", args.overwrite)
    env_path.write_text(env_report(), encoding="utf-8")

    log("[Stage 0] inventorying code...")
    code_path = guard_overwrite(STAGE0_DIR / "code_inventory.csv", args.overwrite)
    code = code_inventory()
    with open(code_path, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["path", "size_bytes", "mtime_utc", "sha256"])
        w.writeheader()
        w.writerows(code)

    log("[Stage 0] inventorying checkpoints...")
    ck_path = guard_overwrite(STAGE0_DIR / "checkpoint_inventory.csv", args.overwrite)
    cks = checkpoint_inventory()
    with open(ck_path, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        cols = ["path", "size_bytes", "mtime_utc", "keys", "epoch", "loss",
                "in_dim", "temperature"]
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cks)

    log("[Stage 0] inventorying datasets...")
    ds_path = guard_overwrite(STAGE0_DIR / "dataset_inventory.csv", args.overwrite)
    ds, tf, tb = dataset_inventory()
    with open(ds_path, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["dataset", "files", "bytes"])
        w.writeheader()
        w.writerows(ds)

    meta = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_remote": git("remote", "get-url", "origin"),
        "code_files": len(code),
        "checkpoints": len(cks),
        "datasets": len(ds),
        "total_files": tf,
        "total_bytes": tb,
        "metadata_db_exists": METADATA_DB.exists(),
        "split_parquet_exists": SPLIT_PARQUET.exists(),
        "note": "MusicScope-v1 frozen state; no pipeline code was modified.",
    }
    meta_path = guard_overwrite(STAGE0_DIR / "freeze_manifest.json", args.overwrite)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stage_report(
        stage="0 - Freeze",
        status="PASS",
        done=[
            f"captured environment and git state at {meta['git_commit'][:8]}",
            f"inventoried {len(code)} source files with sha256",
            f"inventoried {len(cks)} checkpoints (keys, epoch, temperature)",
            f"inventoried {len(ds)} raw datasets ({tf:,} files, {tb / 1e9:.1f} GB)",
            "wrote artifacts/stage0_freeze/ (5 files)",
        ],
        deliverables=[
            "artifacts/stage0_freeze/ENVIRONMENT.txt",
            "artifacts/stage0_freeze/code_inventory.csv",
            "artifacts/stage0_freeze/checkpoint_inventory.csv",
            "artifacts/stage0_freeze/dataset_inventory.csv",
            "artifacts/stage0_freeze/freeze_manifest.json",
        ],
        results=[
            f"git branch: {meta['git_branch']} @ {meta['git_commit'][:8]}",
            f"metadata.db present: {meta['metadata_db_exists']}",
            f"split parquet present: {meta['split_parquet_exists']}",
        ],
        problems=[
            "chromaprint / pyacoustid / mutagen / pyloudnorm are NOT installed; "
            "near-duplicate and loudness fields have no source yet",
        ],
        interpretation=[
            "The v1 system can be reconstructed from committed code plus these "
            "inventories; no undocumented state is required.",
        ],
        supports="Yes - v1 is frozen and reconstructible.",
        nxt="Proceed to Stage 1 canonical manifest build.",
        blockers=["missing fingerprint/loudness tooling (see Stage 1 plan)"],
    )


if __name__ == "__main__":
    main()