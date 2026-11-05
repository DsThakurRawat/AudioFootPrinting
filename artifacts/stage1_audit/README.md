# Stage 1 - Dataset and Provenance Audit

Generated: 2026-10-20T18:05:36.343390+00:00
MusicScope-v1 freeze commit: see `../stage0_freeze/freeze_manifest.json`

## Objective

Establish a file-level dataset registry with defensible provenance, and resolve
every generator/dataset inconsistency before any headline result is trusted.

## Canonical manifest

- rows: **221,846** (one per referenced physical file or virtual row)
- audio files: **187,557**
- non-extracted virtual rows: **6,500** (AIME)
- sha256 computed for: **221,846** files

Generator identity is **derived from the filesystem path and shipped sidecars**,
not copied from `metadata.db`, which is retained in `db_*` columns for audit.

## Deliverables

| File | Purpose |
|---|---|
| `canonical_manifest.parquet` | authoritative one-row-per-file registry |
| `canonical_manifest.tsv` | TSV mirror of the manifest |
| `generator_registry.csv` | generator x split occupancy |
| `split_registry.csv` | dataset x generator x split counts |
| `duplicate_report.csv` | exact sha256 duplicate groups |
| `near_duplicate_report.csv` | duration+size duplicate candidates |
| `artist_overlap_report.csv` | artist names by human/synthetic class |
| `prompt_overlap_report.csv` | prompts reused across files/splits |
| `provenance_issues.md` | narrative findings with evidence and severity |
| `resolved_label_changes.csv` | proposed old -> new label corrections |
| `README.md` | this file |
| `path_identity.csv` | path/sidecar-derived identity per file |
| `derived_generator_summary.csv` | dataset x collection x derived generator counts |
| `generator_version_matrix.csv` | generator x version x split matrix |
| `shortcut_metadata_screen.csv` | metadata-only classifier (no audio) |
| `stage1_exit_conditions.csv` | mechanical exit-condition verdicts |
| `overlap_findings.csv` | machine-readable finding summary |

## Key results

- **Metadata-only AUROC 0.9944** using nothing but sample rate, codec, channels, duration and file size.
- **560 exact-duplicate groups; 359 rows cross train/val.**
- **All 8 'unseen generators' are in the train split.** The Phase-1 zero-shot
  evaluation re-scored training data.
- **FakeMusicCaps `udio` is MusicLDM + Stable Audio** (11,042 files), not Udio.
- **No 'Suno v5' exists** in the corpus; the newest SONICS build is chirp-v3.5.
- **AIME has 0 extracted audio** (6,500 virtual parquet rows).

## Exit conditions

| Condition | Verdict |
|---|---|
| every evaluated file has known provenance | **FAIL** |
| exact duplicates across train/test are zero | **FAIL** |
| ambiguous generator labels are resolved | **FAIL** |
| 'unseen generator' claims can be stated precisely | **FAIL** |

**Stage 2 may proceed: NO**

## Blockers / known gaps

- `chromaprint` / `pyacoustid` absent: near-duplicate detection is a
  duration+size proxy, not a true acoustic fingerprint.
- `mutagen` absent: MP3 bitrate is unpopulated.
- `pyloudnorm` absent: `loudness_lufs` and `silence_ratio` are empty.
- `artist_id` exists only for FMA human tracks, so cross-class artist overlap
  is structurally 0 and genre analysis is limited to FMA/SONICS/Echoes.
- Pretraining and calibration partitions were never recorded by the original
  pipeline and cannot be reconstructed from current artifacts.

## Reproduction

```powershell
python Architecture-1/scripts/stage1/s1_freeze.py
python Architecture-1/scripts/stage1/s2_identity.py
python Architecture-1/scripts/stage1/s3_manifest.py --workers 16
python Architecture-1/scripts/stage1/s4_overlap.py
python Architecture-1/scripts/stage1/s5_provenance.py
python Architecture-1/scripts/stage1/s6_report.py
```

Probes are cached in `_cache/audio_probe.csv`, so re-runs skip completed work.
Nothing in this stage mutates `metadata.db`, the split parquet, or any pipeline code.
