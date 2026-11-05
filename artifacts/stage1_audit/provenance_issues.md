# Stage 1 - Provenance Issues

Generated: 2026-10-20T18:05:35.769514+00:00
Manifest: `artifacts/stage1_audit/canonical_manifest.parquet` (221,846 rows, 187,557 audio)

## Summary

| ID | Severity | Issue | Status |
|---|---|---|---|
| [Q3](#q3) | **CRITICAL** | FakeMusicCaps generator composition | RESOLVED_CORRECTION_PROPOSED |
| [Q4](#q4) | **CRITICAL** | Meaning of the 'FMC Udio' stratum | RESOLVED_CORRECTION_PROPOSED |
| [Q5](#q5) | **CRITICAL** | Stable Audio appears in both development and reported zero-shot sets | RESOLVED_CLAIM_INVALID |
| [Q6](#q6) | **CRITICAL** | Phase-1 'unseen generators' were present in training | RESOLVED_CLAIM_INVALID |
| [Q6b](#q6b) | **CRITICAL** | Echoes is a mixed-generator collection treated as one class | RESOLVED_CORRECTION_PROPOSED |
| [Q2](#q2) | **HIGH** | Apparent 'Suno v5' provenance | RESOLVED_NO_SUCH_ROWS |
| [Q3b](#q3b) | **HIGH** | AppleDouble resource forks registered as audio | RESOLVED_EXCLUDED |
| [Q7](#q7) | **HIGH** | AIME registered as 6,500 virtual parquet rows, never extracted | RESOLVED_BLOCKER_DOCUMENTED |
| [Q1](#q1) | **MEDIUM** | SONICS generator and version labels | RESOLVED_IN_MANIFEST |
| [Q8](#q8) | **MEDIUM** | 34,289 audio files are in no split | OPEN |

---

## Q3

**FakeMusicCaps generator composition**

- Severity: **CRITICAL**
- Roadmap question: _FakeMusicCaps generator composition_
- Status: `RESOLVED_CORRECTION_PROPOSED`

**Evidence**

```
on-disk directories -> derived: {'musicgen': 5521, 'audioldm': 5521, 'musicldm': 5521, 'mustango': 5521, 'stableaudio': 5521}; metadata.db labels: {'musicgen': 5521, 'udio': 11042, 'audioldm': 5521, 'mustango': 5521}
```

**Impact**

FMC ships six directories (audioldm2, MusicGen_medium, musicldm, mustango, stable_audio_open, __MACOSX) but metadata.db records only four generators. 'musicldm' and 'stable_audio_open' (11,042 files) were BOTH labelled 'udio'. Those tracks are MusicLDM and Stable Audio, not Udio.

**Resolution**

The ai_fmc_udio stratum is a mislabelled mixture of MusicLDM and Stable Audio. This also explains why Stable Audio (Echoes) recall is the weakest non-Mubert result: it was diluted 50/50 with MusicLDM under a wrong label.

---

## Q4

**Meaning of the 'FMC Udio' stratum**

- Severity: **CRITICAL**
- Roadmap question: _whether 'FMC Udio' is a custom extension or a labeling error_
- Status: `RESOLVED_CORRECTION_PROPOSED`

**Evidence**

```
files labelled udio under fakemusiccaps: 11,042; their true on-disk generators: {'audioldm': 5521, 'stableaudio': 5521}
```

**Impact**

It is a LABELLING ERROR, not a custom extension. No Udio tracks exist in FakeMusicCaps; the stratum is MusicLDM + Stable Audio mislabelled as udio.

**Resolution**

Relabel to musicldm / stableaudio. Any per-generator result attributed to 'FMC Udio' is actually reporting MusicLDM and Stable Audio.

---

## Q5

**Stable Audio appears in both development and reported zero-shot sets**

- Severity: **CRITICAL**
- Roadmap question: _Stable Audio occurrences in development and reported zero-shot partitions_
- Status: `RESOLVED_CLAIM_INVALID`

**Evidence**

```
stableaudio files by split: {'val': 877, 'train': 4838}; by dataset: {'echoes': 194, 'fakemusiccaps': 5521}
```

**Impact**

Stable Audio appears in the TRAINING split (via Echoes TTA and via the mislabelled FMC stratum) AND in the reported Phase-1 'unseen generators' set. It was therefore not unseen.

**Resolution**

Remove Stable Audio from any unseen-generator claim.

---

## Q6

**Phase-1 'unseen generators' were present in training**

- Severity: **CRITICAL**
- Roadmap question: _whether the eight 'unseen' generators were truly unseen at model/version level_
- Status: `RESOLVED_CLAIM_INVALID`

**Evidence**

```
unseen-claim generators: ['acestep', 'brev', 'diffrhythm', 'elevenlabs', 'mubert', 'producer', 'songgen', 'stableaudio']; of these present in train: ['acestep', 'brev', 'diffrhythm', 'elevenlabs', 'mubert', 'producer', 'songgen', 'stableaudio']
```

**Impact**

All eight so-called unseen generators are present in the TRAIN split. The Phase-1 'zero-shot' evaluation re-scored training data. Its 0.9933 AUROC is a training-set measurement, not a generalization measurement.

**Resolution**

The Phase-1 zero-shot claim must be WITHDRAWN. A genuine zero-shot set requires generators absent from all training partitions.

---

## Q6b

**Echoes is a mixed-generator collection treated as one class**

- Severity: **CRITICAL**
- Roadmap question: _dataset composition of the reported zero-shot set_
- Status: `RESOLVED_CORRECTION_PROPOSED`

**Evidence**

```
Echoes files in train split by true generator: {'audioldm': 502, 'diffrhythm': 530, 'musicgen': 481, 'producer': 247, 'songgen': 473, 'acestep': 257, 'brev': 262, 'elevenlabs': 248, 'mubert': 123, 'stableaudio': 164, 'suno': 249, 'udio': 260}
```

**Impact**

metadata.db labels every Echoes row as generator_family='echoes', hiding 12 distinct generators across TTA (text-to-audio) and ATA (audio-to-audio) collections. The Phase-1 zero-shot set is drawn from this same collection.

**Resolution**

Split Echoes by generator and by TTA/ATA modality before any holdout design.

---

## Q2

**Apparent 'Suno v5' provenance**

- Severity: **HIGH**
- Roadmap question: _whether any row labelled 'Suno v5' actually comes from SONICS_
- Status: `RESOLVED_NO_SUCH_ROWS`

**Evidence**

```
rows with any 'v5' version string: 0; all distinct version strings in the corpus: ['chirp-v2-xxl-alpha', 'chirp-v3', 'chirp-v3.5', 'udio-120s', 'udio-30s']
```

**Impact**

There is NO 'Suno v5' anywhere in the corpus. The newest SONICS build is chirp-v3.5. Any external claim of 'Suno v5' coverage is unsupported.

**Resolution**

Report the actual maximum version (chirp-v3.5) or drop the version claim.

---

## Q3b

**AppleDouble resource forks registered as audio**

- Severity: **HIGH**
- Roadmap question: _corrupt/non-audio rows in the dataset_
- Status: `RESOLVED_EXCLUDED`

**Evidence**

```
metadata.db rows with generator_family='__macosx': 27,605
```

**Impact**

__MACOSX/._* AppleDouble resource forks are not audio. They were registered with is_ai=1, inflating the apparent FakeMusicCaps corpus.

**Resolution**

Marked is_audio_file=0 in the manifest; excluded from all analyses.

---

## Q7

**AIME registered as 6,500 virtual parquet rows, never extracted**

- Severity: **HIGH**
- Roadmap question: _whether the designated external OOD set is actually available_
- Status: `RESOLVED_BLOCKER_DOCUMENTED`

**Evidence**

```
virtual rows: 6,500; physical audio files under data/raw/aime: 0
```

**Impact**

metadata.db registers AIME as '<parquet>#row=N' references. No AIME audio was ever extracted to disk, so the designated Stage-8 external set is unusable without extraction. AIME also mislabels AudioLDM 2 and Stable Audio as family 'udio'.

**Resolution**

Extract AIME audio before using it, and correct its generator labels.

---

## Q1

**SONICS generator and version labels**

- Severity: **MEDIUM**
- Roadmap question: _SONICS generator/version labels_
- Status: `RESOLVED_IN_MANIFEST`

**Evidence**

```
metadata.db generator_version: {'chirp-v3.5': 19057, 'udio-120s': 18745, 'chirp-v2-xxl-alpha': 2084, 'udio-30s': 4903, 'chirp-v3': 4285}; sidecar algorithm: {}; split-level generator_family: {'suno': 25426, 'udio': 23648}
```

**Impact**

The split parquet collapses SONICS to only 'suno'/'udio', discarding the chirp-v2-xxl-alpha / chirp-v3 / chirp-v3.5 / udio-120s / udio-30s distinction that metadata.db does record.

**Resolution**

Version granularity is recoverable from metadata.db + sidecar CSVs; the authoritative split must carry it forward.

---

## Q8

**34,289 audio files are in no split**

- Severity: **MEDIUM**
- Roadmap question: _split assignment coverage_
- Status: `OPEN`

**Evidence**

```
audio rows with split='unassigned': 1
```

**Impact**

A large audio population exists outside both train and val. It is neither training data nor evaluation data, so its provenance is unverified.

**Resolution**

Assign explicitly or quarantine.

---

## Unresolved / Blocked

- Acoustic near-duplicate detection is a duration+size proxy; chromaprint and pyacoustid are not installed.
- MP3 bitrate is unpopulated; mutagen is not installed.
- `loudness_lufs` and `silence_ratio` are unpopulated; pyloudnorm is not installed.
- pretraining / calibration partition membership was never recorded by the original pipeline and cannot be reconstructed from current artifacts.
- `artist_id` is populated only for FMA human tracks; AI tracks have no artist identity, so cross-class artist overlap is structurally 0.
