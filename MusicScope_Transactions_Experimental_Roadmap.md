# MusicScope Transactions Research Plan
## Sequential Experimental Instruction Set for Local LLM

**Project:** MusicScope-CL — AI-Generated Music Detection  
**Goal:** Convert the existing MIREX/ISMIR work into a Transactions-level study centered on **generalization, invariance, interpretability, and evidence complementarity**, rather than only a high headline AUROC.

---

# 0. Core Research Thesis

The paper should not be framed primarily as:

> “MusicScope combines acoustic and structural features and achieves 0.997 AUROC.”

The stronger scientific framing is:

> **AI-music detectors rely on multiple classes of evidence with different generalization properties. Acoustic synthesis cues and musical-structural cues fail differently under generator, dataset, and channel shifts. MusicScope is a cross-scale detector designed to exploit this complementarity.**

The study should test four primary claims:

1. **Different evidence generalizes differently.**  
   Acoustic texture and musical-structural evidence should exhibit different degradation patterns under unseen generators, unseen datasets, and post-processing.

2. **The two evidence streams are complementary.**  
   Fusion should improve performance because the branches make meaningfully different errors, not merely because the model has more parameters.

3. **The structural branch should capture musical organization rather than dataset shortcuts.**  
   It must remain informative after controlling for dataset source, codec, sample rate, loudness, genre, duration, and mastering.

4. **Long-range ordering may carry additional forensic information.**  
   If the current 128-D CNE only captures global statistics, an order-aware variant should be tested to determine whether musical development itself contains useful information.

---

# 1. Existing System to Preserve

Treat the current MusicScope-CL implementation as **MusicScope-v1**.

Current architecture:

- **ATE / Acoustic Texture Encoder**
  - 128-band log-Mel spectrogram
  - MobileNetV3-Small
  - SimCLR pretraining
  - 576-D embedding

- **CNE / Compositional Narrative Extractor**
  - deterministic 128-D MIR descriptor
  - harmonic entropy/tension
  - rhythmic microtiming
  - motif/self-similarity features
  - spectral density/white-space features
  - dynamic-envelope statistics

- **Fusion**
  - 576-D ATE + 128-D CNE = 704-D
  - projection to 128-D normalized space
  - SupCon
  - MLP classifier
  - temperature scaling

Important existing pilot results:

- Held-out Macro-AUROC: **0.9970**
- Held-out EER: **1.55%**
- Reported zero-shot Macro-AUROC: **0.9933**
- ATE-only:
  - held-out AUROC: **0.9482**
  - zero-shot AUROC: **0.8841**
- CNE-only:
  - held-out AUROC: **0.9105**
  - zero-shot AUROC: **0.8920**
- Naive fusion:
  - held-out AUROC: **0.9654**
  - zero-shot AUROC: **0.9215**
- Current full MusicScope:
  - held-out AUROC: **0.9970**
  - zero-shot AUROC: **0.9933**
- Mubert thresholded recall: **11.41%**
- Echoes thresholded recall: **71.49%**

These values are **pilot evidence** until the audited experimental pipeline reproduces them.

---

# 2. Operating Rules for the Local LLM

## 2.1 Do not silently change the science

Do not:

- alter train/test splits without recording the change;
- replace generator labels without tracing their provenance;
- tune thresholds on final test sets;
- add datasets into training after they have been declared external test sets;
- report only the best random seed;
- discard failed runs;
- remove difficult generators to improve headline metrics;
- change model architecture before completing the initial audit and reproduction stages.

Every change must be recorded.

## 2.2 Every experiment must generate an artifact package

Every experiment folder should contain, where applicable:

```text
experiment_name/
├── config.yaml
├── manifest.parquet
├── run.sh
├── train.py
├── evaluate.py
├── checkpoints/
├── predictions/
│   ├── track_scores.parquet
│   └── generator_scores.parquet
├── metrics/
│   ├── summary.json
│   ├── per_generator.csv
│   └── confidence_intervals.csv
├── figures/
├── logs/
└── README.md
```

The experiment `README.md` must contain:

- objective;
- hypothesis;
- exact datasets/splits;
- exact checkpoint;
- seed;
- model version;
- hardware;
- runtime;
- metrics;
- unexpected observations;
- whether the hypothesis was supported;
- next recommended action.

## 2.3 Reproducibility

Unless explicitly stated otherwise:

- use at least **5 random seeds** for trainable models;
- save the seed in every config;
- save final per-track scores, not only aggregate metrics;
- preserve the exact commit hash;
- preserve the environment;
- never overwrite previous results.

Recommended naming:

```text
YYYYMMDD_expID_model_dataset_seed/
```

Example:

```text
20261005_E2_ATE_LOGO_MusicGen_seed03/
```

---

# 3. Canonical Evaluation Metrics

Report the following wherever meaningful:

### Primary

- AUROC
- AUPRC
- EER
- TPR @ 1% FPR

### Secondary

- FPR @ 95% TPR
- balanced accuracy
- precision
- recall
- F1

### Calibration

- NLL
- Brier score
- ECE

### OOD degradation

For each condition:

```text
ΔOOD = AUROC_ID - AUROC_OOD
```

### Statistics

Use paired track-level bootstrap intervals where applicable.

Recommended:

```text
10,000 bootstrap resamples
95% confidence intervals
```

For model training:

```text
>= 5 seeds
mean ± std
```

Do not use a single best seed as the main result.

---

# 4. Sequential Experimental Plan

---

# STAGE 0 — Freeze the Existing System

## Objective

Preserve the exact MusicScope-v1 system and all currently reported results before modifying anything.

## Tasks

1. Identify the exact code used for:
   - ATE training;
   - CNE extraction;
   - fusion training;
   - SupCon training;
   - classifier training;
   - calibration;
   - held-out inference;
   - zero-shot inference.

2. Identify all checkpoints.

3. Identify all dataset manifests.

4. Copy the existing output tables and figures.

5. Record:
   - Git commit;
   - Python environment;
   - PyTorch version;
   - CUDA version;
   - GPU;
   - preprocessing settings;
   - random seeds if known.

6. Tag this implementation:

```text
musicscope_v1_frozen
```

## Deliverables

```text
artifacts/stage0_freeze/
├── CODE_VERSION.md
├── ENVIRONMENT.txt
├── checkpoint_inventory.csv
├── dataset_inventory.csv
├── existing_results.csv
├── existing_figures/
└── README.md
```

## Exit Condition

Proceed only when the existing system can be reconstructed without relying on undocumented state.

---

# STAGE 1 — Dataset and Provenance Audit

## Objective

Establish a trustworthy file-level dataset registry and resolve every generator/dataset inconsistency before any major result is trusted.

## Critical Issues to Resolve

Explicitly verify:

- SONICS generator/version labels;
- whether any row labelled “Suno v5” actually comes from SONICS;
- FakeMusicCaps generator composition;
- whether “FMC Udio” is a custom extension or a labeling error;
- Stable Audio occurrences in development and reported zero-shot partitions;
- whether the eight “unseen” generators were truly unseen at model/version level.

## Tasks

Create one canonical row per physical audio file.

Required fields:

```text
track_id
absolute_or_project_path
sha256
chromaprint
dataset
dataset_version
source_platform
generator_family
generator_name
generator_version
checkpoint_if_known
human_or_synthetic
fully_or_partially_synthetic
artist
genre
prompt_id
prompt_text_if_available
duration
original_sample_rate
original_codec
original_bitrate
channels
loudness_lufs
silence_ratio
split
used_for_pretraining
used_for_training
used_for_validation
used_for_calibration
used_for_test
notes
```

Perform:

- SHA256 duplicate detection;
- Chromaprint duplicate/near-duplicate detection;
- artist overlap analysis;
- prompt overlap analysis;
- generator/version overlap analysis;
- dataset overlap analysis.

## Deliverables

```text
artifacts/stage1_audit/
├── canonical_manifest.parquet
├── canonical_manifest.tsv
├── generator_registry.csv
├── split_registry.csv
├── duplicate_report.csv
├── near_duplicate_report.csv
├── artist_overlap_report.csv
├── prompt_overlap_report.csv
├── provenance_issues.md
├── resolved_label_changes.csv
└── README.md
```

Also create the paper-ready matrix:

```text
Generator × Version × Pretrain × Train × Validation × Calibration × Test
```

## Exit Condition

Proceed only when:

- every evaluated file has known provenance;
- exact duplicates across train/test are zero;
- ambiguous generator labels are resolved;
- “unseen generator” claims can be stated precisely.

---

# STAGE 2 — Reproduce MusicScope-v1

## Objective

Determine whether the existing headline results are reproducible under the audited manifest.

## Models

Run:

1. supervised MobileNetV3;
2. ATE-SimCLR;
3. CNE-only;
4. ATE+CNE naive BCE fusion;
5. MusicScope-v1 SupCon fusion;
6. calibrated MusicScope-v1.

## Protocol

Use the audited equivalent of the original held-out split.

Run trainable models over at least:

```text
seeds = [0, 1, 2, 3, 4]
```

Do not tune architecture.

## Deliverables

```text
artifacts/stage2_reproduction/
├── heldout_results_by_seed.csv
├── aggregate_results.csv
├── per_generator_results.csv
├── per_track_predictions.parquet
├── reproduction_delta_vs_original.csv
├── configs/
├── checkpoints/
├── figures/
└── README.md
```

Required figure:

```text
Original reported metric vs reproduced metric
```

Required table:

```text
Model | AUROC | AUPRC | EER | TPR@1%FPR | mean ± std
```

## Exit Condition

Classify the reproduction outcome:

```text
PASS
MINOR_DEVIATION
MAJOR_DEVIATION
```

If major deviation occurs, stop and diagnose before continuing.

---

# STAGE 3 — Shortcut and Dataset-Bias Audit

## Objective

Determine whether the detector can exploit source metadata or preprocessing shortcuts.

## Experiment 3A — Metadata-Only Classifier

Train a simple classifier using only:

- original sample rate;
- codec;
- bitrate;
- duration;
- channels;
- LUFS;
- silence ratio;
- file size;
- dataset identity if testing a diagnostic upper bound.

Try:

- logistic regression;
- gradient boosted trees.

Do not use audio.

### Deliverable

```text
metadata_only_results.csv
metadata_feature_importance.csv
```

## Experiment 3B — Source Prediction

Predict dataset identity from simple metadata/audio summary features.

High source-prediction accuracy indicates strong dataset fingerprints.

### Deliverable

```text
dataset_source_prediction.csv
```

## Experiment 3C — Uniform Transcoding Control

Convert a controlled evaluation subset to:

```text
same PCM format
same sample rate
same bit depth
same channel count
same codec/container
```

Evaluate MusicScope-v1 before and after normalization.

### Deliverable

```text
canonical_transcoding_delta.csv
```

## Experiment 3D — Multi-Source Human Test

Evaluate false-positive rate independently on human music from multiple datasets.

Do not allow:

```text
human ≡ FMA
```

### Deliverable

```text
human_source_fpr.csv
```

## Exit Condition

Create:

```text
SHORTCUT_AUDIT_VERDICT.md
```

The verdict must explicitly answer:

1. Can metadata alone detect AI vs human?
2. Does uniform transcoding significantly reduce performance?
3. Is FPR concentrated in one human dataset?
4. Is any current headline result likely inflated by source shortcuts?

If severe shortcut evidence appears, rebuild/rebalance the development corpus before moving forward.

---

# STAGE 4 — Controlled Fusion and SupCon Study

## Objective

Determine whether the large gain from SupCon is truly due to contrastive metric learning.

## Models

Keep the projection architecture parameter-matched.

Compare:

```text
ATE + CNE + BCE
ATE + CNE + SupCon
ATE + CNE + Triplet Loss
ATE + CNE + cosine-margin / ArcFace-style objective
```

Same:

- input features;
- projection dimensions;
- optimizer;
- training epochs;
- seeds;
- train/validation sets.

## Deliverables

```text
artifacts/stage4_metric_learning/
├── metric_learning_comparison.csv
├── learning_curves.csv
├── seed_variance.csv
├── latent_separation_metrics.csv
├── per_track_predictions/
└── README.md
```

## Required Analysis

Answer:

> Is the improvement caused specifically by SupCon, by metric learning generally, or merely by the extra projection network?

## Exit Condition

Select one fusion objective for all subsequent experiments.

Do not select based only on best held-out AUROC. Include OOD evidence once available.

---

# STAGE 5 — Strong Baseline Suite

## Objective

Establish a credible Transactions-level baseline set.

## Target Baselines

Attempt to reproduce or use official checkpoints for:

1. ResNet-18
2. supervised MobileNetV3
3. CLAP + SVM/MLP
4. SpecTTTra variants where possible
5. MERT-based Segment Transformer
6. Fusion Segment Transformer
7. CLAM
8. MusicDET
9. Afchar et al. decoder-based detector
10. Fourier/interpretable artifact detector
11. ATE
12. CNE
13. naive fusion
14. MusicScope-v1

## Two Evaluation Modes

### Mode A — Native

Use official pretrained checkpoints and preprocessing where available.

Purpose:

```text
How do existing deployable systems behave on our evaluation sets?
```

### Mode B — Common Training

Retrain reproducible systems using the same training corpus.

Purpose:

```text
How do architectures compare under controlled data?
```

Never combine Mode A and Mode B into one leaderboard without labeling the distinction.

## Deliverables

```text
artifacts/stage5_baselines/
├── baseline_inventory.csv
├── baseline_reproduction_notes.md
├── native_protocol_results.csv
├── common_training_results.csv
├── unavailable_or_failed_baselines.md
├── checkpoints/
└── README.md
```

For unavailable systems, record the reason rather than silently excluding them.

---

# STAGE 6 — Leave-One-Generator-Out (LOGO)

## Objective

Measure true generator-level generalization.

## Protocol

For every validated generator `Gi`:

```text
train = all generators except Gi
test = Gi
```

Human training data remains consistent across folds.

Evaluate at minimum:

- ATE;
- CNE;
- naive fusion;
- selected MusicScope fusion;
- strongest feasible external baselines.

## Metrics

For every generator:

- AUROC;
- AUPRC;
- EER;
- TPR@1%FPR;
- ΔOOD.

## Deliverables

```text
artifacts/stage6_logo/
├── logo_full_results.csv
├── logo_summary.csv
├── logo_heatmap.pdf
├── logo_degradation.csv
├── worst_generator_analysis.md
├── predictions/
└── README.md
```

## Required Questions

1. Does CNE degrade less than ATE?
2. Which generators are hardest?
3. Which branch recovers errors made by the other?
4. Is the fusion gain consistent across generators?
5. Does any single generator dominate the aggregate result?

## Exit Condition

The paper's central generalization hypothesis should now either be:

```text
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED
```

Record the verdict before proceeding.

---

# STAGE 7 — Leave-One-Architecture-Family-Out

## Objective

Test generalization beyond individual generators.

## Grouping

Construct defensible generator families such as:

- autoregressive/discrete-token;
- latent diffusion;
- diffusion/flow variants;
- proprietary commercial systems;
- algorithmic/assembly systems.

Do not force uncertain models into incorrect categories.

Mubert should remain explicitly separated if its generation mechanism is fundamentally different.

## Protocol

Hold out one complete architecture family at a time.

## Deliverables

```text
artifacts/stage7_family_ood/
├── family_definition.md
├── family_holdout_results.csv
├── family_holdout_heatmap.pdf
├── family_degradation.csv
└── README.md
```

## Required Interpretation

Determine whether MusicScope learns:

```text
generator fingerprints
```

or something that transfers across synthesis paradigms.

---

# STAGE 8 — External Cross-Dataset Generalization

## Objective

Evaluate without model selection on independent corpora.

## Target External Sets

Where licensing and access permit:

- AIME
- SONICS
- FakeMusicCaps
- MoM / Melody or Machine
- any other independently curated dataset added later

At least two datasets must remain completely untouched during model development.

## Rules

No:

- threshold selection;
- early stopping;
- architecture tuning;
- feature tuning;

using the external test labels.

## Deliverables

```text
artifacts/stage8_external_ood/
├── external_dataset_registry.csv
├── external_results.csv
├── external_per_generator.csv
├── external_human_fpr.csv
├── cross_dataset_summary.csv
├── worst_domain_analysis.md
└── README.md
```

## Main Paper Table

```text
Method | Internal OOD | SONICS | FMC | AIME | MoM | Mean | Worst
```

The `Worst` column must be retained.

---

# STAGE 9 — Delivery-Chain Robustness

## Objective

Measure invariance to realistic post-processing.

## Perturbations

### Compression

```text
MP3: 64, 96, 128, 192, 320 kbps
AAC: 64, 96, 128, 192 kbps
Opus: 32, 64, 96, 128 kbps
```

### Resampling

```text
16 kHz
22.05 kHz
24 kHz
32 kHz
44.1 kHz
48 kHz
```

### Spectral

- low-pass;
- high-pass;
- parametric EQ.

### Production

- loudness normalization;
- dynamic-range compression;
- limiting;
- mild saturation.

### Musical

Pitch shifts:

```text
-2, -1, +1, +2 semitones
```

Tempo changes:

```text
0.90, 0.95, 1.05, 1.10
```

### Environmental

- mild additive noise;
- reverberation;
- optional playback/recording simulation.

## Models

Evaluate separately:

- ATE;
- CNE;
- fusion;
- strongest external baselines.

## Deliverables

```text
artifacts/stage9_robustness/
├── perturbation_manifest.parquet
├── robustness_all_results.csv
├── robustness_delta.csv
├── ate_robustness_curves.pdf
├── cne_robustness_curves.pdf
├── fusion_robustness_curves.pdf
├── branch_comparison.pdf
└── README.md
```

## Key Quantity

```text
ΔAUROC = AUROC_clean - AUROC_perturbed
```

## Core Question

Does:

```text
CNE degrade less than ATE under surface transformations?
```

This is one of the most important tests in the project.

---

# STAGE 10 — Controlled Structural Intervention Study

## Objective

Directly test whether CNE responds to musical organization rather than only surface texture.

This is a central Transactions experiment.

## Experiment 10A — Surface Intervention / Structure Preserved

Apply:

- compression;
- EQ;
- filtering;
- resampling;
- loudness changes.

Measure score changes:

```text
Δscore_ATE
Δscore_CNE
Δscore_fusion
```

Expected hypothesis:

```text
|ΔATE| > |ΔCNE|
```

## Experiment 10B — Section Permutation

Segment tracks into musical sections where possible.

Example:

```text
S1 S2 S3 S4 S5
```

becomes:

```text
S1 S4 S2 S5 S3
```

Use crossfades to avoid trivial discontinuity artifacts.

## Experiment 10C — Section Repetition

Example:

```text
S1 S2 S3 S4
```

becomes:

```text
S1 S2 S2 S2 S4
```

## Experiment 10D — Motif Duplication

Artificially increase recurrence/self-similarity while preserving timbre.

## Experiment 10E — Dynamic Flattening

Reduce macro-level dynamic evolution while retaining local acoustic identity as much as possible.

## Experiment 10F — Timing Quantization

Where technically valid, reduce expressive onset deviations without changing instrumentation.

## Deliverables

```text
artifacts/stage10_interventions/
├── intervention_manifest.parquet
├── intervention_results.csv
├── intervention_score_deltas.csv
├── ate_vs_cne_intervention_matrix.pdf
├── example_audio_registry.csv
├── methodology.md
└── README.md
```

## Desired Paper Figure

A matrix conceptually similar to:

```text
                       ATE         CNE
MP3                    ↓↓          ~
EQ                     ↓           ~
Resampling             ↓           ~
Section shuffle        ~           ↓
Motif repetition       ~           ↓
Dynamic flattening     ~           ↓
```

The actual figure must reflect measured results, not expected results.

---

# STAGE 11 — CNE Feature-Family Dissection

## Objective

Identify which musical feature families provide useful and transferable evidence.

## Groups

1. harmonic entropy/tension;
2. rhythm/microtiming;
3. motivic recurrence / SSM;
4. spectral density;
5. dynamics.

## Experiments

For every group `g`:

### Standalone

```text
CNE_g only
```

### Leave-one-group-out

```text
CNE_full - g
```

Compute:

```text
Δg = AUROC_full - AUROC_without_g
```

Run both:

- held-out;
- LOGO;
- external OOD;
- perturbation subset.

## Deliverables

```text
artifacts/stage11_cne_ablation/
├── standalone_group_results.csv
├── leave_one_group_out.csv
├── per_generator_group_importance.csv
├── ood_group_importance.csv
├── feature_group_heatmap.pdf
└── README.md
```

## Required Interpretation

Do not make one global statement such as:

```text
motivic recurrence is most important
```

unless it is consistent.

Instead identify whether different generators rely on different structural cues.

---

# STAGE 12 — Representation Probing and Complementarity

## Objective

Measure whether ATE and CNE encode different information.

## Experiment 12A — Authenticity Probe

Train a lightweight probe:

```text
z -> human / AI
```

for:

- ATE embedding;
- CNE representation;
- fused representation.

## Experiment 12B — Generator-ID Probe

Train:

```text
z -> generator identity
```

for:

- ATE;
- CNE;
- fused representation.

Key hypothesis:

```text
ATE may contain stronger generator-ID information.
CNE may contain less generator-ID information while remaining useful for authenticity.
```

## Experiment 12C — Error Complementarity

Calculate:

```text
corr(score_ATE, score_CNE)
P(E_ATE ∩ E_CNE)
P(CNE correct | ATE wrong)
P(ATE correct | CNE wrong)
branch disagreement rate
```

Also compute these values per generator.

## Deliverables

```text
artifacts/stage12_probes/
├── authenticity_probe.csv
├── generator_probe.csv
├── branch_score_correlation.csv
├── branch_error_overlap.csv
├── conditional_recovery.csv
├── probe_visualizations/
└── README.md
```

## Exit Condition

The term **complementary** may only be used strongly in the final paper if this stage supports it quantitatively.

---

# STAGE 13 — Temporal Horizon Study

## Objective

Determine whether long-range context materially improves structural detection.

## Context Durations

Evaluate:

```text
5 s
10 s
15 s
30 s
60 s
120 s
full track
```

Use consistent sampling where full-track duration differs.

Evaluate separately:

- ATE;
- CNE;
- fusion.

## Deliverables

```text
artifacts/stage13_context/
├── duration_results.csv
├── per_generator_duration.csv
├── duration_curves.pdf
├── duration_degradation.csv
└── README.md
```

## Key Question

Does CNE benefit disproportionately from increasing context?

---

# STAGE 14 — Order-Aware CNE-v2

## Important Gate

Do not start this stage until Stages 0–13 are complete enough to justify architectural modification.

## Objective

Determine whether actual temporal ordering provides information beyond global CNE statistics.

## Representation

Extract local/section-level structural vectors:

```text
v1, v2, ..., vN
```

instead of immediately collapsing the full track to one 128-D vector.

## Baselines

Compare:

```text
CNE-global       current v1
CNE-mean         mean of local vectors
CNE-TCN          lightweight temporal convolution
CNE-Transformer  lightweight temporal transformer
```

Do not build an excessively large model.

The purpose is to test ordering, not win by parameter count.

## Experiment 14A — Ordered vs Shuffled

Original:

```text
v1 v2 v3 ... vN
```

Shuffled:

```text
vπ(1) vπ(2) ... vπ(N)
```

Evaluate whether order-aware CNE performance degrades under shuffling.

## Deliverables

```text
artifacts/stage14_cnev2/
├── cnev2_architectures.md
├── ordered_results.csv
├── shuffled_results.csv
├── order_sensitivity.csv
├── parameter_counts.csv
├── checkpoints/
└── README.md
```

## Decision Rule

If temporal ordering contributes meaningfully:

```text
promote CNE-v2 to final MusicScope architecture
```

Otherwise:

```text
retain interpretable CNE-v1
```

Do not force CNE-v2 into the paper if the evidence does not justify it.

---

# STAGE 15 — Fixed-Time vs Musical Segmentation

## Objective

Determine whether musically meaningful segmentation is superior to arbitrary time windows.

## Compare

```text
fixed-duration windows
beat-aligned windows
bar-aligned windows
```

Possible bar lengths:

```text
4 bars
8 bars
16 bars
```

Use the same downstream structural model.

## Deliverables

```text
artifacts/stage15_segmentation/
├── segmentation_results.csv
├── segmentation_cost.csv
├── segmentation_failure_cases.md
├── segmentation_comparison.pdf
└── README.md
```

## Decision

Select segmentation based on:

- OOD performance;
- robustness;
- computational cost;
- failure rate of beat/downbeat tracking.

---

# STAGE 16 — Human Genre and Edge-Case Study

## Objective

Understand human false positives.

## Human Strata

Where sufficient metadata exists:

- ambient;
- drone;
- lo-fi;
- minimal techno;
- EDM;
- classical;
- jazz;
- metal;
- acoustic;
- experimental;
- other major categories.

## Metrics

Report:

```text
FPR per genre
mean AI score per genre
95% CI
```

Investigate whether CNE disproportionately flags:

```text
repetitive / low-dynamic / minimalist human music
```

## Optional Follow-Up

Test genre-conditioned normalization.

Do not use genre conditioning in the final system unless it clearly improves generalization without introducing new fragility.

## Deliverables

```text
artifacts/stage16_human_edges/
├── genre_fpr.csv
├── false_positive_catalog.csv
├── false_positive_feature_analysis.csv
├── genre_conditioning_results.csv
└── README.md
```

---

# STAGE 17 — Hybrid Provenance and Mubert Study

## Objective

Clarify what task MusicScope is actually solving.

Distinguish:

```text
Human
Neural synthesis
Algorithmic assembly
Hybrid / AI-assisted
```

## Experiment 17A — Mubert

Treat Mubert explicitly as a separate provenance mechanism.

Analyze:

- ATE scores;
- CNE scores;
- fused scores;
- structural feature distributions.

## Experiment 17B — Controlled Assembly

Create synthetic assembly controls using real human sections/stems:

```text
human sections
+ algorithmic repetition/reordering
+ crossfades
```

This isolates machine-level organization from neural synthesis artifacts.

## Experiment 17C — Synthetic-Fraction Sweep

Where technically and scientifically valid, construct:

```text
0%
25%
50%
75%
100%
```

synthetic-content tracks.

Measure model score against synthetic fraction.

## Deliverables

```text
artifacts/stage17_hybrid/
├── mubert_analysis.csv
├── assembly_control_results.csv
├── synthetic_fraction_results.csv
├── provenance_taxonomy.md
└── README.md
```

## Required Outcome

The final paper must explicitly state what the detector is designed to identify.

Do not describe all algorithmically produced music as equivalent to neural synthesis.

---

# STAGE 18 — Label-Noise Robustness

## Optional / Conditional Stage

Only execute if the final paper intends to claim robustness to label contamination.

## Noise Rates

Inject symmetric corruption:

```text
0%
5%
10%
20%
30%
```

Compare:

- fully supervised MobileNet;
- SimCLR ATE + classifier;
- MusicScope.

## Deliverables

```text
artifacts/stage18_label_noise/
├── label_noise_results.csv
├── label_noise_curves.pdf
└── README.md
```

## Decision

If robustness is weak:

```text
remove strong label-noise-immunity claims
```

Do not retain the phrase “mathematical immunity.”

---

# STAGE 19 — Calibration Under Domain Shift

## Objective

Measure whether probabilities remain trustworthy outside the calibration domain.

## Calibration Methods

Compare:

```text
none
temperature scaling
Platt scaling
optional isotonic regression
```

Use separate:

```text
train
validation
calibration
test
```

partitions.

## Metrics

- NLL;
- Brier;
- ECE;
- reliability curves.

Evaluate on:

- held-out;
- generator-OOD;
- external OOD;
- perturbed audio.

## Deliverables

```text
artifacts/stage19_calibration/
├── calibration_results.csv
├── ood_calibration.csv
├── reliability_diagrams/
├── selected_calibration_method.md
└── README.md
```

---

# STAGE 20 — Efficiency and Deployment Cost

## Objective

Quantify whether MusicScope provides meaningful accuracy-per-compute advantages.

## Report

For every serious baseline:

```text
parameter count
FLOPs
peak GPU memory
peak CPU memory
feature extraction time
GPU inference time
CPU inference time
real-time factor
tracks/minute
```

Evaluate at:

```text
5 s
30 s
120 s
full-track where feasible
```

Use the same hardware and batch conditions.

## Deliverables

```text
artifacts/stage20_efficiency/
├── efficiency_results.csv
├── hardware_spec.md
├── latency_distribution.csv
├── accuracy_compute_tradeoff.pdf
└── README.md
```

---

# STAGE 21 — Final Statistical Consolidation

## Objective

Produce one canonical set of paper-ready numbers from all experiments.

## Tasks

1. Collect final per-track predictions.
2. Verify no test-set threshold tuning.
3. Calculate bootstrap confidence intervals.
4. Calculate seed variation.
5. Recompute all paper tables from raw predictions.
6. Ensure every number is machine-generated from saved outputs.
7. Never manually type final result numbers into tables if avoidable.

## Deliverables

```text
artifacts/stage21_statistics/
├── final_metrics.parquet
├── final_metrics.csv
├── bootstrap_ci.csv
├── seed_statistics.csv
├── final_per_generator.csv
├── final_per_dataset.csv
├── final_per_perturbation.csv
└── README.md
```

---

# STAGE 22 — Final Figure and Table Package

## Objective

Prepare the complete scientific evidence package before writing the paper.

## Required Main Figures

### Figure 1
Final architecture.

### Figure 2
Leave-One-Generator-Out heatmap.

### Figure 3
ATE vs CNE robustness curves.

### Figure 4
Controlled intervention matrix.

### Figure 5
Representation probing / generator-ID information.

### Figure 6
Context-duration curves.

### Figure 7
Optional ordered-vs-shuffled CNE-v2 result.

## Required Main Tables

### Table I
Dataset and provenance registry.

### Table II
Main baseline comparison.

### Table III
External cross-dataset generalization.

### Table IV
CNE feature-group ablation.

### Table V
Calibration / low-FPR operating metrics.

### Table VI
Efficiency comparison.

## Deliverables

```text
artifacts/stage22_paper_package/
├── figures/
├── tables/
├── captions.md
├── figure_source_data/
├── table_source_data/
└── README.md
```

Every figure/table must have machine-readable source data.

---

# 5. Mandatory vs Optional Priority

## Tier A — Mandatory for the Transactions Paper

Do not skip:

```text
Stage 0   Freeze
Stage 1   Provenance audit
Stage 2   Reproduction
Stage 3   Shortcut audit
Stage 4   Controlled SupCon/fusion study
Stage 5   Strong baseline suite
Stage 6   LOGO
Stage 8   External OOD
Stage 9   Delivery-chain robustness
Stage 10  Controlled interventions
Stage 11  CNE feature dissection
Stage 12  Complementarity/probes
Stage 13  Temporal horizon
Stage 20  Efficiency
Stage 21  Statistics
Stage 22  Paper package
```

## Tier B — High-Value Extension

Run if resources permit:

```text
Stage 7   Architecture-family holdout
Stage 14  Order-aware CNE-v2
Stage 15  Musical segmentation
Stage 16  Genre edge cases
Stage 17  Hybrid provenance/Mubert
Stage 19  OOD calibration
```

## Tier C — Conditional

Only if the corresponding claim is retained:

```text
Stage 18  Label-noise study
```

---

# 6. Decision Gates

## Gate A — After Shortcut Audit

If performance collapses after canonical transcoding or metadata-only classification is highly predictive:

```text
STOP
repair dataset construction
repeat Stage 2
```

## Gate B — After LOGO

If CNE does not generalize better or differently than ATE:

Do not force the current “structural invariance” narrative.

Instead investigate:

- whether structural descriptors are too weak;
- whether CNE is dominated by spectral/dynamic features;
- whether temporal ordering is missing;
- whether genre confounding is responsible.

## Gate C — After Controlled Interventions

If CNE does not respond selectively to structural manipulation:

Do not call it a strong “musical narrative” representation.

Reframe it as:

```text
interpretable MIR statistical representation
```

and consider CNE-v2.

## Gate D — Before CNE-v2

Only build CNE-v2 after determining exactly what is missing from CNE-v1.

Do not add a Transformer merely because Transformers are fashionable.

---

# 7. Minimal Final Scientific Success Criteria

The project should be considered Transactions-ready only if most of the following are demonstrated:

1. The main result survives provenance and shortcut auditing.
2. MusicScope is competitive with strong modern baselines.
3. Fusion improves generator-OOD performance consistently.
4. ATE and CNE exhibit measurably different failure behavior.
5. CNE remains useful under surface/channel perturbations.
6. External-dataset generalization remains strong.
7. Performance is not driven primarily by file/sample-rate/codec shortcuts.
8. At least one interpretable structural feature family contributes robust OOD evidence.
9. Low-FPR performance is competitive.
10. Efficiency is meaningfully better than large foundation-model systems, if that claim is retained.
11. All headline results have uncertainty estimates.
12. Every final number can be regenerated directly from saved per-track predictions.

---

# 8. Desired Final Paper Narrative

Do not write the final paper as:

> “We propose another binary AI-music classifier.”

The intended narrative is:

> Modern AI-music detectors exploit evidence at multiple temporal and representational scales. Low-level acoustic evidence is powerful but may be generator- and channel-sensitive, while higher-level musical evidence can provide complementary information under synthesis and domain shift. We systematically characterize these evidence classes through generator holdouts, cross-dataset evaluation, controlled acoustic perturbations, structural interventions, representation probing, and temporal-context experiments. Based on these findings, we develop MusicScope, a lightweight cross-scale detector that combines acoustic and interpretable musical-structural evidence.

The architecture must follow from the evidence.

The experiments are the primary contribution.

---

# 9. Immediate Next Action

Start only with:

```text
STAGE 0
STAGE 1
STAGE 2
```

Do not modify MusicScope-v1 yet.

The first milestone is:

```text
MILESTONE 1
===========
Frozen reproducible MusicScope-v1
+
audited canonical dataset manifest
+
five-seed reproduction of the current core table
```

Expected milestone deliverables:

```text
milestones/M1/
├── canonical_manifest.parquet
├── generator_registry.csv
├── split_registry.csv
├── provenance_issues.md
├── musicscope_v1_reproduction.csv
├── per_generator_results.csv
├── reproduction_report.md
└── NEXT_ACTIONS.md
```

`NEXT_ACTIONS.md` must explicitly state whether the original evidence survives audit and reproduction before Stage 3 begins.

---

# 10. Local LLM Reporting Format

At the end of every stage, respond using:

```text
STAGE:
STATUS: PASS / PARTIAL / FAIL

WHAT WAS DONE:
- ...

DELIVERABLES CREATED:
- path/to/file
- path/to/file

KEY RESULTS:
- ...

PROBLEMS FOUND:
- ...

SCIENTIFIC INTERPRETATION:
- ...

DOES THIS SUPPORT THE CURRENT HYPOTHESIS?
- Yes / Partially / No

NEXT RECOMMENDED ACTION:
- ...

BLOCKERS:
- ...
```

Do not proceed past a failed scientific gate without explicitly documenting why.

---

# End Goal

The desired end product is not merely a higher MusicScope AUROC.

The desired end product is a defensible scientific answer to:

> **Which properties of human and AI-generated music provide robust evidence of provenance, how do those properties fail under distribution shift, and how should a detector combine them to generalize beyond known generators?**

That is the standard against which every experiment in this roadmap should be evaluated.
