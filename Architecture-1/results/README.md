# Architecture-1 Evaluation Results Directory

This directory organizes verified benchmark results for **MusicScope-CL (Architecture-1)**:

```
Architecture-1/results/
|--  README.md                      # Master directory guide & comparison
|--  old_model_5k_baseline/         # Initial 5,000-track prototype baseline results
\--  new_model_retrained_200k/      # Production 200,000-track model benchmark results
```

*(Note: Historical corrupted runs affected by the audio zero-padding bug have been purged from the repository).*

---

## High-Level Comparison Across Iterations

| Milestone | Model Iteration | Audio Pipeline | Calibration | Macro-AUROC | Overall Accuracy | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Old Model Baseline** | 5,000-track Prototype | Zero-padded 30s | Uncalibrated ($\tau = 0.50$) | `0.9782` | `92.62%` | Superseded |
| **Historical Buggy Run** | 200k Interim Model | Zero-padded 30s (Diluted RMS) | Uncalibrated ($\tau = 0.50$) | `0.9814` | `91.07%` (29 NaNs) | Purged |
| **New Model (Phase 1)** | 187k Retrained Production | Decoupled Unpadded | Calibrated ($\tau = 0.18$) | **`0.9933`** |  --  | **Production** |
| **New Model (Phase 2)** | 187k Retrained Production | Decoupled Unpadded | Calibrated ($\tau = 0.18$) | **`0.9970`** | **`98.48%`** | **Production** |
| **New Model (Phase 3)** | 187k Retrained Production | Decoupled Unpadded | Calibrated ($\tau = 0.18$) | *N/A (AI only)* | **`98.89%` AI Caught** | **Production** |

---

## Directory Index

### 1. [`old_model_5k_baseline/`](./old_model_5k_baseline/README.md)
Contains the benchmark results from the initial small-scale prototype trained on only $5,000$ tracks prior to the $200,000$-track scale-up.

### 2. [`new_model_retrained_200k/`](./new_model_retrained_200k/README.md)
Contains the final, verified evaluation results across all 3 phases:
- **Phase 1:** Zero-shot generalization on 8 unseen architectures ($2,876$ tracks, `0.9933` AUROC).
- **Phase 2:** Full held-out benchmark ($28,134$ physical tracks, `0.9970` AUROC, `98.48%` Accuracy, `1.55%` EER).
- **Phase 3:** Large-scale out-of-domain streaming ($50,000$ Suno v5 tracks, `98.89%` AI Detection Rate, AI-only  --  no FP measurement).
