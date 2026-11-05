# Architecture-1 Local Development & Benchmarking Scripts

This directory contains local utilities, data preparation tools, multi-core feature extractors, and benchmark visualization scripts for **Architecture-1 (MusicScope-CL)**.

> [!NOTE]
> These scripts orchestrate offline dataset preparation, large-scale multi-core training, and local analysis. They import the core neural network and DSP modules directly from [`Architecture-1/model/`](../model/).

---

## Script Catalog

### 1. Training & Feature Extraction
| Script | Description | Usage Command |
| :--- | :--- | :--- |
| **[`train_gpu_200k.py`](./train_gpu_200k.py)** | High-throughput GPU training pipeline with FP16 AMP, auto-resume, and batch size 128–256. | `python train_gpu_200k.py --epochs 5` |
| **[`extract_narrative_200k.py`](./extract_narrative_200k.py)** | Multi-process (18–20 workers) 128-D narrative feature extractor with milestone caching. | `python extract_narrative_200k.py --workers 18 --chunk_size 5000` |
| **[`split_dataset_200k.py`](./split_dataset_200k.py)** | Generates clean 85% Train (159k) / 15% Val (28k) stratified partition without corrupt audio. | `python split_dataset_200k.py` |
| **[`train_heads_fast.py`](./train_heads_fast.py)** | Rapidly trains and calibrates SupCon projection and FusionMLP classification heads in-memory. | `python train_heads_fast.py` |

### 2. Evaluation & Benchmarking
| Script | Description | Usage Command |
| :--- | :--- | :--- |
| **[`evaluate_val_28k.py`](./evaluate_val_28k.py)** | Evaluates 200k production model across all 28,134 held-out validation tracks with FastScorer. | `python evaluate_val_28k.py --workers 16` |
| **[`eval_unseen_generators_full.py`](./eval_unseen_generators_full.py)** | Evaluates zero-shot generalization across 8 unseen generators (DiffRhythm, SongGen, etc.). | `python eval_unseen_generators_full.py` |
| **[`eval_all_generators_fast.py`](./eval_all_generators_fast.py)** | Fast stratified benchmark across all generator strata with per-generator AUROC and accuracy. | `python eval_all_generators_fast.py` |
| **[`evaluate_1m_streaming.py`](./evaluate_1m_streaming.py)** | Out-of-domain streaming inference pipeline for massive-scale audio evaluation. | `python evaluate_1m_streaming.py` |

### 3. Visualization & Publication Charts
| Script | Description | Usage Command |
| :--- | :--- | :--- |
| **[`plot_tsne.py`](./plot_tsne.py)** | Generates high-resolution t-SNE latent space projections of narrative and fused features. | `python plot_tsne.py` |
| **[`plot_narrative_rarity.py`](./plot_narrative_rarity.py)** | Computes and plots empirical narrative feature rarity density curves comparing Human vs AI. | `python plot_narrative_rarity.py` |
| **[`visualize_eval_5000.py`](./visualize_eval_5000.py)** | Generates 6-panel publication-quality analysis scorecard (ROC, PR, confusion matrix, errors). | `python visualize_eval_5000.py` |

### 4. Transactions Research Audit ([`stage1/`](./stage1/))
| Script | Description |
| :--- | :--- |
| **`s1_freeze.py`** | Freezes code, environment, dataset sizes, and model weights with SHA-256 hashes. |
| **`s2_identity.py`** | Path-derived generator identity resolution from raw audio directory structures. |
| **`s3_manifest.py`** | Compiles canonical 187k-track dataset manifest. |
| **`s4_overlap.py`** | Exhaustive train/val deduplication and cross-split leakage detector. |
| **`s5_provenance.py`** | Generator version matrix and resolved label mapping. |
| **`s6_report.py`** | Evaluates Stage 1 exit conditions and produces final audit verdicts. |

---

## Environment

All scripts in this directory are configured to run with the project's Python 3.12 virtual environment:
```powershell
d:\mirex\.venv312\Scripts\python.exe <script_name>.py
```
