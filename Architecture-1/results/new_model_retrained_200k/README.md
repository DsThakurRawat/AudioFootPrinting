# Production Results: New Model (187,557-Track Retrained)

> [!NOTE]
> **Production Benchmarks**
> These results represent the official evaluation benchmarks of **MusicScope-CL (Architecture-1)** after full-scale training ($159,423$ training / $28,134$ validation tracks), Supervised Contrastive alignment, decoupled audio padding, and Platt temperature calibration ($T = 0.9083$, $\tau = 0.1800$). All numbers below are recomputed directly from the raw prediction CSVs in this directory.

---

## Directory Artifacts

| Category | Filename | Description |
| :--- | :--- | :--- |
| **Phase 1 Benchmark** | [`eval_unseen_generators_results.csv`](./eval_unseen_generators_results.csv) | Zero-shot evaluation across 8 unseen AI architectures ($2,876$ tracks) |
| **Phase 1 Scorecards** | [`unseen_generators_scorecard.png`](./unseen_generators_scorecard.png) | Visual publication scorecard for unseen generator benchmark |
| **Phase 1 Charts** | [`eval_all_generators_chart.png`](./eval_all_generators_chart.png) | Detection rate bar charts across all generator families |
| **Phase 2 Benchmark** | [`eval_28k_retrained_results.csv`](./eval_28k_retrained_results.csv) | Full 28,134 held-out disjoint validation set predictions |
| **Phase 2 Analysis** | [`eval_28k_retrained_analysis.png`](./eval_28k_retrained_analysis.png) | 4-panel ROC curve, score distribution, and generator breakdown |
| **Phase 2 Verification**| [`benchmark_28k_fixed.png`](./benchmark_28k_fixed.png) | Comparison chart verifying resolution of the short-audio padding bug |
| **Phase 3 Scorecard** | [`phase3_streaming_scorecard.png`](./phase3_streaming_scorecard.png) | 50,000-track out-of-domain streaming evaluation scorecard |
| **Latent Geometry** | [`tsne_latent_space.png`](./tsne_latent_space.png) | 2D t-SNE projection of the 704-D SupCon latent embedding space |
| **Narrative Rarity** | [`narrative_rarity_musicscope.png`](./narrative_rarity_musicscope.png) | Violin distribution of narrative rarity percentiles across generators |
| **Full Report** | [`comprehensive_evaluation_report.md`](./comprehensive_evaluation_report.md) | Comprehensive 3-phase technical scorecard report |

---

## Three-Phase Technical Evaluation Summary

### Phase 1: Zero-Shot Generalization on 8 Unseen AI Architectures
* **Evaluated Dataset:** $2,876$ tracks spanning 8 previously unseen AI generator families + FMA Human controls.
* **Core Metrics:**
  * **Macro-AUROC:** `0.9933`
* **Detection Rates (at $\tau = 0.18$):**

| Generator | Category | Tracks | Caught | Detection Rate |
| :--- | :--- | :---: | :---: | :---: |
| **SongGen** | Autoregressive Transformer | 584 | 552 | **94.52%** |
| **DiffRhythm** | Full-Song Latent Diffusion | 594 | 559 | **94.11%** |
| **AceStep** | Step-based Diffusion | 295 | 254 | **86.10%** |
| **Brev.ai** | Text-to-Music Diffusion | 298 | 252 | **84.56%** |
| **ElevenLabs** | Neural Transformer Audio | 300 | 247 | **82.33%** |
| **Producer (CassetteAI)** | Hybrid Production Engine | 300 | 239 | **79.67%** |
| **Stable Audio** | Continuous Latent Diffusion | 194 | 143 | **73.71%** |
| **Mubert** | Algorithmic Loop Assembler | 149 | 17 | **11.41%** |
| **Human (FMA)** | Real Acoustic Recordings | 162 | 0 | **0% FP** |

> [!NOTE]
> **On Mubert's low detection rate:** Mubert does not synthesize audio via neural generation  --  it deterministically assembles pre-recorded, human-performed acoustic stem loops. Because its underlying stems exhibit genuine human acoustic physics, the model correctly identifies them as acoustically human-like. This is a ground-truth labeling ambiguity, not a model failure.

### Phase 2: Complete Held-Out Disjoint Benchmark (28,134 Tracks)
* **Evaluated Dataset:** $28,134$ physical audio tracks ($12,172$ AI vs. $15,962$ Human) strictly disjoint from the $159,423$ training tracks ($0$ track leakage, verified by path and content hash).
* **Core Metrics:**
  * **Macro-AUROC:** **`0.9970`**
  * **Average Precision (AP):** **`0.9969`**
  * **Overall Accuracy ($P \ge 0.50$):** **`98.48%`**
  * **Equal Error Rate (EER):** **`1.55%`**
  * **F1-Score:** **`0.9822`**
* **Per-Generator Detection Rates (at $\tau = 0.18$):**

| Generator | Tracks | Caught | Detection Rate |
| :--- | :---: | :---: | :---: |
| **AudioLDM** | 828 | 828 | **100.00%** |
| **Mustango** | 828 | 827 | **99.88%** |
| **Suno** | 3,814 | 3,811 | **99.92%** |
| **Udio** | 5,204 | 5,199 | **99.90%** |
| **MusicGen** | 828 | 826 | **99.76%** |
| **Echoes** | 670 | 522 | **77.91%** |
| **Human (FMA)** | 15,962 | 411 FP | **97.43% correct** |

> [!NOTE]
> Human false-positive rate at $\tau = 0.18$: $411 / 15{,}962 = 2.57\%$. At a more conservative threshold of $\tau = 0.50$: $233 / 15{,}962 = 1.46\%$ FP rate (98.54% correct).

### Phase 3: Out-of-Domain 50,000-Track Streaming Benchmark
* **Evaluated Dataset:** $50,000$ unseen streaming AI tracks ingested from Hugging Face (`bolshyC/Muse`, Suno v5 web archive).
* **Core Metrics:**
  * **AI Detection Rate:** **`98.89%`** ($49,444 / 50,000$ tracks detected at $\tau = 0.18$)
  * **Mean Prediction Score:** **`0.9390`**
  * **High-Confidence Score Tier ($[0.80, 1.00]$):** **`93.70%`** ($46,852$ tracks)

> [!NOTE]
> Phase 3 is a single-source (Suno v5) AI-only evaluation with no human tracks. It measures recall/detection rate but cannot measure false-positive rate.

---

## Production Model Architecture

```
Raw Audio (22.05 kHz)
   |--  Branch A (30s Padded) -> Log-Mel (128x1292) -> StyleEncoder (Conv2D) -> z_surf (576-D)
   \--  Branch B (Raw Unpadded) -> Librosa/Scipy Extracts (128-D Moments)   -> x_narr (128-D)
                                                                                  |
Concat [z_surf (576-D) || x_narr (128-D)] = 704-D Fused Latent Representation ---+
   |
   v
SupCon Projection Head (704-D -> 128-D Metric Space)
   |
   v
Calibrated Fusion MLP (128-D -> Logits) -> Temperature Scaler (T = 0.9083) -> Sigmoid
   |
   v
Calibrated Probability p  in  [0, 1]  (Decision Threshold: tau = 0.1800)
```
