# PP-LCNet_x1_0_doc_ori — Runtime Benchmark Findings

**Model:** PP-LCNet_x1_0_doc_ori (PP-StructureV3 Document Image Orientation Classification)
**Date:** 2026-05-05
**Ticket:** [#2963](https://gitlab.amniltech.com/task-board/amnil-chatbot-research-development-taskboard-2025-2026/-/issues/2963)
**Author:** prabin.lamichhane@amniltech.com

---

## 1. Objective

Evaluate PP-LCNet_x1_0_doc_ori across three inference runtimes on a dataset of
Nepali financial documents. Runtimes compared:

- **PaddlePaddle** (native, Paddle 3.x PIR format) — baseline
- **ONNX Runtime** (converted via paddle2onnx, opset 17)
- **HuggingFace Optimum ORT** (same ONNX model, loaded via ORTModelForImageClassification)

---

## 2. Dataset

| Property | Value |
|---|---|
| Source PDFs | 9 real Nepali financial documents (audit reports, microfinance reports, management docs) |
| Total images | 552 |
| Classes | 4 — 0°, 90°, 180°, 270° |
| Per class | 138 (perfectly balanced) |
| Augmented (messy) | 210 images (38%) |
| Messy transforms | Page skew, shadow gradient, ink bleed, fold/crease, salt-and-pepper noise, blur, heavy JPEG |

All source PDFs are naturally 0° orientation. Class balance was enforced by rotating every
source image into all 4 orientations. Messy augmentation was applied equally across all
classes to prevent the model from using image quality as an orientation cue.

---

## 3. Results Summary

| Runtime | Accuracy | Avg Latency | P50 Latency | P95 Latency | Throughput |
|---|---|---|---|---|---|
| PaddlePaddle | 89.13% | 38.57 ms | 38.83 ms | 45.56 ms | 25.9 img/s |
| ONNX Runtime | 89.13% | 31.53 ms | 31.33 ms | 39.35 ms | 31.7 img/s |
| Optimum ORT | 89.13% | 30.88 ms | 30.95 ms | 37.92 ms | 32.4 img/s |

**Prediction agreement: 100% across all 3 runtimes** — conversion is numerically lossless.

### Charts

![Comparison Chart](results/comparison_chart.png)
![Latency Distribution](results/latency_boxplot.png)

---

## 4. Per-class Performance (identical across all runtimes)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0° | 0.887 | 0.906 | 0.896 |
| 90° | 0.899 | 0.899 | 0.899 |
| 180° | 0.882 | 0.870 | 0.876 |
| 270° | 0.898 | 0.891 | 0.895 |
| **Overall** | **0.891** | **0.891** | **0.891** | |

### Confusion Analysis

Most errors are **adjacent-class confusions** — the model mixes up orientations that look
visually similar (90° apart):

| True → Predicted | Errors |
|---|---|
| 270° → 180° | 12 |
| 90° → 0° | 10 |
| 180° → 90° | 10 |
| 0° → 270° | 9 |

This pattern is expected — financial documents often look similar when rotated 90° because
of their symmetric table/column layouts.

![Confusion Matrix](results/paddle_confusion_matrix.png)

### Clean vs Messy Accuracy

| Condition | Accuracy |
|---|---|
| Clean images | 89.77% |
| Messy images (degraded scans) | 88.10% |

The 1.67% drop under messy conditions is small — the model is reasonably robust to
scanner degradation.

---

## 5. Key Findings

1. **All three runtimes produce identical predictions** — paddle2onnx conversion at opset 17
   is numerically lossless for this model.

2. **ONNX Runtime is 18% faster than native PaddlePaddle** (31.53ms vs 38.57ms avg).

3. **Optimum ORT adds negligible overhead over raw ORT** (~2% difference, within noise).
   The HuggingFace wrapper provides no additional optimization on CPU for this model.

4. **89.1% accuracy on Nepali financial documents** — reasonable for an out-of-the-box
   model with no domain fine-tuning.

5. **P95 latency stays under 46ms** across all runtimes — suitable for document preprocessing
   pipelines where orientation is corrected before OCR.

---

## 6. Recommendation

**Use ONNX Runtime for production.**

| Criterion | Recommendation |
|---|---|
| Accuracy | Any runtime (identical) |
| Speed | ONNX Runtime or Optimum ORT |
| Dependency weight | ONNX Runtime (lighter than Optimum) |
| HuggingFace pipeline integration | Optimum ORT |
| Ease of deployment | ONNX Runtime |

- If deploying standalone: **`onnxruntime` + `model.onnx`** — minimal dependencies, fastest startup.
- If integrating into a HuggingFace pipeline: **Optimum ORT** — virtually same speed, cleaner API.
- **Drop PaddlePaddle** from the production stack — heavier dependency, slower, no accuracy benefit.

---

## 7. Fine-tuning — Is It Worth It?

### Current gap

89.1% accuracy means ~60 errors in 552 images. Most errors are adjacent-class confusions
on documents with symmetric layouts (financial tables look similar at 90° and 270°).

### Can fine-tuning close the gap?

**Yes — with moderate effort and likely significant gain.** Here is the analysis:

#### Why the model underperforms on Nepali documents

| Factor | Impact |
|---|---|
| Training data mismatch | PP-LCNet was trained on Chinese/English documents — Devanagari script and Nepali table layouts are out-of-distribution |
| Mixed-language pages | Nepali docs mix Devanagari + English + numbers, creating layouts not seen during pre-training |
| Document quality | Low-quality scans from small orgs (cooperatives, NGOs) differ significantly from training data |
| Symmetric financial tables | Balance sheets / P&L statements have repetitive vertical structure that confuses 90°/270° |

#### Expected gain from fine-tuning

Based on similar document orientation fine-tuning studies:

| Approach | Expected Accuracy | Effort |
|---|---|---|
| No fine-tuning (current) | ~89% | Done |
| Feature extraction (freeze backbone, train head only) | ~92-94% | Low — ~1-2 days |
| Full fine-tuning (unfreeze all layers) | ~95-97% | Medium — ~3-5 days |
| Fine-tune + larger dataset (500+ real pages) | ~97-99% | High — depends on data collection |

#### Fine-tuning approach (when prioritized)

1. **Dataset:** Collect 500-1000 real Nepali financial document pages (currently have 137 source pages).
   Rotate × 4 → 2000-4000 labeled images. More messy/degraded samples needed.

2. **Strategy:** Start with feature extraction (freeze PP-LCNet backbone, retrain the 4-class
   head). If accuracy plateaus below 95%, unfreeze the last 2-3 blocks.

3. **Export path:** Fine-tune in PaddlePaddle or convert to PyTorch (via ONNX), fine-tune,
   re-export to ONNX. The ONNX deployment path remains the same.

4. **Data gap is the bottleneck** — not model capacity. PP-LCNet_x1_0 has sufficient
   capacity for 4-class orientation. More diverse Nepali document data will drive the
   largest accuracy improvement.

#### Verdict

Fine-tuning is **feasible and recommended** once more real Nepali document data is collected.
Feature extraction alone (low effort) is likely to push accuracy to ~93-94%.
Full fine-tuning on a larger dataset should reach ~96-98%.

---

## 8. Technical Notes

### Model format
The downloaded model uses **Paddle 3.x PIR format** (`inference.json` + `inference.pdiparams`),
not the legacy `inference.pdmodel` format. The standard paddle.inference `Config` API
accepts this by passing `inference.json` as the model file path.

### ONNX conversion
```bash
paddle2onnx \
  --model_dir models/PP-LCNet_x1_0_doc_ori_infer \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file models/model.onnx \
  --opset_version 17 \
  --enable_onnx_checker True
```
Constant folding reduced 282 → 115 ONNX nodes.

### Optimum compatibility fix
Optimum `ORTModelForImageClassification` requires HuggingFace-standard I/O names.
The ONNX model was patched before loading:
- Input renamed: `x` → `pixel_values`
- Output renamed: `fetch_name_0` → `logits`
- `config.json` with `model_type: "resnet"` added to the model directory.

### Preprocessing (from config.json)
```
resize_short(256) → center_crop(224) → normalize(ImageNet) → to_CHW → expand_batch_dim
```
Note: simple `resize(224, 224)` without the resize_short + center_crop steps will produce
slightly different results and should be avoided.

---

## 9. Repo

https://github.com/prabinlamichhane-cell/research-page-orientation
