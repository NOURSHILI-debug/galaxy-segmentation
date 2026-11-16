# Galaxy Segmentation — SegFormer-B0

A reproducible pipeline for semantic segmentation of galaxies from the **NASA Galaxy Segmentation** FITS dataset, using a single deep-learning architecture: **SegFormer-B0** with a pretrained **MiT-B0** encoder, fine-tuned through transfer learning.

> **Research question:** to what extent does astronomy-aware preprocessing improve SegFormer-B0 segmentation of faint galaxy structures?

> **Status note.** This repository is built from an implementation plan, not from measured results. Dataset size, HDU layout, mask semantics, class count, channel count, GPU capacity and final hyperparameters are **pending verification** by the data audit and pilot training (see [Pending facts](#pending-facts)). No number in this README should be read as an experimental finding.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Frozen design decisions](#2-frozen-design-decisions)
3. [Pipeline architecture](#3-pipeline-architecture)
4. [Repository structure](#4-repository-structure)
5. [Getting started](#5-getting-started)
6. [Data and splits](#6-data-and-splits)
7. [Preprocessing experiments](#7-preprocessing-experiments)
8. [Model and training](#8-model-and-training)
9. [Evaluation](#9-evaluation)
10. [Classical CV baseline branch](#10-classical-cv-baseline-branch)
11. [Inference and Streamlit app](#11-inference-and-streamlit-app)
12. [Reproducibility](#12-reproducibility)
13. [Team, ownership and workflow](#13-team-ownership-and-workflow)
14. [Integration gates](#14-integration-gates)
15. [Pending facts](#pending-facts)
16. [Limitations](#16-limitations)
17. [References](#17-references)

---

## 1. Project overview

| Item | Value |
|---|---|
| Task | Pixel-wise semantic segmentation of astronomical images |
| Data | NASA Galaxy Segmentation: FITS images and ground-truth masks (AWS Open Data Registry, `apd_galaxysegmentation`) |
| Model | SegFormer-B0, MiT-B0 encoder pretrained on ImageNet-1K, segmentation head adapted to C verified classes |
| Learning strategy | Transfer learning and fine-tuning |
| Core experiment | Controlled comparison of preprocessing variants (E0–E4) with everything else held fixed |
| Side branch | Classical computer vision baselines, reported separately |
| Deliverables | Trained checkpoint, metrics, qualitative error analysis, Streamlit demo, report |

**Scope guardrails**

- Exactly **one** deep-learning architecture. No U-Net, DeepLabV3+, ensemble, cascade or second trainable segmentation model.
- Classical CV is an independent baseline and analysis branch. It is **not** a second model and **not** a cascade stage.
- Raw FITS files and source masks are **immutable**.

---

## 2. Frozen design decisions

These come from the shared integration contract. Any change requires a written proposal approved by both contributors (see [Change control](#change-control)).

| Area | Frozen rule |
|---|---|
| Architecture | SegFormer-B0 only; pretrained MiT-B0 encoder; head outputs C verified classes |
| Pretrained checkpoint | `nvidia/mit-b0` (candidate; verify model card, library compatibility and license before use) |
| Split | Starting 70/15/15 train/val/test, seed 42, by **original sample ID**, before patch extraction |
| Patch size | Start 256×256; reduce only for documented hardware/data constraints |
| Input channels | Inspect FITS first. If single-channel: normalize, then repeat to 3 channels (initial route) |
| Labels | Verify semantics and raw values; map explicitly to contiguous IDs `0..C-1`; define background/ignore policy |
| Dataset item | `image`: `FloatTensor [C,H,W]`, `mask`: `LongTensor [H,W]`, `sample_id`: `str` |
| Model output | Raw logits; resize to target H×W before loss/metrics; `argmax` over the class dimension |
| Training start point | AdamW, batch 2, up to 50 epochs, patience 8, encoder LR ≈ 6e-5 (initial trial, not a claimed optimum) |
| Loss | Cross-Entropy + Dice, only if compatible with the verified mask representation |
| Checkpoint selection | Best **validation macro Dice**; the test set is never used for tuning |

---

## 3. Pipeline architecture

```
NASA FITS images + ground-truth masks
        │
        ▼
Dataset acquisition / sync ──► FITS audit and ID matching ──► Manifest + anomaly reports
        │
        ▼
Mask semantics + class mapping
        │
        ▼
Sample-level train / val / test split        (before any patching)
        │
        ▼
Astronomy-aware preprocessing variants       (images only, never masks)
        │
        ▼
Aligned patching + augmentation (train only)
        │
        ▼
PyTorch Dataset / DataLoader
        │
        ▼
SegFormer-B0 (pretrained MiT-B0 encoder + segmentation head)
        │
        ▼
Fine-tuning ──► full-resolution logits ──► pixel-wise predictions
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                       Metrics (Dice / IoU)     Visual analysis (overlays / errors)
                                  └───────────┬───────────┘
                                              ▼
                         Preprocessing experiments / ablations
                                              ▼
                  Classical CV baselines (separate comparison)
                                              ▼
                                  Streamlit demo + report
```

### SegFormer-B0 internals

```
Input [B,C,H,W]
   → MiT-B0 encoder (hierarchical, 4 stages: high → low resolution)
   → SegFormer all-MLP decoder (multi-scale feature fusion)
   → Segmentation classifier (C classes)
   → Raw logits [B,C,h,w]
   → Upsample to [B,C,H,W]
   → Argmax over class dimension
```

Typical stage strides for a 256×256 input are about 1/4, 1/8, 1/16 and 1/32, but this **must be verified with a forward-pass test** in the selected library version. Do not hard-code shapes.

---

## 4. Repository structure

```
galaxy-segmentation/
├── dataset/
│   ├── raw/                # FITS source — never committed, immutable
│   ├── metadata/           # manifest.csv, audit summaries, mask_audit.json, class_mapping.json
│   ├── processed/          # generated artifacts — normally not committed
│   └── splits/             # train.csv, val.csv, test.csv (frozen sample IDs)
├── notebooks/              # EDA and analysis
├── src/
│   ├── data/               # FITS loading, manifest, mask decoder, dataset, patches, splits
│   ├── preprocessing/      # normalization, background, asinh, enhancement, augmentation
│   ├── classical_cv/       # filtering, edges, thresholding, morphology, watershed
│   ├── models/             # SegFormer-B0 wrapper
│   ├── training/           # losses, trainer, checkpointing, train entry point
│   ├── evaluation/         # metrics and evaluation
│   ├── inference/          # prediction and app-facing API (pending contract approval)
│   ├── visualization/      # overlays and error maps
│   └── utils/
├── configs/
│   ├── experiments/        # E0–E4 experiment configs
│   ├── augmentation.yaml
│   ├── model.yaml
│   └── training.yaml
├── experiments/segformer/  # run artifacts (run_001/, run_002/, ...)
├── app/                    # Streamlit application
├── tests/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

### Run artifact layout

```
experiments/segformer/run_001/
├── config.yaml
├── environment.json
├── history.csv
├── metrics.json
├── checkpoint_best/
├── checkpoint_last/
├── predictions/            # saved by sample_id
├── overlays/
└── notes.md
```

---

## 5. Getting started

### 5.1 Prerequisites

- Python 3.x (exact version recorded in `docs/environment.md` after setup)
- PyTorch with CUDA if a GPU is available; CPU is acceptable **for debugging only**
- Git

Record the Python, PyTorch, CUDA, GPU/VRAM and key package versions in `docs/environment.md`.

### 5.2 Installation

```bash
git clone <repository-url> galaxy-segmentation
cd galaxy-segmentation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies (pin exact versions in `requirements.txt`): `torch`, `transformers`, `astropy`, `numpy`, `scipy`, `scikit-image`, `opencv-python`, `pandas`, `pyyaml`, `matplotlib`, `streamlit`, `pytest`.

### 5.3 Sanity check

```bash
python - <<'EOF'
from astropy.io import fits
print("astropy FITS import OK")
EOF
pytest -q
```

### 5.4 Data acquisition

The dataset is published on the AWS Open Data Registry:
<https://registry.opendata.aws/apd_galaxysegmentation/>

1. Follow the access instructions on the registry page and sync into `dataset/raw/`.
2. **Confirm the sync completed** before processing anything.
3. Do not assume file count, HDU structure, dimensions, mask encoding, channel count or class semantics. Establish them with the audit below.

### 5.5 Typical workflow (planned entry points)

The commands below describe the intended interface; exact module names and flags are finalized as each component lands and is documented in `docs/`.

```bash
# 1. Build manifest and audit image FITS files          (src/data/)
python -m src.data.dataset_index
python -m src.data.dataset_audit

# 2. Audit masks and build the class mapping            (src/data/)
python -m src.data.mask_inspector

# 3. Generate and freeze sample-level splits             (src/data/)
python -m src.data.split_dataset --seed 42

# 4. Train a SegFormer-B0 run                            (src/training/)
python -m src.training.train --config configs/training.yaml

# 5. Evaluate                                           (src/evaluation/)
python -m src.evaluation.evaluate --run experiments/segformer/run_001

# 6. Launch the demo                                    (app/)
streamlit run app/main.py
```

> Real training is **blocked** until image–mask pairing, class mapping, split IDs, channel strategy and tensor contracts are jointly approved (Gates G1–G3).

---

## 6. Data and splits

### 6.1 Audit and manifest

For every file the audit records: HDU layout, shape, dtype, header metadata, finite-value status, NaN/Inf counts, min/max, mean and standard deviation.

- Sample IDs are derived from **verified filename patterns**, never from directory order.
- Images and masks are paired by ID. Missing, duplicate, unreadable and mismatched pairs are all recorded with an explicit status in `dataset/metadata/manifest.csv`.
- No anomaly is silently dropped. Inclusion of problematic files is decided jointly.

### 6.2 Mask semantics

Mask semantics are **not assumed**. Class names are assigned only where supported by the dataset documentation and data. Raw values are mapped explicitly to contiguous IDs `0..C-1` in `dataset/metadata/class_mapping.json`, with a documented background/ignore policy. Unknown values are handled explicitly and never silently mapped or merged.

Masks are stored as **integer labels**, not colorized images; visualization palettes are kept separate.

### 6.3 Splitting and leakage prevention

- 70% train / 15% validation / 15% test, seed 42.
- Split by **original sample ID before patch extraction**: all patches from one galaxy stay in one split.
- Split files (`train.csv`, `val.csv`, `test.csv`) are exported, checked for zero overlap, and **frozen before any tuning**.
- A different ratio is allowed only if a dataset constraint requires it, and only with documented joint approval.

### 6.4 Patching

- Start with **256×256** patches. Images are not blindly resized, because small or faint structures may be lost.
- Variable sizes are handled by controlled patching or padding, applied identically to image and mask.
- Inference on full images uses overlapping patches with documented stitching.
- Rare-class coverage is monitored when sampling patches.

---

## 7. Preprocessing experiments

Images are transformed; **masks are never transformed by preprocessing**. The API contract:

```python
processed_image = preprocess_image(image, config)
# deterministic for a fixed config
# preserves geometry; documented dtype/shape; finite float32 output
# does not mutate the raw array in place
# records transform parameters
```

### Experiment matrix

| ID | Variant | Method | Purpose |
|---|---|---|---|
| E0 | Baseline | Documented finite-value handling and normalization | Reference |
| E1 | Background | Background estimation and correction | Faint-structure visibility under background variation |
| E2 | Asinh | Asinh intensity transform | Nonlinear intensity stretch |
| E3 | Contrast | Controlled enhancement (e.g., CLAHE) | Local contrast enhancement |
| E4 | Combined | Only validation-supported choices | Evaluate the selected pipeline |
| Ablation | Remove one component from E4 | All else fixed | Component contribution |

**Controls held constant across all runs:** architecture, pretrained initialization, splits, loss, training budget, metrics and checkpoint-selection rule. No experiment may quietly change architecture or split.

**Augmentation rules:** training-only flips/rotations and cautious photometric augmentation. Geometry is applied identically to image and mask, with nearest-neighbor interpolation for masks. No random augmentation in validation or test.

> Display stretches used for visualization are kept distinct from training preprocessing.

---

## 8. Model and training

### 8.1 Model

- Load the compatible pretrained MiT-B0 checkpoint and record its source, library version, config and a **weight-loading report** (loaded / missing / unexpected keys).
- Configure the segmentation head for C verified classes; set `id2label` / `label2id` from the approved mapping.
- Confirm encoder weights are loaded and head parameters are trainable.
- Pretrained weights must never be silently replaced by random initialization.

```python
model = build_segformer(config)
outputs = model(pixel_values=images)
logits = outputs.logits          # raw scores, not probabilities
# resize logits to target H×W before loss/metrics
prediction = logits.argmax(dim=1)
```

### 8.2 Input channels

Inspect FITS channels first. If samples are single-channel, normalize and repeat the channel three times to match the pretrained three-channel stem. Do not alter the pretrained stem without evidence and tests. For multi-channel data, agree and document a channel mapping.

### 8.3 Fine-tuning recipe (starting values)

| Setting | Starting value |
|---|---|
| Optimizer | AdamW |
| Batch size | 2 |
| Max epochs | 50 |
| Early-stopping patience | 8 (on validation metric) |
| Encoder LR | ≈ 6e-5 |
| Parameter groups | Lower LR for pretrained encoder, higher for new head (if used) |
| Loss | Cross-Entropy + Dice if compatible |
| Warm-up (optional) | Brief frozen-encoder head warm-up, then unfreeze the full model |
| Best checkpoint | Validation macro Dice |

These are starting values, not guaranteed optima. Tune only on validation.

### 8.4 Fallbacks

| Issue | Fallback | Never do |
|---|---|---|
| CUDA OOM | Batch 2 → 1; then patch 256 → 128; document | Change model family |
| Slow training | Short pilot, early stopping, frozen-head warm-up | Claim full fine-tuning if the encoder stayed frozen |
| No local GPU | CPU for debugging; cloud GPU for full training | Present CPU debug runs as final results |
| Variable image sizes | Controlled patching/padding | Blindly resize all images |
| Unknown mask values | Pause final training; inspect docs, values, overlays; ask supervisor | Invent or merge classes |
| Class imbalance | CE + Dice; inspect per-class metrics; weights only with evidence | Discard rare classes without justification |
| Pretrained weights fail | Verify compatibility; another verified MiT-B0 ImageNet checkpoint if needed | Silently train from random init |

---

## 9. Evaluation

| Metric | Reporting |
|---|---|
| Dice / F1 | Per class and macro; state background and absent-class policy |
| IoU | Per class and macro; same class/ignore policy |
| Precision / recall | Per class and macro; define averaging |
| Pixel accuracy | Overall, with a class-imbalance caveat |
| Efficiency | Training/inference time, hardware, patch size, batch size |

```python
results = evaluate_segmentation(predictions, targets, num_classes=C)
# per-class + aggregate metrics
# explicit absent-class and ignore-index behavior
```

**Rules**

- Metrics are validated against hand-computed toy examples, including empty masks.
- Validation is used for model and configuration selection. **Test is used once**, after all choices are frozen.
- Segmentation metrics are never computed when ground truth is absent or invalid.
- Test results are never used to select preprocessing, epochs, hyperparameters or checkpoints.

---

## 10. Classical CV baseline branch

An independent analysis branch, reported separately and fairly:

- **Filtering:** Gaussian, median, bilateral
- **Edges:** Sobel, Canny
- **Segmentation primitives:** global / Otsu / adaptive thresholding, erosion / dilation / opening / closing, connected components, distance transform, watershed

An edge map or binary mask is **not** automatically a semantic segmentation. Metrics are computed only when the output semantics are comparable to the target labels; otherwise results are reported qualitatively. Edge maps are never scored against a multiclass mask.

---

## 11. Inference and Streamlit app

**Inference**

- Loads the selected SegFormer-B0 checkpoint with the exact preprocessing and class-mapping configuration used in training.
- Upsamples logits to target size, applies `argmax`, saves predictions by `sample_id`.
- Full-size images use overlapping patches with documented stitching (blending logits/probabilities before `argmax` if that method is chosen).
- The app uses the **same trained SegFormer-B0** and preprocessing config, never another model.

**Streamlit pages**

1. Dataset explorer: image, metadata, ground truth
2. Preprocessing comparison
3. Segmentation inference
4. Evaluation dashboard

Image, target, prediction, overlay and error map are displayed where applicable. Dice/IoU are shown **only when valid ground truth exists**; otherwise predictions only.

---

## 12. Reproducibility

Every run stores: `config.yaml`, seed, environment and hardware, split version, preprocessing variant, checkpoint provenance, `history.csv` and `metrics.json`.

- Predictions and overlays are stored by `sample_id`.
- Raw data and large checkpoints stay **outside Git**.
- Anomalies, limitations, class imbalance and runtime constraints are reported.
- Stable file names: `manifest.csv`, `class_mapping.json`, `train.csv`, `val.csv`, `test.csv`, `config.yaml`, `metrics.json`, `history.csv`.

---

## 13. Team, ownership and workflow

### Ownership

| Person A — data, preprocessing, classical filtering, metrics | Person B — masks, model, training, inference |
|---|---|
| Repository and environment | Mask audit and semantics |
| FITS loader and manifest | Mask decoder and statistics |
| Image audit and EDA | Enhancement (CLAHE) and augmentation |
| Normalization / background / asinh preprocessing | Threshold / morphology / watershed |
| Gaussian / median / bilateral, Sobel / Canny | SegFormer-B0 wrapper |
| Sample-level splits | Dataset adapter and patches |
| Metrics and evaluation | Fine-tuning pipeline |
| Data and preprocessing contracts, integration tests | Inference, error visualization, app integration, model docs |

### Indicative six-week schedule

| Week | Person A | Person B | Joint checkpoint |
|---|---|---|---|
| 1 | Environment, image audit, manifest, EDA | Mask audit, decoder prototype, synthetic SegFormer forward pass | Approve dataset/mask contract |
| 2 | Normalization, background correction, filtering/edges | Mask statistics, enhancement, augmentation, classical morphology | Approve preprocessing and mask contracts |
| 3 | Splits, data tests, metric API | Dataset adapter, patches, model wrapper, training loop | Integrate real batches |
| 4 | Preprocessing comparisons, classical baselines | Fine-tune and monitor validation | Review first real validation results |
| 5 | Ablations, quantitative analysis | Inference, overlays, error analysis | Freeze final config; final test once |
| 6 | Report, tests, docs, reproducibility | App integration, model figures, demo | Final presentation and handover |

Synthetic data may be used for interface tests only, never for results.

### Git workflow

- Branches: `main`, `develop`, and feature branches:
  - Person A: `feature/person-a-data`, `feature/person-a-preprocessing`, `feature/person-a-classical-cv`, `feature/person-a-metrics`
  - Person B: `feature/person-b-masks`, `feature/person-b-enhancement`, `feature/person-b-segformer`, `feature/person-b-visualization`
- **No direct commits to `main`.** Feature branches merge into `develop` through reviewed pull requests.
- Never commit raw FITS, generated patches, large checkpoints or credentials. Track code, configs, metadata summaries and docs.
- Every PR states: purpose, changed contracts, test command, sample output and known limitations.

### Change control

Before changing a frozen rule, document: the current rule, the proposed change, the evidence/reason, the impact on the other contributor, affected interfaces/files, required tests, and both approvals. Update the contract and configs only after agreement.

---

## 14. Integration gates

All gates require sign-off from **both** contributors.

| Gate | Evidence required |
|---|---|
| G0 Repository | Folder tree, environment, README, branch workflow |
| G1 Data truth | Manifest, image–mask pairing, mask semantics, class mapping, background/ignore policy |
| G2 Split freeze | Split CSVs, no sample overlap, seed recorded |
| G3 Integration | Real batch passes shape/dtype/alignment tests; model emits C logits |
| G4 Pilot | Short run completes; loss and validation metrics inspected |
| G5 Experiment freeze | Preprocessing matrix, training config and selection rule versioned |
| G6 Final test | One final test run with archived config, checkpoint and results |

---

## Pending facts

The following are **not** to be hard-coded or claimed until verified by the audit and pilot training:

- [ ] Exact sample count
- [ ] FITS HDU layout
- [ ] Image dimensions
- [ ] Mask semantics and class count (C)
- [ ] Number of input channels
- [ ] Corrupt/missing file count
- [ ] GPU/VRAM capacity
- [ ] Final optimal hyperparameters
- [ ] `nvidia/mit-b0` model card, library compatibility and license

---

## 16. Limitations

To be completed with measured evidence. Anticipated areas to report: class imbalance and small/faint regions, patch-boundary effects, single-channel-to-3-channel adaptation of an RGB-pretrained encoder, hardware and runtime constraints, and the validity of comparing classical outputs to semantic labels.

---

## 17. References

- Dataset registry: <https://registry.opendata.aws/apd_galaxysegmentation/>
- Dataset documentation: <https://github.com/erinleeryan/nasa_astro_aiml/tree/main/galaxysegmentation>
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Pretrained checkpoint candidate: `nvidia/mit-b0` (verify before use).

---

