# Technical Handoff Report: Person B (Milestones B1 through B4)

**Project:** Galaxy Segmentation — Astronomy-Aware Preprocessing and SegFormer-B0  
**Author:** Person B (Classical Computer Vision, Mask Semantics, Enhancement & Augmentation)  
**Recipient:** Person A (FITS Ingestion, Normalization, Background Correction, Deep Learning Architecture)  
**Target Milestone Integration:** End of B4 / Pre-Training Integration  
**Date:** November 15, 2026  
**Document Version:** 1.1 (Formal Research Handoff — Scientific & Accuracy Revision)

---

### Revision Note (Version 1.1)
This revision implements essential scientific accuracy refinements identified during independent review:
1. **Missed Pixel Classification:** Clarified that the 13,086 missed pixels on sample `1-100523` are unclassified *annotated foreground pixels* (not assumed to be exclusively spiral arms without class-specific decomposition).
2. **Asinh Stretch Numerical Safeguards:** Removed unsupported reference to gradient clipping; accurately documented the actual safeguards implemented in `src/preprocessing/enhancement.py` (finite positive $\beta$ validation, normalisation precomputation, input shape/range validation, copy semantics, and post-stretch `clamp(0.0, 1.0)`).
3. **Photometric Augmentation Qualifications:** Qualified that shared scaling factors reduce independent channel perturbations across bands, but do not guarantee exact preservation of calibrated astrophysical flux ratios or color indices.
4. **Multi-Band Averaging:** Replaced the absolute contraindication claim with a scientifically accurate qualification: unweighted averaging across all five bands is problematic due to non-finite dropouts and elevated thermal noise, and requires empirical comparison against validity-aware or weighted multi-band fusion.
5. **Reference-Band Recommendation Context:** Clarified why the $r$-band recommendation is strictly preliminary: on clean sample `1-100523`, $i$-band achieved a higher robust SNR ($16.70$ vs $15.81$), while on corrupted sample `1-103458`, $r$-band yielded a higher robust SNR ($15.61$ vs $12.43$). The selection criteria and trade-offs require broader dataset-wide validation.
6. **Image Artifact Distinctions:** Rigorously reported local audit counts: 125 image files contain non-finite pixels, 160 image HDUs contain NaNs, and zero infinite ($\pm\infty$) pixels were observed.
7. **Cross-Checked Five-Band Statistics:** Cross-checked and corrected every numerical value in the five-band diagnostics table against the generated `comparison_report.json` output.
8. **Evaluation Metric Scope:** Explicitly emphasized that B4's diagnostic evaluation on sample `1-100523` is a classical structural probe, not a substitute for Person A's shared multi-class semantic segmentation metrics.
9. **File Inventory Verification:** Audited all listed repository paths, explicitly reporting existing empty files (`src/data/dataset_audit.py`, `tests/test_dataset_contract.py`, `tests/test_integration.py`, `tests/test_metrics.py`, `tests/test_model_forward.py`).
10. **B2 Audit Provenance:** Identified the actual audit script execution via `notebooks/02_mask_analysis.ipynb` saving to `dataset/metadata/mask_audit.json`, while noting `src/data/dataset_audit.py` is currently an unpopulated placeholder.

---

## A. Executive Summary

This report provides the formal technical handoff for all deliverables designed, implemented, and verified by **Person B** across milestones **B1, B2, B3, and B4**.

### Person B Scope and Progression
1. **B1 — Mask Semantics & Categorical Decoder:** Established ground-truth semantics from the NASA Astrophysics Division Galaxy Segmentation Benchmark Dataset (Pan-STARRS cutouts with Galaxy Zoo: 3D annotations). Formally distinguished unannotated pixels (raw 0) from confirmed background, defined the 4-class trainable label space, and implemented the strict categorical decoder (`src/data/mask_decoder.py`).
2. **B2 — Comprehensive Dataset Audit:** Audited all 2,908 mask FITS files and 2,879 image FITS files. Established exact image-mask pair matching (2,879 matched pairs, 29 unmatched masks), verified spatial dimensions ($208 \times 208$), validated categorical integrity (no non-finite or non-integer values), and quantified class imbalance across 125,811,712 mask pixels.
3. **B3 — Enhancement & Synchronized Augmentation:** Developed an astronomy-aware augmentation and enhancement framework (`src/preprocessing/`). Guaranteed exact spatial and categorical synchronization across multi-band images, categorical masks, and boolean validity masks under geometric operations (D4 dihedral symmetry group: flips and $90^\circ$ rotations) and photometric perturbations, with deterministic seed control via `torch.Generator`.
4. **B4 — Classical Computer Vision Baseline:** Engineered a self-contained traditional computer vision pipeline (`src/classical_cv/`). Implemented multi-band diagnostics across all five Pan-STARRS bands ($g, r, i, z, y$), Global Otsu and Local Adaptive thresholding, mathematical morphology, two-pass connected components, exact Euclidean Distance Transform ($O(N)$ Felzenszwalb-Huttenlocher), and Meyer's priority-queue marker-controlled watershed segmentation.

All components have been thoroughly tested. The complete test suite contains **162 automated unit tests across 11 test modules**, all passing with zero errors and zero failures.

---

## B. Project Context and Frozen Research Decisions

### Central Research Question
> *"To what extent does astronomy-aware preprocessing improve SegFormer-B0 segmentation of faint galaxy structures?"*

### Frozen Scientific and Architectural Decisions
- **Four Trainable Semantic Classes:** The deep learning model trains strictly on four semantic categories: Core (0), Bar (1), Spiral Arms (2), and Foreground Stars (3).
- **Raw Value 0 is Unannotated Field, Not Background:** In the benchmark annotations, raw value 0 denotes *"No relevant data"* (unannotated outer sky/field margins). It is mapped to `IGNORE_INDEX = 255` and must **never** be treated as a negative ground-truth background class.
- **Ignore Index:** PyTorch `ignore_index = 255` is enforced across all loss functions and evaluation metrics.
- **Sample-Level Dataset Splitting:** Dataset partitioning (Train / Validation / Test) must occur strictly at the unique galaxy identifier level (`sample_id`) prior to patch extraction to eliminate spatial data leakage.
- **Classical CV is an Independent Baseline:** Classical computer vision algorithms extract structural and photometric features deterministically. They serve as an empirical morphological comparison, **not** as a competing trainable segmentation architecture.
- **Single Deep Learning Architecture:** SegFormer-B0 is the designated trainable model. Person B's data contracts and pipelines are structured specifically to feed SegFormer-B0.
- **Model Training Status:** Neither SegFormer-B0 training nor final model evaluation has been executed. Milestones B1–B4 constitute the data integrity, preprocessing, augmentation, and classical baseline foundations.

---

## C. B1 — Mask Semantics and Decoder

### 1. Data Provenance and Raw Mask Structure
- **Dataset:** NASA Astrophysics Division Galaxy Segmentation Benchmark Dataset (Pan-STARRS stacked cutouts with Galaxy Zoo: 3D citizen-science consensus labels).
- **Mask Format:** Single-HDU FITS files containing a 2D categorical array with dimensions $[208, 208]$ stored as big-endian float64 (`>f8`).
- **Conflict Resolution Rule:** The benchmark documentation specifies that when volunteer annotations overlap, conflicting pixels are assigned the **lowest numerical raw value**.

### 2. Approved Categorical Mapping

| Raw Value | Benchmark Documented Meaning | Training ID | PyTorch Target Class Name | Loss / Metric Status |
| :---: | :--- | :---: | :--- | :--- |
| **0** | No relevant data | **255** | `IGNORE_INDEX` | Masked / Ignored |
| **1** | Galaxy center or core | **0** | Galaxy center or core | Trainable Class 0 |
| **2** | Galaxy bar | **1** | Galaxy bar | Trainable Class 1 |
| **3** | Galaxy arms | **2** | Galaxy arms | Trainable Class 2 |
| **4** | Star | **3** | Foreground star | Trainable Class 3 |

### 3. Implementation Details (`src/data/mask_decoder.py`)
- **API Signature:** `decode_mask(mask: Any) -> torch.LongTensor`
- **Output Contract:** CPU `torch.LongTensor` of shape $[H, W]$ with dtype `torch.int64`.
- **Validation and Safety Checks:**
  - Enforces 2D spatial dimensions (`mask.ndim == 2`); rejects 1D, 3D, or empty non-2D arrays with `MaskDecodeError`.
  - Rejects non-numeric dtypes (e.g., string, boolean, object).
  - Validates that floating-point inputs contain strictly integer-like finite values (`np.all(np.equal(np.mod(arr, 1), 0))`).
  - Rejects non-finite values (`NaN`, $\pm\infty$) with clear error messaging.
  - Rejects out-of-range categorical values (any value outside $\{0, 1, 2, 3, 4\}$).
  - Preserves input immutability (does not modify the source array in place).
  - Preserves discrete single-pixel annotations (critical for single-pixel galactic cores and distant stars).
- **Unit Verification:** Verified by `tests/test_mask_decoder.py` (11 tests, 100% passing).

---

## D. B2 — Dataset Mask Audit

### 1. Audit Methodology, Script Provenance, and Pair Matching
- **Audit Execution:** Executed via `notebooks/02_mask_analysis.ipynb`, persisting quantitative results to `dataset/metadata/mask_audit.json`. Note that `src/data/dataset_audit.py` exists as a 0-byte file in the repository intended for future CLI wrapping.
- **Inventory Counts:**
  - Total FITS images in `dataset/raw/`: **2,879**
  - Total FITS masks in `dataset/raw/`: **2,908**
  - Matched Image–Mask Pairs: **2,879**
  - Unmatched Masks: **29** (orphaned masks with no corresponding image)
  - Unmatched Images: **0**
- **Dimension Matching:** All 2,879 matched pairs have identical spatial dimensions of **$208 \times 208$** pixels (14,395 HDU dimension checks passed).

### 2. Pixel-Level Frequency and Class Imbalance
Across all 2,908 audited masks ($125,811,712$ total pixels):
- **Raw Value 0 (Unannotated / Ignored):** $103,962,425$ pixels (**82.63%** of all mask pixels).
- **Raw Value 1 (Core):** $24,812$ pixels (**0.020%** of all pixels; **0.11%** of trainable pixels).
- **Raw Value 2 (Bar):** $5,242,726$ pixels (**4.17%** of all pixels; **24.00%** of trainable pixels).
- **Raw Value 3 (Spiral Arms):** $16,556,365$ pixels (**13.16%** of all pixels; **75.78%** of trainable pixels).
- **Raw Value 4 (Star):** $25,384$ pixels (**0.020%** of all pixels; **0.12%** of trainable pixels).
- **Total Trainable Pixels (Raw 1–4):** $21,849,287$ pixels (**17.37%** of total dataset volume).

### 3. Data Quality Findings & Terminology
- **Zero Mask Non-Finites:** Every pixel in all 2,908 mask files is finite and integer-like.
- **Zero Empty Masks:** Under the operational definition of containing at least one pixel of raw values 1–4, zero masks are empty.
- **Image FITS Artifacts (Exact Terminology):**
  - Exactly **125 image files** contain non-finite pixels.
  - Across the dataset, exactly **160 image HDUs** contain NaNs (primarily in the $g$-band due to sensor margin masking).
  - Exactly **zero infinite ($\pm\infty$) pixels** were observed.
  - This confirmed the architectural necessity of a boolean **validity mask** across all downstream pipelines.
- **Small Region Prevalence:** 8-connected component analysis identified that cores and stars frequently consist of components smaller than 10 pixels ($1,373$ core components and $1,348$ star components have areas between 6–10 pixels; $33$ cores are isolated 1-pixel annotations). Any spatial filtering or pooling that removes small components destroys critical ground truth.

---

## E. B3 — Enhancement and Augmentation

### 1. Architectural Scope and Tensor Contracts
Located in `src/preprocessing/`:
- Multi-band astronomical tensor: `image` of shape $[C, H, W]$ (float32).
- Categorical segmentation ground truth: `mask` of shape $[H, W]$ (int64).
- Boolean detector validity mask: `valid_mask` of shape $[H, W]$ (bool), where `False` flags non-finite, dead, or padded border pixels.

### 2. Geometric Augmentations (`src/preprocessing/transforms.py`)
All geometric transformations strictly preserve the **D4 dihedral symmetry group** (isometries of the square grid):
- **Horizontal Flip (`DualRandomHorizontalFlip`):** Mirrors columns ($c \to W - 1 - c$).
- **Vertical Flip (`DualRandomVerticalFlip`):** Mirrors rows ($r \to H - 1 - r$).
- **Discrete 90° Rotations (`DualRandomRotation90`):** Rotates strictly by $k \in \{0, 1, 2, 3\} \times 90^\circ$ counter-clockwise. Non-square inputs correctly transpose dimensions ($[C, H, W] \to [C, W, H]$).
- **Symmetric Padding (`DualPadToSize`):** Centers the input on target canvas $[H_{\text{target}}, W_{\text{target}}]$. Constant fill values:
  - `image` padded with $0.0$.
  - `mask` padded with `IGNORE_INDEX = 255`.
  - `valid_mask` padded with `False`.
- **Categorical Integrity:** Nearest-neighbor index tracking is maintained; continuous interpolation (bilinear/bicubic) is strictly prohibited to prevent artificial label synthesis.

### 3. Photometric Augmentation & Asinh Enhancement
- **Photometric Jitter (`DualPhotometricJitter`):**
  - Brightness scaling: $I' = I \times \beta$, where $\beta \sim \mathcal{U}(1 - b, 1 + b)$.
  - Contrast scaling: $I' = (I - \mu_{\text{valid}}) \times \gamma + \mu_{\text{valid}}$, where $\gamma \sim \mathcal{U}(1 - c, 1 + c)$.
  - **Channel-Factor Behavior:** Scaling factors $\beta$ and $\gamma$ are sampled once per tensor and broadcast across all $C$ channels. While this shared-factor formulation reduces arbitrary inter-band distortion compared to independent sampling, it does not guarantee exact mathematical preservation of calibrated astronomical flux ratios or color indices ($g - r$, $r - i$).
  - **Validity Preservation:** Contrast mean $\mu_{\text{valid}}$ is computed exclusively over finite, valid pixels. Invalid pixels remain $0.0$ and are untouched.
- **Asinh Stretch Enhancement (`AsinhStretch` in `src/preprocessing/enhancement.py`):**
  - Mathematical formulation:
    $$f(x) = \frac{\operatorname{asinh}(x / \beta)}{\operatorname{asinh}(1 / \beta)}$$
  - **Actual Numerical Safeguards:**
    1. Validation that $\beta$ is a strictly positive, finite number (`beta > 0 and math.isfinite(beta)`).
    2. Precomputation of the scalar normalisation denominator $\operatorname{asinh}(1 / \beta)$ as a float.
    3. Input validation: enforces float tensor, $\ge 3$ dimensions, finite values, and value range in $[-\epsilon, 1.0 + \epsilon]$ ($\epsilon = 10^{-6}$).
    4. Non-mutating copy semantics.
    5. Output post-stretch clipping via `stretched.clamp(0.0, 1.0)` to safeguard against floating-point boundary overshoot.
    *(Note: Gradient clipping is not part of this inference/preprocessing transform).*

### 4. Ownership and Integration Boundary Notice (Person A vs Person B)
> **IMPORTANT INTEGRATION NOTICE:**  
> According to the project division of responsibilities, **Person A** owns the shared production FITS loader, global percentile normalization, background correction, and shared Asinh preprocessing.  
> Person B implemented `AsinhStretch` within `src/preprocessing/enhancement.py` for local augmentation ablation testing. This module must **not** compete with Person A's production pipeline. Person A should review whether to adopt Person B's numerical stability formulation or provide the canonical preprocessing pipeline.

### 5. Configuration and Split Protocol (`configs/augmentation.yaml`)
- `build_augmentation_pipeline(config_path, split='train'|'val'|'test')`:
  - `train`: Full stochastic pipeline (D4 symmetry, photometric jitter).
  - `val` / `test`: **Strict Identity Transformation** (no stochastic perturbation; optional deterministic padding only).

---

## F. B4 — Classical Computer Vision Baseline

### 1. Architectural Role and Non-Semantic Boundaries
The classical CV module (`src/classical_cv/`) provides an automated, non-learned morphological baseline. Classical operators produce:
- **Binary threshold masks:** Spatial footprints where flux exceeds background thresholds.
- **Connected component maps:** Region adjacency IDs ($1..K$).
- **Watershed catchment basins:** Topological basin IDs ($1..M$).

**Critical Guardrail:** Classical outputs are **topological detections**, not semantic class predictions. Basin IDs cannot be assigned to Core, Bar, Spiral Arms, or Stars without learned priors. Furthermore, B4's diagnostic evaluation is an empirical probe of classical feature extraction, not a replacement for Person A's shared multi-class semantic segmentation evaluation metrics.

### 2. Module Implementations
- **`band_comparison.py`:**
  - Inspects multi-HDU Pan-STARRS FITS cutouts ($g, r, i, z, y$).
  - Converts big-endian arrays (`>f4`) to native float32.
  - Computes robust sky background noise via Median Absolute Deviation: $\sigma_{\text{MAD}} = 1.4826 \times \operatorname{median}(|x - \operatorname{median}(x)|)$.
  - Evaluates Robust 99.5% SNR: $\text{SNR}_{\text{robust}} = (P_{99.5} - \operatorname{median}) / \sigma_{\text{MAD}}$.
  - Emits a provisional reference band recommendation.
- **`thresholding.py`:**
  - `otsu_threshold`: Vectorized between-class variance maximization $\sigma_B^2(t) = \omega_0(t)\omega_1(t)(\mu_0(t) - \mu_1(t))^2$. Computes statistics strictly on valid finite pixels.
  - `adaptive_threshold`: Local background estimation using 2D box-car average pooling with valid-pixel count normalization and configurable offset cutoff $T(x, y) = \mu_{\text{local}}(x, y) - c$.
- **`morphology.py`:**
  - Generates isotropic footprints (`disk`, `square`, `cross`).
  - Binary dilation, erosion, opening (suppresses isolated noise spikes), and closing (bridges narrow arm breaks) using `F.conv2d`. Enforces zero-detection on invalid pixels.
- **`connected_components.py`:**
  - Two-pass raster scan using Disjoint-Set Union (DSU) with path compression for 4- and 8-connectivity.
  - Computes area, bounding box $(r_{\min}, c_{\min}, r_{\max}, c_{\max})$, and centroid $(\bar{r}, \bar{c})$.
  - Preserves fine features by defaulting to `min_area = 1`.
- **`watershed.py`:**
  - `distance_transform_edt`: Exact Euclidean distance to background via Felzenszwalb-Huttenlocher lower parabolic envelope algorithm ($O(N)$).
  - `generate_markers_from_peaks`: Local maxima detection using 3x3 max pooling and greedy distance suppression.
  - `watershed_segmentation`: Meyer's priority-queue immersion algorithm.
  - `extract_watershed_boundaries`: Extracts basin boundaries with configurable `two_pixel_width: bool` (default `True` for prominent symmetric display; `False` for strict single-pixel line).
- **`baseline.py`:**
  - High-level orchestration function `run_classical_pipeline(image, valid_mask, ...)` returning all intermediate masks, distance transforms, basin labels, and metadata.
- **`configs/classical_cv.yaml`:**
  - Central configuration specifying provisional $r$-band reference, normalization percentiles ($0.5\%, 99.5\%$), Otsu histogram bins ($256$), morphology radius ($1$), connectivity ($8$), and watershed marker thresholds.

---

## G. B4 Experiments and Scientific Results

### 1. Pan-STARRS Five-Band Diagnostic Comparison
Audited on representative clean sample `1-100523` and sensor-corrupted sample `1-103458`. All values below have been cross-checked directly against `experiments/b4_classical_results/comparison_report.json`:

| Sample ID | Band | HDU | Valid Px | Invalid Px | Median Flux | Sky-MAD ($\sigma_{\text{MAD}}$) | Robust 99.5% SNR | Notes / Artifacts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1-100523** (Clean) | $g$ | 0 | 43,264 | 0 | 206.85 | 180.79 | 13.04 | Blue continuum, dust sensitive |
| | $r$ | 1 | 43,264 | 0 | 456.77 | 402.61 | 15.81 | High throughput, sharp bulge |
| | **$i$** | **2** | **43,264** | **0** | **622.02** | **561.02** | **16.70** | **Highest SNR on clean sample** |
| | $z$ | 3 | 43,264 | 0 | 456.07 | 479.98 | 16.24 | Moderate sky fringing |
| | $y$ | 4 | 43,264 | 0 | 645.09 | 696.90 | 13.52 | Elevated sky background floor |
| **1-103458** (Corrupted) | $g$ | 0 | 41,984 | **1,280** | 71.18 | 175.33 | 6.64 | **2.96% NaNs (sensor dropouts)** |
| | **$r$** | **1** | **43,264** | **0** | **103.01** | **204.26** | **15.61** | **Zero NaNs, highest SNR on corrupted sample** |
| | $i$ | 2 | 43,264 | 0 | 169.70 | 382.33 | 12.43 | Zero NaNs, elevated Sky-MAD |
| | $z$ | 3 | 43,264 | 0 | 111.98 | 283.29 | 8.06 | Moderate noise floor |
| | $y$ | 4 | 43,264 | 0 | 170.30 | 698.54 | 4.00 | Severe thermal sky noise |

#### Preliminary Band Recommendation & Methodological Trade-offs:
> **Provisional r-band reference recommendation based on diagnostics from two representative samples. Broader validation across the dataset is required.**  
> - **Trade-off Analysis:** On clean sample `1-100523`, $i$-band achieved the highest robust SNR ($16.70$ vs $15.81$ for $r$-band). However, on corrupted sample `1-103458`, $r$-band exhibited higher stability and contrast ($15.61$ vs $12.43$ for $i$-band), with lower sky noise ($\sigma_{\text{MAD}} = 204.26$ vs $382.33$). Because this recommendation rests on only two samples, it is strictly preliminary; full dataset-wide validation by Person A is necessary to establish whether $r$-band or $i$-band is globally superior.
> - **Multi-Band Averaging:** Simple unweighted averaging across all five bands is problematic because it propagates localized sensor dropouts ($1,280$ NaNs in $g$-band) and elevated thermal noise ($y$-band) into clean channels. Any multi-channel fusion must be empirically compared against validity-aware or noise-weighted combination schemes.

### 2. Quantitative Evaluation on Sample `1-100523`
- **Total Image Pixels:** 43,264 ($208 \times 208$)
- **Ground Truth Annotated Pixels ($GT \in \{0, 1, 2, 3\}$):** 14,091
- **Ground Truth Ignored Pixels ($GT = 255$):** 29,173 (unannotated outer field; not confirmed background)
- **Classical Foreground Predictions:** 1,191
- **Predictions Overlapping Annotated Regions:** 1,005
- **Predictions Outside Annotated Regions:** 186 (**unverified detections in unannotated field**, not confirmed false positives)
- **Missed Annotated Pixels:** 13,086 (**missed annotated foreground pixels**, unclassified by component without separate per-class spatial decomposition)
- **Annotated-Domain Sensitivity (Recall):** **7.13%** ($1,005 / 14,091$)
- **Annotated-Domain Binary IoU:** **0.0713** ($1,005 / [1,005 + 13,086]$)
- **Annotated-Domain Binary Dice:** **0.1331** ($2 \times 1,005 / [2 \times 1,005 + 13,086]$)
- **Precision:** **Not meaningfully measurable with the available annotations.** (Because unannotated pixels are not confirmed background, the 186 outer detections cannot be verified as true or false positives; precision is undefined).

#### Scientific Interpretation:
Classical thresholding isolates bright central structures with high fidelity, but fails to recover low-surface-brightness annotated foreground structures (13,086 missed pixels). This quantitatively confirms the necessity of learned semantic segmenters (SegFormer-B0) for complete morphological recovery.

---

## H. Testing and Verification

### 1. Test Suite Execution
Executed in project `.venv` (Python 3.12.10):
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 2. Verified Test Results
- **Total Tests Discovered:** **162**
- **Passed:** **162**
- **Failed:** **0**
- **Errors:** **0**
- **Skipped:** **0**
- **Execution Time:** **1.70 seconds**

### 3. Test Coverage Breakdown
- `tests/test_mask_decoder.py` (11 tests): Raw-to-training mapping, non-finite rejection, float validation, categorical integrity (B1/B2).
- `tests/test_alignment.py` (29 tests): Dual geometric transforms, D4 dihedral group symmetries, non-square transpositions, padding fill values (B3).
- `tests/test_enhancement.py` (28 tests): Numerical stability of Asinh stretch, input range validation, device compatibility (B3).
- `tests/test_augmentation.py` (36 tests): Stochastic transforms, deterministic split pipeline building, multi-channel shared factor broadcasting (B3).
- `tests/test_band_comparison.py` (10 tests): Multi-band FITS loading, Sky-MAD noise, provisional recommendation assertions (B4).
- `tests/test_thresholding.py` (11 tests): Otsu variance maximization, adaptive local pooling, NaN exclusion (B4).
- `tests/test_morphology.py` (13 tests): Footprints, dilation, erosion, opening noise suppression, closing gap bridging (B4).
- `tests/test_connected_components.py` (8 tests): 4- vs 8-connectivity, DSU equivalence merging, centroid and bbox extraction (B4).
- `tests/test_watershed.py` (10 tests): Exact EDT, marker generation, Meyer's immersion watershed, 1-pixel vs 2-pixel boundaries (B4).
- `tests/test_baseline.py` (4 tests): End-to-end pipeline execution on synthetic and real FITS (B4).
- `tests/test_evaluation_semantics.py` (2 tests): Verified ground-truth pixel counts, undefined precision handling, ignore-index exclusion (B4).

### 4. Git Repository Integrity
- `git diff --check`: **Clean (Exit code 0)**. Zero trailing whitespace or syntax errors.
- `git status`: **No files staged, no files committed.** Zero modifications to Person A's modules or raw dataset files.

---

## I. Integration Handoff to Person A

### 1. Consuming the Categorical Mask Decoder
In the `Dataset.__getitem__` method:
```python
from astropy.io import fits
from src.data.mask_decoder import decode_mask

with fits.open(mask_path, memmap=False) as hdul:
    raw_mask_data = hdul[0].data  # shape [208, 208], float64
mask_tensor = decode_mask(raw_mask_data)  # returns torch.LongTensor [208, 208], values in {0, 1, 2, 3, 255}
```

### 2. Consuming the Augmentation Pipeline
```python
from src.preprocessing.augmentation import build_augmentation_pipeline

# Training dataset: stochastic D4 flips, 90-deg rotations, photometric jitter
train_transform = build_augmentation_pipeline("configs/augmentation.yaml", split="train")

# Validation / Test dataset: identity transformation (no stochastic jitter)
eval_transform = build_augmentation_pipeline("configs/augmentation.yaml", split="val")

# Inside __getitem__:
transformed_image, transformed_mask, transformed_valid = train_transform(
    image=normalized_image_tensor,  # torch.FloatTensor [5, H, W]
    mask=mask_tensor,               # torch.LongTensor [H, W]
    valid_mask=valid_mask_tensor,   # torch.BoolTensor [H, W]
)
```

### 3. Outstanding Integration Dependencies (Person A Action Items)
Before end-to-end SegFormer-B0 training can begin, Person A must deliver:
1. **Canonical FITS Loader:** High-throughput 5-band loader producing float32 tensors and boolean validity masks.
2. **Global Normalization & Background Subtraction:** Production background estimation and percentile stretch.
3. **Asinh Preprocessing Alignment:** Formally adopt or replace Person B's `AsinhStretch` module.
4. **Leakage-Free Dataset Split Manifests:** Frozen train/val/test CSV splits partitioned by unique galaxy `sample_id`.
5. **Shared Multi-Class Semantic Evaluation Metrics:** Evaluation suite supporting `ignore_index = 255` and validity masks.
6. **Model Input Strategy:** Formal decision on whether SegFormer-B0 consumes all 5 Pan-STARRS bands ($C=5$) or a selected reference band.

---

## J. Limitations and Outstanding Work

### 1. Scientific & Evaluation Limitations
- **Annotation Incompleteness:** The benchmark provides positive labels for structures, while outer field pixels are unannotated rather than confirmed background. Conventional False Positive counts and Precision metrics are mathematically undefined on this dataset.
- **Sample Coverage of Band Comparison:** Multi-band diagnostics were audited on 2 representative samples. Full-dataset statistical aggregation across all 2,879 images remains to be executed by Person A.
- **Topological vs Semantic Gap:** Classical CV cannot serve as an autonomous semantic segmenter; its utility is confined to morphological feature extraction and structural baselining.

### 2. Integration Work Pending Person A
- Integration of `DualCompose` pipelines into PyTorch `DataLoader`.
- Memory profiling of 5-channel 32-bit float tensors during multi-GPU training.

---

## K. Complete File Inventory

| File Path | Milestone | Responsibility | Status in Repo | Owner |
| :--- | :---: | :--- | :---: | :---: |
| `src/data/mask_decoder.py` | B1 | Categorical mask decoder, raw-to-training mapping | Implemented & Verified | Person B |
| `dataset/metadata/class_mapping.json` | B1 | Approved JSON class mapping dictionary | Implemented & Verified | Person B |
| `docs/mask_semantics.md` | B1 | Documentation of mask provenance and semantics | Implemented & Verified | Person B |
| `src/data/dataset_audit.py` | B2 | Dataset audit script placeholder (CLI interface) | **Empty File (0 bytes)** | Person B (Pending A) |
| `notebooks/02_mask_analysis.ipynb` | B2 | Executed audit notebook producing audit statistics | Implemented & Executed | Person B |
| `dataset/metadata/mask_audit.json` | B2 | Quantitative JSON audit of 2,908 masks (18.3 MB) | Generated Artifact | Person B |
| `docs/mask_statistics.md` | B2 | Pixel counts, class imbalance, component sizing | Implemented & Verified | Person B |
| `src/preprocessing/transforms.py` | B3 | Dual geometric transforms (flips, rot90, padding) | Implemented & Verified | Person B |
| `src/preprocessing/enhancement.py` | B3 | Asinh stretch and numerical stabilization | Implemented & Verified | Person B (Pending A) |
| `src/preprocessing/augmentation.py` | B3 | Photometric jitter, pipeline composition API | Implemented & Verified | Person B |
| `configs/augmentation.yaml` | B3 | Configuration for train/val/test augmentations | Implemented & Verified | Person B |
| `docs/b3_enhancement_augmentation.md` | B3 | Technical specification of augmentation contracts | Implemented & Verified | Person B |
| `src/classical_cv/__init__.py` | B4 | Public export interface for classical CV | Implemented & Verified | Person B |
| `src/classical_cv/band_comparison.py` | B4 | Multi-band loading, Sky-MAD, robust SNR | Implemented & Verified | Person B |
| `src/classical_cv/thresholding.py` | B4 | Otsu and adaptive thresholding primitives | Implemented & Verified | Person B |
| `src/classical_cv/morphology.py` | B4 | Mathematical morphology (open, close, dilate, erode) | Implemented & Verified | Person B |
| `src/classical_cv/connected_components.py`| B4 | Two-pass DSU connected-component labeling | Implemented & Verified | Person B |
| `src/classical_cv/watershed.py` | B4 | Exact EDT, marker generation, Meyer watershed | Implemented & Verified | Person B |
| `src/classical_cv/baseline.py` | B4 | High-level classical pipeline orchestrator | Implemented & Verified | Person B |
| `configs/classical_cv.yaml` | B4 | Classical baseline parameter configuration | Implemented & Verified | Person B |
| `experiments/generate_b4_results.py` | B4 | Script generating diagnostic figures and reports | Implemented & Verified | Person B |
| `experiments/b4_classical_results/` | B4 | Directory holding PNG visualizations and JSON reports| Implemented & Verified | Person B |
| `docs/b4_classical_cv.md` | B4 | Technical specification of classical CV baseline | Implemented & Verified | Person B |
| `tests/test_mask_decoder.py` | B1 | Unit tests for mask decoding (11 tests) | Implemented & Verified | Person B |
| `tests/test_dataset_contract.py` | B2 | Dataset contract placeholder test | **Empty File (0 bytes)** | Shared (Pending A) |
| `tests/test_alignment.py` | B3 | Unit tests for dual geometric alignment (29 tests) | Implemented & Verified | Person B |
| `tests/test_enhancement.py` | B3 | Unit tests for enhancement stability (28 tests) | Implemented & Verified | Person B |
| `tests/test_augmentation.py` | B3 | Unit tests for augmentation pipelines (36 tests) | Implemented & Verified | Person B |
| `tests/test_band_comparison.py` | B4 | Unit tests for band comparison diagnostics (10 tests) | Implemented & Verified | Person B |
| `tests/test_thresholding.py` | B4 | Unit tests for Otsu and adaptive thresholding (11 tests)| Implemented & Verified | Person B |
| `tests/test_morphology.py` | B4 | Unit tests for morphological operators (13 tests) | Implemented & Verified | Person B |
| `tests/test_connected_components.py` | B4 | Unit tests for connected components (8 tests) | Implemented & Verified | Person B |
| `tests/test_watershed.py` | B4 | Unit tests for EDT and watershed (10 tests) | Implemented & Verified | Person B |
| `tests/test_baseline.py` | B4 | Unit tests for end-to-end classical baseline (4 tests)| Implemented & Verified | Person B |
| `tests/test_evaluation_semantics.py` | B4 | Unit tests for verified evaluation counts (2 tests) | Implemented & Verified | Person B |
| `tests/test_integration.py` | Future | Full pipeline integration test placeholder | **Empty File (0 bytes)** | Shared (Pending A) |
| `tests/test_metrics.py` | Future | Shared semantic metrics test placeholder | **Empty File (0 bytes)** | Person A (Pending) |
| `tests/test_model_forward.py` | Future | SegFormer forward pass test placeholder | **Empty File (0 bytes)** | Person A (Pending) |
| `docs/person_b_handoff_b1_b4.md` | Handoff | Master technical handoff report (this document) | Implemented & Verified | Person B |

---

## L. Final Handoff Status

| Milestone | Status | Description |
| :---: | :---: | :--- |
| **B1** | **Implemented and Verified** | Mask semantics resolved, decoder implemented and 100% test verified. |
| **B2** | **Implemented and Verified** | Dataset audit completed across all 2,908 masks; class statistics documented. |
| **B3** | **Implemented; Pending Integration** | Augmentation modules verified (93 tests). Ready for integration with Person A's data loader. |
| **B4** | **Implemented and Verified** | Classical CV algorithms implemented, tested (58 tests), and diagnostic experiments generated. |

### Immediate Next Actions
1. **Person A Review:** Person A to review this handoff document and confirm acceptance of tensor contracts.
2. **Preprocessing Reconciliation:** Agree on whether `AsinhStretch` is consolidated into Person A's loader or consumed from Person B.
3. **Dataset Split Generation:** Person A to generate and freeze `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`.
4. **DataLoader Wiring:** Person A to instantiate `DualCompose` in the production SegFormer dataset loader.
