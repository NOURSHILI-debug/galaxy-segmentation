# B4 — Classical Computer Vision Baseline: Comprehensive Documentation

**Project:** Galaxy Segmentation — Astronomy-Aware Preprocessing and SegFormer-B0  
**Module:** `src/classical_cv/`  
**Author:** Person B (Classical Computer Vision & Augmentation)  
**Configuration:** `configs/classical_cv.yaml`  
**Test Suite:** `tests/test_band_comparison.py`, `tests/test_thresholding.py`, `tests/test_morphology.py`, `tests/test_connected_components.py`, `tests/test_watershed.py`, `tests/test_baseline.py`

---

## 1. Overview and Scientific Motivation

The classical computer vision baseline provides an independent, deterministic reference for galaxy morphology analysis. While deep learning segmentation architectures (such as SegFormer-B0) infer subtle, multi-scale semantic representations (core, bar, spiral arm, star), classical morphological pipelines extract direct intensity and topological structures without gradient descent or learned weight matrices.

### Core Objectives
1. **Multi-Band Evaluation**: Rigorously analyze all five Pan-STARRS bands ($g, r, i, z, y$) to determine the optimal morphological reference band based on signal-to-noise ratio (SNR), sky background variance (Sky-MAD), and non-finite pixel distributions.
2. **Deterministic Intensity Thresholding**: Provide robust Global Otsu and Local Adaptive thresholding algorithms that strictly exclude invalid/non-finite pixels and avoid arbitrary binary class assumptions.
3. **Mathematical Morphology**: Provide isotropic structuring elements (disk, square, cross) for opening (high-frequency noise spike suppression) and closing (inter-arm continuity restoration).
4. **Connected Components Analysis**: Provide two-pass disjoint-set labeling (4- and 8-connectivity) to quantify component areas, centroids, and bounding boxes, while preserving faint structures (`min_area=1`).
5. **Marker-Controlled Watershed**: Implement exact Euclidean Distance Transform (EDT) and Meyer's immersion watershed algorithm to separate blended or overlapping astronomical structures.
6. **Strict Scientific Boundaries**: Classical outputs (binary detections, component IDs, catchment basins) are fundamentally structural and **not** semantically equivalent to the 4 ground-truth training classes. Ignore index 255 (unannotated benchmark pixels) is never conflated with confirmed background.

---

## 2. Pan-STARRS Five-Band Diagnostic Analysis

Each Pan-STARRS FITS file comprises five single-band HDUs corresponding to filters spanning the optical and near-infrared spectrum:
- **HDU 0 ($g$-band, $\lambda_{\text{eff}} \approx 481\text{ nm}$)**: Sensitive to star formation and young stellar populations; susceptible to atmospheric scattering and dust extinction.
- **HDU 1 ($r$-band, $\lambda_{\text{eff}} \approx 617\text{ nm}$)**: High system throughput, sharp point spread function (PSF), well-defined galactic disks and bulges.
- **HDU 2 ($i$-band, $\lambda_{\text{eff}} \approx 752\text{ nm}$)**: Excellent stellar mass tracer, minimal dust attenuation, strong contrast for spiral features.
- **HDU 3 ($z$-band, $\lambda_{\text{eff}} \approx 866\text{ nm}$)**: Near-infrared; lower atmospheric transmission and higher sky background.
- **HDU 4 ($y$-band, $\lambda_{\text{eff}} \approx 962\text{ nm}$)**: Deep near-infrared; severe thermal background contamination and atmospheric water vapor emission.

### Quantitative Diagnostics

Using representative samples (`1-100523` clean and `1-103458` containing detector artifacts/NaNs), we computed the following metrics per band:
- **Sky Background Median and Noise ($\sigma_{\text{MAD}}$)**:
  $$\sigma_{\text{MAD}} = 1.4826 \times \text{median}(|x - \text{median}(x)|)$$
- **Robust 99.5% SNR**:
  $$\text{SNR}_{\text{robust}} = \frac{P_{99.5} - \text{median}}{\sigma_{\text{MAD}}}$$

| Band | HDU | Valid Px (Clean) | Sky-MAD (Clean) | Robust SNR (Clean) | Valid Px (NaN Sample) | Invalid Px | Sky-MAD (NaN Sample) | Robust SNR (NaN Sample) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$g$** | 0 | 43,264 | 180.79 | 13.04 | 41,984 | **1,280** (2.96%) | 20.89 | 10.98 |
| **$r$** | **1** | **43,264** | **402.61** | **15.81** | **43,264** | **0** (0.00%) | **106.66** | **15.61** |
| **$i$** | 2 | 43,264 | 561.02 | 16.70 | 43,264 | 0 (0.00%) | 158.33 | 14.86 |
| **$z$** | 3 | 43,264 | 479.98 | 16.24 | 43,264 | 0 (0.00%) | 283.47 | 8.06 |
| **$y$** | 4 | 43,264 | 698.54 | 4.00 | 43,264 | 0 (0.00%) | 698.54 | 4.00 |

### Scientific Verdict and Recommendation
- **Provisional Reference Recommendation:** **$r$-band (HDU 1)**: Provisional r-band reference recommendation based on diagnostics from two representative samples. Broader validation across the dataset is required. In the audited samples, it provided the highest robust structural SNR ($\approx 15.6 - 15.8$), zero invalid pixels, and stable sky background statistics.
- **Alternative Band:** **$i$-band (HDU 2)** is fully supported and recommended as a high-contrast alternative for bulge/disk mass profiling. All 5 bands remain configurable via `configs/classical_cv.yaml`.
- **Excluded Bands:** $g$-band is provisional rejected due to vulnerability to masked sensor dropouts (1,280 NaNs in sample `1-103458`). $y$-band is rejected due to elevated sky noise ($\sigma_{\text{MAD}} \approx 698.5$), reducing SNR to $\approx 4.0$. Averaging all five bands is scientifically contraindicated because it would propagate $g$-band NaNs and $y$-band noise into clean channels.

---

## 3. Pipeline Architecture and Implementation Details

```
+-------------------------------------------------------------------------+
|                  Pan-STARRS Multi-HDU FITS Image                        |
+-------------------------------------------------------------------------+
                                     |
                                     v
                       [Stage 1: Band Selection]
                           Select r-band (HDU 1)
                     Validity Mask Creation (NaN/Inf)
                                     |
                                     v
                        [Provisional Normalization]
                     Flux clipped to [P_0.5, P_99.5] -> [0, 1]
                                     |
                                     v
                        [Stage 2: Thresholding]
              Global Otsu (between-class variance maximization)
               OR Local Adaptive (box-filter local background)
                                     |
                                     v
                        [Stage 3: Morphology]
               Disk kernel (radius=1, 3x3 footprint)
              Binary Opening -> Cleans high-frequency noise
              Binary Closing -> Bridges inter-arm discontinuities
                                     |
                                     v
                  +-----------------------------------+
                  |                                   |
                  v                                   v
      [Stage 4: Connected Components]    [Stage 5: Distance Transform]
      Two-pass Disjoint-Set (Union-Find)  Exact Euclidean Distance (EDT)
      Area, Bounding Box, Centroids       Marker Generation (Local Maxima)
      Preserves small features (min=1)                |
                                                      v
                                         [Stage 6: Meyer Watershed]
                                         Inverted distance relief (-EDT)
                                         Catchment basin immersion
                                         1-pixel boundary extraction
```

### Module Breakdown
- `src/classical_cv/band_comparison.py`: FITS big-endian loading (`>f4`), validity-mask extraction, robust statistical metrics, automatic recommendation engine.
- `src/classical_cv/thresholding.py`:
  - `otsu_threshold`: Computes optimal intensity cutoff maximizing between-class variance strictly over valid pixels.
  - `adaptive_threshold`: Vectorized local box filter using `F.avg_pool2d` with valid pixel normalizers and configurable spatial offset.
- `src/classical_cv/morphology.py`: Structuring element generator (`get_structuring_element`) and morphological filters (`binary_dilation`, `binary_erosion`, `binary_opening`, `binary_closing`). Enforces valid mask bounds.
- `src/classical_cv/connected_components.py`: Two-pass raster scan with path compression Disjoint-Set Union (DSU) supporting 4- and 8-connectivity. Extracts bounding boxes, areas, and centroids.
- `src/classical_cv/watershed.py`:
  - `distance_transform_edt`: Exact $O(N)$ separable parabolic envelope algorithm (Felzenszwalb & Huttenlocher) in pure NumPy, with automatic SciPy acceleration if available.
  - `generate_markers_from_peaks`: True local maxima peak detector using 3x3 spatial pooling and greedy distance suppression.
  - `watershed_segmentation`: Meyer's topographic immersion algorithm via priority queue (`heapq`).
  - `extract_watershed_boundaries`: Vectorized 1-pixel inter-basin boundary extraction.
- `src/classical_cv/baseline.py`: High-level entry point `run_classical_pipeline` orchestrating all stages deterministically.

---

## 4. Evaluation and Ground Truth Semantics

### Critical Semantic Distinction
The project benchmark ground truth defines four distinct morphological semantic classes:
- Training ID 0 (Raw 1): Galaxy center / core
- Training ID 1 (Raw 2): Galaxy bar
- Training ID 2 (Raw 3): Spiral arms
- Training ID 3 (Raw 4): Foreground stars
- Ignore Index 255 (Raw 0): Unannotated regions / sky benchmark boundary

**Why Classical Outputs Cannot Be Mapped to 4 Classes:**
Classical intensity thresholding and watershed segmentation detect contiguous photometric regions and catchment basins. They do not distinguish a stellar point spread function from a galaxy core, nor do they separate spiral arms from galactic bars without semantic priors or supervised classification.

### Quantitative Overlap Metrics (Sample `1-100523`)
When evaluating classical foreground detections against the benchmark ground truth:
- **Total Pixels:** 43,264
- **Ground Truth Annotated Pixels ($GT \in \{0, 1, 2, 3\}$):** 14,091
- **Ground Truth Ignored Pixels ($GT = 255$):** 29,173 (unannotated field/sky margin; not confirmed background)
- **Classical Foreground Predictions:** 1,191
- **Predictions Overlapping Annotated Regions:** 1,005
- **Predictions Outside Annotated Regions:** 186 (unverified detections in unannotated field)
- **Missed Annotated Pixels:** 13,086
- **Annotated-Domain Sensitivity (Recall):** **7.13%** ($1,005 / 14,091$)
- **Annotated-Domain Binary IoU:** **0.0713** ($1,005 / [1,005 + 13,086]$)
- **Annotated-Domain Binary Dice:** **0.1331** ($2 \times 1,005 / [2 \times 1,005 + 13,086]$)
- **Precision:** **Not meaningfully measurable with the available annotations.** (Because unannotated pixels are not confirmed background, the 186 outer detections cannot be classified as true or false positives; hence precision is undefined rather than 100% or 84.38%).

*Analysis:* Classical thresholding captures the high-surface-brightness core and central bar, but misses faint, low-surface-brightness spiral arms in the outer disk (13,086 missed pixels). This quantitatively demonstrates the fundamental necessity of deep learning architectures (SegFormer-B0) for comprehensive galaxy morphology segmentation.

---

## 5. Test Suite Verification

The classical CV test suite was fully executed within the isolated Python 3.12 environment.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Results:**
- **Total Tests Executed:** 160
- **Failures:** 0
- **Errors:** 0
- **Status:** **PASSED (160/160 in 1.78s)**
- **Coverage Breakdown:**
  - `tests/test_mask_decoder.py`: 11 tests (Person B1/B2 regression)
  - `tests/test_alignment.py`: 29 tests (Person B3 regression)
  - `tests/test_enhancement.py`: 28 tests (Person B3 regression)
  - `tests/test_augmentation.py`: 36 tests (Person B3 regression)
  - `tests/test_band_comparison.py`: 10 tests (Person B4)
  - `tests/test_thresholding.py`: 11 tests (Person B4)
  - `tests/test_morphology.py`: 13 tests (Person B4)
  - `tests/test_connected_components.py`: 8 tests (Person B4)
  - `tests/test_watershed.py`: 10 tests (Person B4)
  - `tests/test_baseline.py`: 4 tests (Person B4)
