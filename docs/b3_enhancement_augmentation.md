# B3 — Image Enhancement and Data Augmentation

## 1. Objectives

Person B3 provides two capabilities for the galaxy segmentation pipeline:

1. **Optional nonlinear image enhancement** — an inverse hyperbolic sine (asinh) intensity stretch that compresses high dynamic range while preserving faint structures near the noise floor.
2. **Synchronised data augmentation** — geometric and conservative photometric augmentations that operate identically on image, segmentation mask, and validity mask to prevent spatial misalignment or label corruption.

These modules integrate between Person A's global normalisation (upstream) and the future PyTorch `Dataset` / training loop (downstream).

---

## 2. Enhancement methodology

### 2.1 Asinh stretch

The only enhancement implemented in the initial B3 release is the **asinh stretch**, the standard nonlinear intensity mapping used in professional astronomical imaging (Lupton et al. 2004).

**Module:** `src/preprocessing/enhancement.py`  
**Class:** `AsinhStretch`

### 2.2 Mathematical formulation

For a normalised pixel intensity $x \in [0, 1]$:

$$f(x) = \frac{\operatorname{asinh}(x / \beta)}{\operatorname{asinh}(1 / \beta)}$$

where $\beta > 0$ is the softness (knee) parameter.

**Properties:**

- $f(0) = 0$ and $f(1) = 1$ — the output is bounded in $[0, 1]$.
- Monotonically non-decreasing — preserves the intensity ordering of pixels.
- For $x \ll \beta$: approximately linear ($f(x) \approx x / \beta \cdot 1/\operatorname{asinh}(1/\beta)$) — faint structures near the noise floor are preserved linearly.
- For $x \gg \beta$: approximately logarithmic — high dynamic range (bright cores, stars) is compressed.

**Default:** $\beta = 0.1$. Recommended range: $0.01$–$0.5$.

### 2.3 Input / output contract

| Property | Value |
|---|---|
| Input dtype | Floating-point (`float32` or `float64`) |
| Input range | $[0, 1]$ (must be pre-normalised) |
| Input shape | `[C, H, W]` or `[B, C, H, W]` |
| Output dtype | `float32` |
| Output range | $[0, 1]$ (clamped for numerical safety) |
| Mutates input? | No |

### 2.4 CLAHE (future experiment E3)

CLAHE (Contrast-Limited Adaptive Histogram Equalisation) is **not implemented** in this release. It is documented as a future controlled experiment (E3) because:

- A correct CLAHE requires either OpenCV (`cv2.createCLAHE`) or `scikit-image`, neither of which is currently installed.
- Writing a custom tile-based adaptive histogram equalisation in pure PyTorch is non-trivial and error-prone.
- The scientific benefit for this dataset must be validated experimentally before investing in a custom implementation.

---

## 3. Geometric augmentation

**Module:** `src/preprocessing/augmentation.py`

### 3.1 Implemented transforms

| Transform | Class | Default probability | Description |
|---|---|---|---|
| Horizontal flip | `DualRandomHorizontalFlip` | 0.5 | `torch.flip(…, [-1])` — left ↔ right |
| Vertical flip | `DualRandomVerticalFlip` | 0.5 | `torch.flip(…, [-2])` — top ↔ bottom |
| 90° rotation | `DualRandomRotate90` | 0.5 | `torch.rot90(…, k, [-2, -1])`, $k \in \{1, 2, 3\}$ |
| Deterministic padding | `DualPadToSize` | N/A | Symmetric constant padding to target size |

### 3.2 Scientific justification

- **Flips and 90° rotations** are physically justified because galaxies have no preferred spatial orientation on the sky. These eight isometric operations form the dihedral group $D_4$, the symmetry group of the regular 2D square pixel grid, allowing lossless orientation sampling with zero interpolation error.
- **No continuous rotations** are implemented because they require interpolation, which blurs faint structures and corrupts categorical mask labels.
- **No random cropping** is implemented because 208×208 images already tightly frame the target galaxy, and cropping would discard annotated structures.

### 3.3 Low-level primitives

**Module:** `src/preprocessing/transforms.py`

| Function | Description |
|---|---|
| `functional_hflip(image, mask, valid_mask)` | Synchronised horizontal flip |
| `functional_vflip(image, mask, valid_mask)` | Synchronised vertical flip |
| `functional_rot90(image, mask, k, valid_mask)` | Synchronised 90° rotation |
| `functional_pad(image, mask, target_h, target_w, valid_mask, …)` | Synchronised symmetric padding |

All use pure PyTorch tensor operations — no interpolation, no external libraries.

---

## 4. Photometric augmentation

**Class:** `RandomPhotometricJitter`

| Parameter | Default | Range |
|---|---|---|
| Brightness factor | — | $(0.9, 1.1)$ |
| Contrast factor | — | $(0.9, 1.1)$ |
| Probability | 0.3 | $[0, 1]$ |

### 4.1 Implementation details

1. **Contrast** is applied as scaling around the per-channel spatial mean: $x' = \alpha_c \cdot (x - \mu) + \mu$.
2. **Brightness** is applied as a global multiplicative factor: $x'' = \alpha_b \cdot x'$.
3. A single brightness factor and a single contrast factor are sampled per image and shared across **all channels** to preserve relative band calibration.
4. Output is clamped to $[0, 1]$.
5. Pixels marked invalid by the validity mask are reset to $0.0$ after the jitter.

### 4.2 What is NOT modified

- The **segmentation mask** is never touched by photometric transforms.
- The **validity mask** is never touched by photometric transforms.

---

## 5. Image / mask synchronisation

Every spatial transform follows the same contract:

```python
def transform(image, mask, valid_mask=None):
    # Apply identical geometric operation to all tensors
    return (transformed_image, transformed_mask, transformed_valid_mask)
```

**Guarantees:**

- The same `torch.flip` / `torch.rot90` call is applied to all tensors.
- No interpolation is used — only exact index permutations.
- Mask categorical labels are never corrupted: the set of unique values in the output mask is always a subset of the input set.
- 1-pixel connected components survive all transforms.

---

## 6. Validity-mask handling

| Operation | Image | Mask | Validity mask |
|---|---|---|---|
| Flip / rotation | Transformed | Transformed identically | Transformed identically |
| Padding | Filled with `0.0` | Filled with `255` (ignore) | Filled with `False` |
| Photometric jitter | Modified | **Unchanged** | **Unchanged** |
| After photometric jitter | Invalid pixels reset to `0.0` | — | — |

When `valid_mask=None` is passed, the pipeline preserves its absence — it does not fabricate a validity mask.

---

## 7. Reproducibility

All stochastic transforms accept an optional `torch.Generator`:

```python
gen = torch.Generator().manual_seed(42)
pipeline = build_augmentation_pipeline("configs/augmentation.yaml", split="train", generator=gen)
```

**Guarantees:**

- Two pipelines using the same seed and inputs produce **identical** outputs.
- No reliance on the global `torch.manual_seed()` state.
- Validation and test splits apply zero stochastic transforms.

---

## 8. Configuration

**File:** `configs/augmentation.yaml`

The YAML file has three sections: `train`, `val`, `test`.

```yaml
train:
  horizontal_flip: true
  horizontal_flip_p: 0.5
  vertical_flip: true
  vertical_flip_p: 0.5
  random_rotate90: true
  random_rotate90_p: 0.5
  photometric_jitter: true
  photometric_jitter_p: 0.3
  brightness_min: 0.9
  brightness_max: 1.1
  contrast_min: 0.9
  contrast_max: 1.1
  pad: false

val:
  # All stochastic transforms disabled
  horizontal_flip: false
  vertical_flip: false
  random_rotate90: false
  photometric_jitter: false
  pad: false

test:
  # All stochastic transforms disabled
  horizontal_flip: false
  vertical_flip: false
  random_rotate90: false
  photometric_jitter: false
  pad: false
```

The pipeline is built using:

```python
from src.preprocessing.augmentation import build_augmentation_pipeline

pipeline = build_augmentation_pipeline("configs/augmentation.yaml", split="train")
```

---

## 9. Testing methodology

Three dedicated test modules:

| Module | Coverage |
|---|---|
| `tests/test_alignment.py` | Spatial alignment with asymmetric markers, categorical integrity, 1-pixel components, multi-channel, device compatibility |
| `tests/test_enhancement.py` | Output bounds, monotonicity, numerical stability, non-mutation, invalid inputs, multi-channel, batched, device compatibility |
| `tests/test_augmentation.py` | Pipeline composition, deterministic splits, reproducibility, label preservation, photometric mask invariance, shape contracts, config validation |

Run all B3 tests:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## 10. Known limitations

1. **CLAHE not implemented.** Requires OpenCV or scikit-image, which are not installed.
2. **No continuous rotations.** Excluded by design to avoid interpolation artefacts on categorical masks and faint structures.
3. **No random cropping.** The 208×208 images are already tightly framed.
4. **Enhancement expects [0, 1] input.** The upstream normalisation (Person A) must produce this range before enhancement is applied.
5. **Photometric jitter uses conservative bounds.** Wider ranges could help generalisation but risk drowning faint structures.
6. **No Gaussian/Poisson noise injection.** Excluded because the raw data already contains real calibrated noise, and synthetic noise risks destroying $S/N \sim 1$ arm features.

---

## 11. Integration interface

### For Person A (normalisation → enhancement)

```python
# Person A produces a normalised image in [0, 1]
normalised_image: torch.FloatTensor  # [C, H, W]
valid_mask: torch.BoolTensor         # [H, W]

# Person B3 enhancement (optional)
from src.preprocessing.enhancement import AsinhStretch
stretch = AsinhStretch(beta=0.1)
enhanced_image = stretch(normalised_image)
```

### For the future Dataset implementation

```python
from src.preprocessing.augmentation import build_augmentation_pipeline
from src.data.mask_decoder import decode_mask

# In Dataset.__getitem__:
image: torch.FloatTensor   # [C, H, W], normalised & enhanced
mask: torch.LongTensor     # [H, W], from decode_mask()
valid_mask: torch.BoolTensor  # [H, W], or None

aug_image, aug_mask, aug_valid = self.augmentation(image, mask, valid_mask)
```

### Tensor contracts

| Tensor | dtype | Shape | Values |
|---|---|---|---|
| Image (input) | `float32` | `[C, H, W]` | `[0, 1]` |
| Image (output) | `float32` | `[C, H', W']` | `[0, 1]` |
| Mask (input/output) | `int64` | `[H, W]` / `[H', W']` | `{0, 1, 2, 3, 255}` |
| Validity (input/output) | `bool` | `[H, W]` / `[H', W']` | `{True, False}` |

Where `H' × W'` equals `H × W` for flips/rotations, or `target_h × target_w` after padding.

---

## 12. Files created by B3

| File | Purpose |
|---|---|
| `src/preprocessing/transforms.py` | Pure PyTorch functional spatial primitives |
| `src/preprocessing/enhancement.py` | Asinh stretch enhancement |
| `src/preprocessing/augmentation.py` | Dual augmentation pipeline + YAML builder |
| `configs/augmentation.yaml` | Augmentation configuration |
| `tests/test_alignment.py` | Spatial alignment tests |
| `tests/test_enhancement.py` | Enhancement tests |
| `tests/test_augmentation.py` | Augmentation pipeline tests |
| `docs/b3_enhancement_augmentation.md` | This documentation |
