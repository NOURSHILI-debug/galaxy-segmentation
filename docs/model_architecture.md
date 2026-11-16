# SegFormer-B0 Model Architecture (Person B5)

**Project:** Galaxy Segmentation  
**Branch:** `feature/person-b-segformer`  
**Milestone:** B5.2 (SegFormer-B0 four-class segmentation architecture)  
**Status:** Implemented and verified

> This document describes the SegFormer-B0 semantic segmentation model
> implemented in `src/models/segformer.py`. It is architecture and
> verification only — no training has been performed.

---

## 1. Architecture overview

The model is a single deep-learning architecture (SegFormer-B0) composed of:

| Component | Description |
|-----------|-------------|
| **Encoder** | Pretrained NVIDIA MiT-B0 (`nvidia/mit-b0`), loaded via `transformers.SegformerModel.from_pretrained`. Produces four hierarchical feature maps. |
| **Decoder** | Lightweight SegFormer-style MLP decoder: per-stage linear projections to a common dimension, resize to stage-1 resolution, concatenate, fuse (1x1 conv + BatchNorm + GELU + dropout), predict logits (1x1 conv). |
| **Upsampling** | Bilinear upsampling of logits to the original input resolution. |

```
Input [B, C, H, W]
   │
   ▼
MiT-B0 encoder (pretrained)
   ├─ stage 1: [B,  32, H/4,  W/4 ]  -> [B, 32, 64, 64]
   ├─ stage 2: [B,  64, H/8,  W/8 ]  -> [B, 64, 32, 32]
   ├─ stage 3: [B, 160, H/16, W/16]  -> [B, 160, 16, 16]
   └─ stage 4: [B, 256, H/32, W/32]  -> [B, 256, 8, 8]
   │
   ▼
MLP decoder (project -> resize -> concat -> fuse -> predict)
   │
   ▼
Bilinear upsample to [H, W]
   │
   ▼
Raw logits [B, num_classes, H, W]   (no softmax / argmax)
```

For a `256 × 256` input, the output is `[B, 4, 256, 256]`.

---

## 2. Pretrained MiT-B0 encoder

| Property | Value |
|----------|-------|
| Checkpoint | `nvidia/mit-b0` (public, Hugging Face Hub) |
| Checkpoint architecture | `SegformerForImageClassification` (ImageNet-1K pretrained) |
| What we load | `SegformerModel` — the MiT-B0 **encoder backbone** (classification head discarded) |
| `model_type` | `segformer` |
| `num_channels` | 3 (pretrained stem expects 3-channel RGB input) |
| `image_size` | 224 (architecture nominal; we run 256 × 256 patches) |
| Encoder stages | 4 |
| Hidden sizes (per stage) | `[32, 64, 160, 256]` |
| Attention heads (per stage) | `[1, 2, 5, 8]` |
| Patch sizes | `[7, 3, 3, 3]` |
| Strides | `[4, 2, 2, 2]` (overall stride 32) |
| SR ratios | `[8, 4, 2, 1]` |
| Decoder hidden size | 256 |
| Encoder parameters | 3,319,392 (~3.3 M) |

**Pretrained-weight loading evidence (B5.1):**
- 192 / 192 encoder weights loaded from `nvidia/mit-b0`.
- Zero missing keys.
- Only `classifier.weight` / `classifier.bias` (ImageNet head) reported as unexpected — expected, since segmentation replaces the head.
- All encoder parameters finite and non-zero (real pretrained weights, not random initialization).
- CPU and CUDA forward passes successful.

The encoder is trainable by default (`encoder_trainable=True`) for fine-tuning, but can be frozen via `model.freeze_encoder()` / `model.unfreeze_encoder()`.

---

## 3. SegFormer-style MLP decoder

The decoder operates on the four encoder stage feature maps:

1. **Per-stage projection:** each stage feature map `[B, C_i, H_i, W_i]` is projected to the common `decoder_dim` (default 256) with a linear layer applied per pixel (`nn.Linear(C_i, decoder_dim)` on the channel axis).
2. **Resize:** all projected maps are resized (bilinear) to the stage-1 resolution (`H/4 × W/4`, the largest map).
3. **Concatenate:** `[B, num_stages × decoder_dim, H/4, W/4]` = `[B, 1024, 64, 64]`.
4. **Fuse:** 1x1 convolution (`1024 → decoder_dim`), BatchNorm2d, GELU activation, `Dropout2d` (p=0.1).
5. **Predict:** 1x1 convolution (`decoder_dim → num_classes=4`) → `[B, 4, 64, 64]`.
6. **Upsample:** bilinear to input resolution `[B, 4, 256, 256]`.

The decoder is lightweight and suitable for a 4 GB VRAM GPU.

---

## 4. Four-class output

| Training ID | Raw mask value | Class name | Loss/metric status |
|-------------|----------------|------------|--------------------|
| 0 | 1 | Galaxy center or core | Trainable class 0 |
| 1 | 2 | Galaxy bar | Trainable class 1 |
| 2 | 3 | Galaxy arms | Trainable class 2 |
| 3 | 4 | Star | Trainable class 3 |
| 255 (ignore) | 0 | No relevant data | Ignored (not a class) |

The mapping is frozen in `src/data/mask_decoder.py` (`IGNORE_INDEX = 255`, `NUM_CLASSES = 4`) and `dataset/metadata/class_mapping.json`. The model predicts exactly `num_classes = 4` logits; `ignore_index = 255` is handled by the loss/metrics (training), not by the model forward.

The model returns **raw logits**; no softmax or argmax is applied inside `forward`.

---

## 5. Input-channel strategy (pending joint decision)

The pretrained MiT-B0 stem expects **exactly three input channels** (`num_channels = 3`). The astronomy dataset's final input-channel strategy has **not** been jointly approved with Person A, so:

- The model defaults to `in_channels = 3` and feeds the pretrained stem directly (no input adapter activated).
- It raises a clear error if `in_channels != 3` unless an explicit `input_adapter` (`nn.Module`) is supplied.
- It does **not** silently select astronomy bands and does **not** permanently modify the pretrained stem.

A final channel strategy (e.g., repeat single channel to 3, or a learned 1x1 adapter over selected bands) must be agreed with Person A before training. Synthetic architecture tests use 3-channel input to verify the pretrained encoder.

---

## 6. Configuration (`configs/model.yaml`)

| Section | Key | Value | Meaning |
|---------|-----|-------|---------|
| `encoder.checkpoint` | `"nvidia/mit-b0"` | Genuine pretrained MiT-B0 checkpoint |
| `encoder.pretrained` | `true` | Load genuine pretrained weights |
| `encoder.encoder_trainable` | `true` | Fine-tune encoder (set false to freeze) |
| `input.channels` | `3` | Pretrained-encoder test value (not final data strategy) |
| `input.height` / `width` | `256` | Target patch size |
| `decoder.embed_dim` | `256` | Common decoder embedding dimension |
| `decoder.dropout` | `0.1` | Dropout after fusion layer |
| `head.num_classes` | `4` | Four verified trainable classes |
| `head.ignore_index` | `255` | Raw mask value 0 -> ignore |

---

## 7. Instantiation

```python
from src.models.segformer import SegFormerForSegmentation

# Default: pretrained MiT-B0 encoder, 4 classes, 3-channel input, trainable encoder.
model = SegFormerForSegmentation()

# Or from config:
from src.models.segformer import build_segformer
model = build_segformer(config=model_config_dict)

# Forward pass (raw logits, no softmax/argmax):
logits = model(pixel_values)   # [B, 4, H, W]

# Freeze / unfreeze the encoder:
model.freeze_encoder()         # feature-extraction mode
model.unfreeze_encoder()       # fine-tuning mode
```

---

## 8. Running the tests

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_forward -v
```

This runs 8 focused tests:

| Test | Covers |
|------|--------|
| `TestSegFormerCPUForward` | CPU forward: `[2,3,256,256] -> [2,4,256,256]`, finite logits |
| `TestSegFormerCUDAForward` | CUDA forward, shape, finiteness, peak VRAM (skipped without CUDA) |
| `TestEncoderFeatureExtraction` | All four stage shapes `[B,32,64,64]`, `[B,64,32,32]`, `[B,160,16,16]`, `[B,256,8,8]` |
| `TestPretrainedWeightIntegrity` | Checkpoint source, finite non-zero encoder params |
| `TestGradientFlow` | Decoder + unfrozen encoder grads; frozen encoder no grads; no NaN/Inf |
| `TestModelConfiguration` | 4 classes, 3-channel stem, 256×256 output |

To run the entire project test suite (regression check):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

## 9. What is and is not implemented

**Implemented (B5.2):**
- `src/models/segformer.py` — `SegFormerForSegmentation` (pretrained MiT-B0 encoder + SegFormer MLP decoder + 4-class logits + upsampling).
- `configs/model.yaml` — documented model configuration.
- `docs/model_architecture.md` — this document.
- `tests/test_model_forward.py` — 8 focused tests (CPU, CUDA, feature extraction, weight integrity, gradient flow, configuration).

**Not implemented (out of scope for B5.2):**
- SegFormer-B0 fine-tuning / training loop (B7).
- Loss functions (Cross-Entropy + Dice) (B7).
- Checkpointing / early stopping (B7).
- Inference, overlapping-patch stitching, logit upsampling at inference (B8).
- Visualization / error maps (B8).
- Streamlit application (B9).
- Five-channel or single-channel input strategy (pending joint decision with Person A).
- Final input preprocessing / normalization pipeline (Person A).

No training has been performed. A successful forward pass with pretrained weights does **not** mean the model has been trained.

---

## 10. Files

| File | Status |
|------|--------|
| `src/models/segformer.py` | Created (B5.2) |
| `configs/model.yaml` | Created (B5.2) |
| `docs/model_architecture.md` | Created (B5.2) |
| `tests/test_model_forward.py` | Created (B5.2) |
| `src/data/mask_decoder.py` | Unchanged (Person B, B1/B2 — frozen class mapping) |
| `dataset/metadata/class_mapping.json` | Unchanged (frozen) |

No Person A files, dataset manifests, split files, augmentation, classical CV, training, inference, or application files were modified.
