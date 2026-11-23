# B5.1 — Pretrained MiT-B0 Environment & Checkpoint Verification

**Project:** Galaxy Segmentation (Person B)  
**Branch:** `feature/person-b-segformer`  
**Milestone:** B5.1 (environment setup + pretrained MiT-B0 verification only)  
**Status:** **PASS**

> This document records evidence that the genuine public pretrained NVIDIA
> MiT-B0 checkpoint (`nvidia/mit-b0`) loads successfully into the project
> virtual environment. No SegFormer-B0 segmentation head is built, no training
> is started, and no Person A / Person B mask / augmentation / classical-CV
> files were modified.

---

## 1. Environment audit (Phase 1)

| Item | Value |
|------|-------|
| Working tree | `C:\Users\souro\Desktop\galaxy-segmentation` |
| Git branch | `feature/person-b-segformer` (clean; no commits made) |
| Python executable | `C:\Users\souro\Desktop\galaxy-segmentation\.venv\Scripts\python.exe` |
| Python version | 3.12.10 |
| `.venv` active? | Not auto-activated; venv Python invoked explicitly (`VIRTUAL_ENV` unset) |
| PyTorch version | 2.14.1+cu126 |
| PyTorch CUDA build | 12.6 |
| CUDA available | `True` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (compute capability 8.6) |
| GPU VRAM | 4.0 GB |
| Free disk (C:) | ~38.7 GB |

PyTorch was **preserved** — not upgraded, uninstalled, or replaced.

---

## 2. Dependency preparation (Phase 2)

Only the missing Hugging Face dependencies were installed (no Streamlit, OpenCV, SciPy, or pytest):

| Package | Version | Notes |
|---------|---------|-------|
| `transformers` | 5.23.0 | newly installed |
| `huggingface_hub` | 2.4.0 | newly installed (dependency of transformers) |
| `tokenizers` | 0.23.3 | transitive dependency |
| `safetensors` | 0.8.0 | transitive dependency |
| `torch` | 2.14.1+cu126 | **unchanged** (preserved) |
| `numpy` | 2.5.4 | already present |
| `pyyaml` | 6.0.3 | already present |
| `astropy` | 8.0.2 | already present |

Compatibility: transformers 5.23.0 imports and runs cleanly against torch 2.14.1+cu126 on Python 3.12.10; CUDA forward pass succeeds.

Pin note: add `transformers==5.23.0` and `huggingface_hub==2.4.0` to the project dependency documentation (`requirements.txt` / environment notes) when the dependency file is next updated. Do **not** pin a torch change.

---

## 3. Pretrained checkpoint inspection (Phase 3)

| Item | Value |
|------|-------|
| Checkpoint identifier | `nvidia/mit-b0` |
| Official repository | https://huggingface.co/nvidia/mit-b0 |
| Download / cache location | `C:\Users\souro\.cache\huggingface\hub\models--nvidia--mit-b0\snapshots\80983a413c30d36a39c20203974ae7807835e2b4\` |
| Checkpoint files | `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, `tf_model.h5`, `README.md`, `.gitattributes` |
| Model license | `other` — see https://github.com/NVlabs/SegFormer/blob/master/LICENSE |
| Architecture (config) | `SegformerForImageClassification` (MiT-B0 encoder + ImageNet-1K classification head) |
| `model_type` | `segformer` |
| `num_channels` | **3** (pretrained stem expects 3-channel RGB input) |
| `image_size` | 224 |
| `num_encoder_blocks` | 4 |
| `depths` (stages) | `[2, 2, 2, 2]` |
| `hidden_sizes` | `[32, 64, 160, 256]` (MiT-B0) |
| `num_attention_heads` | `[1, 2, 5, 8]` |
| `patch_sizes` | `[7, 3, 3, 3]` |
| `strides` | `[4, 2, 2, 2]` (overall stride 32) |
| `sr_ratios` | `[8, 4, 2, 1]` |
| `mlp_ratios` | `[4, 4, 4, 4]` |
| `decoder_hidden_size` | 256 |
| Original `transformers_version` | 4.12.0.dev0 |

**Interpretation:** `nvidia/mit-b0` is the genuine MiT-B0 **encoder backbone** pretrained on ImageNet-1K. For SegFormer-B0 segmentation we load the base encoder (`SegformerModel`) and discard/replace the ImageNet classification head. The input-channel decision (3-channel pretrained stem vs. the project's multi-band / repeated-channel strategy) is deliberately **not** made in this milestone (project boundary).

---

## 4. Pretrained-weight loading & forward pass (Phase 4)

Verification script: `experiments/segformer/verify_mit_b0.py`  
Invocation: `.venv\Scripts\python.exe experiments\segformer\verify_mit_b0.py`  
Exit code: **0**

### 4.1 Load report (genuine pretrained weights, not random init)

```
[transformers] SegformerModel LOAD REPORT from: nvidia/mit-b0
Key               | Status
------------------+------------+
classifier.weight | UNEXPECTED
classifier.bias   | UNEXPECTED
```

- Weights loaded: **192 / 192**
- **MISSING keys: none** — every MiT-B0 encoder weight was populated from the checkpoint.
- **UNEXPECTED keys: `classifier.weight`, `classifier.bias`** — these are the ImageNet-1K classification-head weights that correctly do **not** exist in the base encoder. This is **expected and correct**; the head is discarded for segmentation. It is **not** a loading failure.

### 4.2 Parameter & finiteness evidence

| Check | Result |
|-------|--------|
| Model class | `SegformerModel` |
| Total parameters | **3,319,392** (~3.3 M, correct for MiT-B0) |
| Trainable parameters | 3,319,392 |
| All parameters finite | `True` |
| Any all-zero parameter | `False` → real pretrained weights, not random/zero init |

### 4.3 Synthetic forward pass (3-channel input, `[B, C=3, H=256, W=256]`)

| Device | Input shape | Output (`last_hidden_state`) | Status |
|--------|-------------|------------------------------|--------|
| CPU | `(2, 3, 256, 256)` | `(2, 256, 8, 8)` | OK |
| CUDA (`cuda:0`) | `(2, 3, 256, 256)` | `(2, 256, 8, 8)` | OK |

Output `8 × 8` = `256 / 32`, consistent with the MiT-B0 overall stride of 32 (`4 × 2 × 2 × 2`). Output type: `BaseModelOutput`.

---

## 5. Known warnings / limitations

1. **HF Hub symlink warning on Windows** — `huggingface_hub` cache uses symlinks by default, which Windows does not fully support unless Developer Mode is enabled. Caching still works (degraded, may use more disk). Can be silenced with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. This is environmental, not a functional blocker.
2. **Unauthenticated HF Hub requests** — a rate-limit notice is printed; the public `nvidia/mit-b0` checkpoint downloaded successfully without a token. No token was requested or created.
3. **`nvidia/mit-b0` is an ImageNet-1K classification checkpoint** (`SegformerForImageClassification`). Loading the base encoder reports the classification-head keys as `UNEXPECTED` — expected, since segmentation replaces the head. Do not interpret this as a load failure.
4. **Input channels fixed at 3** by the pretrained stem (`num_channels: 3`). The project's final input-channel strategy (3-band / 5-band / repeated single-channel) is a separate, pending decision and must not be conflated with this verification.
5. No training has run; `experiments/segformer/run_001/` artifacts remain 0-byte placeholders.

---

## 6. Result

| Criterion | Status |
|-----------|--------|
| Existing CUDA PyTorch preserved (2.14.1+cu126) | PASS |
| Compatible Transformers installed (5.23.0) | PASS |
| Public pretrained MiT-B0 checkpoint downloaded | PASS |
| Genuine pretrained encoder weights loaded | PASS (192/192, 0 missing keys) |
| Missing/unexpected keys reported | PASS (only ImageNet head keys, expected) |
| Parameters finite & populated | PASS (3,319,392, all finite, none zero) |
| Forward pass succeeds (CPU) | PASS `[2,3,256,256] -> [2,256,8,8]` |
| Forward pass succeeds (CUDA) | PASS `[2,3,256,256] -> [2,256,8,8]` |
| Verification evidence documented | PASS (this doc + `verify_mit_b0.py`) |

**Final status: B5.1 PASS.**

---

## 7. Files changed / added (read-only audit; nothing committed)

| File | Action |
|------|--------|
| `experiments/segformer/verify_mit_b0.py` | **Created** — reproducible verification script |
| `docs/mit_b0_environment_verification.md` | **Created** — this report |
| `.venv\Lib\site-packages\` | `transformers`, `huggingface_hub` and transitive deps installed (environment only, not a repo file) |
| No existing source/docs/test files were modified, overwritten, or deleted. Nothing committed or pushed. |

### Remaining blockers / next steps (B5.2+, not part of B5.1)

- Build the SegFormer-B0 4-class segmentation head on top of the verified MiT-B0 encoder.
- Resolve the input-channel strategy (3-channel pretrained stem vs. project multi-band data) — joint decision, not made here.
- Wire `id2label` / `label2id` from `dataset/metadata/class_mapping.json` (4 classes + `ignore_index=255`).
- Implement fine-tuning (B7) only after Person A delivers `dataset/metadata/manifest.csv` and frozen `dataset/splits/{train,val,test}.csv` (currently 0 bytes / 0 lines).
- Do **not** start training until the dataset contract and splits are verified.
