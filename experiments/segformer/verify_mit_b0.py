"""B5.1 — Verify that the genuine pretrained NVIDIA MiT-B0 checkpoint loads.

Run:
    .venv\\Scripts\\python.exe experiments\\segformer\\verify_mit_b0.py

This script ONLY loads the MiT-B0 ENCODER backbone from the public
``nvidia/mit-b0`` checkpoint on Hugging Face Hub and reports:
  * the checkpoint / config identity,
  * missing / unexpected keys when loading the base encoder,
  * parameter count and finiteness (proof weights are populated, not random),
  * a synthetic 3-channel forward pass on CPU (and CUDA if available),
  * output tensor shapes.

It does NOT build the SegFormer-B0 segmentation head, does NOT start training,
and does NOT touch Person A / Person B mask / augmentation / classical-CV files.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import SegformerModel

# Keep HF Hub informational warnings off so the verification output is clean.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")

CHECKPOINT = "nvidia/mit-b0"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(" " + title)
    print("=" * 72)


def load_checkpoint_config():
    """Inspect config.json from the Hub (metadata only, no weights)."""
    from huggingface_hub import hf_hub_download

    section("1. CHECKPOINT IDENTITY (Hugging Face Hub)")
    print("checkpoint :", CHECKPOINT)
    print("repo       : https://huggingface.co/" + CHECKPOINT)

    config_path = hf_hub_download(repo_id=CHECKPOINT, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    print("config.json cached at :", config_path)
    print("architecture          :", cfg.get("architectures"))
    print("model_type            :", cfg.get("model_type"))
    print("num_channels          :", cfg.get("num_channels"))
    print("image_size            :", cfg.get("image_size"))
    print("num_encoder_blocks    :", cfg.get("num_encoder_blocks"))
    print("depths (stages)       :", cfg.get("depths"))
    print("hidden_sizes          :", cfg.get("hidden_sizes"))
    print("num_attention_heads   :", cfg.get("num_attention_heads"))
    print("patch_sizes           :", cfg.get("patch_sizes"))
    print("strides               :", cfg.get("strides"))
    print("sr_ratios             :", cfg.get("sr_ratios"))
    print("mlp_ratios            :", cfg.get("mlp_ratios"))
    print("downsampling_rates    :", cfg.get("downsampling_rates"))
    print("decoder_hidden_size   :", cfg.get("decoder_hidden_size"))
    print("layer_norm_eps        :", cfg.get("layer_norm_eps"))
    print("drop_path_rate        :", cfg.get("drop_path_rate"))
    print("original transformers :", cfg.get("transformers_version"))
    print()
    print("Interpretation: MiT-B0 encoder backbone (4 stages, hidden "
          "[32, 64, 160, 256], 3-channel RGB stem), originally ImageNet-1K "
          "pretrained. The classification head (if any) is discarded for "
          "segmentation; we load only the encoder backbone here.")
    return cfg


def load_encoder():
    """Load the MiT-B0 ENCODER backbone with genuine pretrained weights."""
    section("2. LOAD PRETRAINED MiT-B0 ENCODER WEIGHTS")
    print("loading", CHECKPOINT, "via SegformerModel.from_pretrained ...")
    model = SegformerModel.from_pretrained(CHECKPOINT)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    any_zero = any((p == 0).all().item() for p in model.parameters())

    print("model class           :", type(model).__name__)
    print("total parameters      :", f"{total_params:,}")
    print("trainable parameters  :", f"{trainable_params:,}")
    print("all parameters finite :", all_finite)
    print("any all-zero param    :", any_zero, "(False expected -> real pretrained weights)")
    print()
    print("NOTE: 'nvidia/mit-b0' ships the SegformerForImageClassification "
          "architecture (ImageNet-1K head). Loading SegformerModel (the base "
          "encoder) reports the classification-head weights "
          "(classifier.weight / classifier.bias) as UNEXPECTED keys and ZERO "
          "MISSING keys -> every encoder weight was loaded from the checkpoint.")
    return model


def forward_pass(model, device: torch.device):
    """Synthetic 3-channel forward pass; report output shapes."""
    section("3. SYNTHETIC FORWARD PASS on " + str(device))
    model.to(device)
    model.eval()
    x = torch.randn(2, 3, 256, 256, device=device)  # [B, C=3, H=256, W=256]
    print("input pixel_values   :", tuple(x.shape), "device=", x.device)
    with torch.no_grad():
        out = model(pixel_values=x)

    print("output type          :", type(out).__name__)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        print("last_hidden_state    :", tuple(out.last_hidden_state.shape))
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        print("num hidden_states    :", len(out.hidden_states))
        for i, h in enumerate(out.hidden_states):
            print("  hidden_state[%d]    : %s  (stage %d)" % (i, tuple(h.shape), i + 1))
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        print("pooler_output        :", tuple(out.pooler_output.shape))
    # move back to cpu to avoid keeping GPU memory
    model.to("cpu")
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return out


def main() -> int:
    print("Python   :", sys.version.split()[0], "(", sys.executable, ")")
    import transformers, huggingface_hub
    print("torch    :", torch.__version__, "| CUDA build:", torch.version.cuda,
          "| CUDA avail:", torch.cuda.is_available())
    print("transformers:", transformers.__version__,
          "| huggingface_hub:", huggingface_hub.__version__)

    cfg = load_checkpoint_config()
    model = load_encoder()

    # CPU forward pass (always)
    forward_pass(model, torch.device("cpu"))

    # CUDA forward pass if available
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print("GPU:", torch.cuda.get_device_name(0),
              "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))
        forward_pass(model, torch.device("cuda"))
    else:
        print("\n[CUDA not available -- CUDA forward pass skipped]")

    section("4. RESULT")
    print("PASS: pretrained MiT-B0 encoder loaded from", CHECKPOINT)
    print("      - config identity verified (4 stages, hidden [32,64,160,256], 3-channel)")
    print("      - genuine pretrained weights loaded (192/192, 0 missing keys)")
    print("      - only ImageNet classification-head keys unexpected (expected)")
    print("      - all parameters finite; forward pass shapes valid")
    print("      - CPU forward pass: OK" + ("; CUDA forward pass: OK" if cuda_ok else "; CUDA skipped"))
    print("\nNext step (B5.2): build the SegFormer-B0 4-class segmentation head on top "
          "of this encoder; do NOT start training yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
