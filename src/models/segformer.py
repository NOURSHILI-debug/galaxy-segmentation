"""SegFormer-B0 semantic segmentation architecture.

Implements a SegFormer-B0 semantic segmentation model composed of:

1. A pretrained NVIDIA MiT-B0 encoder (``nvidia/mit-b0``) loaded via
   :mod:`transformers`. The encoder produces four hierarchical feature maps
   (one per stage). Pretrained weights are preserved by default.

2. A lightweight SegFormer-style MLP decoder that projects each encoder stage
   to a common dimension, resizes all projections to the first-stage
   resolution, concatenates them, fuses them with a 1x1 convolution plus
   normalization and activation, applies dropout, and predicts class logits
   with a final 1x1 convolution.

3. Bilinear upsampling of the logits to the original input resolution.

The model returns raw logits (no softmax / argmax) with shape
``[B, num_classes, H, W]``.

Input-channel note
------------------
The pretrained MiT-B0 stem expects exactly three input channels
(``num_channels = 3`` in ``nvidia/mit-b0``'s config). The astronomy dataset's
final input-channel strategy has not been jointly approved with Person A, so
this model:

* defaults to ``in_channels = 3`` and feeds the pretrained stem directly
  (no input adapter activated);
* raises a clear error if ``in_channels != 3`` unless an explicit
  ``input_adapter`` is supplied;
* does not silently select astronomy bands and does not permanently modify
  the pretrained stem.

See ``docs/model_architecture.md`` and ``configs/model.yaml``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


class SegFormerForSegmentation(nn.Module):
    """SegFormer-B0 with a pretrained MiT-B0 encoder and a four-class decoder.

    Parameters
    ----------
    encoder_checkpoint:
        Hugging Face checkpoint id for the pretrained MiT-B0 encoder.
    in_channels:
        Number of input channels. Must be ``3`` (the pretrained MiT-B0 stem
        requirement) unless an explicit ``input_adapter`` is supplied.
    num_classes:
        Number of semantic classes (frozen to 4 for this project).
    decoder_dim:
        Common decoder embedding dimension used for the per-stage projections
        and the fusion layer.
    decoder_dropout:
        Dropout probability applied after the fusion layer.
    input_adapter:
        Optional ``nn.Module`` that adapts an input with ``in_channels``
        channels to three channels before the pretrained encoder. Activated
        only when explicitly supplied; no adapter is used by default.
    pretrained:
        Whether to load genuine pretrained MiT-B0 weights from
        ``encoder_checkpoint``.
    encoder_trainable:
        Whether encoder parameters require gradients (fine-tuning mode).
        Set to ``False`` to freeze the encoder (feature-extraction mode).
    ignore_index:
        Label value ignored by the loss / metrics (matches the project's
        mask mapping, raw value 0 -> 255).
    """

    def __init__(
        self,
        encoder_checkpoint: str = "nvidia/mit-b0",
        in_channels: int = 3,
        num_classes: int = 4,
        decoder_dim: int = 256,
        decoder_dropout: float = 0.1,
        input_adapter: Optional[nn.Module] = None,
        pretrained: bool = True,
        encoder_trainable: bool = True,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.encoder_checkpoint = encoder_checkpoint
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decoder_dim = decoder_dim
        self.decoder_dropout = decoder_dropout
        self.ignore_index = ignore_index

        # ------------------------------------------------------------------
        # Input adapter (isolated, opt-in). The pretrained MiT-B0 stem expects
        # three channels. No channel-conversion strategy is activated by
        # default; the stem is fed directly when in_channels == 3.
        # ------------------------------------------------------------------
        if in_channels != 3 and input_adapter is None:
            raise ValueError(
                "The pretrained MiT-B0 encoder expects three input channels "
                f"(in_channels=3), but in_channels={in_channels} was requested. "
                "Either set in_channels=3 (matching the pretrained stem) or "
                "supply an explicit input_adapter nn.Module that maps your "
                "input to three channels before the encoder. No "
                "channel-conversion strategy is activated by default."
            )
        self.input_adapter = input_adapter if input_adapter is not None else nn.Identity()

        # ------------------------------------------------------------------
        # Pretrained MiT-B0 encoder backbone (genuine checkpoint by default).
        # ------------------------------------------------------------------
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(encoder_checkpoint)
        else:
            config = SegformerModel.config_class.from_pretrained(encoder_checkpoint)
            self.encoder = SegformerModel(config)

        # Encoder output channels per stage (MiT-B0): [32, 64, 160, 256].
        encoder_hidden_sizes = getattr(self.encoder.config, "hidden_sizes", None)
        if encoder_hidden_sizes is None:
            raise ValueError(
                "Encoder config does not expose hidden_sizes; cannot build decoder."
            )
        self.encoder_hidden_sizes = list(encoder_hidden_sizes)
        self.num_stages = len(self.encoder_hidden_sizes)

        # Freeze / unfreeze encoder parameters.
        for param in self.encoder.parameters():
            param.requires_grad = encoder_trainable

        # ------------------------------------------------------------------
        # SegFormer-style MLP decoder.
        # ------------------------------------------------------------------
        # Per-stage linear projections to the common decoder dimension.
        self.stage_projections = nn.ModuleList(
            [nn.Linear(in_ch, decoder_dim) for in_ch in self.encoder_hidden_sizes]
        )

        # Fusion: concat(num_stages * decoder_dim) -> decoder_dim.
        self.fusion_conv = nn.Conv2d(
            decoder_dim * self.num_stages, decoder_dim, kernel_size=1, bias=False
        )
        self.fusion_norm = nn.BatchNorm2d(decoder_dim)
        self.fusion_act = nn.GELU()
        self.dropout = nn.Dropout2d(decoder_dropout)

        # Prediction head: decoder_dim -> num_classes.
        self.prediction_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw segmentation logits.

        Args:
            pixel_values: Float tensor ``[B, C, H, W]``. When
                ``in_channels == 3`` this is fed directly to the pretrained
                MiT-B0 stem. Otherwise it must already have been adapted to
                three channels by an upstream ``input_adapter``.

        Returns:
            Raw logits tensor ``[B, num_classes, H, W]``. No softmax or
            argmax is applied.
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must be 4D [B, C, H, W]; got {pixel_values.dim()}D."
            )

        # Optional input adaptation (identity by default).
        adapted = self.input_adapter(pixel_values)

        if adapted.size(1) != 3:
            raise ValueError(
                "After the (optional) input adapter, pixel_values must have 3 "
                f"channels for the pretrained MiT-B0 stem; "
                f"got {adapted.size(1)}."
            )

        input_hw = pixel_values.shape[2:]

        # Encoder: retrieve all four stage feature maps.
        encoder_outputs = self.encoder(
            pixel_values=adapted,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Encoder did not return hidden_states; "
                "call with output_hidden_states=True."
            )

        # Stage 1 resolution is the fusion target (largest spatial map).
        target_hw = tuple(hidden_states[0].shape[2:])

        # Project each stage to decoder_dim and resize to stage-1 resolution.
        projected = []
        for feature, projection in zip(hidden_states, self.stage_projections):
            # feature: [B, C_i, H_i, W_i]
            x = feature.permute(0, 2, 3, 1)            # [B, H_i, W_i, C_i]
            x = projection(x)                          # [B, H_i, W_i, decoder_dim]
            x = x.permute(0, 3, 1, 2)                  # [B, decoder_dim, H_i, W_i]
            x = F.interpolate(
                x, size=target_hw, mode="bilinear", align_corners=False
            )
            projected.append(x)

        # Concatenate along channels: [B, num_stages * decoder_dim, H1, W1].
        fused = torch.cat(projected, dim=1)

        # Fuse: conv -> normalization -> activation -> dropout.
        fused = self.fusion_conv(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_act(fused)
        fused = self.dropout(fused)

        # Predict class logits: [B, num_classes, H1, W1].
        logits = self.prediction_head(fused)

        # Upsample to input resolution: [B, num_classes, H, W].
        logits = F.interpolate(
            logits, size=input_hw, mode="bilinear", align_corners=False
        )
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def freeze_encoder(self) -> None:
        """Freeze pretrained encoder weights (feature-extraction mode)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze pretrained encoder weights (fine-tuning mode)."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def build_segformer(
    config: Optional[dict] = None,
    **kwargs,
) -> SegFormerForSegmentation:
    """Build a SegFormer-B0 segmentation model from a config dict / kwargs."""
    if config is None:
        config = {}
    merged = {**config, **kwargs}
    return SegFormerForSegmentation(**merged)


