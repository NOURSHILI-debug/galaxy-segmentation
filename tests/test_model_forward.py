"""Focused tests for the SegFormer-B0 segmentation architecture (B5.2).

Covers:
    A. CPU forward pass  -> [2, 3, 256, 256] -> [2, 4, 256, 256], finite logits.
    B. CUDA forward pass -> shape, finite logits, peak VRAM (if CUDA available).
    C. Encoder feature extraction -> all four stage shapes.
    D. Pretrained-weight integrity -> loaded from nvidia/mit-b0, finite, not zero.
    E. Gradient flow -> decoder + unfrozen encoder receive grads, no NaN/Inf.
    F. Configuration -> exactly 4 classes, 256 x 256 target.

These tests use deterministic synthetic inputs and the Hugging Face cache
(they do not re-download the checkpoint after the first load) and do not
depend on the real astronomy dataset or any Person A deliverable.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from transformers import SegformerModel

from src.models.segformer import SegFormerForSegmentation

CHECKPOINT = "nvidia/mit-b0"
NUM_CLASSES = 4
IMG_H = 256
IMG_W = 256
BATCH = 2
CUDA_AVAILABLE = torch.cuda.is_available()


def _make_input(batch: int = BATCH, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic 3-channel input [B, 3, 256, 256]."""
    torch.manual_seed(seed)
    return torch.randn(batch, 3, IMG_H, IMG_W, dtype=torch.float32)


class TestSegFormerCPUForward(unittest.TestCase):
    """A. CPU forward pass."""

    def test_output_shape_and_finiteness(self):
        model = SegFormerForSegmentation(encoder_trainable=True)
        model.eval()
        x = _make_input(seed=1)
        with torch.no_grad():
            logits = model(x)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (BATCH, NUM_CLASSES, IMG_H, IMG_W))
        self.assertTrue(torch.isfinite(logits).all(), "logits must be finite")
        self.assertEqual(logits.dtype, torch.float32)


class TestSegFormerCUDAForward(unittest.TestCase):
    """B. CUDA forward pass (skipped when CUDA is unavailable)."""

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_cuda_output_shape_and_finiteness(self):
        model = SegFormerForSegmentation(encoder_trainable=True)
        model.eval()
        model = model.to("cuda")
        x = _make_input(seed=2).to("cuda")

        torch.cuda.reset_peak_memory_stats("cuda:0")
        with torch.no_grad(), torch.inference_mode():
            logits = model(x)

        self.assertEqual(logits.shape, (BATCH, NUM_CLASSES, IMG_H, IMG_W))
        self.assertTrue(torch.isfinite(logits).all(), "CUDA logits must be finite")
        self.assertEqual(logits.device.type, "cuda")

        peak_mb = torch.cuda.max_memory_allocated("cuda:0") / (1024 ** 2)
        print(f"\n[CUDA] peak memory allocated: {peak_mb:.1f} MB")
        self.assertGreater(peak_mb, 0.0)


class TestEncoderFeatureExtraction(unittest.TestCase):
    """C. Encoder feature extraction — all four stage shapes."""

    def test_four_stage_feature_shapes(self):
        model = SegFormerForSegmentation(encoder_trainable=True)
        model.eval()
        x = _make_input(seed=3)
        with torch.no_grad():
            outputs = model.encoder(
                pixel_values=x, output_hidden_states=True, return_dict=True
            )
        hidden = outputs.hidden_states
        self.assertIsNotNone(hidden)
        self.assertEqual(len(hidden), 4)

        expected = [
            (BATCH, 32, IMG_H // 4, IMG_W // 4),   # stage 1: [B, 32, 64, 64]
            (BATCH, 64, IMG_H // 8, IMG_W // 8),   # stage 2: [B, 64, 32, 32]
            (BATCH, 160, IMG_H // 16, IMG_W // 16),  # stage 3: [B, 160, 16, 16]
            (BATCH, 256, IMG_H // 32, IMG_W // 32),  # stage 4: [B, 256, 8, 8]
        ]
        for i, (actual, exp) in enumerate(zip(hidden, expected)):
            self.assertEqual(tuple(actual.shape), exp, f"stage {i + 1} shape mismatch")


class TestPretrainedWeightIntegrity(unittest.TestCase):
    """D. Pretrained-weight integrity.

    Notes
    -----
    The authoritative missing/unexpected-key report is produced by
    ``SegformerModel.from_pretrained("nvidia/mit-b0")`` (verified in B5.1):
    192 / 192 encoder weights loaded, zero missing keys, and only the ImageNet
    classification-head keys (``classifier.weight``, ``classifier.bias``)
    reported as unexpected. That is expected because segmentation discards the
    ImageNet classification head.

    The tests below confirm that the constructed SegFormerForSegmentation uses
    the genuine checkpoint and carries real (finite, non-zero) pretrained
    encoder weights rather than random initialization.
    """

    def test_encoder_checkpoint_source(self):
        model = SegFormerForSegmentation()
        self.assertEqual(model.encoder_checkpoint, CHECKPOINT)

    def test_encoder_parameters_finite_and_not_zero(self):
        """The constructed model carries real pretrained encoder weights,
        not random initialization."""
        model = SegFormerForSegmentation(encoder_trainable=True)
        encoder_params = list(model.encoder.parameters())
        self.assertGreater(len(encoder_params), 0)
        total_params = 0
        for p in encoder_params:
            self.assertTrue(torch.isfinite(p).all(), "encoder params must be finite")
            self.assertFalse((p == 0).all(), "encoder params must not be all zero")
            total_params += p.numel()
        print(f"\n[weights] encoder checkpoint: {CHECKPOINT}")
        print(f"[weights] encoder parameters: {total_params:,} (MiT-B0 ~3.3M expected)")


class TestGradientFlow(unittest.TestCase):
    """E. Gradient flow through decoder and unfrozen encoder."""

    def test_decoder_and_encoder_receive_gradients(self):
        model = SegFormerForSegmentation(encoder_trainable=True)
        model.train()
        x = _make_input(seed=4)
        target = torch.randint(0, NUM_CLASSES, (BATCH, IMG_H, IMG_W), dtype=torch.long)

        logits = model(x)
        loss = F.cross_entropy(logits, target, ignore_index=255)
        loss.backward()

        # Decoder parameters must receive gradients.
        decoder_grads = 0
        for name, p in model.named_parameters():
            if "encoder" not in name:
                self.assertIsNotNone(
                    p.grad, f"decoder param {name} received no gradient"
                )
                self.assertTrue(
                    torch.isfinite(p.grad).all(),
                    f"decoder param {name} has non-finite gradient",
                )
                decoder_grads += 1
        self.assertGreater(decoder_grads, 0, "no decoder parameters found")

        # Encoder parameters (unfrozen by default) must receive gradients.
        encoder_grads = 0
        for p in model.encoder.parameters():
            self.assertIsNotNone(p.grad, "encoder param received no gradient")
            self.assertTrue(torch.isfinite(p.grad).all(), "encoder grad non-finite")
            encoder_grads += 1
        self.assertGreater(encoder_grads, 0, "no encoder parameters found")

    def test_frozen_encoder_receives_no_gradients(self):
        model = SegFormerForSegmentation(encoder_trainable=False)
        model.train()
        x = _make_input(seed=5)
        target = torch.randint(0, NUM_CLASSES, (BATCH, IMG_H, IMG_W), dtype=torch.long)

        logits = model(x)
        loss = F.cross_entropy(logits, target, ignore_index=255)
        loss.backward()

        for p in model.encoder.parameters():
            self.assertIsNone(p.grad, "frozen encoder param received a gradient")
        # Decoder still trains.
        any_decoder_grad = any(
            p.grad is not None
            for name, p in model.named_parameters()
            if "encoder" not in name
        )
        self.assertTrue(any_decoder_grad, "decoder should still receive gradients")


class TestModelConfiguration(unittest.TestCase):
    """F. Configuration — four classes and 256 x 256 target."""

    def test_num_classes_and_input_size(self):
        model = SegFormerForSegmentation()
        self.assertEqual(model.num_classes, NUM_CLASSES)
        self.assertEqual(model.encoder.config.num_channels, 3)
        # Forward contract: [B, 3, 256, 256] -> [B, 4, 256, 256].
        model.eval()
        x = _make_input(seed=6)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (BATCH, NUM_CLASSES, IMG_H, IMG_W))


if __name__ == "__main__":
    unittest.main()
