"""Synthetic data-to-model integration contract tests (B6.3).

Validates the complete pipeline on synthetic CPU tensors:

    Synthetic image/mask samples
        -> B6.1 extract_patches
        -> B6.2 PatchDataset + DataLoader (patch_dataset_collate)
        -> B5.2 SegFormerForSegmentation forward pass

This is strictly integration testing and contract verification.  No
training is implemented.  The real astronomy dataset, Person A's data
pipeline, manifests, and split files are NOT used.

Channel strategy (synthetic testing only): three-channel input is used to
match the pretrained MiT-B0 encoder stem.  This does NOT represent the
final astronomy input-channel strategy, which remains pending joint
approval with Person A.
"""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from src.data.patch_dataset import PatchDataset, patch_dataset_collate
from src.data.patches import MASK_IGNORE_INDEX, extract_patches
from src.models.segformer import SegFormerForSegmentation


CHECKPOINT = "nvidia/mit-b0"
NUM_CLASSES = 4
PATCH_SIZE = 256
STRIDE = 256
VALID_LABELS = frozenset({0, 1, 2, 3, MASK_IGNORE_INDEX})


def _make_sample(sample_id: str, H: int, W: int, C: int = 3, seed: int = 0):
    """Deterministic synthetic sample with labels in {0,1,2,3,255}."""
    torch.manual_seed(seed)
    image = torch.randn(C, H, W, dtype=torch.float32)
    # Mask with real labels (0-3) plus ignore-index (255) border region.
    mask = torch.zeros(H, W, dtype=torch.int64)
    rows = torch.arange(H).unsqueeze(1).expand(H, W)
    cols = torch.arange(W).unsqueeze(0).expand(H, W)
    core = ((rows * W + cols) % 4).to(torch.int64)
    mask[...] = core
    # Mark an outer ring as ignore index to exercise 255 handling.
    mask[0:8, :] = MASK_IGNORE_INDEX
    mask[-8:, :] = MASK_IGNORE_INDEX
    mask[:, 0:8] = MASK_IGNORE_INDEX
    mask[:, -8:] = MASK_IGNORE_INDEX
    return {"sample_id": sample_id, "image": image, "mask": mask}


def _assert_mask_contract(mask: torch.Tensor, label: str = "mask") -> None:
    """Explicit validation helper for the project's mask contract.

    Checks dtype is integer and values belong to {0, 1, 2, 3, 255}.
    Raises AssertionError on violation (no silent acceptance).
    """
    assert mask.dtype.is_floating_point is False, (
        f"{label}: mask dtype must be integer, got {mask.dtype}"
    )
    assert mask.dtype in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ), f"{label}: unsupported mask dtype {mask.dtype}"
    unique = torch.unique(mask)
    for v in unique.tolist():
        assert v in VALID_LABELS, (
            f"{label}: invalid label {v}; allowed {sorted(VALID_LABELS)}"
        )


class TestDatasetToModelShapeCompatibility(unittest.TestCase):
    """A. Dataset-to-model shape compatibility."""

    @classmethod
    def setUpClass(cls):
        # Build the model once (CPU) to reuse the HF cache / avoid repeated
        # checkpoint downloads.  Pretrained encoder loaded genuinely.
        cls.model = SegFormerForSegmentation(
            encoder_checkpoint=CHECKPOINT,
            in_channels=3,
            num_classes=NUM_CLASSES,
            encoder_trainable=True,
        )
        cls.model.eval()

    def test_dataloader_batch_to_model_output_shape(self):
        samples = [
            _make_sample("s1", 256, 256, seed=1),
            _make_sample("s2", 256, 256, seed=2),
        ]
        ds = PatchDataset(samples, patch_size=256, stride=256)
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=patch_dataset_collate
        )
        batch = next(iter(loader))

        images = batch["image"]   # [B, C, 256, 256]
        targets = batch["mask"]   # [B, 256, 256]
        self.assertEqual(images.ndim, 4)
        self.assertEqual(images.shape[1], 3)  # 3-channel synthetic input
        self.assertEqual(images.shape[2:], (256, 256))
        self.assertEqual(targets.shape, (2, 256, 256))

        with torch.no_grad():
            logits = self.model(pixel_values=images)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, NUM_CLASSES, 256, 256))
        self.assertTrue(torch.isfinite(logits).all(), "logits must be finite")


class TestImageAndMaskBatchAlignment(unittest.TestCase):
    """B. Image and mask batch alignment."""

    @classmethod
    def setUpClass(cls):
        cls.model = SegFormerForSegmentation(
            encoder_checkpoint=CHECKPOINT,
            in_channels=3,
            num_classes=NUM_CLASSES,
            encoder_trainable=True,
        )
        cls.model.eval()

    def test_batch_counts_and_metadata_preserved(self):
        samples = [
            _make_sample("alpha", 256, 256, seed=3),
            _make_sample("beta", 256, 256, seed=4),
        ]
        ds = PatchDataset(samples, patch_size=256, stride=256)
        loader = DataLoader(
            ds, batch_size=4, shuffle=False, collate_fn=patch_dataset_collate
        )
        batch = next(iter(loader))

        images = batch["image"]
        targets = batch["mask"]
        metadata = batch["metadata"]

        # Image and mask counts match.
        self.assertEqual(images.shape[0], targets.shape[0])
        self.assertEqual(len(metadata), images.shape[0])

        # No sample / patch silently dropped: every dataset patch appears.
        self.assertEqual(images.shape[0], len(ds))

        # Metadata preserves sample_id and coordinates (patch_index is
        # per-sample, so it restarts at 0 for each sample).
        seen_ids = []
        for i, m in enumerate(metadata):
            self.assertIn(m["sample_id"], ("alpha", "beta"))
            self.assertEqual(m["patch_index"], 0)  # one patch per sample
            self.assertEqual(m["top"], 0)
            self.assertEqual(m["left"], 0)
            self.assertEqual(m["original_height"], 256)
            self.assertEqual(m["original_width"], 256)
            seen_ids.append(m["sample_id"])

        # Both samples are represented in the batch.
        self.assertEqual(sorted(seen_ids), ["alpha", "beta"])

        # Coordinates correspond: image[i] / mask[i] share top/left.
        for m in metadata:
            self.assertEqual(m["top"], 0)
            self.assertEqual(m["left"], 0)


class TestMaskContract(unittest.TestCase):
    """C. Mask contract."""

    def test_mask_labels_and_ignore_index_preserved(self):
        samples = [_make_sample("s1", 256, 256, seed=5)]
        ds = PatchDataset(samples, patch_size=256, stride=256)
        for i in range(len(ds)):
            item = ds[i]
            _assert_mask_contract(item["mask"], label=f"patch {i}")
            # Ignore index 255 present (outer ring was set to 255).
            self.assertIn(255, torch.unique(item["mask"]).tolist())

    def test_invalid_label_values_rejected(self):
        # Explicit contract-level check: a mask with an out-of-range value
        # (e.g. 99) is rejected by the validation helper, not silently accepted.
        bad_mask = torch.zeros(16, 16, dtype=torch.int64)
        bad_mask[0, 0] = 99  # not in {0,1,2,3,255}
        with self.assertRaises(AssertionError):
            _assert_mask_contract(bad_mask, label="bad")

    def test_no_softmax_one_hot_or_resize_on_mask(self):
        sample = _make_sample("s1", 256, 256, seed=6)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        item = ds[0]
        mask = item["mask"]
        # Mask remains integer labels [256, 256], not one-hot or probabilities.
        self.assertEqual(mask.ndim, 2)
        self.assertEqual(mask.shape, (256, 256))
        self.assertFalse(mask.dtype.is_floating_point)
        # Values are a subset of the allowed contract labels.
        _assert_mask_contract(mask)

    def test_target_dtype_compatible_with_loss_contract(self):
        # Integer [B, H, W] target suitable for CrossEntropyLoss with
        # ignore_index=255 (the project's intended loss contract).
        samples = [_make_sample("s1", 256, 256, seed=7)]
        ds = PatchDataset(samples, patch_size=256, stride=256)
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, collate_fn=patch_dataset_collate
        )
        batch = next(iter(loader))
        targets = batch["mask"]
        self.assertEqual(targets.dtype, torch.int64)
        self.assertEqual(targets.ndim, 3)  # [B, H, W]
        self.assertIn(255, torch.unique(targets).tolist())


class TestOutputAndTargetCompatibility(unittest.TestCase):
    """D. Output and target compatibility."""

    @classmethod
    def setUpClass(cls):
        cls.model = SegFormerForSegmentation(
            encoder_checkpoint=CHECKPOINT,
            in_channels=3,
            num_classes=NUM_CLASSES,
            encoder_trainable=True,
        )
        cls.model.eval()

    def test_logits_and_target_spatial_match(self):
        samples = [_make_sample("s1", 256, 256, seed=8), _make_sample("s2", 256, 256, seed=9)]
        ds = PatchDataset(samples, patch_size=256, stride=256)
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=patch_dataset_collate
        )
        batch = next(iter(loader))

        with torch.no_grad():
            logits = self.model(pixel_values=batch["image"])

        targets = batch["mask"]
        self.assertEqual(logits.shape[0], targets.shape[0])  # batch match
        self.assertEqual(logits.shape[2], targets.shape[1])  # H match
        self.assertEqual(logits.shape[3], targets.shape[2])  # W match
        self.assertEqual(logits.shape[1], NUM_CLASSES)  # 4 classes
        self.assertEqual(targets.dtype, torch.int64)
        self.assertEqual(logits.dtype, torch.float32)


class TestBoundaryPaddingIntegration(unittest.TestCase):
    """E. Boundary-padding integration (non-divisible dimensions).

    B6.1 pads only when a dimension is <= patch_size; for larger dimensions
    it repositions the last patch to end exactly at the edge (no padding).
    So here we use H=200 (<= 256, triggers vertical padding) with W=400
    (> 256, non-divisible horizontal) to exercise both padding and exact-fit.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = SegFormerForSegmentation(
            encoder_checkpoint=CHECKPOINT,
            in_channels=3,
            num_classes=NUM_CLASSES,
            encoder_trainable=True,
        )
        cls.model.eval()

    def test_non_divisible_sample_model_accepts_padded_patches(self):
        # H=200 (<= 256 -> vertical padding), W=400 (> 256, non-divisible).
        H, W = 200, 400
        samples = [_make_sample("s1", H, W, seed=10)]
        ds = PatchDataset(samples, patch_size=256, stride=256)

        # Direct B6.1 call: same coordinates / count as the dataset.
        sample = samples[0]
        _, _, b61_metadata = extract_patches(
            sample["image"], sample["mask"], sample["sample_id"],
            patch_size=256, stride=256,
        )

        # Every dataset patch comes from B6.1 coordinates.
        self.assertEqual(len(ds), len(b61_metadata))
        # H=200<=256 -> tops=[0]; W=400>256 -> lefts=[0, 144] -> 2 patches.
        self.assertEqual(len(ds), 2)

        images = []
        masks = []
        for i in range(len(ds)):
            item = ds[i]
            images.append(item["image"])
            masks.append(item["mask"])
            _assert_mask_contract(item["mask"], label=f"patch {i}")

        image_batch = torch.stack(images, dim=0)  # [N, 3, 256, 256]
        mask_batch = torch.stack(masks, dim=0)    # [N, 256, 256]

        # Fixed patch size for every patch.
        self.assertEqual(image_batch.shape, (2, 3, 256, 256))
        self.assertEqual(mask_batch.shape, (2, 256, 256))

        # Same coordinates as B6.1.
        for i, m in enumerate(b61_metadata):
            self.assertEqual(ds[i]["metadata"]["top"], m["top"])
            self.assertEqual(ds[i]["metadata"]["left"], m["left"])

        # Vertical padding present (H=200 < 256): bottom 56 rows padded.
        # Image padding uses configured value (default 0.0).
        self.assertTrue((image_batch[0][:, 200:, :] == 0.0).all())
        self.assertTrue((image_batch[1][:, 200:, :] == 0.0).all())
        # Mask padding uses ignore index 255.
        self.assertTrue((mask_batch[0][200:, :] == 255).all())
        self.assertTrue((mask_batch[1][200:, :] == 255).all())
        # Interior (non-padded) rows/cols preserve original values exactly.
        # Patch 0 covers source [0:256, 0:256]; compare the non-padded region.
        self.assertTrue(
            (image_batch[0][:, :200, :256] == sample["image"][:, :200, :256]).all()
        )
        self.assertTrue(
            (mask_batch[0][:200, :256] == sample["mask"][:200, :256]).all()
        )

        with torch.no_grad():
            logits = self.model(pixel_values=image_batch)

        # Model output matches the padded patch spatial size (256x256), not
        # the original image size (200x400). Full-image stitching is B8's
        # concern; here we verify the model accepts the padded patch batch.
        self.assertEqual(logits.shape, (2, NUM_CLASSES, 256, 256))
        self.assertEqual(logits.shape[2:], image_batch.shape[2:])
        self.assertTrue(torch.isfinite(logits).all())


class TestDeterministicIntegration(unittest.TestCase):
    """F. Deterministic integration."""

    def test_repeated_run_identical(self):
        samples = [_make_sample("s1", 256, 256, seed=11)]
        ds1 = PatchDataset(samples, patch_size=256, stride=256)
        ds2 = PatchDataset(samples, patch_size=256, stride=256)

        self.assertEqual(len(ds1), len(ds2))
        for i in range(len(ds1)):
            a, b = ds1[i], ds2[i]
            self.assertTrue(torch.equal(a["image"], b["image"]))
            self.assertTrue(torch.equal(a["mask"], b["mask"]))
            self.assertEqual(a["metadata"], b["metadata"])


class TestGradientFlowSmokeTest(unittest.TestCase):
    """G. Gradient-flow smoke test (CPU, small batch)."""

    def test_decoder_and_encoder_gradients_finite(self):
        model = SegFormerForSegmentation(
            encoder_checkpoint=CHECKPOINT,
            in_channels=3,
            num_classes=NUM_CLASSES,
            encoder_trainable=True,
        )
        model.train()

        samples = [_make_sample("s1", 256, 256, seed=12), _make_sample("s2", 256, 256, seed=13)]
        ds = PatchDataset(samples, patch_size=256, stride=256)
        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=patch_dataset_collate
        )
        batch = next(iter(loader))

        logits = model(pixel_values=batch["image"])
        self.assertEqual(logits.shape, (2, NUM_CLASSES, 256, 256))

        # Synthetic scalar objective: mean of logits (compatible target shape
        # is [B, H, W]; here we use logits.mean() purely as a grad probe).
        loss = logits.mean()
        loss.backward()

        # Decoder gradients finite and present.
        decoder_grads = 0
        for name, p in model.named_parameters():
            if "encoder" not in name:
                self.assertIsNotNone(p.grad, f"decoder param {name} no grad")
                self.assertTrue(
                    torch.isfinite(p.grad).all(),
                    f"decoder param {name} non-finite grad",
                )
                decoder_grads += 1
        self.assertGreater(decoder_grads, 0)

        # Encoder gradients finite and present (trainable by default).
        encoder_grads = 0
        for p in model.encoder.parameters():
            self.assertIsNotNone(p.grad, "encoder param no grad")
            self.assertTrue(torch.isfinite(p.grad).all(), "encoder non-finite grad")
            encoder_grads += 1
        self.assertGreater(encoder_grads, 0)

        # No NaN / Inf anywhere in gradients.
        for p in model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())
                self.assertFalse(torch.isinf(p.grad).any())


if __name__ == "__main__":
    unittest.main()
