"""Tests for the synchronised augmentation pipeline.

Covers all classes in ``src.preprocessing.augmentation``.
"""

import unittest
from pathlib import Path
from typing import Optional, Tuple

import torch

from src.preprocessing.augmentation import (
    DualCompose,
    DualPadToSize,
    DualRandomHorizontalFlip,
    DualRandomRotate90,
    DualRandomVerticalFlip,
    RandomPhotometricJitter,
    build_augmentation_pipeline,
)
from src.preprocessing.transforms import IGNORE_INDEX


# ====================================================================
# Helpers
# ====================================================================

def _make_sample(
    c: int = 3,
    h: int = 8,
    w: int = 8,
    with_valid: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Build a synthetic sample with all four trainable classes."""
    image = torch.rand(c, h, w, dtype=torch.float32)
    mask = torch.full((h, w), IGNORE_INDEX, dtype=torch.int64)
    # Place one pixel per class
    mask[0, 0] = 0
    mask[1, 1] = 1
    mask[2, 2] = 2
    mask[3, 3] = 3
    valid = torch.ones(h, w, dtype=torch.bool) if with_valid else None
    if valid is not None:
        valid[0, 0] = False  # Mark one pixel invalid
    return image, mask, valid


VALID_LABEL_SET = {0, 1, 2, 3, IGNORE_INDEX}

AUGMENTATION_YAML = Path(__file__).resolve().parent.parent / "configs" / "augmentation.yaml"


# ====================================================================
# DualCompose
# ====================================================================

class TestDualCompose(unittest.TestCase):

    def test_empty_compose_is_identity(self):
        img, mask, valid = _make_sample()
        pipeline = DualCompose([])
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertTrue(torch.equal(img, o_img))
        self.assertTrue(torch.equal(mask, o_mask))
        self.assertTrue(torch.equal(valid, o_valid))

    def test_compose_chains_transforms(self):
        gen = torch.Generator().manual_seed(0)
        flip = DualRandomHorizontalFlip(p=1.0, generator=gen)
        pipeline = DualCompose([flip, flip])  # double flip = identity
        img, mask, valid = _make_sample()
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertTrue(torch.equal(img, o_img))


# ====================================================================
# DualRandomHorizontalFlip
# ====================================================================

class TestDualRandomHorizontalFlip(unittest.TestCase):

    def test_always_flips(self):
        flip = DualRandomHorizontalFlip(p=1.0)
        img, mask, _ = _make_sample(c=1, h=4, w=6, with_valid=False)
        f_img, f_mask, _ = flip(img, mask)
        self.assertTrue(torch.equal(f_img, torch.flip(img, [-1])))
        self.assertTrue(torch.equal(f_mask, torch.flip(mask, [-1])))

    def test_never_flips(self):
        flip = DualRandomHorizontalFlip(p=0.0)
        img, mask, valid = _make_sample()
        f_img, f_mask, f_valid = flip(img, mask, valid)
        self.assertTrue(torch.equal(img, f_img))
        self.assertTrue(torch.equal(mask, f_mask))

    def test_labels_preserved(self):
        flip = DualRandomHorizontalFlip(p=1.0)
        img, mask, valid = _make_sample()
        _, f_mask, _ = flip(img, mask, valid)
        self.assertTrue(set(f_mask.unique().tolist()).issubset(VALID_LABEL_SET))


# ====================================================================
# DualRandomVerticalFlip
# ====================================================================

class TestDualRandomVerticalFlip(unittest.TestCase):

    def test_always_flips(self):
        flip = DualRandomVerticalFlip(p=1.0)
        img, mask, _ = _make_sample(c=1, h=4, w=6, with_valid=False)
        f_img, f_mask, _ = flip(img, mask)
        self.assertTrue(torch.equal(f_img, torch.flip(img, [-2])))

    def test_labels_preserved(self):
        flip = DualRandomVerticalFlip(p=1.0)
        img, mask, valid = _make_sample()
        _, f_mask, _ = flip(img, mask, valid)
        self.assertTrue(set(f_mask.unique().tolist()).issubset(VALID_LABEL_SET))


# ====================================================================
# DualRandomRotate90
# ====================================================================

class TestDualRandomRotate90(unittest.TestCase):

    def test_never_rotates(self):
        rot = DualRandomRotate90(p=0.0)
        img, mask, valid = _make_sample()
        r_img, r_mask, r_valid = rot(img, mask, valid)
        self.assertTrue(torch.equal(img, r_img))
        self.assertTrue(torch.equal(mask, r_mask))

    def test_always_rotates_labels_preserved(self):
        gen = torch.Generator().manual_seed(42)
        rot = DualRandomRotate90(p=1.0, generator=gen)
        img, mask, valid = _make_sample()
        _, r_mask, _ = rot(img, mask, valid)
        self.assertTrue(set(r_mask.unique().tolist()).issubset(VALID_LABEL_SET))

    def test_class_pixel_counts_preserved(self):
        gen = torch.Generator().manual_seed(99)
        rot = DualRandomRotate90(p=1.0, generator=gen)
        img, mask, valid = _make_sample()
        original_counts = {v: (mask == v).sum().item() for v in VALID_LABEL_SET}
        _, r_mask, _ = rot(img, mask, valid)
        rotated_counts = {v: (r_mask == v).sum().item() for v in VALID_LABEL_SET}
        self.assertEqual(original_counts, rotated_counts)


# ====================================================================
# RandomPhotometricJitter
# ====================================================================

class TestRandomPhotometricJitter(unittest.TestCase):

    def test_mask_unchanged(self):
        gen = torch.Generator().manual_seed(0)
        jitter = RandomPhotometricJitter(p=1.0, generator=gen)
        img, mask, valid = _make_sample()
        original_mask = mask.clone()
        _, j_mask, _ = jitter(img, mask, valid)
        self.assertTrue(torch.equal(original_mask, j_mask))

    def test_output_in_valid_range(self):
        gen = torch.Generator().manual_seed(0)
        jitter = RandomPhotometricJitter(p=1.0, generator=gen)
        img, mask, valid = _make_sample()
        j_img, _, _ = jitter(img, mask, valid)
        self.assertGreaterEqual(j_img.min().item(), 0.0)
        self.assertLessEqual(j_img.max().item(), 1.0)

    def test_invalid_pixels_remain_zero(self):
        gen = torch.Generator().manual_seed(0)
        jitter = RandomPhotometricJitter(p=1.0, generator=gen)
        img, mask, valid = _make_sample()
        # Zero out invalid pixels in the source image
        img[:, ~valid] = 0.0
        j_img, _, j_valid = jitter(img, mask, valid)
        # All pixels marked invalid must still be zero
        self.assertTrue((j_img[:, ~j_valid] == 0.0).all())

    def test_never_jitters(self):
        jitter = RandomPhotometricJitter(p=0.0)
        img, mask, valid = _make_sample()
        j_img, _, _ = jitter(img, mask, valid)
        self.assertTrue(torch.equal(img, j_img))

    def test_no_nan_or_inf(self):
        gen = torch.Generator().manual_seed(0)
        jitter = RandomPhotometricJitter(p=1.0, generator=gen)
        img, mask, valid = _make_sample()
        j_img, _, _ = jitter(img, mask, valid)
        self.assertTrue(torch.isfinite(j_img).all())

    def test_multi_channel_same_factor(self):
        """All channels must be scaled by the exact same brightness and contrast
        factors to preserve relative band calibration."""
        gen = torch.Generator().manual_seed(123)
        jitter = RandomPhotometricJitter(
            brightness_range=(1.05, 1.05),  # fixed brightness factor: 1.05
            contrast_range=(0.95, 0.95),    # fixed contrast factor: 0.95
            p=1.0,
            generator=gen,
        )
        # 5 distinct channels with different means and spatial structures
        img = torch.stack([
            torch.linspace(0.1, 0.3, 16).reshape(4, 4),
            torch.linspace(0.2, 0.5, 16).reshape(4, 4),
            torch.linspace(0.3, 0.7, 16).reshape(4, 4),
            torch.linspace(0.15, 0.45, 16).reshape(4, 4),
            torch.linspace(0.05, 0.25, 16).reshape(4, 4),
        ], dim=0)  # [5, 4, 4]
        mask = torch.zeros(4, 4, dtype=torch.int64)

        j_img, _, _ = jitter(img, mask)

        # For every channel, output must match contrast * (x - mean) + mean * brightness
        for c in range(5):
            ch = img[c]
            mean_c = ch.mean()
            expected_c = (0.95 * (ch - mean_c) + mean_c) * 1.05
            expected_c = expected_c.clamp(0.0, 1.0)
            self.assertTrue(
                torch.allclose(j_img[c], expected_c, atol=1e-6),
                f"Channel {c} factor mismatch: max diff = {(j_img[c] - expected_c).abs().max().item()}"
            )

    def test_valid_pixels_for_contrast_statistics(self):
        """Contrast mean must be computed over valid pixels only."""
        gen = torch.Generator().manual_seed(0)
        jitter = RandomPhotometricJitter(
            brightness_range=(1.0, 1.0),
            contrast_range=(1.5, 1.5),
            p=1.0,
            generator=gen,
        )
        # 1-channel 4x4 image
        img = torch.full((1, 4, 4), 0.5, dtype=torch.float32)
        mask = torch.zeros(4, 4, dtype=torch.int64)
        valid = torch.ones(4, 4, dtype=torch.bool)
        # Half of the pixels are invalid and set to 0.0
        valid[:, 2:] = False
        img[:, ~valid] = 0.0

        j_img, _, j_val = jitter(img, mask, valid)
        # The valid pixels were all 0.5, so their mean is 0.5.
        # Contrast scaling around 0.5 gives: 1.5 * (0.5 - 0.5) + 0.5 = 0.5.
        # If the invalid 0.0 pixels had been included, the mean would be 0.25,
        # yielding 1.5 * (0.5 - 0.25) + 0.25 = 0.625.
        self.assertAlmostEqual(j_img[0, 0, 0].item(), 0.5, places=5)
        # Invalid pixels remain 0.0
        self.assertEqual(j_img[0, 0, 3].item(), 0.0)

    def test_all_invalid_validity_mask(self):
        """When all pixels are invalid, no division by zero or NaN should occur."""
        gen = torch.Generator().manual_seed(0)
        jitter = RandomPhotometricJitter(p=1.0, generator=gen)
        img = torch.rand(3, 8, 8)
        mask = torch.zeros(8, 8, dtype=torch.int64)
        valid = torch.zeros(8, 8, dtype=torch.bool)  # all invalid

        j_img, _, j_val = jitter(img, mask, valid)
        self.assertTrue(torch.isfinite(j_img).all())
        self.assertEqual((j_img == 0.0).sum().item(), j_img.numel())

    def test_invalid_inputs_rejected(self):
        """Non-finite and out-of-range inputs must raise ValueError."""
        jitter = RandomPhotometricJitter()
        mask = torch.zeros(8, 8, dtype=torch.int64)

        # NaN
        img_nan = torch.rand(1, 8, 8)
        img_nan[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            jitter(img_nan, mask)

        # Inf
        img_inf = torch.rand(1, 8, 8)
        img_inf[0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            jitter(img_inf, mask)

        # Negative
        img_neg = torch.rand(1, 8, 8)
        img_neg[0, 0, 0] = -0.5
        with self.assertRaises(ValueError):
            jitter(img_neg, mask)

        # Out-of-range high
        img_high = torch.rand(1, 8, 8)
        img_high[0, 0, 0] = 2.0
        with self.assertRaises(ValueError):
            jitter(img_high, mask)


# ====================================================================
# Reproducibility
# ====================================================================

class TestReproducibility(unittest.TestCase):

    def _build_seeded_pipeline(self, seed: int) -> DualCompose:
        gen = torch.Generator().manual_seed(seed)
        return DualCompose([
            DualRandomHorizontalFlip(p=0.5, generator=gen),
            DualRandomVerticalFlip(p=0.5, generator=gen),
            DualRandomRotate90(p=0.5, generator=gen),
            RandomPhotometricJitter(p=0.5, generator=gen),
        ])

    def test_same_seed_same_output(self):
        img, mask, valid = _make_sample()
        p1 = self._build_seeded_pipeline(seed=42)
        p2 = self._build_seeded_pipeline(seed=42)
        o1_img, o1_mask, o1_valid = p1(img.clone(), mask.clone(), valid.clone())
        o2_img, o2_mask, o2_valid = p2(img.clone(), mask.clone(), valid.clone())
        self.assertTrue(torch.equal(o1_img, o2_img))
        self.assertTrue(torch.equal(o1_mask, o2_mask))
        self.assertTrue(torch.equal(o1_valid, o2_valid))

    def test_different_seed_different_output(self):
        """Different seeds must produce different output tensors."""
        img, mask, valid = _make_sample()
        p1 = self._build_seeded_pipeline(seed=42)
        p2 = self._build_seeded_pipeline(seed=999)
        o1_img, o1_mask, _ = p1(img.clone(), mask.clone(), valid.clone())
        o2_img, o2_mask, _ = p2(img.clone(), mask.clone(), valid.clone())
        self.assertFalse(
            torch.equal(o1_img, o2_img),
            "Different random seeds must produce different augmented images."
        )


# ====================================================================
# Validation / test pipelines are deterministic
# ====================================================================

class TestDeterministicSplits(unittest.TestCase):

    @unittest.skipUnless(AUGMENTATION_YAML.is_file(), "augmentation.yaml not found")
    def test_val_pipeline_is_identity(self):
        pipeline = build_augmentation_pipeline(AUGMENTATION_YAML, split="val")
        img, mask, valid = _make_sample()
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertTrue(torch.equal(img, o_img))
        self.assertTrue(torch.equal(mask, o_mask))
        self.assertTrue(torch.equal(valid, o_valid))

    @unittest.skipUnless(AUGMENTATION_YAML.is_file(), "augmentation.yaml not found")
    def test_test_pipeline_is_identity(self):
        pipeline = build_augmentation_pipeline(AUGMENTATION_YAML, split="test")
        img, mask, valid = _make_sample()
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertTrue(torch.equal(img, o_img))
        self.assertTrue(torch.equal(mask, o_mask))
        self.assertTrue(torch.equal(valid, o_valid))


# ====================================================================
# build_augmentation_pipeline
# ====================================================================

class TestBuildPipeline(unittest.TestCase):

    @unittest.skipUnless(AUGMENTATION_YAML.is_file(), "augmentation.yaml not found")
    def test_train_pipeline_builds(self):
        gen = torch.Generator().manual_seed(0)
        pipeline = build_augmentation_pipeline(AUGMENTATION_YAML, split="train", generator=gen)
        self.assertIsInstance(pipeline, DualCompose)

    @unittest.skipUnless(AUGMENTATION_YAML.is_file(), "augmentation.yaml not found")
    def test_train_pipeline_runs_on_sample(self):
        gen = torch.Generator().manual_seed(42)
        pipeline = build_augmentation_pipeline(AUGMENTATION_YAML, split="train", generator=gen)
        img, mask, valid = _make_sample(c=5, h=208, w=208)
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertEqual(o_img.shape[0], 5)
        self.assertTrue(set(o_mask.unique().tolist()).issubset(VALID_LABEL_SET))

    def test_dict_config(self):
        cfg = {
            "train": {
                "horizontal_flip": True,
                "horizontal_flip_p": 0.5,
                "vertical_flip": False,
                "random_rotate90": False,
                "photometric_jitter": False,
                "pad": False,
            },
            "val": {
                "horizontal_flip": False,
                "vertical_flip": False,
                "random_rotate90": False,
                "photometric_jitter": False,
                "pad": False,
            },
            "test": {
                "horizontal_flip": False,
                "vertical_flip": False,
                "random_rotate90": False,
                "photometric_jitter": False,
                "pad": False,
            },
        }
        pipeline = build_augmentation_pipeline(cfg, split="train")
        self.assertIsInstance(pipeline, DualCompose)

    def test_invalid_split_raises(self):
        cfg = {"train": {}, "val": {}, "test": {}}
        with self.assertRaises(ValueError):
            build_augmentation_pipeline(cfg, split="invalid")

    def test_missing_split_section_raises(self):
        cfg = {"train": {}}
        with self.assertRaises(ValueError):
            build_augmentation_pipeline(cfg, split="val")

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_augmentation_pipeline("/nonexistent/path.yaml", split="train")

    def test_invalid_probability_in_config(self):
        cfg = {
            "train": {"horizontal_flip": True, "horizontal_flip_p": 1.5},
            "val": {},
            "test": {},
        }
        with self.assertRaises(ValueError):
            build_augmentation_pipeline(cfg, split="train")


# ====================================================================
# Shape contract tests
# ====================================================================

class TestShapeContracts(unittest.TestCase):

    def test_various_sizes(self):
        for h, w in [(8, 8), (16, 32), (208, 208), (7, 13)]:
            for c in (1, 3, 5):
                with self.subTest(h=h, w=w, c=c):
                    gen = torch.Generator().manual_seed(0)
                    pipeline = DualCompose([
                        DualRandomHorizontalFlip(p=1.0, generator=gen),
                    ])
                    img = torch.rand(c, h, w)
                    mask = torch.zeros(h, w, dtype=torch.int64)
                    o_img, o_mask, _ = pipeline(img, mask)
                    self.assertEqual(o_img.shape, (c, h, w))
                    self.assertEqual(o_mask.shape, (h, w))


# ====================================================================
# DualPadToSize
# ====================================================================

class TestDualPadToSize(unittest.TestCase):

    def test_pad_to_256(self):
        pad = DualPadToSize(target_h=256, target_w=256)
        img = torch.rand(5, 208, 208)
        mask = torch.zeros(208, 208, dtype=torch.int64)
        valid = torch.ones(208, 208, dtype=torch.bool)
        p_img, p_mask, p_valid = pad(img, mask, valid)
        self.assertEqual(p_img.shape, (5, 256, 256))
        self.assertEqual(p_mask.shape, (256, 256))
        self.assertEqual(p_valid.shape, (256, 256))
        # Border of mask must be IGNORE_INDEX
        self.assertEqual(p_mask[0, 0].item(), IGNORE_INDEX)
        # Border of valid must be False
        self.assertFalse(p_valid[0, 0].item())

    def test_invalid_target_raises(self):
        with self.assertRaises(ValueError):
            DualPadToSize(target_h=0, target_w=256)


# ====================================================================
# Device compatibility
# ====================================================================

class TestAugmentationDeviceCompat(unittest.TestCase):

    def _run_on_device(self, device: torch.device):
        gen = torch.Generator()  # CPU generator is always valid
        gen.manual_seed(42)
        pipeline = DualCompose([
            DualRandomHorizontalFlip(p=1.0, generator=gen),
            DualRandomRotate90(p=1.0, generator=gen),
        ])
        img, mask, valid = _make_sample(c=3, h=8, w=8)
        img, mask, valid = img.to(device), mask.to(device), valid.to(device)
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertEqual(o_img.device.type, device.type)
        self.assertEqual(o_mask.device.type, device.type)
        self.assertTrue(set(o_mask.cpu().unique().tolist()).issubset(VALID_LABEL_SET))

    def test_cpu(self):
        self._run_on_device(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._run_on_device(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_generator_compatibility(self):
        """Pipeline must function seamlessly with a CUDA generator."""
        gen = torch.Generator(device="cuda")
        gen.manual_seed(42)
        pipeline = DualCompose([
            DualRandomHorizontalFlip(p=0.5, generator=gen),
            DualRandomVerticalFlip(p=0.5, generator=gen),
            DualRandomRotate90(p=0.5, generator=gen),
            RandomPhotometricJitter(p=0.5, generator=gen),
        ])
        img, mask, valid = _make_sample(c=3, h=8, w=8)
        img, mask, valid = img.cuda(), mask.cuda(), valid.cuda()
        o_img, o_mask, o_valid = pipeline(img, mask, valid)
        self.assertEqual(o_img.device.type, "cuda")
        self.assertEqual(o_mask.device.type, "cuda")
        self.assertEqual(o_valid.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
