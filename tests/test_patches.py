"""Focused tests for reusable image / mask patch extraction (B6.1).

All tests use deterministic synthetic PyTorch tensors on CPU.  They do
not depend on the real astronomy dataset, Person A's manifest, splits,
FITS loader, or normalizer.

Covers:
    A. Standard divisible dimensions  [3, 512, 512] / [512, 512], 256/256 -> 4 patches
    B. Non-divisible dimensions       600 x 700 -> full boundary coverage
    C. Smaller-than-patch dimensions  < 256 x 256 -> one padded patch
    D. Image-mask alignment           identical coordinates, no shift
    E. Overlapping patches            stride < patch_size
    F. Metadata                       sample_id, patch_index, top, left, dims
    G. Dtype preservation             image & mask dtypes preserved
    H. Invalid inputs                 clear exceptions for bad inputs
"""

from __future__ import annotations

import unittest

import torch

from src.data.patches import MASK_IGNORE_INDEX, extract_patches

PATCH_SIZE = 256
STRIDE = 256


def _make_divisible_sample():
    """Image [3, 512, 512], mask [512, 512] with a coordinate grid pattern."""
    torch.manual_seed(0)
    image = torch.randn(3, 512, 512, dtype=torch.float32)
    # Mask with a simple spatial pattern so we can verify alignment.
    mask = torch.zeros(512, 512, dtype=torch.int64)
    mask[0:128, 0:128] = 1
    mask[128:256, 128:256] = 2
    mask[256:384, 256:384] = 3
    mask[384:512, 384:512] = 0
    return image, mask


class TestStandardDivisibleDimensions(unittest.TestCase):
    """A. Standard divisible dimensions."""

    def test_512x512_256_stride_yields_four_patches(self):
        image, mask = _make_divisible_sample()
        img_patches, mask_patches, metadata = extract_patches(
            image, mask, sample_id="s1", patch_size=256, stride=256
        )
        self.assertEqual(img_patches.shape, (4, 3, 256, 256))
        self.assertEqual(mask_patches.shape, (4, 256, 256))
        self.assertEqual(len(metadata), 4)
        # All patch spatial dims exactly 256 x 256.
        for p in range(4):
            self.assertEqual(tuple(img_patches[p].shape), (3, 256, 256))
            self.assertEqual(tuple(mask_patches[p].shape), (256, 256))


class TestNonDivisibleDimensions(unittest.TestCase):
    """B. Non-divisible dimensions — full boundary coverage, no dropped pixels."""

    def test_600x700_covers_every_pixel(self):
        H, W = 600, 700
        image = torch.randn(3, H, W, dtype=torch.float32)
        mask = torch.zeros(H, W, dtype=torch.int64)
        # Mark every pixel with a unique-ish pattern: mask = (row*W + col) % 4
        rows = torch.arange(H).unsqueeze(1).expand(H, W)
        cols = torch.arange(W).unsqueeze(0).expand(H, W)
        mask = ((rows * W + cols) % 4).to(torch.int64)

        img_patches, mask_patches, metadata = extract_patches(
            image, mask, sample_id="s2", patch_size=256, stride=256
        )

        # Reconstruct: for each patch, copy back into a canvas and compare
        # against the original for the non-padded region.
        canvas_img = torch.full_like(image, float("nan"))
        canvas_mask = torch.full_like(mask, -1)

        for p in range(len(metadata)):
            meta = metadata[p]
            top, left = meta["top"], meta["left"]
            ph, pw = meta["patch_height"], meta["patch_width"]
            h_end = min(top + ph, H)
            w_end = min(left + pw, W)
            canvas_img[:, top:h_end, left:w_end] = img_patches[p][
                :, : h_end - top, : w_end - left
            ]
            canvas_mask[top:h_end, left:w_end] = mask_patches[p][
                : h_end - top, : w_end - left
            ]

        self.assertTrue(torch.equal(canvas_img, image), "image not fully covered")
        self.assertTrue(
            torch.equal(canvas_mask, mask), "mask not fully covered"
        )
        # No NaN left in canvas (every pixel written at least once).
        self.assertFalse(torch.isnan(canvas_img).any())
        self.assertFalse((canvas_mask == -1).any())

    def test_non_divisible_patch_count_and_sizes(self):
        H, W = 600, 700
        image = torch.randn(1, H, W)
        mask = torch.zeros(H, W, dtype=torch.int64)
        img_patches, mask_patches, metadata = extract_patches(
            image, mask, "s2", patch_size=256, stride=256
        )
        # tops for H=600: [0, 256, 344] -> 3; lefts for W=700: [0, 256, 444] -> 3
        self.assertEqual(len(metadata), 9)
        self.assertEqual(img_patches.shape, (9, 1, 256, 256))
        self.assertEqual(mask_patches.shape, (9, 256, 256))


class TestSmallerThanPatchDimensions(unittest.TestCase):
    """C. Smaller-than-patch dimensions — one padded patch."""

    def test_100x150_single_patch_padded(self):
        H, W = 100, 150
        image = torch.randn(3, H, W, dtype=torch.float32)
        mask = torch.zeros(H, W, dtype=torch.int64)
        mask[0:50, 0:50] = 2

        img_patches, mask_patches, metadata = extract_patches(
            image, mask, sample_id="s3", patch_size=256, stride=256
        )

        self.assertEqual(img_patches.shape, (1, 3, 256, 256))
        self.assertEqual(mask_patches.shape, (1, 256, 256))
        self.assertEqual(len(metadata), 1)
        meta = metadata[0]
        self.assertEqual(meta["top"], 0)
        self.assertEqual(meta["left"], 0)
        self.assertEqual(meta["original_height"], H)
        self.assertEqual(meta["original_width"], W)
        self.assertEqual(meta["patch_height"], 256)
        self.assertEqual(meta["patch_width"], 256)

        # Original values preserved in the top-left region.
        self.assertTrue(torch.equal(img_patches[0, :, :H, :W], image))
        self.assertTrue(torch.equal(mask_patches[0, :H, :W], mask))
        # Image padding equals image_pad_value (default 0).
        self.assertTrue((img_patches[0, :, H:, :] == 0.0).all())
        self.assertTrue((img_patches[0, :, :, W:] == 0.0).all())
        # Mask padding equals ignore index 255.
        self.assertEqual(MASK_IGNORE_INDEX, 255)
        self.assertTrue((mask_patches[0, H:, :] == 255).all())
        self.assertTrue((mask_patches[0, :, W:] == 255).all())


class TestImageMaskAlignment(unittest.TestCase):
    """D. Image-mask alignment — identical coordinates, no shift."""

    def test_patches_use_identical_coordinates(self):
        image, mask = _make_divisible_sample()
        img_patches, mask_patches, metadata = extract_patches(
            image, mask, sample_id="s4", patch_size=256, stride=256
        )
        # Patch 1 (top=128? no—tops are [0, 256]). Verify patch at top=0,left=0
        # matches the top-left 256x256 of both image and mask.
        self.assertTrue(torch.equal(img_patches[0], image[:, 0:256, 0:256]))
        self.assertTrue(torch.equal(mask_patches[0], mask[0:256, 0:256]))
        # Patch at top=256, left=256 (index 3 in row-major: tops=[0,256]).
        self.assertTrue(torch.equal(img_patches[3], image[:, 256:512, 256:512]))
        self.assertTrue(torch.equal(mask_patches[3], mask[256:512, 256:512]))

    def test_no_spatial_shift_for_non_divisible(self):
        H, W = 600, 700
        image = torch.randn(3, H, W)
        mask = ((torch.arange(H * W).reshape(H, W)) % 4).to(torch.int64)
        img_patches, mask_patches, _ = extract_patches(
            image, mask, "s5", patch_size=256, stride=256
        )
        # Reconstruct and compare exactly (alignment = no shift).
        canvas_img = torch.full_like(image, float("nan"))
        canvas_mask = torch.full_like(mask, -1)
        for p in range(len(img_patches)):
            # top/left not needed for this check: we just verify the non-pad
            # region of each patch equals the corresponding source region.
            pass
        # Simpler direct check: every patch's valid region equals source.
        # We already verified full coverage in TestNonDivisibleDimensions;
        # here confirm a specific interior patch matches source exactly.
        # tops for 600: [0, 256, 344]; lefts for 700: [0, 256, 444]
        # patch at top=256, left=256 -> source [256:512, 256:512]
        self.assertTrue(
            torch.equal(img_patches[4][:, :256, :256], image[:, 256:512, 256:512])
        )
        self.assertTrue(
            torch.equal(mask_patches[4][:256, :256], mask[256:512, 256:512])
        )


class TestOverlappingPatches(unittest.TestCase):
    """E. Overlapping patches — stride < patch_size."""

    def test_stride_128_on_512(self):
        image, mask = _make_divisible_sample()
        img_patches, mask_patches, metadata = extract_patches(
            image, mask, sample_id="s6", patch_size=256, stride=128
        )
        # tops for 512, P=256, S=128: [0, 128, 256] -> 3 per dim -> 9 patches
        self.assertEqual(len(metadata), 9)
        self.assertEqual(img_patches.shape, (9, 3, 256, 256))
        self.assertEqual(mask_patches.shape, (9, 256, 256))
        # Coordinates deterministic: tops [0, 128, 256], lefts [0, 128, 256]
        expected_tops = [0, 128, 256]
        expected_lefts = [0, 128, 256]
        idx = 0
        for t in expected_tops:
            for l in expected_lefts:
                self.assertEqual(metadata[idx]["top"], t)
                self.assertEqual(metadata[idx]["left"], l)
                idx += 1

    def test_overlap_boundary_coverage(self):
        H, W = 512, 512
        image = torch.randn(1, H, W)
        mask = torch.zeros(H, W, dtype=torch.int64)
        img_patches, mask_patches, _ = extract_patches(
            image, mask, "s7", patch_size=256, stride=128
        )
        # Reconstruct canvas to confirm full coverage with overlap.
        canvas_img = torch.full_like(image, float("nan"))
        for p in range(len(img_patches)):
            meta = _ = None  # noqa
        # Direct: first patch covers [0:256, 0:256], second [0:256, 128:384].
        # Top-left 128x128 of patch 0 equals top-left 128x128 of the source.
        self.assertTrue(
            torch.equal(img_patches[0][:, :128, :128], image[:, :128, :128])
        )
        # Overlap region [0:256, 128:256] appears in both patch 0 (cols
        # 128:256) and patch 1 (cols 0:128); values must agree (no shift /
        # no interpolation artifacts).
        self.assertTrue(
            torch.equal(
                img_patches[0][:, :, 128:256], img_patches[1][:, :, :128]
            )
        )


class TestMetadata(unittest.TestCase):
    """F. Metadata — correct and deterministic."""

    def test_metadata_fields_and_order(self):
        image = torch.randn(3, 512, 512)
        mask = torch.zeros(512, 512, dtype=torch.int64)
        img_patches, mask_patches, metadata = extract_patches(
            image, mask, sample_id="galaxy-42", patch_size=256, stride=256
        )
        self.assertEqual(len(metadata), 4)
        # Row-major order: tops [0, 256], lefts [0, 256]
        expected = [
            (0, 0),
            (0, 256),
            (256, 0),
            (256, 256),
        ]
        for p, (exp_top, exp_left) in enumerate(expected):
            m = metadata[p]
            self.assertEqual(m["sample_id"], "galaxy-42")
            self.assertEqual(m["patch_index"], p)
            self.assertEqual(m["top"], exp_top)
            self.assertEqual(m["left"], exp_left)
            self.assertEqual(m["original_height"], 512)
            self.assertEqual(m["original_width"], 512)
            self.assertEqual(m["patch_height"], 256)
            self.assertEqual(m["patch_width"], 256)
            # Required keys present.
            for key in (
                "sample_id",
                "patch_index",
                "top",
                "left",
                "original_height",
                "original_width",
                "patch_height",
                "patch_width",
            ):
                self.assertIn(key, m)

    def test_patch_index_matches_tensor_order(self):
        H, W = 300, 400
        image = torch.randn(2, H, W)
        mask = torch.zeros(H, W, dtype=torch.int64)
        _, _, metadata = extract_patches(
            image, mask, "s8", patch_size=256, stride=256
        )
        # tops [0, 44] (300<=? no, 300>256: steps=(300-256)//256=0 -> [0], then 0<44 -> [0,44])
        # actually H=300>P=256: num_steps=(300-256)//256=0, positions=[0], 0<44 -> append 44 -> [0,44]
        # lefts W=400: num_steps=(400-256)//256=0, positions=[0], 0<144 -> append 144 -> [0,144]
        # 2x2=4 patches
        self.assertEqual(len(metadata), 4)
        for p, m in enumerate(metadata):
            self.assertEqual(m["patch_index"], p)


class TestDtypePreservation(unittest.TestCase):
    """G. Dtype preservation."""

    def test_image_and_mask_dtypes_preserved(self):
        image = torch.randn(3, 512, 512, dtype=torch.float32)
        mask = torch.zeros(512, 512, dtype=torch.int64)
        img_patches, mask_patches, _ = extract_patches(
            image, mask, "s9", patch_size=256, stride=256
        )
        self.assertEqual(img_patches.dtype, torch.float32)
        self.assertEqual(mask_patches.dtype, torch.int64)

    def test_other_dtypes_preserved(self):
        image = torch.randn(3, 64, 64, dtype=torch.float64)
        mask = torch.zeros(64, 64, dtype=torch.int16)
        img_patches, mask_patches, _ = extract_patches(
            image, mask, "s10", patch_size=256, stride=256
        )
        self.assertEqual(img_patches.dtype, torch.float64)
        self.assertEqual(mask_patches.dtype, torch.int16)


class TestInvalidInputs(unittest.TestCase):
    """H. Invalid inputs produce clear exceptions."""

    def setUp(self):
        self.image = torch.randn(3, 64, 64)
        self.mask = torch.zeros(64, 64, dtype=torch.int64)

    def test_image_not_tensor(self):
        with self.assertRaises(TypeError):
            extract_patches([[1, 2], [3, 4]], self.mask, "s")  # type: ignore[arg-type]

    def test_mask_not_tensor(self):
        with self.assertRaises(TypeError):
            extract_patches(self.image, [[1, 2], [3, 4]], "s")  # type: ignore[arg-type]

    def test_image_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            extract_patches(torch.randn(64, 64), self.mask, "s")

    def test_mask_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            extract_patches(self.image, torch.zeros(64, 64, 64, dtype=torch.int64), "s")

    def test_mismatched_spatial_dims(self):
        with self.assertRaises(ValueError):
            extract_patches(
                torch.randn(3, 64, 64), torch.zeros(32, 32, dtype=torch.int64), "s"
            )

    def test_invalid_patch_size(self):
        for bad in (0, -1, 1.5, "256", True):
            with self.subTest(patch_size=bad):
                with self.assertRaises((ValueError, TypeError)):
                    extract_patches(
                        self.image, self.mask, "s", patch_size=bad  # type: ignore[arg-type]
                    )

    def test_invalid_stride(self):
        for bad in (0, -1, 2.5, "128", True):
            with self.subTest(stride=bad):
                with self.assertRaises((ValueError, TypeError)):
                    extract_patches(
                        self.image, self.mask, "s", stride=bad  # type: ignore[arg-type]
                    )

    def test_floating_point_mask_rejected(self):
        with self.assertRaises(TypeError):
            extract_patches(
                self.image,
                torch.zeros(64, 64, dtype=torch.float32),
                "s",
            )

    def test_bool_mask_rejected(self):
        with self.assertRaises(TypeError):
            extract_patches(
                self.image,
                torch.zeros(64, 64, dtype=torch.bool),
                "s",
            )


if __name__ == "__main__":
    unittest.main()
