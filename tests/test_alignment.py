"""Tests for spatial alignment of image / mask / validity-mask transforms.

These tests verify that every spatial operation in
``src.preprocessing.transforms`` applies the exact same geometric
transformation to the image, categorical mask, and boolean validity mask.

Synthetic asymmetric markers are used so that any misalignment is
immediately detectable.
"""

import unittest
from typing import Optional, Tuple

import torch

from src.preprocessing.transforms import (
    IGNORE_INDEX,
    functional_hflip,
    functional_pad,
    functional_rot90,
    functional_vflip,
)


# ====================================================================
# Helpers — build small asymmetric test tensors
# ====================================================================

def _make_marker_tensors(
    c: int = 1,
    h: int = 5,
    w: int = 7,
    marker_row: int = 1,
    marker_col: int = 2,
    marker_class: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(image, mask, valid_mask)`` with a single marked pixel.

    The image has value ``42.0`` at ``[:, marker_row, marker_col]``,
    the mask has ``marker_class`` there, and the validity mask is
    ``False`` there (``True`` everywhere else).

    The test grid is intentionally **non-square** so that 90° rotation
    changes the shape and any row/column confusion is caught.
    """
    image = torch.zeros(c, h, w, dtype=torch.float32)
    mask = torch.full((h, w), IGNORE_INDEX, dtype=torch.int64)
    valid = torch.ones(h, w, dtype=torch.bool)

    image[:, marker_row, marker_col] = 42.0
    mask[marker_row, marker_col] = marker_class
    valid[marker_row, marker_col] = False
    return image, mask, valid


def _find_marker(
    image: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    marker_class: int = 2,
) -> Tuple[int, int]:
    """Return the (row, col) of the single marked pixel, asserting alignment."""
    img_pos = (image[0] == 42.0).nonzero(as_tuple=False)
    mask_pos = (mask == marker_class).nonzero(as_tuple=False)
    val_pos = (~valid).nonzero(as_tuple=False)

    # All three must find exactly one pixel
    assert img_pos.shape[0] == 1, f"Expected 1 image marker; found {img_pos.shape[0]}"
    assert mask_pos.shape[0] == 1, f"Expected 1 mask marker; found {mask_pos.shape[0]}"
    assert val_pos.shape[0] == 1, f"Expected 1 valid marker; found {val_pos.shape[0]}"

    # All three must agree on the same position
    assert torch.equal(img_pos, mask_pos), (
        f"Image marker at {img_pos.tolist()} ≠ mask marker at {mask_pos.tolist()}"
    )
    assert torch.equal(img_pos, val_pos), (
        f"Image marker at {img_pos.tolist()} ≠ valid marker at {val_pos.tolist()}"
    )
    return int(img_pos[0, 0]), int(img_pos[0, 1])


# ====================================================================
# Test cases
# ====================================================================

class TestHorizontalFlipAlignment(unittest.TestCase):
    """Verify horizontal flip synchronises image, mask and validity."""

    def test_marker_moves_to_mirrored_column(self):
        h, w, mr, mc = 5, 7, 1, 2
        img, mask, valid = _make_marker_tensors(c=3, h=h, w=w, marker_row=mr, marker_col=mc)
        f_img, f_mask, f_valid = functional_hflip(img, mask, valid)
        r, c = _find_marker(f_img, f_mask, f_valid)
        self.assertEqual(r, mr, "Row should be unchanged by hflip")
        self.assertEqual(c, w - 1 - mc, f"Column should flip: expected {w - 1 - mc}, got {c}")

    def test_double_hflip_is_identity(self):
        img, mask, valid = _make_marker_tensors()
        f_img, f_mask, f_valid = functional_hflip(img, mask, valid)
        r_img, r_mask, r_valid = functional_hflip(f_img, f_mask, f_valid)
        self.assertTrue(torch.equal(img, r_img))
        self.assertTrue(torch.equal(mask, r_mask))
        self.assertTrue(torch.equal(valid, r_valid))

    def test_without_validity_mask(self):
        img, mask, _ = _make_marker_tensors()
        f_img, f_mask, f_valid = functional_hflip(img, mask, None)
        self.assertIsNone(f_valid)
        self.assertEqual(f_img.shape, img.shape)


class TestVerticalFlipAlignment(unittest.TestCase):
    """Verify vertical flip synchronises image, mask and validity."""

    def test_marker_moves_to_mirrored_row(self):
        h, w, mr, mc = 5, 7, 1, 2
        img, mask, valid = _make_marker_tensors(c=3, h=h, w=w, marker_row=mr, marker_col=mc)
        f_img, f_mask, f_valid = functional_vflip(img, mask, valid)
        r, c = _find_marker(f_img, f_mask, f_valid)
        self.assertEqual(r, h - 1 - mr, f"Row should flip: expected {h - 1 - mr}, got {r}")
        self.assertEqual(c, mc, "Column should be unchanged by vflip")

    def test_double_vflip_is_identity(self):
        img, mask, valid = _make_marker_tensors()
        f_img, f_mask, f_valid = functional_vflip(img, mask, valid)
        r_img, r_mask, r_valid = functional_vflip(f_img, f_mask, f_valid)
        self.assertTrue(torch.equal(img, r_img))
        self.assertTrue(torch.equal(mask, r_mask))
        self.assertTrue(torch.equal(valid, r_valid))


class TestRot90Alignment(unittest.TestCase):
    """Verify 90° rotations synchronise image, mask and validity."""

    def test_rot90_marker_position(self):
        """After 90° CCW rotation of (H, W) image, (r, c) -> (W-1-c, r)."""
        h, w, mr, mc = 5, 7, 1, 2
        img, mask, valid = _make_marker_tensors(c=1, h=h, w=w, marker_row=mr, marker_col=mc)
        r_img, r_mask, r_valid = functional_rot90(img, mask, k=1, valid_mask=valid)
        # After rot90(k=1): new shape is (W, H), position (r, c) -> (W-1-c, r)
        self.assertEqual(r_img.shape, (1, w, h))
        r, c = _find_marker(r_img, r_mask, r_valid)
        self.assertEqual(r, w - 1 - mc)
        self.assertEqual(c, mr)

    def test_rot180_marker_position(self):
        h, w, mr, mc = 5, 7, 1, 2
        img, mask, valid = _make_marker_tensors(c=1, h=h, w=w, marker_row=mr, marker_col=mc)
        r_img, r_mask, r_valid = functional_rot90(img, mask, k=2, valid_mask=valid)
        self.assertEqual(r_img.shape, (1, h, w))
        r, c = _find_marker(r_img, r_mask, r_valid)
        self.assertEqual(r, h - 1 - mr)
        self.assertEqual(c, w - 1 - mc)

    def test_rot270_marker_position(self):
        h, w, mr, mc = 5, 7, 1, 2
        img, mask, valid = _make_marker_tensors(c=1, h=h, w=w, marker_row=mr, marker_col=mc)
        r_img, r_mask, r_valid = functional_rot90(img, mask, k=3, valid_mask=valid)
        self.assertEqual(r_img.shape, (1, w, h))
        r, c = _find_marker(r_img, r_mask, r_valid)
        self.assertEqual(r, mc)
        self.assertEqual(c, h - 1 - mr)

    def test_rot0_is_identity(self):
        img, mask, valid = _make_marker_tensors()
        r_img, r_mask, r_valid = functional_rot90(img, mask, k=0, valid_mask=valid)
        self.assertTrue(torch.equal(img, r_img))
        self.assertTrue(torch.equal(mask, r_mask))
        self.assertTrue(torch.equal(valid, r_valid))

    def test_four_rot90_is_identity(self):
        img, mask, valid = _make_marker_tensors(c=5)
        current = (img, mask, valid)
        for _ in range(4):
            current = functional_rot90(current[0], current[1], k=1, valid_mask=current[2])
        self.assertTrue(torch.equal(img, current[0]))
        self.assertTrue(torch.equal(mask, current[1]))
        self.assertTrue(torch.equal(valid, current[2]))

    def test_invalid_k_raises(self):
        img, mask, valid = _make_marker_tensors()
        with self.assertRaises(ValueError):
            functional_rot90(img, mask, k=4, valid_mask=valid)
        with self.assertRaises(ValueError):
            functional_rot90(img, mask, k=-1, valid_mask=valid)


class TestPadAlignment(unittest.TestCase):
    """Verify that padding preserves alignment and fills correctly."""

    def test_pad_symmetric_alignment(self):
        h, w, mr, mc = 4, 6, 1, 2
        img, mask, valid = _make_marker_tensors(c=3, h=h, w=w, marker_row=mr, marker_col=mc)
        th, tw = 8, 10
        p_img, p_mask, p_valid = functional_pad(img, mask, th, tw, valid)

        # Offsets
        pad_top = (th - h) // 2  # 2
        pad_left = (tw - w) // 2  # 2

        # Check image marker position
        img_pos = (p_img[0] == 42.0).nonzero(as_tuple=False)
        self.assertEqual(img_pos.shape[0], 1)
        self.assertEqual(img_pos[0, 0].item(), mr + pad_top)
        self.assertEqual(img_pos[0, 1].item(), mc + pad_left)

        # Check mask marker position
        mask_pos = (p_mask == 2).nonzero(as_tuple=False)
        self.assertEqual(mask_pos.shape[0], 1)
        self.assertTrue(torch.equal(img_pos, mask_pos))

        # Image and mask markers must agree
        self.assertEqual(img_pos[0, 0].item(), mask_pos[0, 0].item())
        self.assertEqual(img_pos[0, 1].item(), mask_pos[0, 1].item())

    def test_pad_fills_mask_with_ignore_index(self):
        img, mask, _ = _make_marker_tensors(c=1, h=4, w=4)
        mask[:] = 0  # fill with class 0
        p_img, p_mask, _ = functional_pad(img, mask, 8, 8)
        # All border values must be IGNORE_INDEX
        self.assertEqual(p_mask[0, 0].item(), IGNORE_INDEX)
        self.assertEqual(p_mask[7, 7].item(), IGNORE_INDEX)
        # Interior preserved
        self.assertEqual(p_mask[2, 2].item(), 0)

    def test_pad_fills_valid_mask_with_false(self):
        img, mask, valid = _make_marker_tensors(c=1, h=4, w=4)
        valid[:] = True
        _, _, p_valid = functional_pad(img, mask, 8, 8, valid)
        self.assertFalse(p_valid[0, 0].item())
        self.assertFalse(p_valid[7, 7].item())
        self.assertTrue(p_valid[2, 2].item())

    def test_pad_noop_when_already_target_size(self):
        img, mask, valid = _make_marker_tensors(c=1, h=8, w=8)
        p_img, p_mask, p_valid = functional_pad(img, mask, 8, 8, valid)
        self.assertTrue(torch.equal(img, p_img))
        self.assertTrue(torch.equal(mask, p_mask))
        self.assertTrue(torch.equal(valid, p_valid))

    def test_pad_rejects_smaller_target(self):
        img, mask, valid = _make_marker_tensors(c=1, h=8, w=8)
        with self.assertRaises(ValueError):
            functional_pad(img, mask, 4, 4, valid)

    def test_pad_without_validity_mask(self):
        img, mask, _ = _make_marker_tensors(c=1, h=4, w=4)
        p_img, p_mask, p_valid = functional_pad(img, mask, 8, 8)
        self.assertIsNone(p_valid)
        self.assertEqual(p_img.shape, (1, 8, 8))


class TestCategoricalIntegrity(unittest.TestCase):
    """Verify that no transform introduces spurious mask values."""

    def _all_label_set(self) -> set:
        return {0, 1, 2, 3, IGNORE_INDEX}

    def _make_multi_class_mask(self, h: int = 8, w: int = 8) -> torch.Tensor:
        mask = torch.full((h, w), IGNORE_INDEX, dtype=torch.int64)
        mask[0, 0] = 0
        mask[1, 1] = 1
        mask[2, 2] = 2
        mask[3, 3] = 3
        return mask

    def test_hflip_preserves_labels(self):
        img = torch.rand(1, 8, 8)
        mask = self._make_multi_class_mask()
        _, f_mask, _ = functional_hflip(img, mask)
        self.assertTrue(set(f_mask.unique().tolist()).issubset(self._all_label_set()))

    def test_vflip_preserves_labels(self):
        img = torch.rand(1, 8, 8)
        mask = self._make_multi_class_mask()
        _, f_mask, _ = functional_vflip(img, mask)
        self.assertTrue(set(f_mask.unique().tolist()).issubset(self._all_label_set()))

    def test_rot90_preserves_labels(self):
        img = torch.rand(1, 8, 8)
        mask = self._make_multi_class_mask()
        for k in (1, 2, 3):
            _, r_mask, _ = functional_rot90(img, mask, k)
            with self.subTest(k=k):
                self.assertTrue(set(r_mask.unique().tolist()).issubset(self._all_label_set()))

    def test_pad_preserves_labels(self):
        img = torch.rand(1, 8, 8)
        mask = self._make_multi_class_mask()
        _, p_mask, _ = functional_pad(img, mask, 12, 12)
        self.assertTrue(set(p_mask.unique().tolist()).issubset(self._all_label_set()))

    def test_one_pixel_components_survive_all_transforms(self):
        """A 1-pixel Class-0 region must survive flips and rotations."""
        img = torch.zeros(1, 10, 10)
        mask = torch.full((10, 10), IGNORE_INDEX, dtype=torch.int64)
        mask[3, 7] = 0  # single pixel
        img[0, 3, 7] = 1.0

        for name, fn in [
            ("hflip", lambda i, m: functional_hflip(i, m)),
            ("vflip", lambda i, m: functional_vflip(i, m)),
            ("rot90", lambda i, m: functional_rot90(i, m, k=1)),
            ("rot180", lambda i, m: functional_rot90(i, m, k=2)),
            ("rot270", lambda i, m: functional_rot90(i, m, k=3)),
        ]:
            with self.subTest(transform=name):
                t_img, t_mask, _ = fn(img, mask)
                self.assertEqual(
                    (t_mask == 0).sum().item(), 1,
                    f"{name} lost the 1-pixel component"
                )


class TestInputValidation(unittest.TestCase):
    """Verify that invalid inputs raise clear errors."""

    def test_image_wrong_ndim(self):
        with self.assertRaises(ValueError):
            functional_hflip(torch.rand(8, 8), torch.zeros(8, 8, dtype=torch.int64))

    def test_mask_wrong_dtype(self):
        with self.assertRaises(TypeError):
            functional_hflip(torch.rand(1, 8, 8), torch.zeros(8, 8, dtype=torch.float32))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            functional_hflip(torch.rand(1, 8, 8), torch.zeros(4, 4, dtype=torch.int64))

    def test_valid_mask_wrong_dtype(self):
        with self.assertRaises(TypeError):
            functional_hflip(
                torch.rand(1, 8, 8),
                torch.zeros(8, 8, dtype=torch.int64),
                torch.zeros(8, 8, dtype=torch.int64),
            )


class TestMultiChannelAlignment(unittest.TestCase):
    """Verify alignment with 1, 3, and 5 channel images."""

    def test_channels_1_3_5(self):
        for c in (1, 3, 5):
            with self.subTest(channels=c):
                h, w, mr, mc = 6, 8, 2, 3
                img, mask, valid = _make_marker_tensors(c=c, h=h, w=w, marker_row=mr, marker_col=mc)
                # Compound transform: hflip then rot90
                t_img, t_mask, t_valid = functional_hflip(img, mask, valid)
                t_img, t_mask, t_valid = functional_rot90(t_img, t_mask, k=1, valid_mask=t_valid)
                # Verify all channels agree with the mask
                for ch in range(c):
                    ch_pos = (t_img[ch] == 42.0).nonzero(as_tuple=False)
                    mask_pos = (t_mask != IGNORE_INDEX).nonzero(as_tuple=False)
                    self.assertTrue(
                        torch.equal(ch_pos, mask_pos),
                        f"Channel {ch} misaligned with mask"
                    )


class TestDeviceCompatibility(unittest.TestCase):
    """Verify transforms work on both CPU and CUDA (if available)."""

    def _run_on_device(self, device: torch.device):
        img, mask, valid = _make_marker_tensors(c=3, h=6, w=8)
        img, mask, valid = img.to(device), mask.to(device), valid.to(device)

        f_img, f_mask, f_valid = functional_hflip(img, mask, valid)
        self.assertEqual(f_img.device.type, device.type)
        self.assertEqual(f_mask.device.type, device.type)
        self.assertEqual(f_valid.device.type, device.type)

        r_img, r_mask, r_valid = functional_rot90(f_img, f_mask, k=1, valid_mask=f_valid)
        self.assertEqual(r_img.device.type, device.type)

    def test_cpu(self):
        self._run_on_device(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._run_on_device(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
