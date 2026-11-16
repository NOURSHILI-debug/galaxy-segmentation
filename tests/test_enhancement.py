"""Tests for the astronomical image enhancement module.

Covers ``AsinhStretch`` from ``src.preprocessing.enhancement``.
"""

import unittest

import torch

from src.preprocessing.enhancement import AsinhStretch


class TestAsinhStretchOutputBounds(unittest.TestCase):
    """Output must always be in [0, 1] for inputs in [0, 1]."""

    def test_zeros(self):
        stretch = AsinhStretch(beta=0.1)
        img = torch.zeros(1, 8, 8)
        out = stretch(img)
        self.assertAlmostEqual(out.min().item(), 0.0, places=6)

    def test_ones(self):
        stretch = AsinhStretch(beta=0.1)
        img = torch.ones(1, 8, 8)
        out = stretch(img)
        self.assertAlmostEqual(out.max().item(), 1.0, places=6)

    def test_random_input_bounded(self):
        stretch = AsinhStretch(beta=0.05)
        img = torch.rand(5, 64, 64)
        out = stretch(img)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_output_dtype_is_float32(self):
        stretch = AsinhStretch(beta=0.1)
        img = torch.rand(3, 16, 16, dtype=torch.float64)
        out = stretch(img)
        self.assertEqual(out.dtype, torch.float32)


class TestAsinhStretchMonotonicity(unittest.TestCase):
    """Asinh stretch must be monotonically non-decreasing."""

    def test_monotonic_on_ramp(self):
        stretch = AsinhStretch(beta=0.1)
        ramp = torch.linspace(0.0, 1.0, 100).unsqueeze(0).unsqueeze(0)  # [1, 1, 100]
        out = stretch(ramp)
        diffs = out[0, 0, 1:] - out[0, 0, :-1]
        self.assertTrue((diffs >= -1e-7).all(), "Asinh stretch is not monotonic")


class TestAsinhStretchNumericalStability(unittest.TestCase):
    """Edge cases for numerical stability."""

    def test_very_small_beta(self):
        stretch = AsinhStretch(beta=0.001)
        img = torch.rand(1, 8, 8)
        out = stretch(img)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_large_beta(self):
        stretch = AsinhStretch(beta=10.0)
        img = torch.rand(1, 8, 8)
        out = stretch(img)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_no_nan_or_inf(self):
        stretch = AsinhStretch(beta=0.1)
        # Include exact boundary values
        img = torch.tensor([[[0.0, 0.5, 1.0]]])
        out = stretch(img)
        self.assertTrue(torch.isfinite(out).all())


class TestAsinhStretchMultiChannel(unittest.TestCase):
    """Multi-channel and batched inputs."""

    def test_single_channel(self):
        stretch = AsinhStretch()
        img = torch.rand(1, 32, 32)
        out = stretch(img)
        self.assertEqual(out.shape, (1, 32, 32))

    def test_three_channels(self):
        stretch = AsinhStretch()
        img = torch.rand(3, 32, 32)
        out = stretch(img)
        self.assertEqual(out.shape, (3, 32, 32))

    def test_five_channels(self):
        stretch = AsinhStretch()
        img = torch.rand(5, 208, 208)
        out = stretch(img)
        self.assertEqual(out.shape, (5, 208, 208))

    def test_batched_input(self):
        stretch = AsinhStretch()
        img = torch.rand(4, 3, 32, 32)  # [B, C, H, W]
        out = stretch(img)
        self.assertEqual(out.shape, (4, 3, 32, 32))


class TestAsinhStretchNonMutation(unittest.TestCase):
    """Input tensor must not be modified."""

    def test_input_unchanged(self):
        stretch = AsinhStretch()
        img = torch.rand(3, 16, 16)
        original = img.clone()
        _ = stretch(img)
        self.assertTrue(torch.equal(img, original))


class TestAsinhStretchInvalidBeta(unittest.TestCase):
    """Invalid beta values must be rejected."""

    def test_zero_beta(self):
        with self.assertRaises(ValueError):
            AsinhStretch(beta=0.0)

    def test_negative_beta(self):
        with self.assertRaises(ValueError):
            AsinhStretch(beta=-0.1)

    def test_nan_beta(self):
        with self.assertRaises(ValueError):
            AsinhStretch(beta=float("nan"))

    def test_inf_beta(self):
        with self.assertRaises(ValueError):
            AsinhStretch(beta=float("inf"))

    def test_string_beta(self):
        with self.assertRaises(TypeError):
            AsinhStretch(beta="0.1")


class TestAsinhStretchInvalidImage(unittest.TestCase):
    """Invalid image inputs must be rejected."""

    def test_integer_tensor_rejected(self):
        stretch = AsinhStretch()
        with self.assertRaises(TypeError):
            stretch(torch.randint(0, 10, (1, 8, 8)))

    def test_2d_tensor_rejected(self):
        stretch = AsinhStretch()
        with self.assertRaises(ValueError):
            stretch(torch.rand(8, 8))

    def test_non_tensor_rejected(self):
        stretch = AsinhStretch()
        with self.assertRaises(TypeError):
            stretch([[0.1, 0.2], [0.3, 0.4]])

    def test_nan_rejected(self):
        stretch = AsinhStretch()
        img = torch.rand(1, 8, 8)
        img[0, 2, 2] = float("nan")
        with self.assertRaises(ValueError):
            stretch(img)

    def test_inf_rejected(self):
        stretch = AsinhStretch()
        img = torch.rand(1, 8, 8)
        img[0, 2, 2] = float("inf")
        with self.assertRaises(ValueError):
            stretch(img)

    def test_negative_values_rejected(self):
        stretch = AsinhStretch()
        img = torch.rand(1, 8, 8)
        img[0, 2, 2] = -0.1
        with self.assertRaises(ValueError):
            stretch(img)

    def test_out_of_range_high_rejected(self):
        stretch = AsinhStretch()
        img = torch.rand(1, 8, 8)
        img[0, 2, 2] = 1.5
        with self.assertRaises(ValueError):
            stretch(img)


class TestAsinhStretchRepr(unittest.TestCase):
    """Repr must include the beta parameter."""

    def test_repr(self):
        stretch = AsinhStretch(beta=0.05)
        self.assertIn("0.05", repr(stretch))
        self.assertIn("AsinhStretch", repr(stretch))


class TestAsinhStretchDeviceCompatibility(unittest.TestCase):
    """Must work on CPU and CUDA if available."""

    def test_cpu(self):
        stretch = AsinhStretch()
        img = torch.rand(3, 16, 16)
        out = stretch(img)
        self.assertEqual(out.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        stretch = AsinhStretch()
        img = torch.rand(3, 16, 16, device="cuda")
        out = stretch(img)
        self.assertEqual(out.device.type, "cuda")
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
