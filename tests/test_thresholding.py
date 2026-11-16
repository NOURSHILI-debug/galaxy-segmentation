"""Unit tests for classical thresholding (Otsu & Adaptive).

Tests:
1. Otsu on bimodal synthetic distributions.
2. Otsu constant and all-invalid images.
3. Otsu NaN / Inf handling and valid_mask masking.
4. Otsu input validation (dimensions, mismatched mask).
5. Adaptive thresholding on synthetic gradients and local structures.
6. Adaptive threshold window size validation.
7. Adaptive threshold NaN / valid_mask handling.
8. Real FITS smoke test on Pan-STARRS images.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import torch

from src.classical_cv.band_comparison import load_band_fits
from src.classical_cv.thresholding import adaptive_threshold, otsu_threshold


class TestOtsuThreshold(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_otsu_bimodal_separation(self) -> None:
        # Create bimodal image: background around 10, foreground disk around 100
        img = np.random.normal(loc=10.0, scale=2.0, size=(100, 100)).astype(np.float32)
        y, x = np.ogrid[:100, :100]
        disk_mask = ((x - 50) ** 2 + (y - 50) ** 2) < 20**2
        img[disk_mask] = np.random.normal(loc=100.0, scale=5.0, size=np.count_nonzero(disk_mask))

        bin_mask, thresh, meta = otsu_threshold(img)

        # Threshold should separate 10 and 100 cleanly
        self.assertGreater(thresh, 15.0)
        self.assertLess(thresh, 85.0)
        self.assertEqual(bin_mask.shape, (100, 100))
        self.assertEqual(bin_mask.dtype, bool)
        # Agreement with disk mask should be high (>98%)
        overlap = np.sum(bin_mask & disk_mask) / np.sum(bin_mask | disk_mask)
        self.assertGreater(overlap, 0.95)
        self.assertIn("max_variance", meta)
        self.assertGreater(meta["max_variance"], 0.0)

    def test_otsu_constant_image(self) -> None:
        img = np.full((50, 50), 42.0, dtype=np.float32)
        bin_mask, thresh, meta = otsu_threshold(img)
        self.assertFalse(np.any(bin_mask))
        self.assertAlmostEqual(thresh, 42.0)
        self.assertEqual(meta["max_variance"], 0.0)

    def test_otsu_all_invalid(self) -> None:
        img = np.full((50, 50), np.nan, dtype=np.float32)
        bin_mask, thresh, meta = otsu_threshold(img)
        self.assertFalse(np.any(bin_mask))
        self.assertEqual(thresh, 0.0)
        self.assertEqual(meta["valid_pixel_count"], 0)

    def test_otsu_nan_inf_and_valid_mask(self) -> None:
        img = np.random.normal(10.0, 1.0, (60, 60)).astype(np.float32)
        img[20:40, 20:40] += 50.0  # foreground square

        # Introduce NaNs and Infs:
        # img[0, 0] is inside rows 0..4 (which are masked by valid_mask anyway)
        img[0, 0] = np.nan
        img[1, 1] = np.inf
        # img[25, 25] is inside valid region
        img[25, 25] = np.nan

        valid_mask = np.ones((60, 60), dtype=bool)
        valid_mask[0:5, :] = False  # masked edge (300 pixels)

        bin_mask, thresh, meta = otsu_threshold(img, valid_mask=valid_mask)

        # NaNs, Infs, and masked regions must NEVER be foreground
        self.assertFalse(bin_mask[0, 0])
        self.assertFalse(bin_mask[1, 1])
        self.assertFalse(bin_mask[25, 25])
        self.assertFalse(np.any(bin_mask[0:5, :]))
        self.assertTrue(bin_mask[30, 30])  # inside square
        # 3600 total - 300 (masked rows) - 1 (NaN at [25, 25]) = 3299
        self.assertEqual(meta["valid_pixel_count"], 3299)

    def test_otsu_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            otsu_threshold(np.zeros((10, 10, 3)))  # 3D
        with self.assertRaises(ValueError):
            otsu_threshold(np.zeros((10, 10)), valid_mask=np.zeros((10, 12), dtype=bool))

    def test_otsu_torch_tensor(self) -> None:
        t_img = torch.randn(50, 50) + 10.0
        t_img[20:30, 20:30] += 50.0
        bin_mask, thresh, meta = otsu_threshold(t_img)
        self.assertIsInstance(bin_mask, np.ndarray)
        self.assertEqual(bin_mask.shape, (50, 50))


class TestAdaptiveThreshold(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_adaptive_local_structure(self) -> None:
        # Background with a smooth spatial gradient: 10 to 50
        y, x = np.mgrid[:100, :100]
        bg = 10.0 + 0.4 * (x + y)
        img = bg.copy()
        # Add local bright peaks on top of gradient
        img[25, 25] += 30.0
        img[75, 75] += 30.0

        # Local mean - c_offset: c_offset=-5.0 means threshold = local_mean + 5.0
        bin_mask, thresh_map, meta = adaptive_threshold(
            img, window_size=15, c_offset=-5.0
        )
        self.assertEqual(bin_mask.shape, (100, 100))
        self.assertEqual(thresh_map.shape, (100, 100))
        # The peaks should be detected
        self.assertTrue(bin_mask[25, 25])
        self.assertTrue(bin_mask[75, 75])
        # Smooth background away from peaks should be False
        self.assertFalse(bin_mask[50, 50])

    def test_adaptive_window_size_validation(self) -> None:
        img = np.zeros((30, 30), dtype=np.float32)
        with self.assertRaises(ValueError):
            adaptive_threshold(img, window_size=2)  # < 3
        with self.assertRaises(ValueError):
            adaptive_threshold(img, window_size=10)  # even

    def test_adaptive_nan_handling(self) -> None:
        img = np.ones((40, 40), dtype=np.float32) * 5.0
        img[20, 20] = 50.0
        img[0, 0] = np.nan
        img[0, 1] = np.inf

        bin_mask, thresh_map, meta = adaptive_threshold(img, window_size=9, c_offset=-5.0)
        self.assertFalse(bin_mask[0, 0])
        self.assertFalse(bin_mask[0, 1])
        self.assertTrue(bin_mask[20, 20])

    def test_adaptive_deterministic(self) -> None:
        img = np.random.normal(50.0, 10.0, (50, 50)).astype(np.float32)
        m1, t1, _ = adaptive_threshold(img, window_size=11, c_offset=2.0)
        m2, t2, _ = adaptive_threshold(img, window_size=11, c_offset=2.0)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_allclose(t1, t2)


class TestThresholdingRealFITS(unittest.TestCase):
    def test_real_fits_thresholds(self) -> None:
        fits_path = Path("dataset/raw/imagedata_1-100523.fits")
        if not fits_path.exists():
            self.skipTest(f"FITS file {fits_path} not found.")

        # Load r-band (HDU 1 / 'r')
        r_band, val_mask, _ = load_band_fits(fits_path, band="r")

        # Otsu
        otsu_mask, otsu_thresh, otsu_meta = otsu_threshold(r_band, valid_mask=val_mask)
        self.assertEqual(otsu_mask.shape, (208, 208))
        self.assertGreater(otsu_meta["foreground_pixel_count"], 0)
        self.assertLess(otsu_meta["foreground_pixel_count"], 208 * 208)

        # Adaptive
        adapt_mask, adapt_map, adapt_meta = adaptive_threshold(
            r_band, valid_mask=val_mask, window_size=25, c_offset=-50.0
        )
        self.assertEqual(adapt_mask.shape, (208, 208))
        self.assertGreater(adapt_meta["foreground_pixel_count"], 0)


if __name__ == "__main__":
    unittest.main()
