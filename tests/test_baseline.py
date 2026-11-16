"""Unit tests for the end-to-end classical computer vision baseline pipeline.

Tests:
1. End-to-end pipeline execution with Otsu thresholding.
2. End-to-end pipeline execution with Adaptive thresholding.
3. Integrity of output keys, shapes, and data types.
4. Input validation (dimensions, threshold methods, mask alignment).
5. Real Pan-STARRS FITS sample execution and validity preservation.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import torch

from src.classical_cv.band_comparison import load_band_fits
from src.classical_cv.baseline import run_classical_pipeline


class TestClassicalBaseline(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        # Synthetic galaxy image: smooth sky background (mean 10, std 1)
        # with a central bright exponential profile (galaxy core + disk)
        self.img = np.random.normal(10.0, 1.0, (100, 100)).astype(np.float32)
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        galaxy = 80.0 * np.exp(-r / 12.0)
        self.img += galaxy

        self.valid_mask = np.ones((100, 100), dtype=bool)
        self.valid_mask[:5, :] = False  # masked edge

    def test_pipeline_otsu_execution(self) -> None:
        results = run_classical_pipeline(
            self.img,
            valid_mask=self.valid_mask,
            threshold_method="otsu",
            morph_shape="disk",
            morph_radius=1,
            connectivity=8,
        )

        expected_keys = [
            "threshold_mask",
            "morph_mask",
            "labeled_components",
            "component_stats",
            "distance_map",
            "markers",
            "labeled_basins",
            "watershed_boundaries",
            "metadata",
        ]
        for k in expected_keys:
            self.assertIn(k, results)

        self.assertEqual(results["threshold_mask"].shape, (100, 100))
        self.assertEqual(results["threshold_mask"].dtype, bool)
        self.assertEqual(results["morph_mask"].shape, (100, 100))
        self.assertEqual(results["morph_mask"].dtype, bool)
        self.assertEqual(results["labeled_components"].shape, (100, 100))
        self.assertEqual(results["labeled_components"].dtype, np.int32)
        self.assertEqual(results["distance_map"].shape, (100, 100))
        self.assertEqual(results["distance_map"].dtype, np.float32)
        self.assertEqual(results["labeled_basins"].shape, (100, 100))
        self.assertEqual(results["labeled_basins"].dtype, np.int32)

        # Masked rows (0..4) must never be foreground
        self.assertFalse(np.any(results["threshold_mask"][:5, :]))
        self.assertFalse(np.any(results["morph_mask"][:5, :]))
        self.assertTrue(np.all(results["labeled_components"][:5, :] == 0))
        self.assertTrue(np.all(results["distance_map"][:5, :] == 0.0))
        self.assertTrue(np.all(results["labeled_basins"][:5, :] == 0))

        # Central core must be detected
        self.assertTrue(results["morph_mask"][50, 50])
        self.assertGreater(results["metadata"]["connected_components"]["num_components"], 0)

    def test_pipeline_adaptive_execution(self) -> None:
        results = run_classical_pipeline(
            self.img,
            valid_mask=self.valid_mask,
            threshold_method="adaptive",
            adaptive_window=21,
            adaptive_c=-10.0,
            morph_shape="disk",
            morph_radius=1,
        )
        self.assertEqual(results["threshold_mask"].shape, (100, 100))
        self.assertTrue(results["morph_mask"][50, 50])

    def test_pipeline_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            run_classical_pipeline(np.zeros((50, 50, 3)))
        with self.assertRaises(ValueError):
            run_classical_pipeline(self.img, threshold_method="kmeans")
        with self.assertRaises(ValueError):
            run_classical_pipeline(self.img, valid_mask=np.zeros((50, 50), dtype=bool))

    def test_pipeline_real_fits(self) -> None:
        fits_path = Path("dataset/raw/imagedata_1-100523.fits")
        if not fits_path.exists():
            self.skipTest(f"FITS file {fits_path} not found.")

        # Load r-band
        r_band, val_mask, _ = load_band_fits(fits_path, band="r")

        # Provisionally normalize valid flux to [0, 1] using robust percentiles
        p05 = float(np.percentile(r_band[val_mask], 0.5))
        p995 = float(np.percentile(r_band[val_mask], 99.5))
        norm_img = np.clip((r_band - p05) / max(p995 - p05, 1e-8), 0.0, 1.0)

        results = run_classical_pipeline(
            norm_img,
            valid_mask=val_mask,
            threshold_method="otsu",
            morph_shape="disk",
            morph_radius=1,
            watershed_min_distance=5,
            watershed_threshold=2.0,
        )

        self.assertEqual(results["threshold_mask"].shape, (208, 208))
        self.assertGreater(results["metadata"]["connected_components"]["num_components"], 0)
        self.assertGreater(results["metadata"]["watershed"]["num_basins"], 0)


if __name__ == "__main__":
    unittest.main()
