"""Tests for the five-band astronomical diagnostic comparison workflow."""

import unittest
from pathlib import Path

import numpy as np

from src.classical_cv.band_comparison import (
    BAND_NAMES,
    BAND_TO_HDU,
    HDU_TO_BAND,
    compare_five_bands,
    compute_band_metrics,
    load_band_fits,
    recommend_reference_band,
)

SAMPLE_CLEAN = Path("dataset/raw/imagedata_1-100523.fits")
SAMPLE_NAN = Path("dataset/raw/imagedata_1-103458.fits")


class TestBandMetrics(unittest.TestCase):
    """Test compute_band_metrics on synthetic and known distributions."""

    def test_empty_or_all_invalid(self):
        data = np.full((10, 10), np.nan, dtype=np.float32)
        metrics = compute_band_metrics(data)
        self.assertEqual(metrics["valid_count"], 0)
        self.assertEqual(metrics["invalid_fraction"], 1.0)
        self.assertIsNone(metrics["min"])
        self.assertIsNone(metrics["peak_snr"])

    def test_constant_image(self):
        data = np.full((20, 20), 42.0, dtype=np.float32)
        metrics = compute_band_metrics(data)
        self.assertEqual(metrics["valid_count"], 400)
        self.assertEqual(metrics["invalid_count"], 0)
        self.assertAlmostEqual(metrics["min"], 42.0)
        self.assertAlmostEqual(metrics["max"], 42.0)
        self.assertAlmostEqual(metrics["mean"], 42.0)
        self.assertAlmostEqual(metrics["std"], 0.0)

    def test_gaussian_noise_mad(self):
        """Standard normal noise should yield sky_mad ~ 1.0."""
        np.random.seed(42)
        data = np.random.normal(loc=100.0, scale=2.5, size=(100, 100)).astype(np.float32)
        metrics = compute_band_metrics(data)
        self.assertAlmostEqual(metrics["median"], 100.0, delta=0.2)
        self.assertAlmostEqual(metrics["sky_mad"], 2.5, delta=0.2)

    def test_with_validity_mask(self):
        data = np.array([[10.0, 20.0], [np.nan, 40.0]], dtype=np.float32)
        valid = np.array([[True, True], [False, True]], dtype=bool)
        metrics = compute_band_metrics(data, valid_mask=valid)
        self.assertEqual(metrics["valid_count"], 3)
        self.assertEqual(metrics["invalid_count"], 1)
        self.assertAlmostEqual(metrics["min"], 10.0)
        self.assertAlmostEqual(metrics["max"], 40.0)


class TestRecommendationLogic(unittest.TestCase):
    """Verify band recommendation logic."""

    def test_higher_snr_recommended(self):
        reports = {
            "g": {"valid_count": 100, "invalid_fraction": 0.0, "robust_snr": 5.0, "peak_snr": 10.0, "sky_mad": 2.0},
            "r": {"valid_count": 100, "invalid_fraction": 0.0, "robust_snr": 25.0, "peak_snr": 50.0, "sky_mad": 1.5},
            "i": {"valid_count": 100, "invalid_fraction": 0.0, "robust_snr": 15.0, "peak_snr": 30.0, "sky_mad": 2.2},
            "z": {"valid_count": 100, "invalid_fraction": 0.0, "robust_snr": 8.0, "peak_snr": 20.0, "sky_mad": 3.0},
            "y": {"valid_count": 100, "invalid_fraction": 0.0, "robust_snr": 4.0, "peak_snr": 12.0, "sky_mad": 4.5},
        }
        rec = recommend_reference_band(reports)
        self.assertEqual(rec["recommended_band"], "r")
        self.assertEqual(rec["recommended_hdu"], 1)

    def test_corrupted_band_penalized(self):
        """Band with NaNs is penalized over clean band with slightly lower SNR."""
        reports = {
            "g": {"valid_count": 80, "invalid_fraction": 0.20, "robust_snr": 30.0, "peak_snr": 60.0, "sky_mad": 1.0},
            "r": {"valid_count": 100, "invalid_fraction": 0.0, "robust_snr": 20.0, "peak_snr": 40.0, "sky_mad": 1.0},
        }
        rec = recommend_reference_band(reports)
        self.assertEqual(rec["recommended_band"], "r")
        self.assertEqual(rec["status"], "provisional")
        self.assertIn("Provisional", rec["reason"])
        self.assertIn("Broader validation", rec["reason"])


class TestRealFITSBandComparison(unittest.TestCase):
    """Smoke tests on real repository FITS files if present."""

    @unittest.skipUnless(SAMPLE_CLEAN.is_file(), "Clean sample FITS file not found")
    def test_clean_sample_five_bands(self):
        res = compare_five_bands(SAMPLE_CLEAN)
        self.assertEqual(len(res["bands"]), 5)
        for b in BAND_NAMES:
            self.assertIn(b, res["bands"])
            self.assertEqual(res["bands"][b]["invalid_count"], 0)
        self.assertIn(res["recommendation"]["recommended_band"], ("r", "i"))

    @unittest.skipUnless(SAMPLE_NAN.is_file(), "Problematic sample FITS file not found")
    def test_problematic_sample_g_band_nans(self):
        res = compare_five_bands(SAMPLE_NAN)
        self.assertEqual(res["bands"]["g"]["invalid_count"], 1280)
        # Recommended band must not be the corrupted g-band
        self.assertNotEqual(res["recommendation"]["recommended_band"], "g")
        self.assertEqual(res["recommendation"]["recommended_band"], "r")

    def test_invalid_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_band_fits("non_existent_file.fits", band=1)

    def test_invalid_band_name_raises(self):
        with self.assertRaises(ValueError):
            load_band_fits(SAMPLE_CLEAN, band="invalid_band")


if __name__ == "__main__":
    unittest.main()
