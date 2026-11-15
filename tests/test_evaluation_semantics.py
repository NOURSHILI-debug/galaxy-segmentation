"""Unit tests for quantitative evaluation metrics and ground-truth semantics.

Tests:
1. Annotated-domain sensitivity calculation.
2. Handling of ignore index 255 (unannotated pixels, not confirmed background).
3. Unverified predictions outside annotated regions.
4. Correct undefined precision status when negative background ground truth is absent.
5. Exact verified pixel counts on sample 1-100523.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
from astropy.io import fits

from src.classical_cv.band_comparison import load_band_fits
from src.classical_cv.baseline import run_classical_pipeline
from src.data.mask_decoder import decode_mask


class TestQuantitativeEvaluationSemantics(unittest.TestCase):
    def test_synthetic_annotated_domain_metrics(self) -> None:
        # Create a synthetic ground truth mask of 100x100
        # 200 pixels are annotated foreground (classes 0..3)
        # Remaining 9800 pixels are ignore index 255
        gt_mask = np.full((100, 100), 255, dtype=np.int64)
        gt_mask[40:50, 40:60] = 0  # 200 pixels core (class 0)

        # Prediction detects 100 of those core pixels, and 50 pixels outside in ignore region
        pred_mask = np.zeros((100, 100), dtype=bool)
        pred_mask[40:50, 40:50] = True  # 100 pixels in core
        pred_mask[10:15, 10:20] = True  # 50 pixels in ignore region

        gt_annotated = gt_mask != 255
        gt_foreground = gt_annotated & (gt_mask >= 0) & (gt_mask <= 3)

        tp = int(np.count_nonzero(pred_mask & gt_foreground))
        outside = int(np.count_nonzero(pred_mask & ~gt_annotated))
        fn = int(np.count_nonzero(~pred_mask & gt_foreground))

        self.assertEqual(tp, 100)
        self.assertEqual(outside, 50)
        self.assertEqual(fn, 100)

        # Sensitivity within annotated domain
        sensitivity = tp / (tp + fn)
        self.assertAlmostEqual(sensitivity, 0.5)

        # Annotated-domain IoU
        iou = tp / (tp + fn)
        self.assertAlmostEqual(iou, 0.5)

        # Annotated-domain Dice
        dice = (2 * tp) / (2 * tp + fn)
        self.assertAlmostEqual(dice, 200.0 / 300.0)

        # Detections in ignore region must be treated as unverified, NOT confirmed false positives
        # Because ignore region contains no confirmed negative background labels
        confirmed_negatives = np.count_nonzero(gt_annotated & ~gt_foreground)
        self.assertEqual(confirmed_negatives, 0)

    def test_real_sample_100523_verified_numbers(self) -> None:
        fits_path = Path("dataset/raw/imagedata_1-100523.fits")
        mask_path = Path("dataset/raw/mask_1-100523.fits")
        if not fits_path.exists() or not mask_path.exists():
            self.skipTest("Sample files not present.")

        r_band, val_mask, _ = load_band_fits(fits_path, band="r")
        p05 = float(np.percentile(r_band[val_mask], 0.5))
        p995 = float(np.percentile(r_band[val_mask], 99.5))
        norm_img = np.clip((r_band - p05) / max(p995 - p05, 1e-8), 0.0, 1.0)

        results = run_classical_pipeline(
            norm_img,
            valid_mask=val_mask,
            threshold_method="otsu",
            morph_shape="disk",
            morph_radius=1,
            connectivity=8,
        )
        classical_fg = results["morph_mask"]

        with fits.open(mask_path, memmap=False) as hdul:
            raw_mask = hdul[0].data
        gt_np = decode_mask(raw_mask).numpy()

        gt_annotated = gt_np != 255
        gt_foreground = gt_annotated & (gt_np >= 0) & (gt_np <= 3)

        total_pixels = gt_np.size
        annotated_pixels = int(np.count_nonzero(gt_annotated))
        ignored_pixels = int(np.count_nonzero(gt_np == 255))
        classical_predictions = int(np.count_nonzero(classical_fg))
        overlapping_annotated = int(np.count_nonzero(classical_fg & gt_foreground))
        outside_annotated = int(np.count_nonzero(classical_fg & (gt_np == 255)))
        missed_annotated = int(np.count_nonzero(~classical_fg & gt_foreground))

        # Verified ground-truth and prediction numbers
        self.assertEqual(total_pixels, 43264)
        self.assertEqual(annotated_pixels, 14091)
        self.assertEqual(ignored_pixels, 29173)
        self.assertEqual(classical_predictions, 1191)
        self.assertEqual(overlapping_annotated, 1005)
        self.assertEqual(outside_annotated, 186)
        self.assertEqual(missed_annotated, 13086)

        # Annotated sensitivity: 1005 / 14091 = 0.0713
        sensitivity = overlapping_annotated / annotated_pixels
        self.assertAlmostEqual(sensitivity, 0.071322, places=4)

        # Annotated IoU: 1005 / (1005 + 13086) = 0.0713
        iou = overlapping_annotated / (overlapping_annotated + missed_annotated)
        self.assertAlmostEqual(iou, 0.071322, places=4)

        # Annotated Dice: 2010 / (2010 + 13086) = 0.1331
        dice = (2 * overlapping_annotated) / (2 * overlapping_annotated + missed_annotated)
        self.assertAlmostEqual(dice, 0.133148, places=4)

        # Ensure NO confirmed negative background pixels exist in benchmark ground truth
        confirmed_background = np.count_nonzero(gt_annotated & ~gt_foreground)
        self.assertEqual(confirmed_background, 0)


if __name__ == "__main__":
    unittest.main()
