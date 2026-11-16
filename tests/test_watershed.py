"""Unit tests for distance transforms, marker generation, and watershed segmentation.

Tests:
1. Exact Euclidean Distance Transform (EDT) on geometric primitives.
2. EDT validity mask handling (invalid pixels treated as background).
3. Marker generation from surface peaks with distance suppression.
4. Marker-controlled watershed on overlapping disks.
5. Zero-marker handling and background masking.
6. Watershed boundary line extraction.
7. Input validation (dimensions, connectivity, tensor inputs).
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.classical_cv.watershed import (
    distance_transform_edt,
    extract_watershed_boundaries,
    generate_markers_from_peaks,
    watershed_segmentation,
)


class TestDistanceTransform(unittest.TestCase):
    def test_edt_disk(self) -> None:
        mask = np.zeros((41, 41), dtype=bool)
        y, x = np.ogrid[:41, :41]
        disk = ((x - 20) ** 2 + (y - 20) ** 2) <= 10**2
        mask[disk] = True

        dt = distance_transform_edt(mask)
        self.assertEqual(dt.shape, (41, 41))
        # Center should have distance ~ 10.0
        self.assertAlmostEqual(dt[20, 20], 10.0, delta=0.5)
        # Background pixels outside disk must have distance 0.0
        self.assertEqual(dt[0, 0], 0.0)
        self.assertEqual(dt[20, 0], 0.0)

    def test_edt_empty(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        dt = distance_transform_edt(mask)
        self.assertTrue(np.all(dt == 0.0))

    def test_edt_valid_mask(self) -> None:
        mask = np.ones((30, 30), dtype=bool)
        # Valid mask cuts out left half: cols 0..14 are invalid
        val_mask = np.ones((30, 30), dtype=bool)
        val_mask[:, :15] = False

        dt = distance_transform_edt(mask, valid_mask=val_mask)
        # Left half must have distance 0.0
        self.assertTrue(np.all(dt[:, :15] == 0.0))
        # Distance should increase into the right half
        self.assertGreater(dt[15, 25], 0.0)


class TestMarkerGeneration(unittest.TestCase):
    def test_two_distinct_peaks(self) -> None:
        surface = np.zeros((50, 50), dtype=np.float32)
        surface[15, 15] = 10.0
        surface[35, 35] = 8.0

        markers, num = generate_markers_from_peaks(surface, min_distance=10, threshold=5.0)
        self.assertEqual(num, 2)
        self.assertEqual(markers[15, 15], 1)
        self.assertEqual(markers[35, 35], 2)
        self.assertEqual(np.count_nonzero(markers), 2)

    def test_min_distance_suppression(self) -> None:
        surface = np.zeros((30, 30), dtype=np.float32)
        # Two peaks only 3 pixels apart
        surface[10, 10] = 10.0
        surface[10, 13] = 9.0

        # With min_distance=5, the lower peak at (10, 13) should be suppressed
        markers, num = generate_markers_from_peaks(surface, min_distance=5, threshold=5.0)
        self.assertEqual(num, 1)
        self.assertEqual(markers[10, 10], 1)
        self.assertEqual(markers[10, 13], 0)

    def test_no_peaks_above_threshold(self) -> None:
        surface = np.ones((20, 20), dtype=np.float32) * 2.0
        markers, num = generate_markers_from_peaks(surface, threshold=5.0)
        self.assertEqual(num, 0)
        self.assertTrue(np.all(markers == 0))


class TestWatershedSegmentation(unittest.TestCase):
    def test_two_overlapping_disks(self) -> None:
        # Create two overlapping disks (like a binary galaxy or core + companion)
        y, x = np.ogrid[:60, :60]
        disk1 = ((x - 20) ** 2 + (y - 30) ** 2) <= 12**2
        disk2 = ((x - 40) ** 2 + (y - 30) ** 2) <= 12**2
        mask = disk1 | disk2

        dt = distance_transform_edt(mask)
        # Surface for watershed: inverted distance
        surface = -dt

        markers, num_markers = generate_markers_from_peaks(
            dt, min_distance=10, threshold=5.0, mask=mask
        )
        self.assertEqual(num_markers, 2)

        basins, meta = watershed_segmentation(surface, markers, mask=mask, connectivity=8)
        self.assertEqual(meta["num_basins"], 2)
        # Background must be 0
        self.assertTrue(np.all(basins[~mask] == 0))
        # Center of disk 1 and disk 2 must have distinct positive labels
        lbl1 = basins[30, 20]
        lbl2 = basins[30, 40]
        self.assertGreater(lbl1, 0)
        self.assertGreater(lbl2, 0)
        self.assertNotEqual(lbl1, lbl2)

    def test_watershed_zero_markers(self) -> None:
        surface = np.zeros((20, 20), dtype=np.float32)
        markers = np.zeros((20, 20), dtype=np.int32)
        basins, meta = watershed_segmentation(surface, markers)
        self.assertEqual(meta["num_basins"], 0)
        self.assertTrue(np.all(basins == 0))

    def test_extract_boundaries(self) -> None:
        # 10x10 map with left half = 1, right half = 2
        labeled = np.zeros((10, 10), dtype=np.int32)
        labeled[:, :5] = 1
        labeled[:, 5:] = 2

        # Default 2-pixel width (symmetric interface)
        boundaries_2px = extract_watershed_boundaries(labeled, two_pixel_width=True)
        self.assertEqual(boundaries_2px.shape, (10, 10))
        # Boundary line between column 4 and 5 should be True
        self.assertTrue(np.all(boundaries_2px[:, 4]))
        self.assertTrue(np.all(boundaries_2px[:, 5]))
        self.assertFalse(np.any(boundaries_2px[:, 0:3]))
        self.assertFalse(np.any(boundaries_2px[:, 7:10]))

        # Strict single-pixel boundary width
        boundaries_1px = extract_watershed_boundaries(labeled, two_pixel_width=False)
        self.assertTrue(np.all(boundaries_1px[:, 4]))
        self.assertFalse(np.any(boundaries_1px[:, 5]))

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            watershed_segmentation(np.zeros((10, 10)), np.zeros((10, 10)), connectivity=6)
        with self.assertRaises(ValueError):
            watershed_segmentation(np.zeros((10, 10)), np.zeros((10, 12)))


if __name__ == "__main__":
    unittest.main()
