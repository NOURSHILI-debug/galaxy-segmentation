"""Unit tests for connected-component labeling and region analysis.

Tests:
1. Empty and single-component masks.
2. 4-connectivity vs 8-connectivity behavior.
3. Label equivalence resolving on complex/U-shaped structures.
4. Component statistics (area, bounding box, centroid).
5. Minimum area filtering with contiguous relabeling.
6. Validity mask masking.
7. Input validation (connectivity, dimensions, tensor compatibility).
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.classical_cv.connected_components import label_connected_components


class TestConnectedComponents(unittest.TestCase):
    def test_empty_mask(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        lbl_map, num_feats, stats = label_connected_components(mask)
        self.assertEqual(num_feats, 0)
        self.assertEqual(len(stats), 0)
        self.assertTrue(np.all(lbl_map == 0))

    def test_single_component_stats(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:15, 20:26] = True  # 5 rows x 6 cols = 30 pixels

        lbl_map, num_feats, stats = label_connected_components(mask)
        self.assertEqual(num_feats, 1)
        self.assertEqual(len(stats), 1)

        st = stats[0]
        self.assertEqual(st["label"], 1)
        self.assertEqual(st["area"], 30)
        # Bounding box: min_r=10, min_c=20, max_r=14, max_c=25
        self.assertEqual(st["bbox"], (10, 20, 14, 25))
        # Centroid: row=(10+14)/2 = 12.0, col=(20+25)/2 = 22.5
        self.assertAlmostEqual(st["centroid"][0], 12.0)
        self.assertAlmostEqual(st["centroid"][1], 22.5)

    def test_connectivity_4_vs_8(self) -> None:
        # Two diagonal pixels
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 3] = True
        mask[4, 4] = True

        # In 4-connectivity, diagonals are distinct components
        lbl_4, num_4, _ = label_connected_components(mask, connectivity=4)
        self.assertEqual(num_4, 2)
        self.assertNotEqual(lbl_4[3, 3], lbl_4[4, 4])

        # In 8-connectivity, diagonals are part of the same component
        lbl_8, num_8, _ = label_connected_components(mask, connectivity=8)
        self.assertEqual(num_8, 1)
        self.assertEqual(lbl_8[3, 3], lbl_8[4, 4])

    def test_u_shape_equivalence_merging(self) -> None:
        # A U-shaped structure that starts as two separate branches at the top
        # and joins at the bottom:
        # 1 0 1
        # 1 0 1
        # 1 1 1
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:5, 2] = True  # left prong
        mask[1:5, 6] = True  # right prong
        mask[5, 2:7] = True  # bottom bar

        lbl_map, num_feats, stats = label_connected_components(mask, connectivity=4)
        self.assertEqual(num_feats, 1)
        # All foreground pixels must share label 1
        foreground_labels = set(lbl_map[mask])
        self.assertEqual(foreground_labels, {1})

    def test_area_filtering(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        # Small component (area 2)
        mask[2, 2] = True
        mask[2, 3] = True
        # Large component (area 16)
        mask[10:14, 10:14] = True

        # With min_area=5, small component should be dropped
        lbl_map, num_feats, stats = label_connected_components(mask, min_area=5)
        self.assertEqual(num_feats, 1)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["area"], 16)
        self.assertEqual(lbl_map[2, 2], 0)
        self.assertEqual(lbl_map[10, 10], 1)

    def test_valid_mask_enforcement(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[2:4, 2:4] = True

        lbl_map, num_feats, _ = label_connected_components(mask, valid_mask=valid_mask)
        self.assertEqual(num_feats, 1)
        self.assertEqual(np.count_nonzero(lbl_map), 4)
        self.assertEqual(lbl_map[0, 0], 0)
        self.assertEqual(lbl_map[2, 2], 1)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            label_connected_components(np.zeros((5, 5)), connectivity=6)
        with self.assertRaises(ValueError):
            label_connected_components(np.zeros((5, 5, 2)))

    def test_torch_tensor_input(self) -> None:
        t_mask = torch.zeros(10, 10, dtype=torch.bool)
        t_mask[3:6, 3:6] = True
        lbl_map, num_feats, stats = label_connected_components(t_mask)
        self.assertIsInstance(lbl_map, np.ndarray)
        self.assertEqual(num_feats, 1)


if __name__ == "__main__":
    unittest.main()
