"""Unit tests for classical mathematical morphology operators.

Tests:
1. Structuring element generation (disk, square, cross).
2. Binary dilation (expansion, multiple iterations).
3. Binary erosion (shrinking, elimination of small structures).
4. Morphological opening (noise spike removal, shape preservation).
5. Morphological closing (hole filling, gap bridging).
6. Validity mask enforcement (invalid pixels never become foreground).
7. Input validation (dimensions, kernel shapes, tensor compatibility).
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.classical_cv.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    get_structuring_element,
)


class TestStructuringElements(unittest.TestCase):
    def test_disk_element(self) -> None:
        se = get_structuring_element("disk", radius=2)
        self.assertEqual(se.shape, (5, 5))
        self.assertEqual(se.dtype, bool)
        # Center should be True
        self.assertTrue(se[2, 2])
        # Corners (0, 0) should be False for radius 2 disk on 5x5
        self.assertFalse(se[0, 0])
        self.assertFalse(se[0, 4])
        self.assertFalse(se[4, 0])
        self.assertFalse(se[4, 4])

    def test_square_element(self) -> None:
        se = get_structuring_element("square", radius=1)
        self.assertEqual(se.shape, (3, 3))
        self.assertTrue(np.all(se))

    def test_cross_element(self) -> None:
        se = get_structuring_element("cross", radius=1)
        self.assertEqual(se.shape, (3, 3))
        expected = np.array(
            [[False, True, False], [True, True, True], [False, True, False]], dtype=bool
        )
        np.testing.assert_array_equal(se, expected)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            get_structuring_element("triangle", radius=2)
        with self.assertRaises(ValueError):
            get_structuring_element("disk", radius=0)


class TestBinaryDilationErosion(unittest.TestCase):
    def test_dilation_single_point(self) -> None:
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 5] = True
        se = get_structuring_element("disk", radius=1)  # 3x3 cross/disk

        dilated = binary_dilation(mask, structuring_element=se, iterations=1)
        self.assertEqual(dilated.shape, (11, 11))
        # Footprint should appear centered at [5, 5]
        self.assertTrue(dilated[5, 5])
        self.assertTrue(dilated[4, 5])
        self.assertTrue(dilated[6, 5])
        self.assertTrue(dilated[5, 4])
        self.assertTrue(dilated[5, 6])
        # Outside footprint should remain False
        self.assertFalse(dilated[0, 0])

    def test_dilation_multiple_iterations(self) -> None:
        mask = np.zeros((15, 15), dtype=bool)
        mask[7, 7] = True
        se = get_structuring_element("square", radius=1)  # 3x3

        dil_1 = binary_dilation(mask, se, iterations=1)
        dil_2 = binary_dilation(mask, se, iterations=2)
        self.assertEqual(np.count_nonzero(dil_1), 9)   # 3x3
        self.assertEqual(np.count_nonzero(dil_2), 25)  # 5x5

    def test_erosion_shrinking(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True  # 10x10 square
        se = get_structuring_element("square", radius=1)

        eroded = binary_erosion(mask, se, iterations=1)
        self.assertEqual(np.count_nonzero(eroded), 8 * 8)  # shrunk by 1 on each side
        # Shrunk box boundaries
        self.assertFalse(eroded[5, 5])
        self.assertTrue(eroded[6, 6])

    def test_erosion_eliminates_isolated_points(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[2, 2] = True  # 1-pixel noise
        mask[10:18, 10:18] = True  # solid object
        se = get_structuring_element("disk", radius=1)

        eroded = binary_erosion(mask, se)
        self.assertFalse(eroded[2, 2])
        self.assertTrue(eroded[14, 14])


class TestMorphologicalOpeningClosing(unittest.TestCase):
    def test_opening_removes_noise(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        # Large central disk
        y, x = np.ogrid[:30, :30]
        disk = ((x - 15) ** 2 + (y - 15) ** 2) <= 6**2
        mask[disk] = True
        # Isolated single-pixel noise spikes
        mask[3, 3] = True
        mask[27, 27] = True

        se = get_structuring_element("disk", radius=1)
        opened = binary_opening(mask, se)

        # Noise spikes should be gone
        self.assertFalse(opened[3, 3])
        self.assertFalse(opened[27, 27])
        # Central disk should be preserved
        self.assertTrue(opened[15, 15])
        self.assertGreater(np.count_nonzero(opened), 80)

    def test_closing_fills_hole(self) -> None:
        mask = np.zeros((25, 25), dtype=bool)
        mask[5:20, 5:20] = True  # solid 15x15 square
        mask[12, 12] = False     # 1-pixel hole in the center

        se = get_structuring_element("square", radius=1)
        closed = binary_closing(mask, se)

        # Hole should be filled
        self.assertTrue(closed[12, 12])
        self.assertTrue(np.all(closed[5:20, 5:20]))


class TestValidityMaskAndValidation(unittest.TestCase):
    def test_dilation_respects_valid_mask(self) -> None:
        mask = np.zeros((15, 15), dtype=bool)
        mask[5, 5] = True
        # Mark row 5, col 6 as INVALID
        valid_mask = np.ones((15, 15), dtype=bool)
        valid_mask[5, 6] = False

        se = get_structuring_element("square", radius=1)
        dilated = binary_dilation(mask, se, valid_mask=valid_mask)

        # [5, 6] must NEVER become True because valid_mask is False
        self.assertFalse(dilated[5, 6])
        self.assertTrue(dilated[5, 4])  # other neighbor should be True

    def test_torch_tensor_inputs(self) -> None:
        t_mask = torch.zeros(10, 10, dtype=torch.bool)
        t_mask[4:7, 4:7] = True
        out = binary_dilation(t_mask)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, bool)

    def test_input_dimension_errors(self) -> None:
        with self.assertRaises(ValueError):
            binary_dilation(np.zeros((5, 5, 2)))  # 3D
        with self.assertRaises(ValueError):
            # Even dimension kernel
            binary_dilation(np.zeros((10, 10)), structuring_element=np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
