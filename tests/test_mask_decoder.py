import unittest

import numpy as np
import torch

from src.data.mask_decoder import (
    IGNORE_INDEX,
    NUM_CLASSES,
    RAW_TO_TRAINING_ID,
    MaskDecodeError,
    decode_mask,
)


class TestMaskDecoder(unittest.TestCase):
    def test_each_raw_value_decodes_to_approved_id(self):
        expected = {0: IGNORE_INDEX, 1: 0, 2: 1, 3: 2, 4: 3}
        for raw_value, expected_training_id in expected.items():
            with self.subTest(raw_value=raw_value):
                decoded = decode_mask(np.array([[raw_value]], dtype=np.int16))
                self.assertIsInstance(decoded, torch.Tensor)
                self.assertEqual(decoded.dtype, torch.int64)
                self.assertEqual(decoded.shape, (1, 1))
                self.assertEqual(decoded.tolist(), [[expected_training_id]])

    def test_mapping_has_four_contiguous_trainable_ids_and_ignore_index(self):
        self.assertEqual(RAW_TO_TRAINING_ID, {0: 255, 1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(NUM_CLASSES, 4)
        self.assertEqual(IGNORE_INDEX, 255)

    def test_decode_preserves_shape_and_does_not_modify_input(self):
        raw = np.array([[0, 1, 2], [3, 4, 1]], dtype=np.float64)
        original = raw.copy()

        decoded = decode_mask(raw)

        self.assertIsInstance(decoded, torch.Tensor)
        self.assertEqual(decoded.shape, (2, 3))
        self.assertEqual(decoded.dtype, torch.int64)
        self.assertEqual(decoded.tolist(), [[255, 0, 1], [2, 3, 0]])
        np.testing.assert_array_equal(raw, original)

    def test_repeated_decoding_is_deterministic(self):
        raw = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.float64)

        first = decode_mask(raw)
        second = decode_mask(raw)

        self.assertTrue(torch.equal(first, second))

    def test_integer_like_float_values_are_accepted_without_rounding(self):
        raw = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64)

        decoded = decode_mask(raw)

        self.assertEqual(decoded.tolist(), [[255, 0, 1, 2, 3]])
        self.assertEqual(decoded.dtype, torch.int64)

    def test_unknown_raw_values_raise_clear_error(self):
        for unknown_value in (-1, 5, 99):
            with self.subTest(unknown_value=unknown_value):
                with self.assertRaisesRegex(MaskDecodeError, "unexpected raw value"):
                    decode_mask(np.array([[unknown_value]], dtype=np.int16))

    def test_non_integer_like_values_raise_clear_error(self):
        for non_integer_value in (1.5, 2.25, -0.5):
            with self.subTest(non_integer_value=non_integer_value):
                with self.assertRaisesRegex(MaskDecodeError, "non-integer-like"):
                    decode_mask(np.array([[non_integer_value]], dtype=np.float32))

    def test_non_finite_values_raise_clear_error(self):
        for non_finite_value in (np.nan, np.inf, -np.inf):
            with self.subTest(non_finite_value=non_finite_value):
                with self.assertRaisesRegex(MaskDecodeError, "non-finite"):
                    decode_mask(np.array([[non_finite_value]], dtype=np.float64))

    def test_non_numeric_or_boolean_mask_dtype_is_rejected(self):
        with self.assertRaisesRegex(MaskDecodeError, "integer or floating-point"):
            decode_mask(np.array([[True, False]], dtype=bool))

    def test_empty_mask_preserves_empty_shape_and_output_dtype(self):
        decoded = decode_mask(np.empty((0, 3), dtype=np.uint8))

        self.assertEqual(decoded.shape, (0, 3))
        self.assertEqual(decoded.dtype, torch.int64)

    def test_non_2d_mask_is_rejected(self):
        with self.assertRaisesRegex(MaskDecodeError, "two-dimensional"):
            decode_mask(np.array([0, 1, 2], dtype=np.int16))


if __name__ == "__main__":
    unittest.main()
