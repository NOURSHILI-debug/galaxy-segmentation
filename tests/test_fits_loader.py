"""Focused tests for the real FITS ingestion layer (D1).

Deterministic edge cases use temporary synthetic FITS files.  A small smoke
test class runs only when the real dataset is present and never scans the whole
dataset.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from src.data.fits_loader import (
    BAND_ORDER_IS_HEADER_VERIFIED,
    IMAGE_BAND_COUNT,
    IMAGE_BAND_NAMES,
    FitsLoadError,
    load_image_fits,
    load_mask_fits,
)
from src.data.mask_decoder import IGNORE_INDEX, MaskDecodeError

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"


def _write_image(path: Path, bands) -> None:
    """Write a multi-HDU image FITS file."""
    arrays = [np.asarray(band) for band in bands]
    hdus = [fits.PrimaryHDU(arrays[0])]
    hdus.extend(fits.ImageHDU(array) for array in arrays[1:])
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_mask(path: Path, mask) -> None:
    fits.PrimaryHDU(np.asarray(mask, dtype=np.float64)).writeto(path, overwrite=True)


def _five_bands(h: int = 12, w: int = 12) -> list[np.ndarray]:
    """Five distinct finite float32 bands with a stable value pattern."""
    bands = []
    for index in range(IMAGE_BAND_COUNT):
        band = (np.arange(h * w, dtype=np.float32).reshape(h, w) + 100.0 * index)
        bands.append(band)
    return bands


class _TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="fits_loader_test_")
        self.tmp = Path(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestImageShapeAndDtype(_TempDirTest):
    """1. Correct image tensor shape and dtype."""

    def test_all_bands_shape_and_dtype(self):
        path = self.tmp / "imagedata_syn-001.fits"
        _write_image(path, _five_bands(12, 10))

        result = load_image_fits(path)

        self.assertEqual(result["image"].shape, (5, 12, 10))
        self.assertEqual(result["image"].dtype, torch.float32)
        self.assertEqual(result["valid_mask"].shape, (12, 10))
        self.assertEqual(result["valid_mask"].dtype, torch.bool)
        self.assertTrue(bool(result["valid_mask"].all()))
        self.assertEqual(result["shape"], (12, 10))
        self.assertEqual(result["band_indices"], (0, 1, 2, 3, 4))
        self.assertEqual(result["band_names"], IMAGE_BAND_NAMES)
        self.assertEqual(result["sample_id"], "syn-001")


class TestNativeEndianConversion(_TempDirTest):
    """2. Correct native-endian float32 conversion."""

    def test_big_endian_fits_becomes_native_float32(self):
        path = self.tmp / "imagedata_syn-002.fits"
        _write_image(path, _five_bands())

        with fits.open(path, mode="readonly") as hdul:
            self.assertEqual(hdul[0].data.dtype.str[0], ">")
            self.assertEqual(hdul[0].data.dtype, np.dtype(">f4"))

        result = load_image_fits(path)

        self.assertEqual(result["image"].numpy().dtype, np.dtype(np.float32))
        self.assertNotEqual(result["image"].numpy().dtype, np.dtype(">f4"))
        self.assertEqual(result["image"].numpy().dtype.str[0], "<")


class TestRawFluxPreservation(_TempDirTest):
    """3. Preservation of finite raw flux values (no normalization)."""

    def test_signed_and_large_values_preserved_exactly(self):
        h = w = 4
        bands = []
        for index in range(IMAGE_BAND_COUNT):
            band = np.full((h, w), -222.637, dtype=np.float32)
            band[0, 0] = 6547.291
            band[0, 1] = -331.0023
            band[1, 0] = 1.0e6
            band[1, 1] = 0.0
            band[2, 2] = 34872.25 + index
            bands.append(band)
        path = self.tmp / "imagedata_syn-003.fits"
        _write_image(path, bands)

        result = load_image_fits(path)

        expected = np.stack([np.asarray(band, dtype=np.float32) for band in bands])
        np.testing.assert_array_equal(result["image"].numpy(), expected)
        # Not normalized into [0, 1].
        self.assertGreater(float(result["image"].max()), 1.0)
        self.assertLess(float(result["image"].min()), 0.0)


class TestValidityMask(_TempDirTest):
    """4. Validity mask when one selected band contains NaNs."""

    def test_nan_in_one_band_invalidates_only_that_pixel(self):
        bands = _five_bands()
        bands[0][2, 3] = np.nan
        path = self.tmp / "imagedata_syn-004.fits"
        _write_image(path, bands)

        result = load_image_fits(path)

        self.assertFalse(bool(result["valid_mask"][2, 3]))
        self.assertEqual(int((~result["valid_mask"]).sum()), 1)
        self.assertTrue(bool(result["valid_mask"][0, 0]))
        # The NaN itself is preserved, not replaced.
        self.assertTrue(torch.isnan(result["image"][0, 2, 3]))

    def test_unaffected_band_selection_is_fully_valid(self):
        bands = _five_bands()
        bands[0][2, 3] = np.nan
        path = self.tmp / "imagedata_syn-004.fits"
        _write_image(path, bands)

        result = load_image_fits(path, band_indices=(1,))

        self.assertTrue(bool(result["valid_mask"].all()))
        self.assertEqual(result["band_indices"], (1,))
        self.assertEqual(result["band_names"], ("r",))


class TestNonFiniteCounts(_TempDirTest):
    """5. Correct per-band non-finite counts."""

    def test_counts_are_per_selected_band(self):
        bands = _five_bands()
        bands[0][0, 0] = np.nan
        bands[0][1, 1] = np.inf
        bands[2][2, 2] = np.nan
        bands[2][3, 3] = -np.inf
        path = self.tmp / "imagedata_syn-005.fits"
        _write_image(path, bands)

        result = load_image_fits(path)
        self.assertEqual(result["non_finite_counts"], {0: 2, 1: 0, 2: 2, 3: 0, 4: 0})

        subset = load_image_fits(path, band_indices=(0, 4))
        self.assertEqual(subset["non_finite_counts"], {0: 2, 4: 0})
        self.assertEqual(int((~subset["valid_mask"]).sum()), 2)


class TestBandSelection(_TempDirTest):
    """6. Loading selected bands and all five bands."""

    def test_explicit_selection_preserves_order(self):
        bands = _five_bands()
        path = self.tmp / "imagedata_syn-006.fits"
        _write_image(path, bands)

        result = load_image_fits(path, band_indices=(2, 0))

        self.assertEqual(result["image"].shape, (2, 12, 12))
        self.assertEqual(result["band_indices"], (2, 0))
        self.assertEqual(result["band_names"], ("i", "g"))
        np.testing.assert_array_equal(result["image"][0].numpy(), bands[2])
        np.testing.assert_array_equal(result["image"][1].numpy(), bands[0])

    def test_none_loads_all_five(self):
        path = self.tmp / "imagedata_syn-006.fits"
        _write_image(path, _five_bands())

        result = load_image_fits(path, band_indices=None)

        self.assertEqual(result["image"].shape, (5, 12, 12))
        self.assertEqual(result["band_indices"], (0, 1, 2, 3, 4))

    def test_explicit_sample_id_overrides_filename(self):
        path = self.tmp / "imagedata_syn-006.fits"
        _write_image(path, _five_bands())

        result = load_image_fits(path, sample_id="override-9")

        self.assertEqual(result["sample_id"], "override-9")


class TestMaskDecoding(_TempDirTest):
    """7. Correct decoded mask shape, dtype, and allowed values."""

    def test_values_decoded_with_frozen_mapping(self):
        raw = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 0.0],
            ]
        )
        path = self.tmp / "mask_syn-007.fits"
        _write_mask(path, raw)

        result = load_mask_fits(path)

        expected = torch.tensor(
            [
                [IGNORE_INDEX, 0, 1],
                [2, 3, IGNORE_INDEX],
            ],
            dtype=torch.int64,
        )
        self.assertEqual(result["mask"].shape, (2, 3))
        self.assertEqual(result["mask"].dtype, torch.int64)
        self.assertTrue(torch.equal(result["mask"], expected))
        self.assertEqual(set(result["mask"].tolist()[0]) | set(result["mask"].tolist()[1]),
                         {IGNORE_INDEX, 0, 1, 2, 3})
        self.assertEqual(result["raw_unique"], [0, 1, 2, 3, 4])
        self.assertEqual(result["shape"], (2, 3))
        self.assertEqual(result["sample_id"], "syn-007")


class TestMissingFiles(_TempDirTest):
    """8. Missing-file errors."""

    def test_missing_image_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_image_fits(self.tmp / "imagedata_absent.fits")

    def test_missing_mask_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_mask_fits(self.tmp / "mask_absent.fits")


class TestInvalidHduStructure(_TempDirTest):
    """9. Invalid HDU structure."""

    def test_too_few_image_hdus(self):
        path = self.tmp / "imagedata_syn-009.fits"
        _write_image(path, _five_bands()[:3])

        with self.assertRaises(FitsLoadError):
            load_image_fits(path)

    def test_image_hdu_without_data(self):
        path = self.tmp / "imagedata_syn-009b.fits"
        bands = _five_bands()
        hdus = [fits.PrimaryHDU(bands[0])]
        hdus.extend(fits.ImageHDU(band) for band in bands[1:4])
        hdus.append(fits.ImageHDU())  # HDU 4 has no data.
        fits.HDUList(hdus).writeto(path, overwrite=True)

        with self.assertRaises(FitsLoadError):
            load_image_fits(path)

    def test_mismatched_band_shapes(self):
        path = self.tmp / "imagedata_syn-009c.fits"
        bands = _five_bands()
        bands[3] = np.zeros((6, 12), dtype=np.float32)
        _write_image(path, bands)

        with self.assertRaises(FitsLoadError):
            load_image_fits(path)

    def test_mask_without_data(self):
        path = self.tmp / "mask_syn-009.fits"
        fits.PrimaryHDU().writeto(path, overwrite=True)

        with self.assertRaises(FitsLoadError):
            load_mask_fits(path)

    def test_mask_with_extra_data_hdu(self):
        path = self.tmp / "mask_syn-009b.fits"
        array = np.zeros((4, 4), dtype=np.float64)
        fits.HDUList(
            [fits.PrimaryHDU(array), fits.ImageHDU(array)]
        ).writeto(path, overwrite=True)

        with self.assertRaises(FitsLoadError):
            load_mask_fits(path)

    def test_path_is_directory(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.tmp)


class TestInvalidDimensionsAndValues(_TempDirTest):
    """10. Invalid dimensionality and malformed mask values."""

    def test_three_dimensional_image_hdu(self):
        path = self.tmp / "imagedata_syn-010.fits"
        bands = _five_bands()
        bands[0] = np.zeros((2, 12, 12), dtype=np.float32)
        _write_image(path, bands)

        with self.assertRaises(FitsLoadError):
            load_image_fits(path)

    def test_non_floating_image_hdu(self):
        path = self.tmp / "imagedata_syn-010b.fits"
        bands = [band.astype(np.int16) for band in _five_bands()]

        with self.assertRaises(FitsLoadError):
            _write_image(path, bands)
            load_image_fits(path)

    def test_three_dimensional_mask(self):
        path = self.tmp / "mask_syn-010.fits"
        _write_mask(path, np.zeros((2, 4, 4)))

        with self.assertRaises(FitsLoadError):
            load_mask_fits(path)

    def test_non_integer_mask_values(self):
        path = self.tmp / "mask_syn-010b.fits"
        _write_mask(path, np.array([[0.0, 1.0], [2.5, 3.0]]))

        with self.assertRaises(MaskDecodeError):
            load_mask_fits(path)

    def test_out_of_range_mask_values(self):
        path = self.tmp / "mask_syn-010c.fits"
        _write_mask(path, np.array([[0.0, 7.0], [1.0, 2.0]]))

        with self.assertRaises(MaskDecodeError):
            load_mask_fits(path)


class TestInvalidBandSelection(_TempDirTest):
    """11. Empty or invalid band selection."""

    def setUp(self) -> None:
        super().setUp()
        self.path = self.tmp / "imagedata_syn-011.fits"
        _write_image(self.path, _five_bands())

    def test_empty_selection(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=())

    def test_out_of_range_index(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=(5,))

    def test_negative_index(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=(-1,))

    def test_duplicate_index(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=(1, 1))

    def test_boolean_index(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=(True,))

    def test_non_integer_index(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=(1.0,))

    def test_bare_integer_not_iterable(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices=1)

    def test_string_selection(self):
        with self.assertRaises(FitsLoadError):
            load_image_fits(self.path, band_indices="01234")


class TestContractConstants(unittest.TestCase):
    """The documented band-order assumption is explicit, not header-verified."""

    def test_constants(self):
        self.assertEqual(IMAGE_BAND_COUNT, 5)
        self.assertEqual(IMAGE_BAND_NAMES, ("g", "r", "i", "z", "y"))
        self.assertFalse(BAND_ORDER_IS_HEADER_VERIFIED)


_NORMAL_IMAGE = RAW / "imagedata_1-100523.fits"
_NORMAL_MASK = RAW / "mask_1-100523.fits"
_NAN_IMAGE = RAW / "imagedata_1-103458.fits"
_NAN_MASK = RAW / "mask_1-103458.fits"


@unittest.skipUnless(
    _NORMAL_IMAGE.exists() and _NORMAL_MASK.exists(),
    "real dataset not present",
)
class TestRealFitsSmoke(unittest.TestCase):
    """Small real-data smoke test; no full-dataset scan."""

    def test_normal_pair(self):
        image = load_image_fits(_NORMAL_IMAGE)
        mask = load_mask_fits(_NORMAL_MASK)

        self.assertEqual(image["image"].shape, (5, 208, 208))
        self.assertEqual(image["image"].dtype, torch.float32)
        self.assertEqual(image["band_names"], ("g", "r", "i", "z", "y"))
        self.assertEqual(image["non_finite_counts"], {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
        self.assertTrue(bool(image["valid_mask"].all()))
        self.assertEqual(mask["mask"].shape, (208, 208))
        self.assertEqual(mask["mask"].dtype, torch.int64)
        observed = set(mask["mask"].tolist()[0]) | set(mask["mask"].tolist()[1])
        self.assertTrue(observed.issubset({0, 1, 2, 3, IGNORE_INDEX}))
        self.assertTrue(mask["raw_unique"])
        self.assertTrue(set(mask["raw_unique"]).issubset({0, 1, 2, 3, 4}))

    @unittest.skipUnless(_NAN_IMAGE.exists(), "NaN sample not present")
    def test_known_nan_sample(self):
        image = load_image_fits(_NAN_IMAGE)

        self.assertEqual(image["image"].shape, (5, 208, 208))
        self.assertGreater(image["non_finite_counts"][0], 0)
        self.assertGreater(int((~image["valid_mask"]).sum()), 0)
        self.assertTrue(bool(torch.isnan(image["image"][0]).any()))

    def test_mask_matches_image_shape(self):
        image = load_image_fits(_NORMAL_IMAGE, band_indices=(1, 2, 3))
        mask = load_mask_fits(_NORMAL_MASK)
        self.assertEqual(tuple(image["shape"]), tuple(mask["shape"]))


if __name__ == "__main__":
    unittest.main()
