import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.data.dataset_index import (
    MANIFEST_COLUMNS,
    STATUS_DUPLICATE,
    STATUS_MISSING_IMAGE,
    STATUS_MISSING_MASK,
    STATUS_OK,
    STATUS_SHAPE_MISMATCH,
    STATUS_UNREADABLE,
    build_index,
    main as dataset_index_main,
)


def _write_fits(path: Path, hdus):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul = fits.HDUList([fits.PrimaryHDU()] if False else [])
    # build explicitly
    hdul = fits.HDUList(hdus)
    hdul.writeto(path, overwrite=True)


def _image_hdus(shape=(8, 8), n=5, dtype=np.float32, with_nan=False):
    hdus = []
    for i in range(n):
        data = np.ones(shape, dtype=dtype) * (i + 1)
        if with_nan and i == 0:
            data = data.copy()
            data[0, 0] = np.nan
        if i == 0:
            hdus.append(fits.PrimaryHDU(data=data))
        else:
            hdus.append(fits.ImageHDU(data=data, name=f"B{i}"))
    return hdus


def _mask_hdu(shape=(8, 8), values=None, dtype=np.float64):
    if values is None:
        values = np.zeros(shape, dtype=dtype)
        values[0, 0] = 1
        values[0, 1] = 2
        values[1, 0] = 3
        values[1, 1] = 4
    else:
        values = np.asarray(values, dtype=dtype)
    return [fits.PrimaryHDU(data=values)]


class TestDatasetIndex(unittest.TestCase):
    def test_normal_pair_ok(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "imagedata_001.fits", _image_hdus())
            _write_fits(root / "mask_001.fits", _mask_hdu())
            rows, audit = build_index(root, deep=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "001")
            self.assertEqual(rows[0]["status"], STATUS_OK)
            self.assertEqual(audit["n_unique_sample_ids"], 1)
            self.assertEqual(audit["status_counts"].get(STATUS_OK), 1)

    def test_image_without_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "imagedata_002.fits", _image_hdus())
            rows, audit = build_index(root)
            self.assertEqual(rows[0]["status"], STATUS_MISSING_MASK)
            self.assertEqual(audit["status_counts"].get(STATUS_MISSING_MASK), 1)

    def test_mask_without_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "mask_003.fits", _mask_hdu())
            rows, audit = build_index(root)
            self.assertEqual(rows[0]["status"], STATUS_MISSING_IMAGE)
            self.assertEqual(audit["status_counts"].get(STATUS_MISSING_IMAGE), 1)

    def test_duplicate_sample_id_no_row_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "a" / "imagedata_004.fits", _image_hdus())
            _write_fits(root / "b" / "imagedata_004.fits", _image_hdus())
            _write_fits(root / "a" / "mask_004.fits", _mask_hdu())
            _write_fits(root / "b" / "mask_004.fits", _mask_hdu())
            rows, audit = build_index(root)
            self.assertEqual(audit["n_unique_sample_ids"], 1)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["status"] == STATUS_DUPLICATE for r in rows))
            # paths relative to root
            for r in rows:
                self.assertTrue(r["image_path"])
                self.assertTrue(r["mask_path"])
                self.assertFalse(Path(r["image_path"]).is_absolute())

    def test_garbage_fits_unreadable_no_crash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "imagedata_005.fits"
            bad.write_bytes(b"THIS IS NOT A FITS FILE")
            _write_fits(root / "mask_005.fits", _mask_hdu())
            rows, audit = build_index(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], STATUS_UNREADABLE)
            self.assertTrue(rows[0]["image_error"])
            self.assertGreaterEqual(audit["n_unreadable_files"], 1)

    def test_shape_mismatch_image_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "imagedata_006.fits", _image_hdus(shape=(8, 8)))
            _write_fits(root / "mask_006.fits", _mask_hdu(shape=(6, 6)))
            rows, audit = build_index(root)
            self.assertEqual(rows[0]["status"], STATUS_SHAPE_MISMATCH)

    def test_shape_mismatch_between_image_hdus(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hdus = _image_hdus(shape=(8, 8), n=5)
            hdus[2] = fits.ImageHDU(data=np.ones((7, 7), dtype=np.float32), name="B2")
            _write_fits(root / "imagedata_007.fits", hdus)
            _write_fits(root / "mask_007.fits", _mask_hdu(shape=(8, 8)))
            rows, audit = build_index(root)
            self.assertEqual(rows[0]["status"], STATUS_SHAPE_MISMATCH)

    def test_unknown_prefix_unrecognized_and_non_fits_counted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "imagedata_008.fits", _image_hdus())
            _write_fits(root / "mask_008.fits", _mask_hdu())
            _write_fits(root / "weird_999.fits", _mask_hdu())  # unknown prefix
            (root / "notes.txt").write_text("hello", encoding="utf-8")
            rows, audit = build_index(root)
            sample_ids = [r["sample_id"] for r in rows]
            self.assertNotIn("999", sample_ids)
            self.assertIn("weird_999.fits", audit["unrecognized_fits_files"])
            self.assertEqual(audit["n_unrecognized_fits_files"], 1)
            self.assertEqual(audit["n_non_fits_files"], 1)

    def test_deep_records_nan_and_mask_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "imagedata_009.fits", _image_hdus(with_nan=True))
            _write_fits(root / "mask_009.fits", _mask_hdu())
            rows, audit = build_index(root, deep=True)
            self.assertEqual(rows[0]["status"], STATUS_OK)
            self.assertTrue(int(rows[0]["image_nonfinite"]) >= 1)
            self.assertEqual(int(rows[0]["mask_nonfinite"]), 0)
            vals = json.loads(rows[0]["mask_raw_values"])
            self.assertEqual(sorted(vals), [0, 1, 2, 3, 4])
            self.assertEqual(sorted(audit["union_mask_raw_values"]), [0, 1, 2, 3, 4])

    def test_rows_equal_unique_ids_and_paths_relative(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "sub" / "imagedata_A.fits", _image_hdus())
            _write_fits(root / "sub" / "mask_A.fits", _mask_hdu())
            _write_fits(root / "mask_B.fits", _mask_hdu())  # missing image
            rows, audit = build_index(root)
            self.assertEqual(len(rows), audit["n_unique_sample_ids"])
            for r in rows:
                for k in ("image_path", "mask_path"):
                    if r[k]:
                        self.assertNotIn(":", r[k].split("/")[0] if "/" in r[k] else "")
                        self.assertFalse(r[k].startswith("/"))

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "raw"
            out_m = Path(td) / "out" / "manifest.csv"
            out_a = Path(td) / "out" / "audit.json"
            _write_fits(root / "imagedata_010.fits", _image_hdus())
            _write_fits(root / "mask_010.fits", _mask_hdu())
            rc = dataset_index_main(["--root", str(root), "--manifest", str(out_m), "--audit", str(out_a), "--deep"])
            self.assertEqual(rc, 0)
            self.assertTrue(out_m.exists())
            self.assertTrue(out_a.exists())
            with out_m.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, MANIFEST_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
