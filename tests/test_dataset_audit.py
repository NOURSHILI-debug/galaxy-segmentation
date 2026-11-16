import csv
import filecmp
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.data.dataset_audit import IMAGE_STATS_COLUMNS, run_audit


def _write_image(path: Path, hdus):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _make_hdu(data, index=0, header_extra=None):
    hdr = fits.Header()
    if header_extra:
        for k, v in header_extra.items():
            hdr[k] = v
    if index == 0:
        h = fits.PrimaryHDU(data=np.asarray(data), header=hdr)
    else:
        h = fits.ImageHDU(data=np.asarray(data), header=hdr, name=f"H{index}")
    return h


class TestDatasetAudit(unittest.TestCase):
    def _make_manifest(self, root: Path, rows):
        mpath = root.parent / "manifest.csv"
        with mpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "status", "image_nonfinite"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return mpath

    def test_counts_and_stats_known_array(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "raw"
            out = Path(td) / "out"
            # small 2x2 array with NaN and +Inf: finite values [1,2,5]
            a0 = np.array([[1.0, np.nan], [2.0, np.inf]], dtype=np.float32)
            # constant HDU
            a1 = np.full((2, 2), 3.0, dtype=np.float32)
            hdr0 = {"EXPTIME": 10.0, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CDELT1": 1.0, "CDELT2": 2.0, "FPA.FILTER": "g.00000"}
            _write_image(root / "imagedata_S1.fits", [_make_hdu(a0, 0, hdr0), _make_hdu(a1, 1, {"FPA.FILTER": "r.00000"})])
            manifest = self._make_manifest(root, [{"sample_id": "S1", "image_path": "imagedata_S1.fits", "status": "ok", "image_nonfinite": "2"}])
            run_audit(manifest, root, out)
            with (out / "image_stats.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            r0 = [r for r in rows if r["hdu_index"] == "0"][0]
            self.assertEqual(r0["n_pixels"], "4")
            self.assertEqual(r0["n_nan"], "1")
            self.assertEqual(r0["n_posinf"], "1")
            self.assertEqual(r0["n_neginf"], "0")
            self.assertEqual(r0["n_finite"], "2")
            self.assertAlmostEqual(float(r0["min"]), 1.0)
            self.assertAlmostEqual(float(r0["max"]), 2.0)
            self.assertAlmostEqual(float(r0["mean"]), 1.5)
            self.assertAlmostEqual(float(r0["median"]), 1.5)
            self.assertEqual(r0["EXPTIME"], "10.0")
            self.assertIn("FPA.FILTER=g.00000", r0["FILTER_KEYS"])
            r1 = [r for r in rows if r["hdu_index"] == "1"][0]
            self.assertEqual(r1["n_finite"], "4")
            self.assertAlmostEqual(float(r1["std"]), 0.0)
            self.assertAlmostEqual(float(r1["sigma_mad"]), 0.0)

    def test_all_nan_hdu(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "raw"
            out = Path(td) / "out"
            a = np.full((3, 3), np.nan, dtype=np.float32)
            _write_image(root / "imagedata_S2.fits", [_make_hdu(a, 0, {"FPA.FILTER": "i.00000"})])
            manifest = self._make_manifest(root, [{"sample_id": "S2", "image_path": "imagedata_S2.fits", "status": "ok", "image_nonfinite": "9"}])
            run_audit(manifest, root, out)
            with (out / "image_stats.csv").open(newline="", encoding="utf-8") as f:
                r = list(csv.DictReader(f))[0]
            self.assertEqual(r["n_nan"], "9")
            self.assertEqual(r["n_finite"], "0")
            self.assertEqual(r["min"], "")
            self.assertEqual(r["median"], "")
            self.assertEqual(r["sigma_mad"], "")
            with (out / "image_audit_summary.json").open(encoding="utf-8") as f:
                s = json.load(f)
            self.assertEqual(s["nonfinite_summary"]["n_all_nan_hdus"], 1)

    def test_constant_and_header_captured(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "raw"
            out = Path(td) / "out"
            a = np.zeros((2, 2), dtype=np.float32)
            hdr = {"CTYPE1": "RA", "CTYPE2": "DEC", "CDELT1": 0.1, "CDELT2": 0.2, "MYBAND": "z"}
            _write_image(root / "imagedata_S3.fits", [_make_hdu(a, 0, hdr)])
            manifest = self._make_manifest(root, [{"sample_id": "S3", "image_path": "imagedata_S3.fits", "status": "ok", "image_nonfinite": "0"}])
            run_audit(manifest, root, out)
            with (out / "image_stats.csv").open(newline="", encoding="utf-8") as f:
                r = list(csv.DictReader(f))[0]
            self.assertEqual(r["CTYPE1"], "RA")
            self.assertIn("MYBAND=z", r["FILTER_KEYS"])
            with (out / "image_audit_summary.json").open(encoding="utf-8") as f:
                s = json.load(f)
            self.assertEqual(s["nonfinite_summary"]["n_constant_hdus"], 1)
            self.assertEqual(s["nonfinite_summary"]["n_all_zero_hdus"], 1)

    def test_rerun_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "raw"
            out1 = Path(td) / "out1"
            out2 = Path(td) / "out2"
            _write_image(root / "imagedata_S4.fits", [_make_hdu(np.array([[1, 2], [3, 4]], dtype=np.float32), 0, {"FPA.FILTER": "y.00000"})])
            manifest = self._make_manifest(root, [{"sample_id": "S4", "image_path": "imagedata_S4.fits", "status": "ok", "image_nonfinite": "0"}])
            run_audit(manifest, root, out1)
            run_audit(manifest, root, out2)
            self.assertTrue(filecmp.cmp(out1 / "image_stats.csv", out2 / "image_stats.csv", shallow=False))
            self.assertTrue(filecmp.cmp(out1 / "image_audit_summary.json", out2 / "image_audit_summary.json", shallow=False))

    def test_limit_works(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "raw"
            out = Path(td) / "out"
            rows_m = []
            for sid in ["B", "A", "C"]:
                _write_image(root / f"imagedata_{sid}.fits", [_make_hdu(np.ones((2, 2), dtype=np.float32), 0)])
                rows_m.append({"sample_id": sid, "image_path": f"imagedata_{sid}.fits", "status": "ok", "image_nonfinite": "0"})
            manifest = self._make_manifest(root, rows_m)
            run_audit(manifest, root, out, limit=2)
            with (out / "image_stats.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # sorted sample_id order: A,B then limit 2
            self.assertEqual(sorted(set(r["sample_id"] for r in rows)), ["A", "B"])
            self.assertEqual(reader_fieldnames_ok(out / "image_stats.csv"), True)


def reader_fieldnames_ok(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _ = next(reader, None)
        return reader.fieldnames == IMAGE_STATS_COLUMNS


if __name__ == "__main__":
    unittest.main()
