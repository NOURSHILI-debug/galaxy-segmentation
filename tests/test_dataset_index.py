"""Focused tests for deterministic sample-level indexing and manifests (D2)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.data.dataset_index import (
    MANIFEST_COLUMNS,
    DatasetIndexError,
    build_manifest,
    read_manifest,
    scan_dataset,
    validate_manifest,
    write_manifest,
)


def _write_image(
    path: Path,
    *,
    height: int = 8,
    width: int = 8,
    hdu_count: int = 5,
    value: float = 1.5,
) -> None:
    data = np.full((height, width), value, dtype=np.float32)
    hdus = [fits.PrimaryHDU(data)]
    for _ in range(hdu_count - 1):
        hdus.append(fits.ImageHDU(data.copy()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_mask(
    path: Path,
    *,
    height: int = 8,
    width: int = 8,
    values: tuple[int, ...] = (0, 1, 2, 3),
) -> None:
    mask = np.zeros((height, width), dtype=np.float64)
    flat = mask.reshape(-1)
    for index, value in enumerate(values):
        flat[index] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(mask).writeto(path, overwrite=True)


class _TempDataset(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="d2_index_")
        self.root = Path(self._tmp)
        self.raw = self.root / "raw"
        self.raw.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def add_pair(
        self,
        sample_id: str,
        *,
        image_shape: tuple[int, int] = (8, 8),
        mask_shape: tuple[int, int] | None = None,
    ) -> None:
        _write_image(self.raw / f"imagedata_{sample_id}.fits", height=image_shape[0], width=image_shape[1])
        mask_shape = image_shape if mask_shape is None else mask_shape
        _write_mask(self.raw / f"mask_{sample_id}.fits", height=mask_shape[0], width=mask_shape[1])


class TestPairingBySampleId(_TempDataset):
    def test_pairs_by_id_not_directory_order(self):
        self.add_pair("b-2", image_shape=(7, 9))
        self.add_pair("a-1", image_shape=(8, 6))

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertEqual(len(index["matched_population"]), 2)
        self.assertEqual([row["sample_id"] for row in manifest["rows"]], ["a-1", "b-2"])
        self.assertEqual(manifest["rows"][0]["image_path"], "raw/imagedata_a-1.fits")
        self.assertEqual(manifest["rows"][0]["mask_path"], "raw/mask_a-1.fits")
        self.assertEqual((manifest["rows"][0]["height"], manifest["rows"][0]["width"]), (8, 6))
        self.assertEqual((manifest["rows"][1]["height"], manifest["rows"][1]["width"]), (7, 9))
        self.assertEqual(manifest["columns"], list(MANIFEST_COLUMNS))
        self.assertEqual(manifest["row_count"], 2)
        self.assertEqual(manifest["missing_referenced_files"], [])

    def test_recursive_discovery(self):
        nested = self.raw / "nested"
        _write_image(nested / "imagedata_n-1.fits")
        _write_mask(nested / "mask_n-1.fits")

        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertEqual(manifest["row_count"], 1)
        self.assertEqual(manifest["rows"][0]["image_path"], "raw/nested/imagedata_n-1.fits")


class TestDeterminism(_TempDataset):
    def test_repeated_manifest_bytes_are_identical(self):
        self.add_pair("a-1")
        self.add_pair("b-2")
        first = build_manifest(self.raw, repo_root=self.root)
        second = build_manifest(self.raw, repo_root=self.root)
        self.assertEqual(first["rows"], second["rows"])

        out1 = self.root / "manifest1.csv"
        out2 = self.root / "manifest2.csv"
        write_manifest(first["rows"], out1)
        write_manifest(second["rows"], out2)
        self.assertEqual(out1.read_bytes(), out2.read_bytes())

    def test_csv_uses_lf_and_expected_header(self):
        self.add_pair("a-1")
        out = self.root / "manifest.csv"
        write_manifest(build_manifest(self.raw, repo_root=self.root)["rows"], out)

        data = out.read_bytes()
        self.assertNotIn(b"\r", data)
        self.assertEqual(data.decode("utf-8").splitlines()[0], "sample_id,image_path,mask_path,height,width")

    def test_read_manifest_round_trip(self):
        self.add_pair("a-1")
        out = self.root / "manifest.csv"
        manifest = build_manifest(self.raw, repo_root=self.root)
        write_manifest(manifest["rows"], out)

        self.assertEqual(read_manifest(out), manifest["rows"])

    def test_read_manifest_rejects_bad_header(self):
        out = self.root / "bad.csv"
        out.write_text("sample_id,image_path\nx,y\n", encoding="utf-8")
        with self.assertRaises(DatasetIndexError):
            read_manifest(out)


class TestUnmatchedFiles(_TempDataset):
    def test_unmatched_image_excluded(self):
        _write_image(self.raw / "imagedata_only-image.fits")

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertEqual(list(index["unmatched_image_ids"]), ["only-image"])
        self.assertEqual(manifest["row_count"], 0)
        self.assertEqual(manifest["unique_pair_count"], 0)

    def test_unmatched_mask_excluded(self):
        _write_mask(self.raw / "mask_only-mask.fits")

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertEqual(list(index["unmatched_mask_ids"]), ["only-mask"])
        self.assertEqual(manifest["row_count"], 0)


class TestDuplicateAndAmbiguousIds(_TempDataset):
    def test_duplicate_image_id_is_ambiguous_and_excluded(self):
        self.add_pair("ok-1")
        _write_image(self.raw / "imagedata_dup-1.fits")
        _write_image(self.raw / "sub" / "imagedata_dup-1.fits")
        _write_mask(self.raw / "mask_dup-1.fits")

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertIn("dup-1", index["duplicate_image_ids"])
        self.assertIn("dup-1", index["ambiguous_ids"])
        self.assertNotIn("dup-1", [row["sample_id"] for row in manifest["rows"]])
        self.assertEqual([row["sample_id"] for row in manifest["rows"]], ["ok-1"])

    def test_duplicate_mask_id_is_ambiguous_and_excluded(self):
        _write_image(self.raw / "imagedata_dup-2.fits")
        _write_mask(self.raw / "mask_dup-2.fits")
        _write_mask(self.raw / "sub" / "mask_dup-2.fits")

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertIn("dup-2", index["duplicate_mask_ids"])
        self.assertIn("dup-2", index["ambiguous_ids"])
        self.assertEqual(manifest["row_count"], 0)


class TestMalformedFilenames(_TempDataset):
    def test_malformed_fits_are_reported_not_paired(self):
        self.add_pair("ok-1")
        _write_image(self.raw / "notes.fits")
        _write_image(self.raw / "imagedata_.fits")
        (self.raw / "readme.txt").write_text("not fits", encoding="utf-8")

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        malformed = {item["path"].name for item in index["malformed_files"]}
        self.assertEqual(malformed, {"notes.fits", "imagedata_.fits"})
        self.assertEqual([path.name for path in index["ignored_non_fits_files"]], ["readme.txt"])
        self.assertEqual(manifest["row_count"], 1)


class TestMissingFiles(_TempDataset):
    def test_validate_manifest_reports_missing_file(self):
        self.add_pair("a-1")
        manifest = build_manifest(self.raw, repo_root=self.root)
        (self.raw / "mask_a-1.fits").unlink()

        problems = validate_manifest(manifest["rows"], repo_root=self.root)
        self.assertTrue(
            any(item.get("field") == "mask_path" and item["reason"] == "file_missing" for item in problems)
        )

    def test_build_manifest_excludes_file_deleted_after_scan(self):
        self.add_pair("a-1")
        index = scan_dataset(self.raw)
        (self.raw / "imagedata_a-1.fits").unlink()

        manifest = build_manifest(self.raw, repo_root=self.root, index=index)

        self.assertEqual(manifest["row_count"], 0)
        self.assertEqual(manifest["excluded_pair_count"], 1)
        codes = {reason["code"] for reason in manifest["excluded_pairs"][0]["reasons"]}
        self.assertIn("image_file_missing", codes)

    def test_validate_manifest_reports_duplicate_absolute_and_non_posix_paths(self):
        rows = [
            {
                "sample_id": "a-1",
                "image_path": "raw/imagedata_a-1.fits",
                "mask_path": "raw/mask_a-1.fits",
                "height": 8,
                "width": 8,
            },
            {
                "sample_id": "a-1",
                "image_path": "C:/absolute.fits",
                "mask_path": "raw\\mask_a-1.fits",
                "height": 8,
                "width": 8,
            },
        ]
        problems = validate_manifest(rows, repo_root=self.root)
        reasons = {(item.get("field"), item["reason"]) for item in problems}
        self.assertIn((None, "duplicate_sample_id"), reasons)
        self.assertIn(("image_path", "absolute_path"), reasons)
        self.assertIn(("mask_path", "non_posix_path"), reasons)


class TestDimensionMismatch(_TempDataset):
    def test_mismatched_dimensions_are_excluded_and_reported(self):
        self.add_pair("bad-1", image_shape=(8, 8), mask_shape=(8, 7))
        self.add_pair("ok-1", image_shape=(6, 6))

        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertEqual([row["sample_id"] for row in manifest["rows"]], ["ok-1"])
        self.assertEqual(manifest["excluded_pair_count"], 1)
        excluded = manifest["excluded_pairs"][0]
        self.assertEqual(excluded["sample_id"], "bad-1")
        self.assertIn("dimension_mismatch", {reason["code"] for reason in excluded["reasons"]})

    def test_mismatched_extracted_ids_are_excluded(self):
        self.add_pair("a-1")
        index = scan_dataset(self.raw)
        index["matched_population"][0]["image"]["sample_id"] = "different"

        manifest = build_manifest(self.raw, repo_root=self.root, index=index)

        self.assertEqual(manifest["row_count"], 0)
        codes = {reason["code"] for reason in manifest["excluded_pairs"][0]["reasons"]}
        self.assertIn("sample_id_mismatch", codes)


class TestCorruptFits(_TempDataset):
    def test_corrupt_image_is_reported_and_excluded(self):
        (self.raw / "imagedata_bad-1.fits").write_bytes(b"not a fits file")
        _write_mask(self.raw / "mask_bad-1.fits")

        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        record = index["image_records"][0]
        self.assertIsNotNone(record["read_error"])
        self.assertEqual(manifest["row_count"], 0)
        codes = {reason["code"] for reason in manifest["excluded_pairs"][0]["reasons"]}
        self.assertIn("image_read_error", codes)


class TestEmptyAndInvalidDirectories(_TempDataset):
    def test_empty_dataset(self):
        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root)

        self.assertEqual(index["image_records"], [])
        self.assertEqual(index["mask_records"], [])
        self.assertEqual(manifest["row_count"], 0)
        self.assertEqual(manifest["rows"], [])

    def test_nonexistent_dataset(self):
        missing = self.root / "absent"
        with self.assertRaises(FileNotFoundError):
            scan_dataset(missing)
        with self.assertRaises(FileNotFoundError):
            build_manifest(missing, repo_root=self.root)

    def test_dataset_path_is_file(self):
        file_path = self.root / "not-a-dir"
        file_path.write_text("x", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            scan_dataset(file_path)

    def test_raw_directory_outside_repo_root_raises(self):
        self.add_pair("a-1")
        with self.assertRaises(DatasetIndexError):
            build_manifest(self.raw, repo_root=self.root / "elsewhere")


if __name__ == "__main__":
    unittest.main()
