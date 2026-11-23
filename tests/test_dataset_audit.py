"""Focused tests for the deterministic dataset audit summary (D2)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.data.dataset_audit import (
    DATASET_AUDIT_SCHEMA_VERSION,
    build_audit_summary,
    generate_manifest_and_audit,
    write_audit_summary,
)
from src.data.dataset_index import build_manifest, scan_dataset


def _write_image(path: Path, *, height: int = 8, width: int = 8, hdu_count: int = 5) -> None:
    data = np.ones((height, width), dtype=np.float32)
    hdus = [fits.PrimaryHDU(data)]
    for _ in range(hdu_count - 1):
        hdus.append(fits.ImageHDU(data.copy()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_mask(path: Path, *, height: int = 8, width: int = 8) -> None:
    mask = np.zeros((height, width), dtype=np.float64)
    mask.reshape(-1)[:4] = [0, 1, 2, 3]
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(mask).writeto(path, overwrite=True)


def _write_artifact(path: Path, *, image_count: int, mask_count: int, non_finite: bool = True) -> None:
    artifact = {
        "audit_summary": {
            "raw_directory": "raw",
            "image_file_count": image_count,
            "mask_file_count": mask_count,
        },
        "image_statistics": {"files": [{} for _ in range(image_count)]},
        "mask_statistics": {
            "files": [{} for _ in range(mask_count)],
            "total_pixel_count": 64 * mask_count,
            "non_finite_pixel_count": 0,
            "raw_value_frequencies": [
                {"value": 0.0, "pixel_count": 50 * mask_count, "mask_count": mask_count},
                {"value": 1.0, "pixel_count": 14 * mask_count, "mask_count": mask_count},
            ],
        },
        "data_quality": {
            "image_non_finite_hdus": (
                [{"path": "raw/imagedata_a-1.fits", "hdu_index": 0, "non_finite_count": 2}]
                if non_finite
                else []
            ),
            "mask_non_finite_hdus": [],
            "mask_non_integer_like_hdus": [],
        },
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")


class _TempDataset(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="d2_audit_")
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
        _write_image(
            self.raw / f"imagedata_{sample_id}.fits",
            height=image_shape[0],
            width=image_shape[1],
        )
        mask_shape = image_shape if mask_shape is None else mask_shape
        _write_mask(
            self.raw / f"mask_{sample_id}.fits",
            height=mask_shape[0],
            width=mask_shape[1],
        )


class TestCountsAndStatus(_TempDataset):
    def test_counts_and_advisory_unmatched_mask(self):
        self.add_pair("a-1")
        self.add_pair("b-2")
        _write_mask(self.raw / "mask_orphan-1.fits")

        summary = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)

        self.assertEqual(summary["schema_version"], DATASET_AUDIT_SCHEMA_VERSION)
        self.assertEqual(summary["dataset"]["image_file_count"], 2)
        self.assertEqual(summary["dataset"]["mask_file_count"], 3)
        self.assertEqual(summary["dataset"]["fits_file_count"], 5)
        self.assertEqual(summary["pairing"]["one_to_one_pair_count"], 2)
        self.assertEqual(summary["pairing"]["unmatched_mask_count"], 1)
        self.assertEqual(summary["pairing"]["unmatched_masks"][0]["sample_id"], "orphan-1")
        self.assertEqual(summary["manifest"]["row_count"], 2)
        self.assertEqual(summary["manifest"]["height_width_distribution"], {"8x8": 2})
        self.assertEqual(summary["fits_structure"]["image"]["hdu_count_distribution"], {"5": 2})
        self.assertEqual(summary["fits_structure"]["image"]["dtype_distribution"], {"float32": 10})
        self.assertEqual(summary["fits_structure"]["mask"]["hdu_count_distribution"], {"1": 3})
        self.assertEqual(summary["fits_structure"]["mask"]["dtype_distribution"], {"float64": 3})
        self.assertEqual(summary["validation"]["status"], "ok")
        self.assertEqual(summary["validation"]["blocking_failures"], [])
        self.assertIn("no_unmatched_masks", summary["validation"]["advisory_failures"])
        self.assertTrue(summary["validation"]["checks"]["no_unmatched_masks"]["passed"] is False)

    def test_json_serializable_and_no_unstable_timestamps(self):
        self.add_pair("a-1")
        summary = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)
        json.dumps(summary)  # must not raise

        keys: set[str] = set()

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    keys.add(key)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(summary)
        forbidden = {"timestamp", "generated_at", "created_at", "date"}
        self.assertFalse({key.lower() for key in keys} & forbidden)


class TestDimensionMismatchStatus(_TempDataset):
    def test_dimension_mismatch_is_blocking(self):
        self.add_pair("bad-1", image_shape=(8, 8), mask_shape=(9, 8))

        summary = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)

        self.assertEqual(summary["manifest"]["row_count"], 0)
        self.assertEqual(summary["manifest"]["excluded_by_reason"], {"dimension_mismatch": 1})
        self.assertEqual(summary["validation"]["status"], "failed")
        self.assertIn("all_unique_pairs_valid", summary["validation"]["blocking_failures"])
        self.assertIn("manifest_dimensions_match", summary["validation"]["blocking_failures"])


class TestMalformedFilenames(_TempDataset):
    def test_malformed_filename_is_reported_and_advisory(self):
        self.add_pair("ok-1")
        _write_image(self.raw / "notes.fits")

        summary = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)

        self.assertEqual(summary["dataset"]["malformed_filename_count"], 1)
        self.assertEqual(
            summary["dataset"]["malformed_filenames"][0]["path"], "raw/notes.fits"
        )
        self.assertEqual(summary["validation"]["status"], "ok")
        self.assertIn("all_fits_files_recognized", summary["validation"]["advisory_failures"])


class TestDuplicateAmbiguousStatus(_TempDataset):
    def test_duplicate_ids_are_blocking(self):
        _write_image(self.raw / "imagedata_dup-1.fits")
        _write_image(self.raw / "sub" / "imagedata_dup-1.fits")
        _write_mask(self.raw / "mask_dup-1.fits")

        summary = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)

        self.assertEqual(summary["manifest"]["row_count"], 0)
        self.assertEqual(summary["validation"]["status"], "failed")
        self.assertIn("no_duplicate_image_ids", summary["validation"]["blocking_failures"])
        self.assertIn("no_ambiguous_ids", summary["validation"]["blocking_failures"])

    def test_empty_dataset_fails(self):
        summary = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)

        self.assertEqual(summary["dataset"]["image_file_count"], 0)
        self.assertEqual(summary["manifest"]["row_count"], 0)
        self.assertEqual(summary["validation"]["status"], "failed")
        self.assertIn("dataset_non_empty", summary["validation"]["blocking_failures"])
        self.assertIn("unique_pairs_found", summary["validation"]["blocking_failures"])


class TestArtifactReuse(_TempDataset):
    def test_valid_artifact_provides_pixel_facts(self):
        self.add_pair("a-1")
        artifact = self.root / "artifact.json"
        _write_artifact(artifact, image_count=1, mask_count=1)

        summary = build_audit_summary(
            self.raw,
            repo_root=self.root,
            audit_artifact_path=artifact,
            reuse_audit_artifact=True,
        )

        artifact_status = summary["data_quality"]["reused_audit_artifact"]
        self.assertTrue(artifact_status["valid"])
        self.assertEqual(artifact_status["problems"], [])
        non_finite = summary["data_quality"]["non_finite_images"]
        self.assertTrue(non_finite["available"])
        self.assertEqual(non_finite["unique_file_count"], 1)
        self.assertEqual(non_finite["hdu_entry_count"], 1)
        self.assertEqual(non_finite["total_non_finite_pixels"], 2)
        self.assertEqual(non_finite["entries"][0]["sample_id"], "a-1")
        self.assertEqual(summary["data_quality"]["mask_labels"]["observed_raw_values"], [0, 1])
        self.assertTrue(
            summary["validation"]["checks"]["reused_audit_artifact_valid"]["passed"]
        )

    def test_stale_artifact_is_rejected(self):
        self.add_pair("a-1")
        artifact = self.root / "artifact.json"
        _write_artifact(artifact, image_count=99, mask_count=99)

        summary = build_audit_summary(
            self.raw,
            repo_root=self.root,
            audit_artifact_path=artifact,
            reuse_audit_artifact=True,
        )

        status = summary["data_quality"]["reused_audit_artifact"]
        self.assertFalse(status["valid"])
        self.assertTrue(any("image_file_count" in problem for problem in status["problems"]))
        self.assertFalse(summary["data_quality"]["non_finite_images"]["available"])
        self.assertFalse(summary["data_quality"]["mask_labels"]["available"])
        self.assertFalse(summary["validation"]["checks"]["reused_audit_artifact_valid"]["passed"])
        # Advisory only: the manifest itself is still valid.
        self.assertEqual(summary["validation"]["status"], "ok")

    def test_missing_or_unparseable_artifact_does_not_crash(self):
        self.add_pair("a-1")
        missing = self.root / "absent.json"
        summary = build_audit_summary(
            self.raw,
            repo_root=self.root,
            audit_artifact_path=missing,
            reuse_audit_artifact=True,
        )
        self.assertFalse(summary["data_quality"]["reused_audit_artifact"]["available"])

        empty = self.root / "empty.json"
        empty.write_text("", encoding="utf-8")
        summary = build_audit_summary(
            self.raw,
            repo_root=self.root,
            audit_artifact_path=empty,
            reuse_audit_artifact=True,
        )
        self.assertFalse(summary["data_quality"]["reused_audit_artifact"]["valid"])

    def test_artifact_entry_referencing_unknown_file_is_rejected(self):
        self.add_pair("a-1")
        artifact = self.root / "artifact.json"
        _write_artifact(artifact, image_count=1, mask_count=1)
        data = json.loads(artifact.read_text(encoding="utf-8"))
        data["data_quality"]["image_non_finite_hdus"][0]["path"] = "raw/imagedata_unknown.fits"
        artifact.write_text(json.dumps(data), encoding="utf-8")

        summary = build_audit_summary(
            self.raw,
            repo_root=self.root,
            audit_artifact_path=artifact,
            reuse_audit_artifact=True,
        )
        self.assertFalse(summary["data_quality"]["reused_audit_artifact"]["valid"])
        self.assertFalse(summary["data_quality"]["non_finite_images"]["available"])


class TestMissingReferencedFiles(_TempDataset):
    def test_deleted_file_fails_manifest_file_check(self):
        self.add_pair("a-1")
        index = scan_dataset(self.raw)
        manifest = build_manifest(self.raw, repo_root=self.root, index=index)
        (self.raw / "imagedata_a-1.fits").unlink()

        summary = build_audit_summary(
            self.raw,
            repo_root=self.root,
            index=index,
            manifest_result=manifest,
            reuse_audit_artifact=False,
        )

        missing = summary["manifest"]["missing_referenced_files"]
        self.assertTrue(any(item["reason"] == "file_missing" for item in missing))
        self.assertEqual(summary["validation"]["status"], "failed")
        self.assertIn("manifest_files_exist", summary["validation"]["blocking_failures"])


class TestDeterminism(_TempDataset):
    def test_repeated_audit_bytes_are_identical(self):
        self.add_pair("a-1")
        self.add_pair("b-2")
        first = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)
        second = build_audit_summary(self.raw, repo_root=self.root, reuse_audit_artifact=False)
        out1 = self.root / "audit1.json"
        out2 = self.root / "audit2.json"
        write_audit_summary(first, out1)
        write_audit_summary(second, out2)
        self.assertEqual(out1.read_bytes(), out2.read_bytes())

    def test_generate_both_artifacts(self):
        self.add_pair("a-1")
        manifest_path = self.root / "out" / "manifest.csv"
        audit_path = self.root / "out" / "audit.json"

        result = generate_manifest_and_audit(
            self.raw,
            manifest_path=manifest_path,
            audit_path=audit_path,
            repo_root=self.root,
            reuse_audit_artifact=False,
        )

        self.assertTrue(manifest_path.is_file())
        self.assertTrue(audit_path.is_file())
        self.assertEqual(result["manifest"]["row_count"], 1)
        written = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(written["manifest"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
