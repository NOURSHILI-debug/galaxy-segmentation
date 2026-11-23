"""Deterministic, machine-readable dataset audit for the manifest (D2).

The audit deliberately separates three kinds of evidence:

1. **Current header facts** — file counts, HDU counts, shapes, and storage
   dtypes are re-derived by scanning FITS headers of the current dataset.
   Pixel arrays are not read.
2. **Manifest facts** — pairing, exclusions, relative paths, and validation
   results come from :mod:`src.data.dataset_index`.
3. **Reused pixel-level facts** — non-finite image inventory and mask-label
   frequencies are reused from the verified ``mask_audit.json`` artifact only
   after its counts and referenced paths are checked against the current
   dataset.  If the artifact is missing or stale, the summary says so instead
   of pretending the facts are available.

No timestamps or machine-specific absolute paths are written.  The same
dataset produces byte-identical output.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.data.dataset_index import (
    MANIFEST_COLUMNS,
    build_manifest,
    repo_root_default,
    scan_dataset,
    validate_manifest,
    write_manifest,
)
from src.data.mask_decoder import RAW_TO_TRAINING_ID

__all__ = [
    "DATASET_AUDIT_SCHEMA_VERSION",
    "DEFAULT_AUDIT_ARTIFACT",
    "build_audit_summary",
    "generate_manifest_and_audit",
    "write_audit_summary",
]

DATASET_AUDIT_SCHEMA_VERSION = "d2.1"
DEFAULT_AUDIT_ARTIFACT = "dataset/metadata/mask_audit.json"

#: Raw mask values declared by the frozen decoder mapping.
DECLARED_MASK_RAW_VALUES: tuple[int, ...] = tuple(sorted(RAW_TO_TRAINING_ID))

#: Documented contract counts; reported as advisory checks, not hard-coded facts.
EXPECTED_IMAGE_HDU_COUNT = 5
EXPECTED_MASK_DATA_HDU_COUNT = 1


def _rel_posix(path: Path, root: Path) -> str:
    """Return a repository-relative POSIX path, or absolute if outside the root."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _shape_key(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _distribution(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _id_paths(
    mapping: dict[str, list[dict[str, Any]]],
    root: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "paths": [_rel_posix(record["path"], root) for record in records],
        }
        for sample_id, records in sorted(mapping.items())
    ]


def _structure_summary(
    records: list[dict[str, Any]],
    root: Path,
    *,
    expected_hdu_count: int,
    expected_data_hdu_count: int,
) -> dict[str, Any]:
    """Summarize header-level structure across one file population."""
    readable = [record for record in records if record["read_error"] is None]
    hdu_counts: Counter[Any] = Counter()
    data_hdu_counts: Counter[Any] = Counter()
    primary_shapes: Counter[str] = Counter()
    data_shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    bitpix_values: Counter[Any] = Counter()
    scaled_hdu_count = 0
    nonstandard_hdu_count_paths: list[str] = []
    nonstandard_data_hdu_count_paths: list[str] = []

    for record in readable:
        if record["hdu_count"] is not None:
            hdu_counts[record["hdu_count"]] += 1
            if record["hdu_count"] != expected_hdu_count:
                nonstandard_hdu_count_paths.append(_rel_posix(record["path"], root))
        if record["data_hdu_count"] is not None:
            data_hdu_counts[record["data_hdu_count"]] += 1
            if record["data_hdu_count"] != expected_data_hdu_count:
                nonstandard_data_hdu_count_paths.append(_rel_posix(record["path"], root))

        if record["hdus"]:
            first = record["hdus"][0]
            if first["has_data"] and first["naxis"] == 2:
                primary_shapes[_shape_key(first["shape"])] += 1

        for hdu in record["hdus"]:
            if not hdu["has_data"]:
                continue
            data_shapes[_shape_key(hdu["shape"])] += 1
            if hdu["dtype"] is not None:
                dtypes[hdu["dtype"]] += 1
            if hdu["bitpix"] is not None:
                bitpix_values[hdu["bitpix"]] += 1
            if hdu["scaled"]:
                scaled_hdu_count += 1

    read_errors = [
        {"path": _rel_posix(record["path"], root), "error": record["read_error"]}
        for record in records
        if record["read_error"]
    ]
    read_errors.sort(key=lambda item: item["path"])

    return {
        "file_count": len(records),
        "readable_file_count": len(readable),
        "read_error_count": len(records) - len(readable),
        "read_errors": read_errors,
        "hdu_count_distribution": _distribution(hdu_counts),
        "data_hdu_count_distribution": _distribution(data_hdu_counts),
        "primary_shape_distribution": _distribution(primary_shapes),
        "data_hdu_shape_distribution": _distribution(data_shapes),
        "dtype_distribution": _distribution(dtypes),
        "bitpix_distribution": _distribution(bitpix_values),
        "scaled_hdu_count": scaled_hdu_count,
        "nonstandard_hdu_count_paths": sorted(nonstandard_hdu_count_paths),
        "nonstandard_data_hdu_count_paths": sorted(nonstandard_data_hdu_count_paths),
    }


def _load_reused_artifact(
    artifact_path: Path | None,
    root: Path,
    image_records: list[dict[str, Any]],
    mask_records: list[dict[str, Any]],
    *,
    reuse: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate and extract pixel-level facts from ``mask_audit.json``.

    Returns ``(status, non_finite_images, mask_labels)``.  The latter two are
    marked unavailable whenever the artifact cannot be trusted.
    """
    if not reuse:
        status = {
            "path": None,
            "available": False,
            "valid": False,
            "problems": ["reuse_audit_artifact=False"],
        }
        return (
            status,
            {"available": False, "reason": "reuse disabled"},
            {"available": False, "reason": "reuse disabled"},
        )

    if artifact_path is None:
        status = {
            "path": None,
            "available": False,
            "valid": False,
            "problems": ["no audit artifact path provided"],
        }
        return (
            status,
            {"available": False, "reason": "no audit artifact path provided"},
            {"available": False, "reason": "no audit artifact path provided"},
        )

    artifact_path = Path(artifact_path)
    status: dict[str, Any] = {
        "path": _rel_posix(artifact_path, root),
        "available": artifact_path.is_file(),
        "valid": False,
        "problems": [],
    }
    if not status["available"]:
        status["problems"].append(f"artifact not found: {status['path']}")
        return (
            status,
            {"available": False, "reason": "artifact not found"},
            {"available": False, "reason": "artifact not found"},
        )

    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except Exception as exc:
        status["problems"].append(f"artifact could not be parsed: {type(exc).__name__}: {exc}")
        return (
            status,
            {"available": False, "reason": "artifact unreadable"},
            {"available": False, "reason": "artifact unreadable"},
        )

    if not isinstance(artifact, dict):
        status["problems"].append("artifact root is not a JSON object")
        return (
            status,
            {"available": False, "reason": "artifact root is not an object"},
            {"available": False, "reason": "artifact root is not an object"},
        )

    problems: list[str] = status["problems"]
    audit_summary = artifact.get("audit_summary")
    image_statistics = artifact.get("image_statistics")
    mask_statistics = artifact.get("mask_statistics")
    data_quality = artifact.get("data_quality")

    if not isinstance(audit_summary, dict):
        problems.append("missing or invalid audit_summary")
    else:
        if audit_summary.get("image_file_count") != len(image_records):
            problems.append(
                "audit_summary.image_file_count "
                f"{audit_summary.get('image_file_count')!r} != current {len(image_records)}"
            )
        if audit_summary.get("mask_file_count") != len(mask_records):
            problems.append(
                "audit_summary.mask_file_count "
                f"{audit_summary.get('mask_file_count')!r} != current {len(mask_records)}"
            )

    if not isinstance(image_statistics, dict):
        problems.append("missing or invalid image_statistics")
    elif isinstance(image_statistics.get("files"), list) and len(image_statistics["files"]) != len(
        image_records
    ):
        problems.append(
            "image_statistics.files length "
            f"{len(image_statistics['files'])} != current {len(image_records)}"
        )

    if not isinstance(mask_statistics, dict):
        problems.append("missing or invalid mask_statistics")
    elif isinstance(mask_statistics.get("files"), list) and len(mask_statistics["files"]) != len(
        mask_records
    ):
        problems.append(
            "mask_statistics.files length "
            f"{len(mask_statistics['files'])} != current {len(mask_records)}"
        )

    if not isinstance(data_quality, dict):
        problems.append("missing or invalid data_quality")

    image_path_map = {record["path"].resolve(): record for record in image_records}
    hdu_lookup = {
        record["path"].resolve(): {
            hdu["index"]: hdu for hdu in record["hdus"] if hdu["has_data"]
        }
        for record in image_records
    }

    non_finite_entries = (
        data_quality.get("image_non_finite_hdus") if isinstance(data_quality, dict) else None
    )
    if not isinstance(non_finite_entries, list):
        problems.append("data_quality.image_non_finite_hdus is missing or not a list")
        non_finite_entries = None
    else:
        for index, entry in enumerate(non_finite_entries):
            if not isinstance(entry, dict):
                problems.append(f"non-finite entry {index} is not an object")
                continue
            raw_path = entry.get("path")
            hdu_index = entry.get("hdu_index")
            count = entry.get("non_finite_count")
            if not isinstance(raw_path, str):
                problems.append(f"non-finite entry {index} has no path")
                continue
            if isinstance(hdu_index, bool) or not isinstance(hdu_index, int):
                problems.append(f"non-finite entry {index} has invalid hdu_index")
                continue
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                problems.append(f"non-finite entry {index} has invalid non_finite_count")
                continue
            entry_path = Path(raw_path)
            resolved = (
                entry_path.resolve()
                if entry_path.is_absolute()
                else (root / entry_path).resolve()
            )
            if resolved not in image_path_map:
                problems.append(f"non-finite entry path is not a current image: {raw_path}")
                continue
            hdu = hdu_lookup.get(resolved, {}).get(hdu_index)
            if hdu is None:
                problems.append(f"non-finite entry references absent HDU {hdu_index} in {raw_path}")
                continue
            pixel_count = 1
            for dimension in hdu["shape"]:
                pixel_count *= int(dimension)
            if count > pixel_count:
                problems.append(
                    f"non-finite count {count} exceeds HDU pixel count {pixel_count} in {raw_path}"
                )

    raw_frequencies = (
        mask_statistics.get("raw_value_frequencies") if isinstance(mask_statistics, dict) else None
    )
    if not isinstance(raw_frequencies, list) or not raw_frequencies:
        problems.append("mask_statistics.raw_value_frequencies is missing or empty")
        raw_frequencies = None
    else:
        for index, entry in enumerate(raw_frequencies):
            if not isinstance(entry, dict):
                problems.append(f"raw frequency entry {index} is not an object")
                continue
            value = entry.get("value")
            count = entry.get("pixel_count")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"raw frequency entry {index} has invalid value")
                continue
            if not float(value).is_integer():
                problems.append(f"raw frequency entry {index} value is not integer-like")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                problems.append(f"raw frequency entry {index} has invalid pixel_count")

    if problems:
        status["valid"] = False
        return (
            status,
            {"available": False, "reason": "artifact failed validation"},
            {"available": False, "reason": "artifact failed validation"},
        )

    status["valid"] = True

    # --- Non-finite image inventory -------------------------------------
    non_finite_records: list[dict[str, Any]] = []
    for entry in non_finite_entries or []:
        raw_path = entry["path"]
        entry_path = Path(raw_path)
        resolved = (
            entry_path.resolve() if entry_path.is_absolute() else (root / entry_path).resolve()
        )
        record = image_path_map[resolved]
        non_finite_records.append(
            {
                "sample_id": record["sample_id"],
                "path": _rel_posix(resolved, root),
                "hdu_index": int(entry["hdu_index"]),
                "non_finite_count": int(entry["non_finite_count"]),
            }
        )
    non_finite_records.sort(key=lambda item: (item["path"], item["hdu_index"]))

    entries_by_band = Counter(str(item["hdu_index"]) for item in non_finite_records)
    pixels_by_band: Counter[str] = Counter()
    for item in non_finite_records:
        pixels_by_band[str(item["hdu_index"])] += item["non_finite_count"]

    non_finite_images = {
        "available": True,
        "source": status["path"],
        "unique_file_count": len({item["path"] for item in non_finite_records}),
        "hdu_entry_count": len(non_finite_records),
        "total_non_finite_pixels": sum(item["non_finite_count"] for item in non_finite_records),
        "hdu_entry_count_by_band_index": _distribution(entries_by_band),
        "non_finite_pixel_count_by_band_index": _distribution(pixels_by_band),
        "entries": non_finite_records,
    }

    # --- Mask label inventory -------------------------------------------
    frequency_records: list[dict[str, Any]] = []
    for entry in raw_frequencies or []:
        item: dict[str, Any] = {
            "value": int(entry["value"]),
            "pixel_count": int(entry["pixel_count"]),
        }
        if isinstance(entry.get("mask_count"), int) and not isinstance(entry["mask_count"], bool):
            item["mask_count"] = int(entry["mask_count"])
        frequency_records.append(item)
    frequency_records.sort(key=lambda item: item["value"])
    observed_values = [item["value"] for item in frequency_records]
    unexpected_values = [
        value for value in observed_values if value not in DECLARED_MASK_RAW_VALUES
    ]
    non_integer_like_hdus = (
        data_quality.get("mask_non_integer_like_hdus") if isinstance(data_quality, dict) else []
    )
    non_finite_hdus = (
        data_quality.get("mask_non_finite_hdus") if isinstance(data_quality, dict) else []
    )
    mask_labels = {
        "available": True,
        "source": status["path"],
        "declared_allowed_raw_values": list(DECLARED_MASK_RAW_VALUES),
        "observed_raw_values": observed_values,
        "unexpected_raw_values": unexpected_values,
        "non_finite_pixel_count": int(mask_statistics.get("non_finite_pixel_count") or 0),
        "non_finite_hdu_count": len(non_finite_hdus) if isinstance(non_finite_hdus, list) else None,
        "non_integer_like_hdu_count": (
            len(non_integer_like_hdus) if isinstance(non_integer_like_hdus, list) else None
        ),
        "total_pixel_count": int(mask_statistics.get("total_pixel_count") or 0),
        "raw_value_frequencies": frequency_records,
    }
    return status, non_finite_images, mask_labels


def _check(passed: bool, severity: str, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "severity": severity, "details": details}


def build_audit_summary(
    raw_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    index: dict[str, Any] | None = None,
    manifest_result: dict[str, Any] | None = None,
    audit_artifact_path: str | Path | None = None,
    reuse_audit_artifact: bool = True,
) -> dict[str, Any]:
    """Build a deterministic audit summary for the current dataset.

    Args:
        raw_dir: Dataset directory containing FITS files.
        repo_root: Repository root for relative paths.
        index: Optional :func:`scan_dataset` result to avoid rescanning headers.
        manifest_result: Optional :func:`build_manifest` result to avoid
            rebuilding the manifest.
        audit_artifact_path: Existing pixel-level audit artifact.  Defaults to
            ``dataset/metadata/mask_audit.json`` under ``repo_root``.
        reuse_audit_artifact: Set false to skip reused pixel-level facts.
    """
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_default()
    index = scan_dataset(raw_dir) if index is None else index
    manifest_result = (
        build_manifest(raw_dir, repo_root=root, index=index)
        if manifest_result is None
        else manifest_result
    )

    if audit_artifact_path is None:
        audit_artifact_path = root / DEFAULT_AUDIT_ARTIFACT
    artifact_status, non_finite_images, mask_labels = _load_reused_artifact(
        Path(audit_artifact_path),
        root,
        index["image_records"],
        index["mask_records"],
        reuse=reuse_audit_artifact,
    )

    manifest_rows = manifest_result["rows"]
    missing_referenced_files = validate_manifest(manifest_rows, repo_root=root)
    duplicate_sample_ids = [
        sample_id
        for sample_id, count in Counter(row["sample_id"] for row in manifest_rows).items()
        if count > 1
    ]
    duplicate_sample_ids.sort()

    excluded_pairs = manifest_result["excluded_pairs"]
    excluded_by_reason: Counter[str] = Counter()
    dimension_mismatch_count = 0
    for pair in excluded_pairs:
        codes = {reason["code"] for reason in pair["reasons"]}
        if "dimension_mismatch" in codes:
            dimension_mismatch_count += 1
        for code in codes:
            excluded_by_reason[code] += 1

    image_structure = _structure_summary(
        index["image_records"],
        root,
        expected_hdu_count=EXPECTED_IMAGE_HDU_COUNT,
        expected_data_hdu_count=EXPECTED_IMAGE_HDU_COUNT,
    )
    mask_structure = _structure_summary(
        index["mask_records"],
        root,
        expected_hdu_count=1,
        expected_data_hdu_count=EXPECTED_MASK_DATA_HDU_COUNT,
    )

    malformed_filenames = [
        {
            "path": _rel_posix(item["path"], root),
            "reason": item["reason"],
        }
        for item in index["malformed_files"]
    ]
    malformed_filenames.sort(key=lambda item: item["path"])

    dimension_distribution: Counter[str] = Counter(
        _shape_key((row["height"], row["width"])) for row in manifest_rows
    )

    checks: dict[str, dict[str, Any]] = {
        "dataset_non_empty": _check(
            len(index["image_records"]) > 0 and len(index["mask_records"]) > 0,
            "blocking",
            image_file_count=len(index["image_records"]),
            mask_file_count=len(index["mask_records"]),
        ),
        "unique_pairs_found": _check(
            len(index["matched_population"]) > 0,
            "blocking",
            one_to_one_pair_count=len(index["matched_population"]),
        ),
        "all_fits_files_recognized": _check(
            len(malformed_filenames) == 0,
            "advisory",
            malformed_filename_count=len(malformed_filenames),
        ),
        "image_files_readable": _check(
            image_structure["read_error_count"] == 0,
            "blocking",
            read_error_count=image_structure["read_error_count"],
        ),
        "mask_files_readable": _check(
            mask_structure["read_error_count"] == 0,
            "blocking",
            read_error_count=mask_structure["read_error_count"],
        ),
        "no_duplicate_image_ids": _check(
            len(index["duplicate_image_ids"]) == 0,
            "blocking",
            duplicate_image_id_count=len(index["duplicate_image_ids"]),
        ),
        "no_duplicate_mask_ids": _check(
            len(index["duplicate_mask_ids"]) == 0,
            "blocking",
            duplicate_mask_id_count=len(index["duplicate_mask_ids"]),
        ),
        "no_ambiguous_ids": _check(
            len(index["ambiguous_ids"]) == 0,
            "blocking",
            ambiguous_id_count=len(index["ambiguous_ids"]),
        ),
        "all_unique_pairs_valid": _check(
            len(excluded_pairs) == 0,
            "blocking",
            excluded_pair_count=len(excluded_pairs),
            excluded_by_reason=dict(sorted(excluded_by_reason.items())),
        ),
        "manifest_dimensions_match": _check(
            dimension_mismatch_count == 0,
            "blocking",
            dimension_mismatch_count=dimension_mismatch_count,
        ),
        "manifest_files_exist": _check(
            len(missing_referenced_files) == 0,
            "blocking",
            missing_referenced_file_count=len(missing_referenced_files),
        ),
        "manifest_ids_unique": _check(
            len(duplicate_sample_ids) == 0,
            "blocking",
            duplicate_sample_id_count=len(duplicate_sample_ids),
        ),
        "manifest_row_count_consistent": _check(
            len(manifest_rows) + len(excluded_pairs) == len(index["matched_population"]),
            "blocking",
            manifest_row_count=len(manifest_rows),
            excluded_pair_count=len(excluded_pairs),
            unique_pair_count=len(index["matched_population"]),
        ),
        "no_unmatched_images": _check(
            len(index["unmatched_image_ids"]) == 0,
            "advisory",
            unmatched_image_count=len(index["unmatched_image_ids"]),
        ),
        "no_unmatched_masks": _check(
            len(index["unmatched_mask_ids"]) == 0,
            "advisory",
            unmatched_mask_count=len(index["unmatched_mask_ids"]),
        ),
        "image_hdu_count_matches_contract": _check(
            not image_structure["nonstandard_hdu_count_paths"],
            "advisory",
            expected_hdu_count=EXPECTED_IMAGE_HDU_COUNT,
            nonstandard_file_count=len(image_structure["nonstandard_hdu_count_paths"]),
        ),
        "mask_data_hdu_count_matches_contract": _check(
            not mask_structure["nonstandard_data_hdu_count_paths"],
            "advisory",
            expected_data_hdu_count=EXPECTED_MASK_DATA_HDU_COUNT,
            nonstandard_file_count=len(mask_structure["nonstandard_data_hdu_count_paths"]),
        ),
        "reused_audit_artifact_valid": _check(
            artifact_status["valid"],
            "advisory",
            available=artifact_status["available"],
            problems=artifact_status["problems"],
        ),
        "non_finite_inventory_available": _check(
            non_finite_images["available"],
            "advisory",
            reason=non_finite_images.get("reason"),
        ),
        "mask_label_inventory_available": _check(
            mask_labels["available"],
            "advisory",
            reason=mask_labels.get("reason"),
        ),
        "mask_raw_values_within_declared_set": _check(
            mask_labels["available"] and not mask_labels.get("unexpected_raw_values"),
            "advisory",
            declared_allowed_raw_values=list(DECLARED_MASK_RAW_VALUES),
            unexpected_raw_values=mask_labels.get("unexpected_raw_values"),
        ),
        "mask_pixels_finite": _check(
            mask_labels["available"] and mask_labels.get("non_finite_pixel_count") == 0,
            "advisory",
            non_finite_pixel_count=mask_labels.get("non_finite_pixel_count"),
        ),
    }

    blocking_failures = sorted(
        name
        for name, check in checks.items()
        if check["severity"] == "blocking" and not check["passed"]
    )
    advisory_failures = sorted(
        name
        for name, check in checks.items()
        if check["severity"] == "advisory" and not check["passed"]
    )

    return {
        "schema_version": DATASET_AUDIT_SCHEMA_VERSION,
        "dataset": {
            "raw_directory": _rel_posix(index["raw_directory"], root),
            "fits_file_count": (
                len(index["image_records"])
                + len(index["mask_records"])
                + len(index["malformed_files"])
            ),
            "image_file_count": len(index["image_records"]),
            "mask_file_count": len(index["mask_records"]),
            "image_sample_id_count": len(index["images_by_id"]),
            "mask_sample_id_count": len(index["masks_by_id"]),
            "malformed_filename_count": len(malformed_filenames),
            "malformed_filenames": malformed_filenames,
            "ignored_non_fits_file_count": len(index["ignored_non_fits_files"]),
            "ignored_non_fits_files": [
                _rel_posix(path, root) for path in index["ignored_non_fits_files"]
            ],
        },
        "pairing": {
            "one_to_one_pair_count": len(index["matched_population"]),
            "unmatched_image_count": len(index["unmatched_image_ids"]),
            "unmatched_images": _id_paths(index["unmatched_image_ids"], root),
            "unmatched_mask_count": len(index["unmatched_mask_ids"]),
            "unmatched_masks": _id_paths(index["unmatched_mask_ids"], root),
            "duplicate_image_id_count": len(index["duplicate_image_ids"]),
            "duplicate_image_ids": _id_paths(index["duplicate_image_ids"], root),
            "duplicate_mask_id_count": len(index["duplicate_mask_ids"]),
            "duplicate_mask_ids": _id_paths(index["duplicate_mask_ids"], root),
            "ambiguous_id_count": len(index["ambiguous_ids"]),
            "ambiguous_ids": [
                {
                    "sample_id": sample_id,
                    "image_paths": [
                        _rel_posix(path, root) for path in item["image_paths"]
                    ],
                    "mask_paths": [_rel_posix(path, root) for path in item["mask_paths"]],
                }
                for sample_id, item in sorted(index["ambiguous_ids"].items())
            ],
        },
        "manifest": {
            "columns": list(MANIFEST_COLUMNS),
            "row_count": len(manifest_rows),
            "unique_pair_count": len(index["matched_population"]),
            "excluded_pair_count": len(excluded_pairs),
            "excluded_pairs": excluded_pairs,
            "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
            "missing_referenced_files": missing_referenced_files,
            "duplicate_sample_id_count": len(duplicate_sample_ids),
            "duplicate_sample_ids": duplicate_sample_ids,
            "height_width_distribution": _distribution(dimension_distribution),
        },
        "fits_structure": {
            "image": image_structure,
            "mask": mask_structure,
            "declared_contract": {
                "expected_image_hdu_count": EXPECTED_IMAGE_HDU_COUNT,
                "expected_mask_data_hdu_count": EXPECTED_MASK_DATA_HDU_COUNT,
                "band_order": "documented g,r,i,z,y by image HDU index",
                "band_order_header_verified": False,
            },
        },
        "data_quality": {
            "reused_audit_artifact": artifact_status,
            "non_finite_images": non_finite_images,
            "mask_labels": mask_labels,
        },
        "validation": {
            "status": "ok" if not blocking_failures else "failed",
            "blocking_failures": blocking_failures,
            "advisory_failures": advisory_failures,
            "checks": checks,
        },
    }


def write_audit_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write a deterministic UTF-8 JSON audit summary with ``\\n`` endings."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.write_text(text, encoding="utf-8", newline="\n")
    return output


def generate_manifest_and_audit(
    raw_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    audit_artifact_path: str | Path | None = None,
    reuse_audit_artifact: bool = True,
) -> dict[str, Any]:
    """Scan once, then write both deterministic D2 artifacts."""
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_default()
    index = scan_dataset(raw_dir)
    manifest_result = build_manifest(raw_dir, repo_root=root, index=index)

    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else root / "dataset/metadata/manifest.csv"
    )
    audit_output = (
        Path(audit_path) if audit_path is not None else root / "dataset/metadata/audit_summary.json"
    )

    write_manifest(manifest_result["rows"], manifest_output)
    summary = build_audit_summary(
        raw_dir,
        repo_root=root,
        index=index,
        manifest_result=manifest_result,
        audit_artifact_path=audit_artifact_path,
        reuse_audit_artifact=reuse_audit_artifact,
    )
    write_audit_summary(summary, audit_output)
    return {
        "manifest": manifest_result,
        "audit": summary,
        "manifest_path": str(manifest_output),
        "audit_path": str(audit_output),
    }
