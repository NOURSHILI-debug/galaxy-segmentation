"""Read-only FITS dataset auditing utilities.

This module reports numeric mask values as observed. It does not assign class
names, infer image-HDU meanings, or select a background/ignore policy.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


_IMAGE_NAME = re.compile(r"^imagedata_(.+)\.fits(?:\.gz)?$", re.IGNORECASE)
_MASK_NAME = re.compile(r"^mask_(.+)\.fits(?:\.gz)?$", re.IGNORECASE)


def _json_safe(value: Any, issues: list[dict[str, str]], location: str) -> Any:
    """Convert values recursively to strict-JSON-safe primitives.

    Non-finite or unsupported values become ``None`` and are recorded for the
    caller so sanitization cannot silently hide a data-quality issue.
    """
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.number):
            return _json_number(value, issues, location)
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        issues.append({"location": location, "issue": "non-finite numeric value replaced with null"})
        return None
    if isinstance(value, complex):
        issues.append({"location": location, "issue": "complex value is not representable as a JSON scalar"})
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = key if isinstance(key, str) else str(key)
            result[safe_key] = _json_safe(item, issues, f"{location}.{safe_key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, issues, f"{location}[{index}]") for index, item in enumerate(value)]
    issues.append({"location": location, "issue": f"unsupported value type {type(value).__name__} replaced with null"})
    return None


def _json_number(value: Any, issues: list[dict[str, str]], location: str) -> int | float | None:
    """Convert a numeric scalar exactly where possible, otherwise return null."""
    if isinstance(value, np.generic):
        original = value
        value = value.item()
        if isinstance(value, np.generic):
            # Extended precision NumPy scalars may remain NumPy scalars from item().
            try:
                narrowed = float(value)
                if not math.isfinite(narrowed) or np.asarray(narrowed, dtype=original.dtype) != original:
                    issues.append({"location": location, "issue": "numeric value is not exactly representable in JSON number range"})
                    return None
                value = narrowed
            except (TypeError, ValueError, OverflowError):
                issues.append({"location": location, "issue": "numeric value is not representable as a JSON number"})
                return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        issues.append({"location": location, "issue": "non-finite numeric statistic replaced with null"})
        return None
    if isinstance(value, complex):
        issues.append({"location": location, "issue": "complex value is not representable as a JSON scalar"})
        return None
    issues.append({"location": location, "issue": f"unsupported numeric type {type(value).__name__}"})
    return None


def _filename_id(path: Path, pattern: re.Pattern[str]) -> str | None:
    """Return the sample ID from a supported FITS filename, if it matches."""
    match = pattern.match(path.name)
    return match.group(1) if match else None


def _inspect_fits(path: Path, *, inspect_mask_values: bool) -> dict[str, Any]:
    """Inspect one FITS file read-only, recording file- and HDU-level errors."""
    pattern = _MASK_NAME if inspect_mask_values else _IMAGE_NAME
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "sample_id": _filename_id(path, pattern),
        "hdu_count": None,
        "hdus": [],
        "read_error": None,
        "hdu_errors": [],
        "numeric_sanitization_issues": [],
    }

    try:
        hdus = fits.open(path, mode="readonly", memmap=True)
    except Exception as exc:
        record["read_error"] = f"{type(exc).__name__}: {exc}"
        return record

    try:
        with hdus:
            try:
                record["hdu_count"] = len(hdus)
                iterator = enumerate(hdus)
                for index, hdu in iterator:
                    location = f"{path.name}:HDU {index}"
                    item: dict[str, Any] = {
                        "index": index,
                        "name": None,
                        "has_data": None,
                        "shape": None,
                        "dtype": None,
                        "bitpix": None,
                        "bunit": None,
                        "finite_count": None,
                        "non_finite_count": None,
                        "min_finite": None,
                        "max_finite": None,
                    }
                    try:
                        item["name"] = _json_safe(hdu.name, record["numeric_sanitization_issues"], f"{location}.name")
                        item["bitpix"] = _json_safe(hdu.header.get("BITPIX"), record["numeric_sanitization_issues"], f"{location}.BITPIX")
                        item["bunit"] = _json_safe(hdu.header.get("BUNIT"), record["numeric_sanitization_issues"], f"{location}.BUNIT")
                        data = hdu.data
                        item["has_data"] = data is not None
                        if data is not None:
                            array = np.asarray(data)
                            item["shape"] = [int(dimension) for dimension in array.shape]
                            item["dtype"] = str(array.dtype)
                            finite = np.isfinite(array)
                            finite_count = int(np.count_nonzero(finite))
                            total_count = int(array.size)
                            item["finite_count"] = finite_count
                            item["non_finite_count"] = total_count - finite_count
                            if finite_count:
                                finite_values = array[finite]
                                item["min_finite"] = _json_number(
                                    np.min(finite_values), record["numeric_sanitization_issues"], f"{location}.min_finite"
                                )
                                item["max_finite"] = _json_number(
                                    np.max(finite_values), record["numeric_sanitization_issues"], f"{location}.max_finite"
                                )
                                del finite_values
                            if inspect_mask_values:
                                item["integer_like"] = None
                                item["unique_value_count"] = 0
                                item["raw_value_frequencies"] = []
                                item["unrepresented_finite_value_pixel_count"] = 0
                                if finite_count:
                                    # Unique runs over the current HDU only. The original
                                    # array is never rounded, cast, normalized, or changed.
                                    unique_values, frequencies = np.unique(array, return_counts=True)
                                    finite_unique = np.isfinite(unique_values)
                                    unique_values = unique_values[finite_unique]
                                    frequencies = frequencies[finite_unique]
                                    rounded = np.rint(unique_values)
                                    item["integer_like"] = bool(np.all(unique_values == rounded))
                                    raw_entries = []
                                    represented_count = 0
                                    for value, count in zip(unique_values, frequencies):
                                        safe_value = _json_number(
                                            value, record["numeric_sanitization_issues"],
                                            f"{location}.raw_mask_value",
                                        )
                                        if safe_value is None:
                                            item["unrepresented_finite_value_pixel_count"] += int(count)
                                            continue
                                        raw_entries.append({"value": safe_value, "pixel_count": int(count)})
                                        represented_count += int(count)
                                    item["raw_value_frequencies"] = raw_entries
                                    item["unique_value_count"] = int(unique_values.size)
                                    item["represented_finite_value_pixel_count"] = represented_count
                        record["hdus"].append(item)
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        record["hdu_errors"].append({"path": path.as_posix(), "hdu_index": index, "error": message})
                        item["inspection_error"] = message
                        record["hdus"].append(item)
            except Exception as exc:
                record["read_error"] = f"HDU traversal failed: {type(exc).__name__}: {exc}"
    except Exception as exc:
        record["read_error"] = record["read_error"] or f"FITS close/traversal failed: {type(exc).__name__}: {exc}"

    return record


def _inspect_collection(paths: list[Path], *, inspect_mask_values: bool) -> list[dict[str, Any]]:
    """Inspect paths sequentially, retaining metadata but never pixel arrays."""
    return [_inspect_fits(path, inspect_mask_values=inspect_mask_values) for path in paths]


def _group_by_id(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group all discovered records by ID without discarding duplicate files."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["sample_id"] is not None:
            grouped[record["sample_id"]].append(record)
    return dict(grouped)


def _aggregate_masks(mask_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate finite raw values across readable HDUs in all mask files."""
    pixel_counts: Counter[int | float] = Counter()
    mask_occurrences: Counter[int | float] = Counter()
    total_pixels = finite_pixels = non_finite_pixels = 0
    unknown_pixels = 0
    incomplete_files = 0

    for mask in mask_records:
        if mask["read_error"]:
            incomplete_files += 1
        values_in_file: set[int | float] = set()
        file_complete = not mask["hdu_errors"]
        for hdu in mask["hdus"]:
            if hdu["has_data"] is not True:
                if hdu["has_data"] is None:
                    file_complete = False
                continue
            if hdu["finite_count"] is None or hdu["non_finite_count"] is None:
                file_complete = False
                continue
            finite_pixels += hdu["finite_count"]
            non_finite_pixels += hdu["non_finite_count"]
            total_pixels += hdu["finite_count"] + hdu["non_finite_count"]
            unknown_pixels += hdu.get("unrepresented_finite_value_pixel_count", 0)
            for entry in hdu.get("raw_value_frequencies", []):
                value = entry["value"]
                pixel_counts[value] += entry["pixel_count"]
                values_in_file.add(value)
        if not file_complete:
            incomplete_files += 1
        for value in values_in_file:
            mask_occurrences[value] += 1

    values = []
    for value in sorted(pixel_counts):
        count = int(pixel_counts[value])
        values.append({
            "value": value,
            "pixel_count": count,
            "percentage_of_total_pixels": 100.0 * count / total_pixels if total_pixels else None,
            "percentage_of_finite_pixels": 100.0 * count / finite_pixels if finite_pixels else None,
            "mask_count": int(mask_occurrences[value]),
        })
    return {
        "total_pixel_count": total_pixels,
        "finite_pixel_count": finite_pixels,
        "non_finite_pixel_count": non_finite_pixels,
        "unrepresented_finite_value_pixel_count": unknown_pixels,
        "statistics_incomplete_mask_file_count": incomplete_files,
        "percentage_denominators": {
            "percentage_of_total_pixels": "total_pixel_count",
            "percentage_of_finite_pixels": "finite_pixel_count",
        },
        "raw_value_frequencies": values,
    }


def _dimension_checks(
    image_groups: dict[str, list[dict[str, Any]]],
    mask_groups: dict[str, list[dict[str, Any]]],
    common_ids: list[str],
) -> dict[str, Any]:
    """Compare all image-mask candidate combinations sharing each common ID."""
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for sample_id in common_ids:
        ambiguous = len(image_groups[sample_id]) != 1 or len(mask_groups[sample_id]) != 1
        for image in image_groups[sample_id]:
            for mask in mask_groups[sample_id]:
                pair_info = {
                    "sample_id": sample_id,
                    "image_path": image["path"],
                    "mask_path": mask["path"],
                    "ambiguous": ambiguous,
                    "pairing_status": "ambiguous_candidate" if ambiguous else "unique_match",
                }
                if image["read_error"] or mask["read_error"]:
                    unavailable.append({**pair_info, "reason": "one or both FITS files could not be read"})
                    continue
                image_hdus = [h for h in image["hdus"] if h["has_data"] is True]
                mask_hdus = [h for h in mask["hdus"] if h["has_data"] is True]
                if not image_hdus or not mask_hdus:
                    unavailable.append({**pair_info, "reason": "image or mask has no data-bearing HDU"})
                for image_hdu in image_hdus:
                    for mask_hdu in mask_hdus:
                        if image_hdu.get("shape") is None or mask_hdu.get("shape") is None:
                            unavailable.append({
                                **pair_info,
                                "image_hdu_index": image_hdu["index"],
                                "mask_hdu_index": mask_hdu["index"],
                                "reason": "shape unavailable because HDU inspection failed",
                            })
                            continue
                        comparison = {
                            **pair_info,
                            "image_hdu_index": image_hdu["index"],
                            "image_shape": image_hdu["shape"],
                            "mask_hdu_index": mask_hdu["index"],
                            "mask_shape": mask_hdu["shape"],
                            "dimensions_match": image_hdu["shape"] == mask_hdu["shape"],
                        }
                        comparisons.append(comparison)
                        if not comparison["dimensions_match"]:
                            mismatches.append(comparison)

    return {
        "candidate_combination_count": len({(item["sample_id"], item["image_path"], item["mask_path"]) for item in comparisons + unavailable}),
        "comparison_count": len(comparisons),
        "matched_dimensions": sum(item["dimensions_match"] for item in comparisons),
        "ambiguous_candidate_combinations": [item for item in comparisons if item["ambiguous"]],
        "comparisons": comparisons,
        "mismatches": mismatches,
        "unavailable": unavailable,
    }


def audit_dataset(raw_dir: Path) -> dict[str, Any]:
    """Audit filenames and FITS data under ``raw_dir`` without modifying files.

    Discovery is recursive and accepts plain or gzip-compressed FITS filenames.
    Returned mask values are raw numeric observations with no semantic mapping.
    """
    raw_dir = Path(raw_dir).expanduser()
    try:
        raw_stat = raw_dir.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw FITS directory does not exist: {raw_dir}") from exc
    except OSError as exc:
        raise PermissionError(f"Cannot access raw FITS path {raw_dir}: {exc}") from exc
    if not stat.S_ISDIR(raw_stat.st_mode):
        raise NotADirectoryError(f"Raw FITS path is not a directory: {raw_dir}")

    try:
        discovered: list[Path] = []

        def fail_on_scan_error(error: OSError) -> None:
            raise error

        for directory, _, filenames in os.walk(raw_dir, onerror=fail_on_scan_error):
            discovered.extend(
                Path(directory) / filename
                for filename in filenames
                if filename.lower().endswith((".fits", ".fits.gz"))
            )
        fits_paths = sorted(discovered, key=lambda path: path.as_posix().lower())
    except OSError as exc:
        raise PermissionError(f"Could not scan raw FITS directory {raw_dir}: {exc}") from exc

    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    unexpected: list[str] = []
    for path in fits_paths:
        if _IMAGE_NAME.match(path.name):
            image_paths.append(path)
        elif _MASK_NAME.match(path.name):
            mask_paths.append(path)
        else:
            unexpected.append(path.as_posix())

    image_records = _inspect_collection(image_paths, inspect_mask_values=False)
    mask_records = _inspect_collection(mask_paths, inspect_mask_values=True)
    image_groups = _group_by_id(image_records)
    mask_groups = _group_by_id(mask_records)
    image_ids, mask_ids = set(image_groups), set(mask_groups)
    common_ids = sorted(image_ids & mask_ids)
    unique_matched_ids = [
        sample_id for sample_id in common_ids
        if len(image_groups[sample_id]) == 1 and len(mask_groups[sample_id]) == 1
    ]
    duplicate_images = {
        sample_id: [record["path"] for record in records]
        for sample_id, records in sorted(image_groups.items()) if len(records) > 1
    }
    duplicate_masks = {
        sample_id: [record["path"] for record in records]
        for sample_id, records in sorted(mask_groups.items()) if len(records) > 1
    }
    ambiguous_ids = {
        sample_id: {
            "image_paths": [record["path"] for record in image_groups[sample_id]],
            "mask_paths": [record["path"] for record in mask_groups[sample_id]],
            "candidate_combination_count": len(image_groups[sample_id]) * len(mask_groups[sample_id]),
        }
        for sample_id in common_ids
        if len(image_groups[sample_id]) != 1 or len(mask_groups[sample_id]) != 1
    }
    ambiguous_combinations = [
        {
            "sample_id": sample_id,
            "image_path": image["path"],
            "mask_path": mask["path"],
            "ambiguous": True,
        }
        for sample_id in sorted(ambiguous_ids)
        for image in image_groups[sample_id]
        for mask in mask_groups[sample_id]
    ]
    unmatched_images = [
        {"sample_id": sample_id, "paths": [r["path"] for r in image_groups[sample_id]]}
        for sample_id in sorted(image_ids - mask_ids)
    ]
    unmatched_masks = [
        {"sample_id": sample_id, "paths": [r["path"] for r in mask_groups[sample_id]]}
        for sample_id in sorted(mask_ids - image_ids)
    ]

    image_layouts: Counter[str] = Counter()
    image_shapes: Counter[str] = Counter()
    image_missing_data: list[dict[str, Any]] = []
    image_non_finite: list[dict[str, Any]] = []
    for record in image_records:
        if record["read_error"]:
            continue
        image_layouts[repr(tuple(
            (h["name"], h["has_data"], tuple(h["shape"]) if h["shape"] else None, h["dtype"], h["bitpix"])
            for h in record["hdus"]
        ))] += 1
        for hdu in record["hdus"]:
            if hdu["has_data"] is False:
                image_missing_data.append({"path": record["path"], "hdu_index": hdu["index"]})
            elif hdu["has_data"] is True:
                image_shapes[str(hdu["shape"])] += 1
                if hdu["non_finite_count"]:
                    image_non_finite.append({"path": record["path"], "hdu_index": hdu["index"], "non_finite_count": hdu["non_finite_count"]})

    mask_layouts: Counter[str] = Counter()
    mask_shapes: Counter[str] = Counter()
    mask_missing_data: list[dict[str, Any]] = []
    mask_non_finite: list[dict[str, Any]] = []
    mask_non_integer_like: list[dict[str, Any]] = []
    for record in mask_records:
        if record["read_error"]:
            continue
        mask_layouts[repr(tuple(
            (h["name"], h["has_data"], tuple(h["shape"]) if h["shape"] else None, h["dtype"], h["bitpix"])
            for h in record["hdus"]
        ))] += 1
        for hdu in record["hdus"]:
            if hdu["has_data"] is False:
                mask_missing_data.append({"path": record["path"], "hdu_index": hdu["index"]})
            elif hdu["has_data"] is True:
                mask_shapes[str(hdu["shape"])] += 1
                if hdu["non_finite_count"]:
                    mask_non_finite.append({"path": record["path"], "hdu_index": hdu["index"], "non_finite_count": hdu["non_finite_count"]})
                if hdu.get("integer_like") is False:
                    mask_non_integer_like.append({"path": record["path"], "hdu_index": hdu["index"]})

    file_errors = [
        {"path": record["path"], "error": record["read_error"]}
        for record in image_records + mask_records if record["read_error"]
    ]
    hdu_errors = [error for record in image_records + mask_records for error in record["hdu_errors"]]
    sanitization_issues = [
        issue for record in image_records + mask_records
        for issue in record["numeric_sanitization_issues"]
    ]
    representative_id = "1-100523"
    representative = {
        "sample_id": representative_id,
        "image": next((r for r in image_records if r["sample_id"] == representative_id), None),
        "mask": next((r for r in mask_records if r["sample_id"] == representative_id), None),
    }

    image_hdu_counts = Counter(r["hdu_count"] for r in image_records if r["hdu_count"] is not None)
    mask_hdu_counts = Counter(r["hdu_count"] for r in mask_records if r["hdu_count"] is not None)
    mask_data_hdu_counts = Counter(
        sum(h["has_data"] is True for h in r["hdus"])
        for r in mask_records if not r["read_error"]
    )
    data_quality = {
        "image_missing_data_hdus": image_missing_data,
        "image_non_finite_hdus": image_non_finite,
        "mask_missing_data_hdus": mask_missing_data,
        "mask_non_finite_hdus": mask_non_finite,
        "mask_non_integer_like_hdus": mask_non_integer_like,
        "numeric_sanitization_issues": sanitization_issues,
        "hdu_inspection_errors": hdu_errors,
        "unusual_image_hdu_counts": {str(k): v for k, v in sorted(image_hdu_counts.items()) if k != 5},
        "unusual_mask_data_hdu_counts": {str(k): v for k, v in sorted(mask_data_hdu_counts.items()) if k != 1},
        "nonstandard_image_shapes": {shape: count for shape, count in sorted(image_shapes.items()) if shape != "[208, 208]"},
        "nonstandard_mask_shapes": {shape: count for shape, count in sorted(mask_shapes.items()) if shape != "[208, 208]"},
    }
    dimension_results = _dimension_checks(image_groups, mask_groups, common_ids)
    return {
        "audit_summary": {
            "raw_directory": raw_dir.as_posix(),
            "fits_file_count": len(fits_paths),
            "image_file_count": len(image_paths),
            "mask_file_count": len(mask_paths),
            "unexpected_fits_filenames": unexpected,
            "image_files_successfully_inspected": sum(not r["read_error"] for r in image_records),
            "mask_files_successfully_inspected": sum(not r["read_error"] for r in mask_records),
        },
        "pairing": {
            "matched_pairs": [
                {"sample_id": sid, "image_path": image_groups[sid][0]["path"], "mask_path": mask_groups[sid][0]["path"]}
                for sid in unique_matched_ids
            ],
            "ambiguous_ids": ambiguous_ids,
            "ambiguous_candidate_combinations": ambiguous_combinations,
            "unmatched_images": unmatched_images,
            "unmatched_masks": unmatched_masks,
            "duplicate_image_ids": duplicate_images,
            "duplicate_mask_ids": duplicate_masks,
        },
        "image_statistics": {
            "hdu_count_distribution": {str(k): v for k, v in sorted(image_hdu_counts.items())},
            "layout_distribution": [{"count": v, "layout": k} for k, v in sorted(image_layouts.items())],
            "data_shape_distribution": dict(sorted(image_shapes.items())),
            "files": image_records,
        },
        "mask_statistics": {
            "hdu_count_distribution": {str(k): v for k, v in sorted(mask_hdu_counts.items())},
            "layout_distribution": [{"count": v, "layout": k} for k, v in sorted(mask_layouts.items())],
            "data_shape_distribution": dict(sorted(mask_shapes.items())),
            **_aggregate_masks(mask_records),
            "files": mask_records,
        },
        "dimension_checks": dimension_results,
        "data_quality": data_quality,
        "representative_sample": representative,
        "errors": {"file_errors": file_errors, "hdu_errors": hdu_errors},
    }
