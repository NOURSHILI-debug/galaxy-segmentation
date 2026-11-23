"""Deterministic sample-level indexing, pairing, and manifest generation.

This module discovers FITS files in a raw dataset directory, extracts sample IDs
from filenames, and pairs image and mask files strictly by sample ID.  It does
not use directory ordering, glob adjacency, or filename similarity.

The manifest is sample-level: one row per uniquely paired image-mask sample.
Patch extraction, train/validation/test splitting, normalization, and all other
preprocessing are deliberately out of scope.

Only FITS headers are inspected here (HDU counts and ``NAXIS*`` values); pixel
arrays are never read.  Pixel-level facts such as non-finite counts and mask
label frequencies are handled by :mod:`src.data.dataset_audit` using the
existing verified audit artifact.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from astropy.io import fits

__all__ = [
    "DATASET_INDEX_SCHEMA_VERSION",
    "IMAGE_FILENAME_PATTERN",
    "MASK_FILENAME_PATTERN",
    "MANIFEST_COLUMNS",
    "DatasetIndexError",
    "build_manifest",
    "read_manifest",
    "repo_root_default",
    "scan_dataset",
    "validate_manifest",
    "write_manifest",
]

DATASET_INDEX_SCHEMA_VERSION = "d2.1"

IMAGE_FILENAME_PATTERN = re.compile(r"^imagedata_(.+)\.fits(?:\.gz)?$", re.IGNORECASE)
MASK_FILENAME_PATTERN = re.compile(r"^mask_(.+)\.fits(?:\.gz)?$", re.IGNORECASE)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "image_path",
    "mask_path",
    "height",
    "width",
)

#: Canonical names for the FITS ``BITPIX`` storage codes.
BITPIX_TO_DTYPE: dict[int, str] = {
    8: "uint8",
    16: "int16",
    32: "int32",
    64: "int64",
    -32: "float32",
    -64: "float64",
}


class DatasetIndexError(ValueError):
    """Raised when a dataset directory or manifest cannot be indexed safely."""


def repo_root_default() -> Path:
    """Return the repository root inferred from this module's location."""
    return Path(__file__).resolve().parents[2]


def _as_directory(raw_dir: str | Path) -> Path:
    """Validate and return a dataset directory."""
    path = Path(raw_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    return path


def _discover_files(raw_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return sorted FITS paths and sorted ignored non-FITS files."""
    fits_paths: list[Path] = []
    ignored_paths: list[Path] = []

    def fail_on_scan_error(error: OSError) -> None:
        raise error

    for directory, _, filenames in os.walk(raw_dir, onerror=fail_on_scan_error):
        for filename in filenames:
            path = Path(directory) / filename
            if filename.lower().endswith((".fits", ".fits.gz")):
                fits_paths.append(path)
            else:
                ignored_paths.append(path)

    fits_paths.sort(key=lambda path: path.as_posix().lower())
    ignored_paths.sort(key=lambda path: path.as_posix().lower())
    return fits_paths, ignored_paths


def _inspect_fits_header(path: Path) -> dict[str, Any]:
    """Inspect one FITS file's HDUs without reading pixel arrays."""
    record: dict[str, Any] = {
        "path": path,
        "hdu_count": None,
        "data_hdu_count": None,
        "hdus": [],
        "read_error": None,
    }
    memmap = not path.name.lower().endswith(".gz")
    try:
        with fits.open(path, mode="readonly", memmap=memmap) as hdul:
            record["hdu_count"] = len(hdul)
            for index, hdu in enumerate(hdul):
                header = hdu.header
                naxis = int(header.get("NAXIS", 0) or 0)
                shape = (
                    tuple(int(header[f"NAXIS{axis}"]) for axis in range(naxis, 0, -1))
                    if naxis > 0
                    else ()
                )
                bitpix = header.get("BITPIX")
                record["hdus"].append(
                    {
                        "index": index,
                        "naxis": naxis,
                        "shape": shape,
                        "dtype": BITPIX_TO_DTYPE.get(int(bitpix)) if bitpix is not None else None,
                        "bitpix": int(bitpix) if bitpix is not None else None,
                        "has_data": naxis > 0,
                        "scaled": ("BSCALE" in header or "BZERO" in header),
                    }
                )
            record["data_hdu_count"] = sum(hdu["has_data"] for hdu in record["hdus"])
    except Exception as exc:  # astropy read/verification failure at the boundary
        record["read_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _classify_files(
    fits_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify FITS paths into image, mask, and malformed records."""
    images: list[dict[str, Any]] = []
    masks: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []

    for path in fits_paths:
        image_match = IMAGE_FILENAME_PATTERN.match(path.name)
        mask_match = MASK_FILENAME_PATTERN.match(path.name)
        if image_match:
            kind = "image"
            sample_id = image_match.group(1)
        elif mask_match:
            kind = "mask"
            sample_id = mask_match.group(1)
        else:
            malformed.append(
                {
                    "path": path,
                    "reason": (
                        "filename does not match imagedata_<sample_id>.fits "
                        "or mask_<sample_id>.fits"
                    ),
                }
            )
            continue

        record = _inspect_fits_header(path)
        record["sample_id"] = sample_id
        record["kind"] = kind
        (images if kind == "image" else masks).append(record)

    images.sort(key=lambda record: record["path"].as_posix().lower())
    masks.sort(key=lambda record: record["path"].as_posix().lower())
    malformed.sort(key=lambda record: record["path"].as_posix().lower())
    return images, masks, malformed


def scan_dataset(raw_dir: str | Path) -> dict[str, Any]:
    """Discover and header-inspect every FITS file under ``raw_dir``.

    Returns a structured index with image and mask records keyed by sample ID.
    Pairing is strictly ID-based; ambiguous, duplicate, and unmatched IDs are
    reported rather than silently resolved.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
        NotADirectoryError: If ``raw_dir`` is not a directory.
        PermissionError: If the directory tree cannot be walked.
    """
    raw_path = _as_directory(raw_dir)
    try:
        fits_paths, ignored_paths = _discover_files(raw_path)
    except OSError as exc:
        raise PermissionError(f"Could not scan dataset directory {raw_path}: {exc}") from exc

    image_records, mask_records, malformed = _classify_files(fits_paths)

    images_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    masks_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_records:
        images_by_id[record["sample_id"]].append(record)
    for record in mask_records:
        masks_by_id[record["sample_id"]].append(record)

    for records in images_by_id.values():
        records.sort(key=lambda record: record["path"].as_posix().lower())
    for records in masks_by_id.values():
        records.sort(key=lambda record: record["path"].as_posix().lower())

    duplicate_image_ids = {
        sample_id: list(records)
        for sample_id, records in sorted(images_by_id.items())
        if len(records) > 1
    }
    duplicate_mask_ids = {
        sample_id: list(records)
        for sample_id, records in sorted(masks_by_id.items())
        if len(records) > 1
    }

    image_ids = set(images_by_id)
    mask_ids = set(masks_by_id)
    common_ids = image_ids & mask_ids

    ambiguous_ids: dict[str, dict[str, Any]] = {}
    matched_population: list[dict[str, Any]] = []
    for sample_id in sorted(common_ids):
        image_group = images_by_id[sample_id]
        mask_group = masks_by_id[sample_id]
        if len(image_group) != 1 or len(mask_group) != 1:
            ambiguous_ids[sample_id] = {
                "image_paths": [record["path"] for record in image_group],
                "mask_paths": [record["path"] for record in mask_group],
            }
        else:
            matched_population.append(
                {
                    "sample_id": sample_id,
                    "image": image_group[0],
                    "mask": mask_group[0],
                }
            )

    unmatched_image_ids = {
        sample_id: list(images_by_id[sample_id])
        for sample_id in sorted(image_ids - mask_ids)
    }
    unmatched_mask_ids = {
        sample_id: list(masks_by_id[sample_id])
        for sample_id in sorted(mask_ids - image_ids)
    }

    return {
        "schema_version": DATASET_INDEX_SCHEMA_VERSION,
        "raw_directory": raw_path,
        "image_records": image_records,
        "mask_records": mask_records,
        "malformed_files": malformed,
        "ignored_non_fits_files": [path for path in ignored_paths],
        "images_by_id": dict(images_by_id),
        "masks_by_id": dict(masks_by_id),
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_mask_ids": duplicate_mask_ids,
        "ambiguous_ids": ambiguous_ids,
        "unmatched_image_ids": unmatched_image_ids,
        "unmatched_mask_ids": unmatched_mask_ids,
        "matched_population": matched_population,
    }


def _relative_posix(path: Path, repo_root: Path) -> str:
    """Return a repository-relative POSIX path or raise a clear error."""
    try:
        relative = path.resolve().relative_to(repo_root)
    except ValueError as exc:
        raise DatasetIndexError(
            f"Path {path} is outside the repository root {repo_root}; "
            "manifest paths must be repository-relative."
        ) from exc
    return relative.as_posix()


def _shape_problems(record: dict[str, Any], role: str) -> tuple[list[dict[str, str]], tuple[int, int] | None]:
    """Validate one file's data-HDU geometry for manifest inclusion."""
    problems: list[dict[str, str]] = []
    if record["read_error"]:
        problems.append(
            {"code": f"{role}_read_error", "message": f"{role} read error: {record['read_error']}"}
        )
        return problems, None

    data_hdus = [hdu for hdu in record["hdus"] if hdu["has_data"]]
    if not data_hdus:
        problems.append(
            {"code": f"{role}_no_data_hdu", "message": f"{role} contains no data-bearing HDU"}
        )
        return problems, None

    non_2d = [hdu for hdu in data_hdus if hdu["naxis"] != 2]
    if non_2d:
        problems.append(
            {
                "code": f"{role}_non_2d_hdu",
                "message": (
                    f"{role} HDU(s) {[hdu['index'] for hdu in non_2d]} are not 2D "
                    f"(shapes {[hdu['shape'] for hdu in non_2d]})"
                ),
            }
        )
        return problems, None

    shapes = {tuple(hdu["shape"]) for hdu in data_hdus}
    if len(shapes) != 1:
        problems.append(
            {
                "code": f"{role}_inconsistent_hdu_shapes",
                "message": f"{role} data HDUs have inconsistent shapes: {sorted(shapes)}",
            }
        )
        return problems, None

    shape = shapes.pop()
    if shape[0] < 1 or shape[1] < 1:
        problems.append(
            {"code": f"{role}_empty_shape", "message": f"{role} has empty spatial shape {shape}"}
        )
        return problems, None
    return problems, (int(shape[0]), int(shape[1]))


def _pair_problems(
    image: dict[str, Any],
    mask: dict[str, Any],
) -> tuple[list[dict[str, str]], tuple[int, int] | None]:
    """Return structured validity problems and the common spatial shape."""
    problems, image_shape = _shape_problems(image, "image")
    mask_problems, mask_shape = _shape_problems(mask, "mask")
    problems.extend(mask_problems)

    if len([hdu for hdu in mask["hdus"] if hdu["has_data"]]) != 1 and not mask_problems:
        problems.append(
            {
                "code": "mask_multiple_data_hdus",
                "message": "mask must contain exactly one data-bearing primary HDU",
            }
        )

    if image_shape is not None and mask_shape is not None and image_shape != mask_shape:
        problems.append(
            {
                "code": "dimension_mismatch",
                "message": f"image shape {image_shape} != mask shape {mask_shape}",
            }
        )
    return problems, image_shape


def build_manifest(
    raw_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic sample-level manifest from the dataset index.

    Args:
        raw_dir: Dataset directory containing image and mask FITS files.
        repo_root: Repository root used to make paths relative.  Defaults to the
            root inferred from this module.
        index: Optional result of :func:`scan_dataset` to avoid rescanning.

    Returns:
        A dictionary with ``columns``, ``rows`` (sorted by ``sample_id``),
        ``row_count``, ``unique_pair_count``, ``excluded_pairs``, and
        ``missing_referenced_files``.  Each row contains ``sample_id``,
        ``image_path``, ``mask_path``, ``height``, and ``width``; paths are
        repository-relative POSIX strings.
    """
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_default()
    index = scan_dataset(raw_dir) if index is None else index
    _relative_posix(index["raw_directory"], root)  # validate relative location

    rows: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []

    for pair in index["matched_population"]:
        sample_id = pair["sample_id"]
        image = pair["image"]
        mask = pair["mask"]
        problems, shape = _pair_problems(image, mask)

        if image.get("sample_id") != sample_id or mask.get("sample_id") != sample_id:
            problems.append(
                {
                    "code": "sample_id_mismatch",
                    "message": (
                        f"pair ID {sample_id!r} does not match extracted IDs "
                        f"image={image.get('sample_id')!r}, mask={mask.get('sample_id')!r}"
                    ),
                }
            )

        image_path = _relative_posix(image["path"], root)
        mask_path = _relative_posix(mask["path"], root)

        if not image["path"].is_file():
            problems.append(
                {"code": "image_file_missing", "message": f"image file does not exist: {image_path}"}
            )
        if not mask["path"].is_file():
            problems.append(
                {"code": "mask_file_missing", "message": f"mask file does not exist: {mask_path}"}
            )

        if problems or shape is None:
            excluded_pairs.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "reasons": problems,
                }
            )
            continue

        height, width = shape
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "height": height,
                "width": width,
            }
        )

    rows.sort(key=lambda row: row["sample_id"])
    excluded_pairs.sort(key=lambda item: item["sample_id"])

    return {
        "columns": list(MANIFEST_COLUMNS),
        "rows": rows,
        "row_count": len(rows),
        "unique_pair_count": len(index["matched_population"]),
        "excluded_pairs": excluded_pairs,
        "excluded_pair_count": len(excluded_pairs),
        "missing_referenced_files": validate_manifest(rows, repo_root=root),
    }


def validate_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Check manifest rows for missing files, duplicate IDs, and path policy."""
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_default()
    problems: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            problems.append({"sample_id": sample_id, "reason": "empty_sample_id"})
        elif sample_id in seen_ids:
            problems.append({"sample_id": sample_id, "reason": "duplicate_sample_id"})
        seen_ids.add(sample_id)

        for field in ("image_path", "mask_path"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                problems.append(
                    {"sample_id": sample_id, "field": field, "reason": "missing_path_value"}
                )
                continue
            path = Path(value)
            if path.is_absolute():
                problems.append(
                    {"sample_id": sample_id, "field": field, "path": value, "reason": "absolute_path"}
                )
                continue
            if "\\" in value:
                problems.append(
                    {"sample_id": sample_id, "field": field, "path": value, "reason": "non_posix_path"}
                )
            if not (root / path).is_file():
                problems.append(
                    {"sample_id": sample_id, "field": field, "path": value, "reason": "file_missing"}
                )

        for field in ("height", "width"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append(
                    {"sample_id": sample_id, "field": field, "reason": "invalid_dimension"}
                )

    return problems


def write_manifest(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Write manifest rows as deterministic UTF-8 CSV with ``\\n`` endings."""
    row_list = list(rows)
    row_list.sort(key=lambda row: str(row.get("sample_id", "")))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(MANIFEST_COLUMNS),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in row_list:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                }
            )
    return output


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest CSV and validate its header and value types."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != MANIFEST_COLUMNS:
            raise DatasetIndexError(
                f"Unexpected manifest columns {fieldnames}; expected {MANIFEST_COLUMNS}."
            )
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "sample_id": raw["sample_id"],
                        "image_path": raw["image_path"],
                        "mask_path": raw["mask_path"],
                        "height": int(raw["height"]),
                        "width": int(raw["width"]),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetIndexError(
                    f"Invalid manifest row at line {line_number}: {raw!r} ({exc})."
                ) from exc
    return rows
