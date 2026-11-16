"""Build a dataset manifest by scanning a root directory for FITS files.

Usage:
    python -m src.data.dataset_index --root dataset/raw --deep

Outputs dataset/metadata/manifest.csv and dataset/metadata/audit_summary.json
(relative to current working directory unless paths overridden).

Read-only with respect to dataset/raw/. Never modifies, moves, or adds files there.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .fits_loader import open_fits_info

IMAGE_PREFIX = "imagedata_"
MASK_PREFIX = "mask_"
FITS_SUFFIX = ".fits"

MANIFEST_COLUMNS = [
    "sample_id",
    "status",
    "image_path",
    "mask_path",
    "image_n_hdus",
    "mask_n_hdus",
    "image_shape",
    "mask_shape",
    "image_hdu_shapes",
    "mask_hdu_shapes",
    "image_nonfinite",
    "mask_nonfinite",
    "mask_raw_values",
    "image_error",
    "mask_error",
]

STATUS_OK = "ok"
STATUS_MISSING_MASK = "missing_mask"
STATUS_MISSING_IMAGE = "missing_image"
STATUS_DUPLICATE = "duplicate"
STATUS_UNREADABLE = "unreadable"
STATUS_SHAPE_MISMATCH = "shape_mismatch"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FITS dataset manifest and audit summary.")
    p.add_argument("--root", type=Path, default=Path("dataset/raw"), help="Root directory to scan for FITS files.")
    p.add_argument("--manifest", type=Path, default=Path("dataset/metadata/manifest.csv"), help="Output manifest CSV path.")
    p.add_argument("--audit", type=Path, default=Path("dataset/metadata/audit_summary.json"), help="Output audit summary JSON path.")
    p.add_argument("--deep", action="store_true", help="Record NaN/non-finite counts and raw mask values.")
    return p.parse_args(argv)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _scan_files(root: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], List[str], int]:
    """Scan root recursively for files.

    Returns (images_by_id, masks_by_id, unrecognized_fits_relpaths, n_non_fits_files).
    A FITS file is recognized if its name starts with IMAGE_PREFIX or MASK_PREFIX (case-sensitive)
    and ends with .fits. ID is text after prefix up to .fits.
    """
    images: Dict[str, List[Path]] = defaultdict(list)
    masks: Dict[str, List[Path]] = defaultdict(list)
    unrecognized_fits: List[str] = []
    n_non_fits = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        lower = name.lower()
        if not lower.endswith(FITS_SUFFIX):
            n_non_fits += 1
            continue
        # recognized by prefix
        if name.startswith(IMAGE_PREFIX):
            sample_id = name[len(IMAGE_PREFIX):-len(FITS_SUFFIX)]
            if sample_id:
                images[sample_id].append(p)
            else:
                unrecognized_fits.append(_relpath(p, root))
        elif name.startswith(MASK_PREFIX):
            sample_id = name[len(MASK_PREFIX):-len(FITS_SUFFIX)]
            if sample_id:
                masks[sample_id].append(p)
            else:
                unrecognized_fits.append(_relpath(p, root))
        else:
            unrecognized_fits.append(_relpath(p, root))
    return images, masks, sorted(unrecognized_fits), n_non_fits


def _shape_str(shape: Optional[Tuple[int, ...]]) -> str:
    if shape is None:
        return ""
    return "x".join(str(int(x)) for x in shape)


def _hdu_shapes_str(shapes: List[Any]) -> str:
    parts = []
    for s in shapes:
        if s is None:
            parts.append("None")
        else:
            parts.append(_shape_str(tuple(s)))
    return ";".join(parts)


def _check_shapes_match(img_info: Dict[str, Any], mask_info: Dict[str, Any]) -> bool:
    """Return True if shapes/HDUs are consistent, False if mismatch detected."""
    if not img_info.get("readable") or not mask_info.get("readable"):
        return True  # unreadable handled separately
    img_shapes = [s for s in img_info.get("hdu_shapes", []) if s is not None]
    mask_shapes = [s for s in mask_info.get("hdu_shapes", []) if s is not None]
    if not img_shapes or not mask_shapes:
        return False
    # image HDUs must all have same shape
    if len(set(tuple(s) for s in img_shapes)) != 1:
        return False
    # mask HDUs (or primary mask shape) must match image spatial shape
    img_shape = tuple(img_shapes[0])
    mask_shape = tuple(mask_shapes[0])
    # compare last two dims if needed, but generally expect exact match
    if img_shape != mask_shape:
        # allow if mask matches spatial dims of image (e.g. image [C,H,W]? unlikely here but be careful)
        # for 2D masks vs 2D image HDUs expect equal
        if len(img_shape) == 2 and len(mask_shape) == 2:
            return False
        if img_shape[-2:] != mask_shape[-2:]:
            return False
    return True


def build_index(root: Path, deep: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(root)
    images, masks, unrecognized_fits, n_non_fits = _scan_files(root)
    all_ids = sorted(set(images.keys()) | set(masks.keys()))

    rows: List[Dict[str, Any]] = []
    status_counts: Counter = Counter()
    image_shapes_counter: Counter = Counter()
    mask_shapes_counter: Counter = Counter()
    image_hdu_counts_counter: Counter = Counter()
    mask_hdu_counts_counter: Counter = Counter()
    union_mask_raw_values = set()
    n_image_files = sum(len(v) for v in images.values())
    n_mask_files = sum(len(v) for v in masks.values())
    n_unreadable_files = 0

    for sid in all_ids:
        img_paths = images.get(sid, [])
        msk_paths = masks.get(sid, [])
        # determine pairings / duplicates
        pair_count = max(len(img_paths), len(msk_paths))
        is_dup = len(img_paths) > 1 or len(msk_paths) > 1
        for i in range(pair_count):
            img_p = img_paths[i] if i < len(img_paths) else None
            msk_p = msk_paths[i] if i < len(msk_paths) else None
            row: Dict[str, Any] = {c: "" for c in MANIFEST_COLUMNS}
            row["sample_id"] = sid
            row["image_path"] = _relpath(img_p, root) if img_p else ""
            row["mask_path"] = _relpath(msk_p, root) if msk_p else ""

            img_info = open_fits_info(img_p, deep=deep) if img_p else None
            msk_info = open_fits_info(msk_p, deep=deep) if msk_p else None

            if img_info:
                row["image_n_hdus"] = img_info.get("n_hdus") if img_info.get("n_hdus") is not None else ""
                row["image_shape"] = _shape_str(img_info.get("image_shape"))
                row["image_hdu_shapes"] = _hdu_shapes_str(img_info.get("hdu_shapes", []))
                row["image_error"] = img_info.get("error") or ""
                if deep and img_info.get("nonfinite_count") is not None:
                    row["image_nonfinite"] = img_info.get("nonfinite_count")
                if img_info.get("readable"):
                    if img_info.get("image_shape") is not None:
                        image_shapes_counter[_shape_str(img_info.get("image_shape"))] += 1
                    if img_info.get("n_hdus") is not None:
                        image_hdu_counts_counter[int(img_info.get("n_hdus"))] += 1
                else:
                    n_unreadable_files += 1
            if msk_info:
                row["mask_n_hdus"] = msk_info.get("n_hdus") if msk_info.get("n_hdus") is not None else ""
                row["mask_shape"] = _shape_str(msk_info.get("image_shape"))
                row["mask_hdu_shapes"] = _hdu_shapes_str(msk_info.get("hdu_shapes", []))
                row["mask_error"] = msk_info.get("error") or ""
                if deep and msk_info.get("nonfinite_count") is not None:
                    row["mask_nonfinite"] = msk_info.get("nonfinite_count")
                if deep and msk_info.get("raw_mask_values") is not None:
                    vals = msk_info.get("raw_mask_values") or []
                    row["mask_raw_values"] = json.dumps(vals)
                    union_mask_raw_values.update(vals)
                if msk_info.get("readable"):
                    if msk_info.get("image_shape") is not None:
                        mask_shapes_counter[_shape_str(msk_info.get("image_shape"))] += 1
                    if msk_info.get("n_hdus") is not None:
                        mask_hdu_counts_counter[int(msk_info.get("n_hdus"))] += 1
                else:
                    n_unreadable_files += 1

            # status determination
            if img_p is None and msk_p is not None:
                status = STATUS_MISSING_IMAGE
            elif img_p is not None and msk_p is None:
                status = STATUS_MISSING_MASK
            elif is_dup:
                status = STATUS_DUPLICATE
            else:
                # check readability
                img_bad = img_info is not None and not img_info.get("readable")
                msk_bad = msk_info is not None and not msk_info.get("readable")
                if img_bad or msk_bad:
                    status = STATUS_UNREADABLE
                elif img_info and msk_info and not _check_shapes_match(img_info, msk_info):
                    status = STATUS_SHAPE_MISMATCH
                else:
                    status = STATUS_OK

            row["status"] = status
            status_counts[status] += 1
            rows.append(row)

    # unique sample IDs count = len(all_ids)
    audit: Dict[str, Any] = {
        "root": str(root),
        "deep": bool(deep),
        "n_image_files": n_image_files,
        "n_mask_files": n_mask_files,
        "n_total_fits_files": n_image_files + n_mask_files + len(unrecognized_fits),
        "n_unrecognized_fits_files": len(unrecognized_fits),
        "unrecognized_fits_files": unrecognized_fits,
        "n_non_fits_files": n_non_fits,
        "n_unique_sample_ids": len(all_ids),
        "n_manifest_rows": len(rows),
        "status_counts": dict(status_counts),
        "n_unreadable_files": n_unreadable_files,
        "image_shape_distribution": dict(image_shapes_counter),
        "mask_shape_distribution": dict(mask_shapes_counter),
        "image_hdu_count_distribution": {str(k): v for k, v in sorted(image_hdu_counts_counter.items())},
        "mask_hdu_count_distribution": {str(k): v for k, v in sorted(mask_hdu_counts_counter.items())},
    }
    if deep:
        audit["union_mask_raw_values"] = sorted(int(v) for v in union_mask_raw_values)
        # counts of images with nonfinite > 0 from rows
        n_img_nonfinite = 0
        n_msk_nonfinite = 0
        for r in rows:
            try:
                if r.get("image_nonfinite") not in ("", None) and int(r.get("image_nonfinite")) > 0:
                    n_img_nonfinite += 1
            except Exception:
                pass
            try:
                if r.get("mask_nonfinite") not in ("", None) and int(r.get("mask_nonfinite")) > 0:
                    n_msk_nonfinite += 1
            except Exception:
                pass
        audit["n_image_rows_with_nonfinite"] = n_img_nonfinite
        audit["n_mask_rows_with_nonfinite"] = n_msk_nonfinite

    return rows, audit


def write_manifest(rows: List[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})


def write_audit(audit: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    rows, audit = build_index(args.root, deep=args.deep)
    write_manifest(rows, args.manifest)
    write_audit(audit, args.audit)
    elapsed = time.time() - t0
    audit["elapsed_seconds"] = round(elapsed, 3)
    # rewrite audit with elapsed
    write_audit(audit, args.audit)
    # print summary
    print(json.dumps({
        "root": str(args.root),
        "deep": args.deep,
        "n_manifest_rows": len(rows),
        "n_unique_sample_ids": audit.get("n_unique_sample_ids"),
        "status_counts": audit.get("status_counts"),
        "n_image_files": audit.get("n_image_files"),
        "n_mask_files": audit.get("n_mask_files"),
        "n_total_fits_files": audit.get("n_total_fits_files"),
        "n_unrecognized_fits_files": audit.get("n_unrecognized_fits_files"),
        "n_non_fits_files": audit.get("n_non_fits_files"),
        "elapsed_seconds": round(elapsed, 3),
        "manifest": str(args.manifest),
        "audit": str(args.audit),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
