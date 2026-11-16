"""Image (FITS) audit for dataset/raw (read-only).

Computes per-sample, per-HDU statistics on RAW image values and writes:
- image_stats.csv
- image_audit_summary.json

Usage:
    python -m src.data.dataset_audit --manifest dataset/metadata/manifest.csv --root dataset/raw --out-dir dataset/metadata [--limit N]

Deterministic and rerunnable (no timestamps). Processes samples in sorted sample_id order.
Never modifies dataset/raw/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

# Columns for per-(sample_id, hdu_index) rows
IMAGE_STATS_COLUMNS = [
    "sample_id",
    "hdu_index",
    "hdu_type",
    "dtype",
    "BITPIX",
    "shape",
    "n_pixels",
    "n_nan",
    "n_posinf",
    "n_neginf",
    "n_finite",
    # finite-only stats (empty if none finite)
    "min",
    "max",
    "mean",
    "std",
    "median",
    "sigma_mad",
    "p1",
    "p99",
    "p99_5",
    "fraction_nonpositive",
    # header values
    "EXPTIME",
    "CTYPE1",
    "CTYPE2",
    "CDELT1",
    "CDELT2",
    "FILTER_KEYS",
]

FILTER_KEY_SUBSTRINGS = ("FILTER", "BAND", "FILT")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit image FITS files (raw values).")
    p.add_argument("--manifest", type=Path, default=Path("dataset/metadata/manifest.csv"))
    p.add_argument("--root", type=Path, default=Path("dataset/raw"), help="Root directory for manifest-relative paths.")
    p.add_argument("--out-dir", type=Path, default=Path("dataset/metadata"))
    p.add_argument("--limit", type=int, default=None, help="Audit only first N samples (sorted by sample_id).")
    return p.parse_args(argv)


def _fmt_float(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        if isinstance(v, (float, np.floating)) and (math.isnan(float(v)) or math.isinf(float(v))):
            return str(v)
        # use repr-like but stable
        fv = float(v)
        # avoid noisy binary float repr; 10 significant digits stable enough
        return format(fv, ".10g")
    except Exception:
        return str(v)


def _shape_str(shape: Optional[Tuple[int, ...]]) -> str:
    if shape is None:
        return ""
    return "x".join(str(int(x)) for x in shape)


def _compute_finite_stats(arr: np.ndarray) -> Dict[str, Any]:
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]
    n_finite = int(finite_vals.size)
    out: Dict[str, Any] = {k: "" for k in ["min", "max", "mean", "std", "median", "sigma_mad", "p1", "p99", "p99_5", "fraction_nonpositive"]}
    if n_finite == 0:
        return out
    # compute on finite values only
    v = finite_vals.astype(np.float64, copy=False)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    mean = float(np.mean(v))
    std = float(np.std(v))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma_mad = float(1.4826 * mad)
    p1 = float(np.percentile(v, 1))
    p99 = float(np.percentile(v, 99))
    p99_5 = float(np.percentile(v, 99.5))
    frac_nonpos = float(np.count_nonzero(v <= 0) / n_finite)
    out.update({
        "min": _fmt_float(vmin),
        "max": _fmt_float(vmax),
        "mean": _fmt_float(mean),
        "std": _fmt_float(std),
        "median": _fmt_float(med),
        "sigma_mad": _fmt_float(sigma_mad),
        "p1": _fmt_float(p1),
        "p99": _fmt_float(p99),
        "p99_5": _fmt_float(p99_5),
        "fraction_nonpositive": _fmt_float(frac_nonpos),
    })
    return out


def _filter_keys_from_header(header: fits.Header) -> str:
    items = []
    for k in header.keys():
        ku = str(k).upper()
        if any(s in ku for s in FILTER_KEY_SUBSTRINGS):
            try:
                val = header[k]
            except Exception:
                val = "<unreadable>"
            items.append(f"{k}={val}")
    # deterministic order
    items = sorted(items)
    return "; ".join(items)


def _header_val(header: fits.Header, key: str) -> str:
    if key in header:
        try:
            return str(header[key])
        except Exception:
            return ""
    return ""


def _nan_border_summary(mask_nan: np.ndarray) -> Optional[Dict[str, int]]:
    """Return counts of NaN pixels by distance to nearest border (Chebyshev on 2D)."""
    if mask_nan.ndim != 2:
        return None
    n_nan = int(np.count_nonzero(mask_nan))
    if n_nan == 0:
        return {"n_nan": 0, "by_border_distance": {}}
    h, w = mask_nan.shape
    ys, xs = np.where(mask_nan)
    # distance to nearest edge
    d_top = ys
    d_bottom = (h - 1 - ys)
    d_left = xs
    d_right = (w - 1 - xs)
    d = np.minimum(np.minimum(d_top, d_bottom), np.minimum(d_left, d_right))
    counts: Dict[str, int] = {}
    for val in d.tolist():
        k = str(int(val))
        counts[k] = counts.get(k, 0) + 1
    # sort keys numerically in output stage; keep as dict here
    return {"n_nan": n_nan, "by_border_distance": counts}


def audit_image_file(path: Path, sample_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """Audit one image FITS file.

    Returns (rows, file_meta, error). file_meta includes per-hdu info for aggregation.
    """
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "sample_id": sample_id,
        "path": str(path),
        "readable": False,
        "n_hdus": 0,
        "has_any_nonfinite": False,
        "hdus": [],
    }
    try:
        with fits.open(path, memmap=False) as hdul:
            meta["readable"] = True
            meta["n_hdus"] = len(hdul)
            for i, hdu in enumerate(hdul):
                data = hdu.data
                header = hdu.header
                hdu_type = hdu.__class__.__name__
                bitpix = header.get("BITPIX", "")
                # header keys for census (strings)
                header_keys = [str(k) for k in header.keys()]
                filter_items = []
                for k in header.keys():
                    ku = str(k).upper()
                    if any(s in ku for s in FILTER_KEY_SUBSTRINGS):
                        try:
                            filter_items.append((str(k), header[k]))
                        except Exception:
                            filter_items.append((str(k), "<unreadable>"))

                base = {
                    "sample_id": sample_id,
                    "hdu_index": i,
                    "hdu_type": hdu_type,
                    "dtype": "",
                    "BITPIX": str(bitpix),
                    "shape": "",
                    "n_pixels": "",
                    "n_nan": "",
                    "n_posinf": "",
                    "n_neginf": "",
                    "n_finite": "",
                    "EXPTIME": _header_val(header, "EXPTIME"),
                    "CTYPE1": _header_val(header, "CTYPE1"),
                    "CTYPE2": _header_val(header, "CTYPE2"),
                    "CDELT1": _header_val(header, "CDELT1"),
                    "CDELT2": _header_val(header, "CDELT2"),
                    "FILTER_KEYS": _filter_keys_from_header(header),
                }
                for k in ["min", "max", "mean", "std", "median", "sigma_mad", "p1", "p99", "p99_5", "fraction_nonpositive"]:
                    base[k] = ""

                hmeta: Dict[str, Any] = {
                    "hdu_index": i,
                    "hdu_type": hdu_type,
                    "dtype": None,
                    "BITPIX": bitpix,
                    "shape": None,
                    "n_nan": 0,
                    "n_posinf": 0,
                    "n_neginf": 0,
                    "n_pixels": 0,
                    "all_nan": False,
                    "constant": False,
                    "all_zero": False,
                    "header_keys": header_keys,
                    "filter_items": [(k, str(v)) for k, v in filter_items],
                    "nan_border": None,
                    "finite_stats_numeric": None,
                }

                if data is None:
                    rows.append(base)
                    meta["hdus"].append(hmeta)
                    continue

                arr = np.asarray(data)
                shape = tuple(int(x) for x in arr.shape)
                n_pixels = int(arr.size)
                n_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
                n_posinf = int(np.isposinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
                n_neginf = int(np.isneginf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
                finite_mask = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite_mask))

                base["dtype"] = str(arr.dtype)
                base["shape"] = _shape_str(shape)
                base["n_pixels"] = str(n_pixels)
                base["n_nan"] = str(n_nan)
                base["n_posinf"] = str(n_posinf)
                base["n_neginf"] = str(n_neginf)
                base["n_finite"] = str(n_finite)

                finite_stats = _compute_finite_stats(arr)
                base.update(finite_stats)

                if n_nan + n_posinf + n_neginf > 0:
                    meta["has_any_nonfinite"] = True

                # constant/all-zero/all-nan flags (on raw array)
                all_nan = (n_nan == n_pixels and n_pixels > 0)
                all_zero = False
                constant = False
                if n_finite > 0:
                    fv = arr[finite_mask]
                    try:
                        vmin = fv.min()
                        vmax = fv.max()
                        constant = bool(vmin == vmax)
                        all_zero = bool(constant and float(vmin) == 0.0)
                    except Exception:
                        constant = False
                        all_zero = False
                elif n_pixels > 0:
                    # no finite values; not "constant" in finite sense
                    constant = False
                    all_zero = False

                # spatial NaN summary (2D only meaningful)
                nan_border = None
                if n_nan > 0 and arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
                    nan_border = _nan_border_summary(np.isnan(arr))

                # numeric finite stats for aggregation
                numeric_stats = None
                if n_finite > 0:
                    v = arr[finite_mask].astype(np.float64, copy=False)
                    med = float(np.median(v))
                    mad = float(np.median(np.abs(v - med)))
                    numeric_stats = {
                        "median": med,
                        "sigma_mad": float(1.4826 * mad),
                        "max": float(np.max(v)),
                        "fraction_nonpositive": float(np.count_nonzero(v <= 0) / n_finite),
                    }

                hmeta.update({
                    "dtype": str(arr.dtype),
                    "shape": shape,
                    "n_nan": n_nan,
                    "n_posinf": n_posinf,
                    "n_neginf": n_neginf,
                    "n_pixels": n_pixels,
                    "all_nan": all_nan,
                    "constant": constant,
                    "all_zero": all_zero,
                    "nan_border": nan_border,
                    "finite_stats_numeric": numeric_stats,
                })

                rows.append(base)
                meta["hdus"].append(hmeta)
        return rows, meta, None
    except Exception as e:  # noqa: BLE001
        return rows, meta, f"{type(e).__name__}: {e}"


def _percentiles_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"min": None, "p5": None, "median": None, "p95": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def build_summary(all_file_metas: List[Dict[str, Any]], n_manifest_samples: int, errors: List[Dict[str, str]]) -> Dict[str, Any]:
    dtype_counter: Counter = Counter()
    bitpix_counter: Counter = Counter()
    shape_counter: Counter = Counter()

    n_files_with_any_nonfinite = 0
    n_hdus_with_nan = 0
    total_nan_pixels = 0
    total_posinf_pixels = 0
    total_neginf_pixels = 0
    n_all_nan_hdus = 0
    n_constant_hdus = 0
    n_all_zero_hdus = 0

    per_hdu_nan_file_count: Dict[int, int] = defaultdict(int)
    per_hdu_max_nan_fraction: Dict[int, float] = defaultdict(float)
    per_hdu_stats_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # header census
    per_hdu_key_sets: Dict[int, List[set]] = defaultdict(list)
    per_hdu_key_value_sets: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    per_hdu_filter_values: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    # spatial NaN by border distance per hdu
    per_hdu_border_counts: Dict[int, Counter] = defaultdict(Counter)
    per_hdu_nan_pixels_total: Dict[int, int] = defaultdict(int)

    for fm in all_file_metas:
        if fm.get("has_any_nonfinite"):
            n_files_with_any_nonfinite += 1
        for h in fm.get("hdus", []):
            idx = int(h.get("hdu_index", -1))
            if h.get("dtype") is not None:
                dtype_counter[str(h.get("dtype"))] += 1
            if h.get("BITPIX") is not None and h.get("BITPIX") != "":
                bitpix_counter[str(h.get("BITPIX"))] += 1
            if h.get("shape") is not None:
                shape_counter[_shape_str(tuple(h.get("shape")))] += 1

            n_nan = int(h.get("n_nan", 0) or 0)
            n_posinf = int(h.get("n_posinf", 0) or 0)
            n_neginf = int(h.get("n_neginf", 0) or 0)
            n_pix = int(h.get("n_pixels", 0) or 0)
            if n_nan > 0:
                n_hdus_with_nan += 1
                per_hdu_nan_file_count[idx] += 1
                frac = (n_nan / n_pix) if n_pix else 0.0
                if frac > per_hdu_max_nan_fraction[idx]:
                    per_hdu_max_nan_fraction[idx] = frac
            total_nan_pixels += n_nan
            total_posinf_pixels += n_posinf
            total_neginf_pixels += n_neginf
            if h.get("all_nan"):
                n_all_nan_hdus += 1
            if h.get("constant"):
                n_constant_hdus += 1
            if h.get("all_zero"):
                n_all_zero_hdus += 1

            fs = h.get("finite_stats_numeric")
            if fs:
                for k in ("median", "sigma_mad", "max", "fraction_nonpositive"):
                    if fs.get(k) is not None and np.isfinite(fs.get(k)):
                        per_hdu_stats_values[idx][k].append(float(fs[k]))

            # header census
            keys = h.get("header_keys") or []
            per_hdu_key_sets[idx].append(set(keys))
            # not storing all values for all keys to keep small; track variability via representative? but requirement states "which keys vary"
            # We'll approximate key variability by checking presence; for values of filter/band-like keys we store values.
            for fk, fv in (h.get("filter_items") or []):
                per_hdu_filter_values[idx][fk].add(str(fv))

            nb = h.get("nan_border")
            if nb and nb.get("n_nan", 0) > 0:
                per_hdu_nan_pixels_total[idx] += int(nb.get("n_nan", 0))
                for dk, dv in (nb.get("by_border_distance") or {}).items():
                    per_hdu_border_counts[idx][str(dk)] += int(dv)

    # header census aggregation per hdu_index
    header_census: Dict[str, Any] = {"per_hdu_index": {}}
    # global (across all hdus of all images) key presence
    all_key_sets = []
    for idx, sets_list in per_hdu_key_sets.items():
        all_key_sets.extend(sets_list)
    if all_key_sets:
        keys_in_every = set.intersection(*all_key_sets) if all_key_sets else set()
        keys_in_any = set.union(*all_key_sets) if all_key_sets else set()
        header_census["keys_in_every_hdu_of_every_image"] = sorted(keys_in_every)
        header_census["keys_in_some_but_not_all_hdus"] = sorted(keys_in_any - keys_in_every)
        header_census["n_distinct_keys_seen"] = len(keys_in_any)
    else:
        header_census["keys_in_every_hdu_of_every_image"] = []
        header_census["keys_in_some_but_not_all_hdus"] = []
        header_census["n_distinct_keys_seen"] = 0

    for idx in sorted(per_hdu_key_sets.keys()):
        sets_list = per_hdu_key_sets[idx]
        if sets_list:
            inter = set.intersection(*sets_list)
            uni = set.union(*sets_list)
            vary = sorted(uni - inter)
            always = sorted(inter)
        else:
            vary = []
            always = []
        filt_vals = {}
        for fk, vals in sorted(per_hdu_filter_values[idx].items()):
            filt_vals[fk] = sorted(vals)
        header_census["per_hdu_index"][str(idx)] = {
            "keys_in_every_hdu": always,
            "keys_that_vary_across_images": vary,
            "filter_band_like_keys_values": filt_vals,
        }

    # per-hdu stats distributions
    per_hdu_distributions = {}
    for idx in sorted(set(list(per_hdu_stats_values.keys()) + list(per_hdu_nan_file_count.keys()) + list(per_hdu_nan_pixels_total.keys()))):
        stat_dists = {}
        for stat_name in ("median", "sigma_mad", "max", "fraction_nonpositive"):
            vals = per_hdu_stats_values.get(idx, {}).get(stat_name, [])
            stat_dists[stat_name] = _percentiles_summary(vals)
        per_hdu_distributions[str(idx)] = {
            "n_hdus_with_nan": int(per_hdu_nan_file_count.get(idx, 0)),
            "max_nan_fraction": float(per_hdu_max_nan_fraction.get(idx, 0.0)),
            "total_nan_pixels": int(per_hdu_nan_pixels_total.get(idx, 0)),
            "distributions_across_images": stat_dists,
        }

    # spatial NaN summary per hdu (sorted distances)
    spatial_nan = {"per_hdu_index": {}}
    for idx in sorted(per_hdu_border_counts.keys()):
        counts = per_hdu_border_counts[idx]
        by_dist = {str(k): int(counts[k]) for k in sorted(counts.keys(), key=lambda x: int(x))}
        total = int(per_hdu_nan_pixels_total.get(idx, 0))
        border0 = int(counts.get("0", 0))
        border_le1 = int(sum(v for k, v in counts.items() if int(k) <= 1))
        spatial_nan["per_hdu_index"][str(idx)] = {
            "total_nan_pixels": total,
            "nan_at_border_distance_0": border0,
            "nan_at_border_distance_le_1": border_le1,
            "fraction_at_border_distance_0": (border0 / total) if total else 0.0,
            "fraction_at_border_distance_le_1": (border_le1 / total) if total else 0.0,
            "by_border_distance": by_dist,
        }

    summary = {
        "n_manifest_samples_requested": n_manifest_samples,
        "n_files_audited": len(all_file_metas),
        "n_read_errors": len(errors),
        "read_errors": errors,
        "dtype_distribution": dict(sorted(dtype_counter.items())),
        "BITPIX_distribution": {k: int(v) for k, v in sorted(bitpix_counter.items(), key=lambda x: str(x[0]))},
        "shape_distribution": dict(sorted(shape_counter.items())),
        "nonfinite_summary": {
            "n_image_files_with_any_nonfinite_pixel": int(n_files_with_any_nonfinite),
            "n_hdus_containing_nan": int(n_hdus_with_nan),
            "total_nan_pixels": int(total_nan_pixels),
            "total_posinf_pixels": int(total_posinf_pixels),
            "total_neginf_pixels": int(total_neginf_pixels),
            "n_all_nan_hdus": int(n_all_nan_hdus),
            "n_constant_hdus": int(n_constant_hdus),
            "n_all_zero_hdus": int(n_all_zero_hdus),
        },
        "per_hdu_index": per_hdu_distributions,
        "header_census": header_census,
        "spatial_nan_summary": spatial_nan,
    }
    return summary


def run_audit(manifest_path: Path, root: Path, out_dir: Path, limit: Optional[int] = None) -> Tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    # deterministic: sorted by sample_id
    manifest_rows.sort(key=lambda r: str(r.get("sample_id", "")))
    if limit is not None:
        manifest_rows = manifest_rows[: int(limit)]

    stats_path = out_dir / "image_stats.csv"
    summary_path = out_dir / "image_audit_summary.json"

    all_file_metas: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    with stats_path.open("w", newline="", encoding="utf-8") as fstats:
        writer = csv.DictWriter(fstats, fieldnames=IMAGE_STATS_COLUMNS)
        writer.writeheader()
        for row in manifest_rows:
            sample_id = str(row.get("sample_id", ""))
            rel = str(row.get("image_path", ""))
            if not rel:
                errors.append({"sample_id": sample_id, "error": "missing image_path in manifest"})
                continue
            img_path = root / rel
            per_hdu_rows, file_meta, err = audit_image_file(img_path, sample_id)
            file_meta["path"] = rel.replace("\\", "/")
            if err:
                errors.append({"sample_id": sample_id, "path": rel, "error": err})
                file_meta["error"] = err
            all_file_metas.append(file_meta)
            for r in per_hdu_rows:
                writer.writerow({k: r.get(k, "") for k in IMAGE_STATS_COLUMNS})

    summary = build_summary(all_file_metas, n_manifest_samples=len(manifest_rows), errors=errors)
    with summary_path.open("w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2, sort_keys=True)
        fsum.write("\n")

    return stats_path, summary_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stats_path, summary_path = run_audit(args.manifest, args.root, args.out_dir, limit=args.limit)
    # brief console summary (no timestamps)
    print(json.dumps({
        "manifest": str(args.manifest),
        "root": str(args.root),
        "out_dir": str(args.out_dir),
        "limit": args.limit,
        "image_stats_csv": str(stats_path),
        "image_audit_summary_json": str(summary_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
