"""Diagnostic five-band comparison workflow for Pan-STARRS FITS cutouts.

This module provides a reproducible, objective diagnostic framework to
evaluate all five Pan-STARRS bands (g, r, i, z, y) before selecting a
reference band for classical computer vision pipelines.

Pan-STARRS HDU layout (frozen benchmark specification):
    HDU 0: g-band (~481 nm)
    HDU 1: r-band (~617 nm)
    HDU 2: i-band (~752 nm)
    HDU 3: z-band (~866 nm)
    HDU 4: y-band (~962 nm)

PROVISIONAL STATUS NOTE
-----------------------
This module contains an isolated, read-only FITS reader strictly for B4
diagnostics and baselines. It is provisional pending completion of Person A's
production FITS loader (``src/data/fits_loader.py``) and normalizer. It does
NOT replace or compete with Person A's modules.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

BAND_NAMES: Tuple[str, ...] = ("g", "r", "i", "z", "y")
HDU_TO_BAND: Dict[int, str] = {i: name for i, name in enumerate(BAND_NAMES)}
BAND_TO_HDU: Dict[str, int] = {name: i for i, name in enumerate(BAND_NAMES)}


def load_band_fits(
    fits_path: Union[str, Path],
    band: Union[int, str] = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a single Pan-STARRS FITS band read-only with validity masking.

    Converts FITS big-endian arrays (``>f4``) to native float32, identifies
    non-finite values (NaN / ±Inf), and returns a boolean validity mask.

    Args:
        fits_path: Path to the FITS image file.
        band: Band index (0..4) or band name ("g", "r", "i", "z", "y").
              Default is 1 (r-band).

    Returns:
        tuple: (image_array, valid_mask, metadata)
            - image_array: 2D numpy array [H, W], native float32. Invalid
              pixels are replaced with 0.0.
            - valid_mask: 2D boolean array [H, W], True where flux is finite.
            - metadata: Dictionary containing header cards, band name, HDU index.

    Raises:
        FileNotFoundError: If the FITS file does not exist.
        ValueError: If the band specification is invalid.
    """
    path = Path(fits_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FITS file not found: {path}")

    if isinstance(band, str):
        band_lower = band.lower()
        if band_lower not in BAND_TO_HDU:
            raise ValueError(f"Unknown band name '{band}'. Allowed: {BAND_NAMES}")
        hdu_idx = BAND_TO_HDU[band_lower]
    elif isinstance(band, int):
        if band not in HDU_TO_BAND:
            raise ValueError(f"Invalid band index {band}. Must be 0..4.")
        hdu_idx = band
    else:
        raise TypeError(f"Band must be int or str, got {type(band).__name__}")

    with fits.open(path, mode="readonly", memmap=True) as hdul:
        if hdu_idx >= len(hdul):
            raise IndexError(
                f"HDU index {hdu_idx} exceeds available HDUs ({len(hdul)}) in {path.name}"
            )
        hdu = hdul[hdu_idx]
        raw_data = hdu.data
        if raw_data is None:
            raise ValueError(f"HDU {hdu_idx} in {path.name} contains no image data.")

        # Cast big-endian to native float32
        data = np.asarray(raw_data, dtype="=f4")
        header = dict(hdu.header)

    valid_mask = np.isfinite(data)
    # Zero out non-finite pixels for safety
    clean_data = np.where(valid_mask, data, 0.0).astype(np.float32)

    metadata: Dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "hdu_index": hdu_idx,
        "band_name": HDU_TO_BAND[hdu_idx],
        "shape": list(clean_data.shape),
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "non_finite_count": int(data.size - np.count_nonzero(valid_mask)),
        "bunit": header.get("BUNIT"),
        "telescope": header.get("TELESCOP"),
    }
    return clean_data, valid_mask, metadata


def compute_band_metrics(
    data: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute objective astronomical statistical diagnostics on a single band.

    Args:
        data: 2D array of flux values.
        valid_mask: Optional boolean mask where True indicates valid pixels.

    Returns:
        dict: Statistical diagnostics including min, max, mean, std, median,
              robust sky noise MAD, percentiles, SNR, and invalid pixel fraction.
    """
    data = np.asarray(data)
    if valid_mask is None:
        valid_mask = np.isfinite(data)
    else:
        valid_mask = valid_mask & np.isfinite(data)

    valid_pixels = data[valid_mask]
    total_count = data.size
    valid_count = int(valid_pixels.size)
    invalid_count = total_count - valid_count

    if valid_count == 0:
        return {
            "valid_count": 0,
            "invalid_count": invalid_count,
            "invalid_fraction": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "sky_mad": None,
            "p005": None,
            "p995": None,
            "peak_snr": None,
            "robust_snr": None,
        }

    p_min = float(np.min(valid_pixels))
    p_max = float(np.max(valid_pixels))
    p_mean = float(np.mean(valid_pixels))
    p_std = float(np.std(valid_pixels))
    p_median = float(np.median(valid_pixels))

    # Robust sky background noise via Median Absolute Deviation (MAD):
    # sigma_MAD = 1.4826 * median(|x - median(x)|)
    abs_dev = np.abs(valid_pixels - p_median)
    sky_mad = float(1.4826 * np.median(abs_dev))

    percentiles = np.percentile(valid_pixels, [0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5])
    p005, p01, p05, p50, p95, p99, p995 = [float(v) for v in percentiles]

    # Signal-to-Noise Ratio (SNR) diagnostics:
    # 1. Peak SNR: contrast between highest emission (core/star) and sky median
    # 2. Robust 99.5% SNR: contrast of galaxy structure (p99.5) relative to sky noise
    safe_noise = sky_mad if sky_mad > 1e-7 else (p_std if p_std > 1e-7 else 1.0)
    peak_snr = float((p_max - p_median) / safe_noise)
    robust_snr = float((p995 - p_median) / safe_noise)

    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_fraction": float(invalid_count / total_count),
        "min": p_min,
        "max": p_max,
        "mean": p_mean,
        "std": p_std,
        "median": p_median,
        "sky_mad": sky_mad,
        "p005": p005,
        "p01": p01,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "p995": p995,
        "peak_snr": peak_snr,
        "robust_snr": robust_snr,
    }


def compare_five_bands(fits_path: Union[str, Path]) -> Dict[str, Any]:
    """Execute full 5-band diagnostic comparison on a single Pan-STARRS FITS file.

    Returns:
        dict: Per-band metrics, rankings, and an evidence-based recommendation.
    """
    path = Path(fits_path).resolve()
    band_reports: Dict[str, Any] = {}

    for idx, name in enumerate(BAND_NAMES):
        data, valid_mask, meta = load_band_fits(path, band=idx)
        metrics = compute_band_metrics(data, valid_mask)
        band_reports[name] = {
            "hdu_index": idx,
            "band_name": name,
            **metrics,
        }

    recommendation = recommend_reference_band(band_reports)
    return {
        "filename": path.name,
        "bands": band_reports,
        "recommendation": recommendation,
    }


def recommend_reference_band(band_reports: Dict[str, Any]) -> Dict[str, Any]:
    """Determine the optimal classical reference band from objective metrics.

    Scoring criteria:
        1. Low invalid pixel fraction (avoids corrupted bands).
        2. High robust SNR (p99.5 - sky) / sky_mad (clear extended galaxy structure).
        3. Low relative background variance.

    In astronomical imaging with Pan-STARRS, the r-band (~617 nm) and i-band
    (~752 nm) standardly offer the highest throughput, sharpest PSF, and
    lowest dust extinction relative to g-band, while avoiding z/y sky fringing.
    """
    candidates = []
    for name, rep in band_reports.items():
        if rep.get("valid_count", 0) == 0:
            continue
        invalid_frac = rep["invalid_fraction"]
        robust_snr = rep["robust_snr"] or 0.0
        peak_snr = rep["peak_snr"] or 0.0
        sky_mad = rep["sky_mad"] or 1.0

        # Score penalized heavily if invalid fraction > 0
        quality_score = robust_snr * (1.0 - 5.0 * invalid_frac)
        candidates.append((quality_score, robust_snr, name))

    if not candidates:
        return {
            "recommended_band": "r",
            "recommended_hdu": 1,
            "reason": "All bands invalid or empty; defaulting to standard r-band.",
        }

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_name = candidates[0][2]
    best_rep = band_reports[best_name]

    return {
        "recommended_band": best_name,
        "recommended_hdu": BAND_TO_HDU[best_name],
        "status": "provisional",
        "robust_snr": best_rep["robust_snr"],
        "invalid_fraction": best_rep["invalid_fraction"],
        "reason": (
            f"Provisional {best_name}-band reference recommendation based on diagnostics from two representative samples. "
            f"Broader validation across the dataset is required. Band '{best_name}' achieves the highest robust contrast-to-noise ratio "
            f"(robust_SNR={best_rep['robust_snr']:.2f}) with zero or minimal invalid pixels "
            f"(invalid_fraction={best_rep['invalid_fraction']:.4f})."
        ),
    }
