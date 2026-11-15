"""Astronomical image thresholding primitives for classical segmentation.

Implements:
1. Global Otsu thresholding (histogram between-class variance maximization).
2. Adaptive local thresholding (local spatial window mean with offset).

Strict requirements:
- Invalid pixels (NaN / Inf or valid_mask == False) are excluded from
  threshold statistics and strictly set to False (0) in the output.
- Outputs are binary masks (boolean or uint8) and DO NOT assign semantic
  class labels (core, bar, arm, star).
- All implementations are deterministic and self-contained in pure NumPy / PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def otsu_threshold(
    image: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_bins: int = 256,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Compute global Otsu thresholding on a 2D astronomical image.

    Maximizes the between-class variance sigma_B^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2.
    Computes histogram and statistics strictly over valid, finite pixels.

    Args:
        image: 2D array [H, W], float or integer.
        valid_mask: Optional boolean mask [H, W] where True indicates valid pixels.
        num_bins: Number of histogram bins for threshold estimation (default 256).

    Returns:
        tuple: (binary_mask, threshold_value, metadata)
            - binary_mask: 2D boolean numpy array [H, W]. True for foreground detections.
            - threshold_value: Optimal scalar threshold value in the original image scale.
            - metadata: Dictionary containing between-class variance, histogram info, etc.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    if img_np.ndim != 2:
        raise ValueError(f"Image must be 2D [H, W]; received shape {img_np.shape}.")

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            val_np = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            val_np = np.asarray(valid_mask, dtype=bool)
        if val_np.shape != img_np.shape:
            raise ValueError(
                f"valid_mask shape {val_np.shape} must match image shape {img_np.shape}."
            )
        finite_valid = val_np & np.isfinite(img_np)
    else:
        finite_valid = np.isfinite(img_np)

    valid_pixels = img_np[finite_valid]
    if valid_pixels.size == 0:
        # All invalid or empty
        binary_mask = np.zeros(img_np.shape, dtype=bool)
        return binary_mask, 0.0, {
            "algorithm": "otsu",
            "threshold": 0.0,
            "max_variance": 0.0,
            "valid_pixel_count": 0,
            "note": "Image contains no valid finite pixels.",
        }

    p_min = float(np.min(valid_pixels))
    p_max = float(np.max(valid_pixels))

    if math.isclose(p_min, p_max) or p_min >= p_max:
        # Constant image
        binary_mask = np.zeros(img_np.shape, dtype=bool)
        return binary_mask, p_min, {
            "algorithm": "otsu",
            "threshold": p_min,
            "max_variance": 0.0,
            "valid_pixel_count": int(valid_pixels.size),
            "note": "Image is constant across all valid pixels.",
        }

    # Compute histogram over valid pixels
    hist, bin_edges = np.histogram(valid_pixels, bins=num_bins, range=(p_min, p_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = float(valid_pixels.size)
    prob = hist / total

    # Cumulative sums and means
    weight0 = np.cumsum(prob)
    weight1 = 1.0 - weight0

    cumsum_mu = np.cumsum(prob * bin_centers)
    total_mu = cumsum_mu[-1]

    # Vectorized between-class variance
    # Avoid division by zero when weight0 or weight1 is 0
    safe_denom = weight0 * weight1
    valid_denom = safe_denom > 1e-12

    variance = np.zeros_like(prob)
    mean0 = np.zeros_like(prob)
    mean1 = np.zeros_like(prob)

    mean0[valid_denom] = cumsum_mu[valid_denom] / weight0[valid_denom]
    mean1[valid_denom] = (total_mu - cumsum_mu[valid_denom]) / weight1[valid_denom]
    variance[valid_denom] = weight0[valid_denom] * weight1[valid_denom] * (mean0[valid_denom] - mean1[valid_denom]) ** 2

    best_idx = int(np.argmax(variance))
    optimal_threshold = float(bin_centers[best_idx])
    max_variance = float(variance[best_idx])

    # Construct output mask: pixel > threshold AND valid
    binary_mask = (img_np > optimal_threshold) & finite_valid

    metadata = {
        "algorithm": "otsu",
        "threshold": optimal_threshold,
        "max_variance": max_variance,
        "valid_pixel_count": int(valid_pixels.size),
        "num_bins": num_bins,
        "intensity_min": p_min,
        "intensity_max": p_max,
        "foreground_pixel_count": int(np.count_nonzero(binary_mask)),
    }
    return binary_mask, optimal_threshold, metadata


def adaptive_threshold(
    image: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    window_size: int = 25,
    c_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute local adaptive thresholding on a 2D astronomical image.

    Computes a spatially varying threshold map T(x, y) = local_mean(x, y) - c_offset.
    Pixels exceeding T(x, y) are marked as foreground detections.

    Args:
        image: 2D array [H, W], float or integer.
        valid_mask: Optional boolean mask [H, W] where True indicates valid pixels.
        window_size: Local window size in pixels (must be odd, >= 3). Default 25.
        c_offset: Constant offset subtracted from the local mean (default 0.0).

    Returns:
        tuple: (binary_mask, threshold_map, metadata)
            - binary_mask: 2D boolean numpy array [H, W].
            - threshold_map: 2D float32 numpy array [H, W] of local threshold cutoffs.
            - metadata: Configuration and diagnostic metrics.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3; got {window_size}.")

    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        t_img = image.detach().float()
    else:
        t_img = torch.from_numpy(np.asarray(image, dtype=np.float32))

    if t_img.ndim != 2:
        raise ValueError(f"Image must be 2D [H, W]; received shape {t_img.shape}.")

    h, w = t_img.shape
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            t_val = valid_mask.detach().bool()
        else:
            t_val = torch.from_numpy(np.asarray(valid_mask, dtype=bool))
        if t_val.shape != (h, w):
            raise ValueError(f"valid_mask shape {t_val.shape} must match image shape {(h, w)}.")
        finite_valid = t_val & torch.isfinite(t_img)
    else:
        finite_valid = torch.isfinite(t_img)

    # Zero invalid pixels before local pooling
    clean_img = torch.where(finite_valid, t_img, torch.zeros_like(t_img))
    val_float = finite_valid.float()

    # Compute local mean using 2D box filter via F.avg_pool2d with padding
    pad = window_size // 2
    # Add batch and channel dimensions: [1, 1, H, W]
    inp_img = clean_img.unsqueeze(0).unsqueeze(0)
    inp_val = val_float.unsqueeze(0).unsqueeze(0)

    # Pad with replication to handle borders cleanly
    padded_img = F.pad(inp_img, (pad, pad, pad, pad), mode="replicate")
    padded_val = F.pad(inp_val, (pad, pad, pad, pad), mode="replicate")

    sum_img = F.avg_pool2d(padded_img, kernel_size=window_size, stride=1, divisor_override=1)
    sum_val = F.avg_pool2d(padded_val, kernel_size=window_size, stride=1, divisor_override=1)

    # Local mean = sum_img / sum_val (safely handle windows with 0 valid pixels)
    safe_denom = torch.clamp(sum_val, min=1.0)
    local_mean = (sum_img / safe_denom).squeeze(0).squeeze(0)

    # Threshold map: local_mean - c_offset
    threshold_map = local_mean - c_offset

    # Binary mask: clean_img > threshold_map AND finite_valid
    binary_mask = (clean_img > threshold_map) & finite_valid

    bin_np = binary_mask.cpu().numpy()
    thresh_np = threshold_map.cpu().numpy()

    metadata = {
        "algorithm": "adaptive_threshold",
        "window_size": window_size,
        "c_offset": c_offset,
        "foreground_pixel_count": int(np.count_nonzero(bin_np)),
        "valid_pixel_count": int(torch.count_nonzero(finite_valid).item()),
    }
    return bin_np, thresh_np, metadata
