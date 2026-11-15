"""Marker-controlled watershed segmentation and distance transforms.

Implements:
1. Exact Euclidean Distance Transform (EDT) with automatic SciPy delegation
   if installed, or self-contained O(N) separable algorithm (Felzenszwalb & Huttenlocher).
2. Astronomical marker generation from distance peaks or intensity local maxima.
3. Marker-controlled watershed segmentation with automatic SciPy/scikit-image
   delegation if installed, or Meyer's priority-queue immersion algorithm.
4. Watershed boundary extraction.

Requirements:
- Preserves input dimensions [H, W].
- Excludes invalid pixels (valid_mask == False) from being labeled as regions.
- Does not assign semantic classes to watershed basin IDs.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Check for optional external libraries (SciPy / scikit-image)
try:
    from scipy.ndimage import distance_transform_edt as _scipy_edt
    _HAS_SCIPY_EDT = True
except ImportError:
    _scipy_edt = None
    _HAS_SCIPY_EDT = False

try:
    from skimage.segmentation import watershed as _skimage_watershed
    _HAS_SKIMAGE_WATERSHED = True
except ImportError:
    _skimage_watershed = None
    _HAS_SKIMAGE_WATERSHED = False


# ---------------------------------------------------------------------------
# 1. Exact Euclidean Distance Transform
# ---------------------------------------------------------------------------

def _edt_1d(f: np.ndarray) -> np.ndarray:
    """1D squared Euclidean distance transform via lower parabolic envelope (O(N))."""
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    k = 0
    v = [0] * n
    z = [0.0] * (n + 1)
    z[0] = -1e20
    z[1] = 1e20

    for q in range(1, n):
        denom = 2.0 * q - 2.0 * v[k]
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / denom
        while s <= z[k]:
            k -= 1
            denom = 2.0 * q - 2.0 * v[k]
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / denom
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = 1e20

    k = 0
    d = np.empty(n, dtype=np.float32)
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]
    return d


def distance_transform_edt(
    binary_mask: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Calculate the exact Euclidean Distance Transform of a 2D binary mask.

    Computes the distance from each foreground pixel to the nearest background pixel.
    Delegates to `scipy.ndimage.distance_transform_edt` if installed, or falls
    back to the exact O(N) separable parabolic envelope algorithm.

    Args:
        binary_mask: 2D array [H, W], True inside foreground.
        valid_mask: Optional boolean mask [H, W]. Invalid pixels are treated as background.

    Returns:
        np.ndarray: 2D float32 array [H, W] of exact Euclidean distances in pixels.
    """
    if isinstance(binary_mask, torch.Tensor):
        bin_np = binary_mask.detach().cpu().numpy().astype(bool)
    else:
        bin_np = np.asarray(binary_mask, dtype=bool)

    if bin_np.ndim != 2:
        raise ValueError(f"binary_mask must be 2D [H, W]; received {bin_np.shape}.")

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            val_np = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            val_np = np.asarray(valid_mask, dtype=bool)
        bin_np = bin_np & val_np

    if _HAS_SCIPY_EDT:
        return _scipy_edt(bin_np).astype(np.float32)

    # Pure NumPy implementation (Felzenszwalb-Huttenlocher 2D EDT)
    h, w = bin_np.shape
    if not np.any(bin_np):
        return np.zeros((h, w), dtype=np.float32)

    f = np.where(~bin_np, 0.0, 1e10).astype(np.float32)
    d = np.empty((h, w), dtype=np.float32)
    for c in range(w):
        d[:, c] = _edt_1d(f[:, c])

    dt = np.empty((h, w), dtype=np.float32)
    for r in range(h):
        dt[r, :] = _edt_1d(d[r, :])

    return np.sqrt(np.maximum(dt, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Marker Generation
# ---------------------------------------------------------------------------

def generate_markers_from_peaks(
    surface: Union[np.ndarray, torch.Tensor],
    min_distance: int = 5,
    threshold: float = 1.0,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[np.ndarray, int]:
    """Generate regional seed markers from true local maxima on a 2D surface.

    Finds local spatial peaks (pixels that equal the local maximum in a 3x3 window)
    that exceed `threshold`, and suppresses secondary peaks within `min_distance`.

    Args:
        surface: 2D array [H, W] (e.g. distance transform or normalized flux).
        min_distance: Minimum separation distance between distinct markers.
        threshold: Minimum surface value to qualify as a marker seed.
        mask: Optional boolean mask restricting marker search to foreground.

    Returns:
        tuple: (marker_map, num_markers)
            - marker_map: 2D int32 array [H, W] where 0 is background and
              1..M are distinct marker seed IDs.
            - num_markers: Integer count of markers found.
    """
    if isinstance(surface, torch.Tensor):
        surf_np = surface.detach().cpu().numpy().astype(np.float32)
    else:
        surf_np = np.asarray(surface, dtype=np.float32)

    h, w = surf_np.shape
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)
        candidate = (surf_np >= threshold) & mask_np
    else:
        candidate = (surf_np >= threshold)

    if not np.any(candidate):
        return np.zeros((h, w), dtype=np.int32), 0

    # True local maxima detection using 3x3 max pooling
    t_surf = torch.from_numpy(surf_np).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t_surf, kernel_size=3, stride=1, padding=1).squeeze().numpy()
    is_local_max = (surf_np == pooled) & candidate

    coords = np.argwhere(is_local_max)
    if coords.size == 0:
        return np.zeros((h, w), dtype=np.int32), 0

    # Sort genuine local maxima by descending surface height
    values = surf_np[coords[:, 0], coords[:, 1]]
    order = np.argsort(-values)
    sorted_coords = coords[order]

    accepted_peaks: List[Tuple[int, int]] = []
    min_dist_sq = min_distance * min_distance

    for r, c in sorted_coords:
        too_close = False
        for pr, pc in accepted_peaks:
            dist_sq = (r - pr) * (r - pr) + (c - pc) * (c - pc)
            if dist_sq < min_dist_sq:
                too_close = True
                break
        if not too_close:
            accepted_peaks.append((int(r), int(c)))

    marker_map = np.zeros((h, w), dtype=np.int32)
    for idx, (pr, pc) in enumerate(accepted_peaks, start=1):
        marker_map[pr, pc] = idx

    return marker_map, len(accepted_peaks)


# ---------------------------------------------------------------------------
# 3. Marker-Controlled Watershed Segmentation
# ---------------------------------------------------------------------------

def _meyer_watershed(
    surface: np.ndarray,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """Meyer's priority-queue topographic immersion watershed algorithm."""
    h, w = surface.shape
    labeled = markers.copy().astype(np.int32)

    if connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    heap: List[Tuple[float, int, int, int]] = []

    # Push initial boundary pixels of marker seeds into the min-heap
    marker_coords = np.argwhere(labeled > 0)
    for r, c in marker_coords:
        lbl = int(labeled[r, c])
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if mask[nr, nc] and labeled[nr, nc] == 0:
                    heapq.heappush(heap, (float(surface[nr, nc]), nr, nc, lbl))

    while heap:
        val, r, c, lbl = heapq.heappop(heap)
        if labeled[r, c] != 0:
            continue
        labeled[r, c] = lbl
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if mask[nr, nc] and labeled[nr, nc] == 0:
                    heapq.heappush(heap, (float(surface[nr, nc]), nr, nc, lbl))

    return labeled


def watershed_segmentation(
    surface: Union[np.ndarray, torch.Tensor],
    markers: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    connectivity: int = 8,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute marker-controlled watershed segmentation.

    Floods catchment basins starting from regional markers along the topography
    of `surface` (e.g. inverted distance transform or gradient map).

    Args:
        surface: 2D array [H, W] defining topographic relief.
        markers: 2D int32 array [H, W] containing marker seeds (1..M).
        mask: Optional boolean mask [H, W] restricting watershed to foreground.
        connectivity: Pixel neighborhood connectivity (4 or 8, default 8).
        valid_mask: Optional boolean mask [H, W] to exclude invalid pixels.

    Returns:
        tuple: (labeled_basins, metadata)
            - labeled_basins: 2D int32 array [H, W] of segmented regional IDs.
            - metadata: Algorithm configuration and basin statistics.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity}.")

    if isinstance(surface, torch.Tensor):
        surf_np = surface.detach().cpu().numpy().astype(np.float32)
    else:
        surf_np = np.asarray(surface, dtype=np.float32)

    if isinstance(markers, torch.Tensor):
        mark_np = markers.detach().cpu().numpy().astype(np.int32)
    else:
        mark_np = np.asarray(markers, dtype=np.int32)

    h, w = surf_np.shape
    if mark_np.shape != (h, w):
        raise ValueError(f"markers shape {mark_np.shape} must match surface shape {(h, w)}.")

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)
    else:
        mask_np = np.ones((h, w), dtype=bool)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            val_np = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            val_np = np.asarray(valid_mask, dtype=bool)
        mask_np = mask_np & val_np

    # Exclude markers that lie outside the valid mask
    clean_markers = np.where(mask_np, mark_np, 0).astype(np.int32)
    num_markers = int(np.max(clean_markers))

    if num_markers == 0:
        return np.zeros((h, w), dtype=np.int32), {
            "algorithm": "watershed",
            "num_basins": 0,
            "backend": "none",
            "note": "No valid markers present.",
        }

    # Execute watershed
    if _HAS_SKIMAGE_WATERSHED:
        conn_grid = 1 if connectivity == 4 else 2
        labeled = _skimage_watershed(surf_np, clean_markers, mask=mask_np, connectivity=conn_grid)
        backend = "skimage"
    else:
        labeled = _meyer_watershed(surf_np, clean_markers, mask_np, connectivity=connectivity)
        backend = "meyer_heapq"

    # Enforce background for pixels outside mask
    labeled = np.where(mask_np, labeled, 0).astype(np.int32)

    unique_basins = np.unique(labeled)
    unique_basins = unique_basins[unique_basins > 0]

    metadata = {
        "algorithm": "watershed",
        "backend": backend,
        "connectivity": connectivity,
        "num_markers": num_markers,
        "num_basins": int(unique_basins.size),
        "basin_labels": [int(b) for b in unique_basins],
    }
    return labeled, metadata


# ---------------------------------------------------------------------------
# 4. Watershed Boundary Extraction
# ---------------------------------------------------------------------------

def extract_watershed_boundaries(
    labeled_basins: np.ndarray,
    two_pixel_width: bool = True,
) -> np.ndarray:
    """Extract boundary lines separating distinct watershed basins.

    By default (`two_pixel_width=True`), marks pixels on both sides of an
    interface between two adjacent positive basin IDs, yielding a prominent
    symmetric boundary visualization. When `two_pixel_width=False`, marks only
    the lower-coordinate side (single-pixel line).

    Args:
        labeled_basins: 2D int32 array [H, W] from watershed segmentation.
        two_pixel_width: If True, marks both adjacent pixels along the interface.
            If False, marks a strict single-pixel-wide line.

    Returns:
        np.ndarray: 2D boolean array [H, W] where True marks boundary pixels.
    """
    labels = np.asarray(labeled_basins, dtype=np.int32)
    h, w = labels.shape
    boundaries = np.zeros((h, w), dtype=bool)

    # Check horizontal neighbors
    diff_h = labels[:, 1:] != labels[:, :-1]
    # Mark boundaries only between two positive labels (not background)
    valid_h = (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    b_h = diff_h & valid_h
    boundaries[:, :-1] |= b_h
    if two_pixel_width:
        boundaries[:, 1:] |= b_h

    # Check vertical neighbors
    diff_v = labels[1:, :] != labels[:-1, :]
    valid_v = (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    b_v = diff_v & valid_v
    boundaries[:-1, :] |= b_v
    if two_pixel_width:
        boundaries[1:, :] |= b_v

    return boundaries
