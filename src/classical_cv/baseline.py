"""End-to-end classical computer vision segmentation baseline pipeline.

Orchestrates:
1. Band selection and input validation.
2. Global Otsu or adaptive local thresholding.
3. Morphological opening and closing.
4. Connected-component labeling and region statistics.
5. Exact Euclidean distance transform and marker-controlled watershed segmentation.

Strict scientific considerations:
- Operates deterministically without stochastic augmentation.
- Retains boolean validity masks throughout all processing stages.
- Does not assign semantic class labels to binary or watershed outputs.
- Retains independence from the SegFormer deep learning pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from src.classical_cv.connected_components import label_connected_components
from src.classical_cv.morphology import (
    binary_closing,
    binary_opening,
    get_structuring_element,
)
from src.classical_cv.thresholding import adaptive_threshold, otsu_threshold
from src.classical_cv.watershed import (
    distance_transform_edt,
    extract_watershed_boundaries,
    generate_markers_from_peaks,
    watershed_segmentation,
)


def run_classical_pipeline(
    image: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold_method: str = "otsu",
    otsu_bins: int = 256,
    adaptive_window: int = 25,
    adaptive_c: float = 0.0,
    morph_shape: str = "disk",
    morph_radius: int = 1,
    connectivity: int = 8,
    min_area: int = 1,
    watershed_min_distance: int = 5,
    watershed_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Execute the complete classical computer vision baseline on a 2D scalar field.

    Args:
        image: 2D array [H, W], float32 in [0, 1] or raw normalized flux.
        valid_mask: Optional boolean mask [H, W] marking valid pixels (True = valid).
        threshold_method: 'otsu' or 'adaptive' (default 'otsu').
        otsu_bins: Number of histogram bins for Otsu (default 256).
        adaptive_window: Local window size for adaptive threshold (default 25).
        adaptive_c: Offset constant for adaptive threshold (default 0.0).
        morph_shape: Structuring element shape ('disk', 'square', 'cross').
        morph_radius: Structuring element radius in pixels (default 1).
        connectivity: Neighborhood connectivity (4 or 8, default 8).
        min_area: Minimum connected component area (default 1, retains small structures).
        watershed_min_distance: Minimum separation between watershed markers (default 5).
        watershed_threshold: Minimum distance transform height for marker seeds (default 2.0).

    Returns:
        dict: Complete results bundle containing:
            - 'threshold_mask': 2D boolean array [H, W] from initial thresholding.
            - 'morph_mask': 2D boolean array [H, W] after opening and closing.
            - 'labeled_components': 2D int32 array [H, W] of connected components.
            - 'component_stats': List of dicts with area, bbox, centroid per component.
            - 'distance_map': 2D float32 array [H, W] of Euclidean distance to background.
            - 'markers': 2D int32 array [H, W] of watershed seed markers.
            - 'labeled_basins': 2D int32 array [H, W] of watershed regions.
            - 'watershed_boundaries': 2D boolean array [H, W] of watershed boundary lines.
            - 'metadata': Detailed parameters and intermediate statistics.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy().astype(np.float32)
    else:
        img_np = np.asarray(image, dtype=np.float32)

    if img_np.ndim != 2:
        raise ValueError(f"Input image must be 2D [H, W]; received {img_np.shape}.")

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            val_np = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            val_np = np.asarray(valid_mask, dtype=bool)
        if val_np.shape != img_np.shape:
            raise ValueError(f"valid_mask shape {val_np.shape} must match image {img_np.shape}.")
        finite_valid = val_np & np.isfinite(img_np)
    else:
        finite_valid = np.isfinite(img_np)

    # Clean input: replace NaNs with 0.0
    clean_img = np.where(finite_valid, img_np, 0.0)

    # 1. Thresholding
    thresh_lower = threshold_method.lower()
    if thresh_lower == "otsu":
        thresh_mask, thresh_val, thresh_meta = otsu_threshold(
            clean_img, valid_mask=finite_valid, num_bins=otsu_bins
        )
    elif thresh_lower == "adaptive":
        thresh_mask, thresh_val, thresh_meta = adaptive_threshold(
            clean_img, valid_mask=finite_valid, window_size=adaptive_window, c_offset=adaptive_c
        )
    else:
        raise ValueError(f"Unknown threshold_method '{threshold_method}'. Choose 'otsu' or 'adaptive'.")

    # 2. Morphology
    se = get_structuring_element(morph_shape, radius=morph_radius)
    # Opening removes isolated high-frequency noise spikes
    opened = binary_opening(thresh_mask, structuring_element=se, iterations=1, valid_mask=finite_valid)
    # Closing bridges narrow gaps and micro-holes in extended galaxy arms
    closed = binary_closing(opened, structuring_element=se, iterations=1, valid_mask=finite_valid)
    morph_mask = closed

    # 3. Connected Components
    labeled_comp, num_comp, comp_stats = label_connected_components(
        morph_mask, connectivity=connectivity, min_area=min_area, valid_mask=finite_valid
    )

    # 4. Distance Transform and Watershed
    dist_map = distance_transform_edt(morph_mask, valid_mask=finite_valid)
    markers, num_markers = generate_markers_from_peaks(
        dist_map, min_distance=watershed_min_distance, threshold=watershed_threshold, mask=morph_mask
    )

    labeled_basins, ws_meta = watershed_segmentation(
        surface=-dist_map,
        markers=markers,
        mask=morph_mask,
        connectivity=connectivity,
        valid_mask=finite_valid,
    )
    boundaries = extract_watershed_boundaries(labeled_basins)

    return {
        "threshold_mask": thresh_mask,
        "morph_mask": morph_mask,
        "labeled_components": labeled_comp,
        "component_stats": comp_stats,
        "distance_map": dist_map,
        "markers": markers,
        "labeled_basins": labeled_basins,
        "watershed_boundaries": boundaries,
        "metadata": {
            "threshold": thresh_meta,
            "morphology": {
                "shape": morph_shape,
                "radius": morph_radius,
                "foreground_pixel_count": int(np.count_nonzero(morph_mask)),
            },
            "connected_components": {
                "connectivity": connectivity,
                "num_components": num_comp,
                "min_area": min_area,
            },
            "watershed": ws_meta,
            "valid_pixel_count": int(np.count_nonzero(finite_valid)),
        },
    }
