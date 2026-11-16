"""Connected-component labeling and spatial region analysis for binary masks.

Implements:
1. Two-pass connected-component labeling with disjoint-set (Union-Find)
   supporting both 4-connectivity and 8-connectivity.
2. Component statistics computation (label ID, area, bounding box, centroid).
3. Optional configurable minimum area filtering.

Requirements:
- Faint and small galaxy features are preserved by default (min_area=1).
- Invalid pixels (valid_mask == False) are excluded and remain 0.
- Fully self-contained in pure NumPy without requiring external C libraries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class _DisjointSet:
    """Disjoint-set (Union-Find) with path compression for label equivalences."""

    def __init__(self) -> None:
        self.parent: List[int] = [0]

    def make_set(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        return idx

    def find(self, i: int) -> int:
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i: int, j: int) -> int:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if root_i < root_j:
                self.parent[root_j] = root_i
                return root_i
            else:
                self.parent[root_i] = root_j
                return root_j
        return root_i


def label_connected_components(
    binary_mask: Union[np.ndarray, torch.Tensor],
    connectivity: int = 8,
    min_area: int = 1,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    """Label connected components in a 2D binary mask.

    Args:
        binary_mask: 2D array [H, W] of boolean or binary values.
        connectivity: Pixel neighborhood connectivity, 4 or 8 (default 8).
        min_area: Minimum area in pixels to retain a component. Default is 1
                  (preserves all features, including single-pixel components).
        valid_mask: Optional boolean mask [H, W] where False marks invalid pixels.

    Returns:
        tuple: (labeled_map, num_features, stats)
            - labeled_map: 2D int32 array [H, W] where 0 is background and
              1..K are contiguous feature IDs.
            - num_features: Total number of labeled components retained.
            - stats: List of dictionaries per component containing:
                - 'label': int, component ID (1..num_features)
                - 'area': int, pixel count
                - 'bbox': tuple (min_row, min_col, max_row, max_col)
                - 'centroid': tuple (row_centroid, col_centroid)
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity}.")

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

    h, w = bin_np.shape
    raw_labels = np.zeros((h, w), dtype=np.int32)
    ds = _DisjointSet()

    # Pass 1: Raster scan and equivalence recording
    for r in range(h):
        for c in range(w):
            if not bin_np[r, c]:
                continue

            neighbor_labels = []
            # Left
            if c > 0 and raw_labels[r, c - 1] > 0:
                neighbor_labels.append(raw_labels[r, c - 1])
            # Top
            if r > 0 and raw_labels[r - 1, c] > 0:
                neighbor_labels.append(raw_labels[r - 1, c])

            if connectivity == 8:
                # Top-Left
                if r > 0 and c > 0 and raw_labels[r - 1, c - 1] > 0:
                    neighbor_labels.append(raw_labels[r - 1, c - 1])
                # Top-Right
                if r > 0 and c + 1 < w and raw_labels[r - 1, c + 1] > 0:
                    neighbor_labels.append(raw_labels[r - 1, c + 1])

            if not neighbor_labels:
                new_lbl = ds.make_set()
                raw_labels[r, c] = new_lbl
            else:
                first_root = ds.find(neighbor_labels[0])
                for other in neighbor_labels[1:]:
                    first_root = ds.union(first_root, other)
                raw_labels[r, c] = first_root

    # Pass 2: Canonical resolution and relabeling
    root_to_new: Dict[int, int] = {}
    current_label = 1
    labeled_map = np.zeros((h, w), dtype=np.int32)

    for r in range(h):
        for c in range(w):
            lbl = raw_labels[r, c]
            if lbl > 0:
                canonical = ds.find(lbl)
                if canonical not in root_to_new:
                    root_to_new[canonical] = current_label
                    current_label += 1
                labeled_map[r, c] = root_to_new[canonical]

    total_components = len(root_to_new)
    if total_components == 0:
        return labeled_map, 0, []

    # Compute bounding boxes, areas, centroids
    component_stats: Dict[int, Dict[str, Any]] = {
        k: {
            "label": k,
            "area": 0,
            "min_r": h,
            "min_c": w,
            "max_r": -1,
            "max_c": -1,
            "sum_r": 0,
            "sum_c": 0,
        }
        for k in range(1, total_components + 1)
    }

    for r in range(h):
        for c in range(w):
            k = int(labeled_map[r, c])
            if k > 0:
                st = component_stats[k]
                st["area"] += 1
                st["sum_r"] += r
                st["sum_c"] += c
                if r < st["min_r"]:
                    st["min_r"] = r
                if r > st["max_r"]:
                    st["max_r"] = r
                if c < st["min_c"]:
                    st["min_c"] = c
                if c > st["max_c"]:
                    st["max_c"] = c

    # Filter by minimum area if requested
    filtered_map = np.zeros((h, w), dtype=np.int32)
    final_stats: List[Dict[str, Any]] = []
    final_id = 1

    for k in sorted(component_stats):
        st = component_stats[k]
        area = st["area"]
        if area >= min_area:
            filtered_map[labeled_map == k] = final_id
            final_stats.append({
                "label": final_id,
                "area": area,
                "bbox": (st["min_r"], st["min_c"], st["max_r"], st["max_c"]),
                "centroid": (float(st["sum_r"] / area), float(st["sum_c"] / area)),
            })
            final_id += 1

    return filtered_map, len(final_stats), final_stats
