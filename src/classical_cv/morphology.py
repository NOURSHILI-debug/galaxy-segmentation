"""Astronomical morphological image processing primitives.

Implements:
1. Structuring element generation (disk, square, cross).
2. Binary and grayscale erosion and dilation.
3. Morphological opening (erosion -> dilation).
4. Morphological closing (dilation -> erosion).

Requirements:
- Preserves spatial dimensions [H, W].
- Excludes invalid pixels (valid_mask == False) from being falsely marked as detections.
- Deterministic, implemented with pure PyTorch / NumPy tensor operations.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


def get_structuring_element(
    shape: str = "disk",
    radius: int = 2,
) -> np.ndarray:
    """Generate a 2D symmetric binary structuring element.

    Args:
        shape: Shape of the footprint: 'disk', 'square', or 'cross'.
        radius: Radius in pixels (kernel size will be 2 * radius + 1).

    Returns:
        np.ndarray: 2D boolean array of shape [2*radius + 1, 2*radius + 1].
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1; got {radius}.")

    size = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]

    shape_lower = shape.lower()
    if shape_lower == "disk":
        kernel = (x * x + y * y) <= (radius * radius)
    elif shape_lower == "square":
        kernel = np.ones((size, size), dtype=bool)
    elif shape_lower == "cross":
        kernel = (x == 0) | (y == 0)
    else:
        raise ValueError(f"Unknown structuring element shape '{shape}'. Choose 'disk', 'square', or 'cross'.")

    return kernel.astype(bool)


def binary_dilation(
    binary_mask: Union[np.ndarray, torch.Tensor],
    structuring_element: Optional[np.ndarray] = None,
    iterations: int = 1,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Apply binary dilation to expand foreground regions.

    Args:
        binary_mask: 2D boolean or uint8 mask [H, W].
        structuring_element: 2D boolean array. Defaults to disk of radius 2 (5x5).
        iterations: Number of consecutive dilation iterations (default 1).
        valid_mask: Optional boolean mask [H, W] to zero out invalid pixels.

    Returns:
        np.ndarray: 2D boolean mask [H, W].
    """
    if iterations < 1:
        return np.asarray(binary_mask, dtype=bool)

    if structuring_element is None:
        structuring_element = get_structuring_element("disk", radius=2)

    kernel = np.asarray(structuring_element, dtype=bool)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Structuring element must be a 2D array with odd dimensions.")

    is_torch = isinstance(binary_mask, torch.Tensor)
    if is_torch:
        t_mask = binary_mask.detach().float()
    else:
        t_mask = torch.from_numpy(np.asarray(binary_mask, dtype=np.float32))

    if t_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2D [H, W]; received {t_mask.shape}.")

    # Kernel tensor [1, 1, Kh, Kw]
    t_kernel = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    cur = t_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    for _ in range(iterations):
        # Convolution sum > 0 means structuring element touches at least one foreground pixel
        conv = F.conv2d(cur, t_kernel, padding=(pad_h, pad_w))
        cur = (conv > 0.0).float()

    out_bool = cur.squeeze(0).squeeze(0).bool()

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            v_mask = valid_mask.detach().bool()
        else:
            v_mask = torch.from_numpy(np.asarray(valid_mask, dtype=bool))
        out_bool = out_bool & v_mask

    return out_bool.cpu().numpy()


def binary_erosion(
    binary_mask: Union[np.ndarray, torch.Tensor],
    structuring_element: Optional[np.ndarray] = None,
    iterations: int = 1,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Apply binary erosion to shrink foreground regions and remove small spikes.

    Args:
        binary_mask: 2D boolean or uint8 mask [H, W].
        structuring_element: 2D boolean array. Defaults to disk of radius 2 (5x5).
        iterations: Number of consecutive erosion iterations (default 1).
        valid_mask: Optional boolean mask [H, W] to zero out invalid pixels.

    Returns:
        np.ndarray: 2D boolean mask [H, W].
    """
    if iterations < 1:
        return np.asarray(binary_mask, dtype=bool)

    if structuring_element is None:
        structuring_element = get_structuring_element("disk", radius=2)

    kernel = np.asarray(structuring_element, dtype=bool)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Structuring element must be a 2D array with odd dimensions.")

    is_torch = isinstance(binary_mask, torch.Tensor)
    if is_torch:
        t_mask = binary_mask.detach().float()
    else:
        t_mask = torch.from_numpy(np.asarray(binary_mask, dtype=np.float32))

    if t_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2D [H, W]; received {t_mask.shape}.")

    kernel_sum = float(np.sum(kernel))
    t_kernel = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    cur = t_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    for _ in range(iterations):
        # Convolution sum == kernel_sum means structuring element is completely covered by foreground
        conv = F.conv2d(cur, t_kernel, padding=(pad_h, pad_w))
        cur = (conv >= (kernel_sum - 1e-4)).float()

    out_bool = cur.squeeze(0).squeeze(0).bool()

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            v_mask = valid_mask.detach().bool()
        else:
            v_mask = torch.from_numpy(np.asarray(valid_mask, dtype=bool))
        out_bool = out_bool & v_mask

    return out_bool.cpu().numpy()


def binary_opening(
    binary_mask: Union[np.ndarray, torch.Tensor],
    structuring_element: Optional[np.ndarray] = None,
    iterations: int = 1,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Apply morphological opening (erosion followed by dilation).

    Removes small isolated noise peaks while preserving overall object geometry.
    """
    eroded = binary_erosion(binary_mask, structuring_element, iterations=iterations, valid_mask=valid_mask)
    opened = binary_dilation(eroded, structuring_element, iterations=iterations, valid_mask=valid_mask)
    return opened


def binary_closing(
    binary_mask: Union[np.ndarray, torch.Tensor],
    structuring_element: Optional[np.ndarray] = None,
    iterations: int = 1,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion).

    Fills small holes and gaps along galactic arms and boundaries.
    """
    dilated = binary_dilation(binary_mask, structuring_element, iterations=iterations, valid_mask=valid_mask)
    closed = binary_erosion(dilated, structuring_element, iterations=iterations, valid_mask=valid_mask)
    return closed
