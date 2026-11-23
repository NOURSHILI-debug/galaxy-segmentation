"""Reusable image / mask patch extraction for galaxy segmentation.

Extracts spatially aligned image and segmentation-mask patches from a
single image / mask pair.  Designed to work with synthetic PyTorch tensors
and independent of the real astronomy dataset or any Person A deliverable
(manifest, splits, FITS loader, normalizer).

The extractor covers the *entire* original image: boundary pixels are never
silently discarded.  When a patch would extend past the right or bottom
edge, the image / mask are padded (constant value) rather than resized or
interpolated.  Image padding uses a configurable fill value; mask padding
always uses the project ignore index (255) so padded locations are excluded
from loss / metrics downstream.

Image and mask patches are extracted from identical coordinates, so
pixel-level alignment is guaranteed.  Original pixel values are never
modified; dtypes are preserved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Mask padding / ignore value.  Kept consistent with
# src.data.mask_decoder.IGNORE_INDEX (raw mask value 0 -> 255).
MASK_IGNORE_INDEX = 255

# Image dtypes we accept: any floating-point or integer type.
# Bool and complex are rejected.
_IMAGE_DTYPES = frozenset(
    {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

# Mask dtypes we accept: integer only (bool and float are rejected).
_MASK_DTYPES = frozenset({torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64})


def _positions(length: int, patch_size: int, stride: int) -> List[int]:
    """Return the top / left start positions that cover ``length``.

    Positions begin at 0 and advance by ``stride``.  A final position at
    ``length - patch_size`` is appended when needed so that the last patch
    reaches the bottom / right edge exactly.  When ``length <= patch_size``
    a single position (0) is returned and the patch is padded.
    """
    if length <= patch_size:
        return [0]
    num_steps = (length - patch_size) // stride
    positions = [i * stride for i in range(num_steps + 1)]
    if positions[-1] < length - patch_size:
        positions.append(length - patch_size)
    return positions


def extract_patches(
    image: torch.Tensor,
    mask: torch.Tensor,
    sample_id: Any,
    patch_size: int = 256,
    stride: int = 256,
    image_pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Extract spatially aligned image and mask patches.

    Args:
        image: Float or integer tensor ``[C, H, W]``.
        mask: Integer tensor ``[H, W]`` with categorical training IDs
            (0..3) and ignore index 255.
        sample_id: Identifier for the original image / mask pair.  Stored
            verbatim in every patch metadata entry.
        patch_size: Patch height / width in pixels (default 256).
        stride: Stride between consecutive patch starts (default 256).
        image_pad_value: Fill value used when padding the image
            (default 0.0).

    Returns:
        A tuple ``(image_patches, mask_patches, metadata)`` where:

        * ``image_patches``: ``torch.Tensor`` ``[N, C, patch_size,
          patch_size]`` with the same dtype as ``image``.
        * ``mask_patches``: ``torch.Tensor`` ``[N, patch_size,
          patch_size]`` with the same dtype as ``mask``.
        * ``metadata``: list of ``N`` dicts, one per patch, in row-major
          order (top-major, then left).  Each dict contains:
          ``sample_id``, ``patch_index``, ``top``, ``left``,
          ``original_height``, ``original_width``, ``patch_height``,
          ``patch_width``.

    Raises:
        TypeError: If ``image`` / ``mask`` are not tensors, or use an
            unsupported dtype (mask must be integer; bool / float rejected).
        ValueError: If spatial dimensions are incompatible, or
            ``patch_size`` / ``stride`` are not positive integers.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"image must be a torch.Tensor; received {type(image).__name__}."
        )
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"mask must be a torch.Tensor; received {type(mask).__name__}."
        )
    if image.ndim != 3:
        raise ValueError(
            f"image must be 3-dimensional [C, H, W]; got {image.ndim}D "
            f"with shape {tuple(image.shape)}."
        )
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-dimensional [H, W]; got {mask.ndim}D "
            f"with shape {tuple(mask.shape)}."
        )
    if image.shape[1:] != mask.shape:
        raise ValueError(
            f"image spatial shape {tuple(image.shape[1:])} does not match "
            f"mask shape {tuple(mask.shape)}."
        )
    if not isinstance(patch_size, int) or isinstance(patch_size, bool):
        raise ValueError(
            f"patch_size must be a positive integer; got {patch_size!r}."
        )
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1; got {patch_size}.")
    if not isinstance(stride, int) or isinstance(stride, bool):
        raise ValueError(
            f"stride must be a positive integer; got {stride!r}."
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}.")
    if image.dtype not in _IMAGE_DTYPES:
        raise TypeError(
            f"image dtype {image.dtype} is not supported.  "
            f"Use a floating-point or integer tensor."
        )
    if mask.dtype not in _MASK_DTYPES:
        raise TypeError(
            f"mask dtype {mask.dtype} is not supported.  "
            f"Mask must be an integer tensor (e.g. torch.int64); "
            f"bool and floating-point masks are rejected."
        )

    _, orig_h, orig_w = image.shape

    # ------------------------------------------------------------------
    # Determine start positions covering the full height / width.
    # ------------------------------------------------------------------
    tops = _positions(orig_h, patch_size, stride)
    lefts = _positions(orig_w, patch_size, stride)

    # Pad once (enough for the furthest-reaching patch) rather than per
    # patch, then slice.  No resizing or interpolation is involved.
    max_top = tops[-1]
    max_left = lefts[-1]
    pad_h = max(0, max_top + patch_size - orig_h)
    pad_w = max(0, max_left + patch_size - orig_w)

    padded_image = F.pad(
        image, (0, pad_w, 0, pad_h), mode="constant", value=image_pad_value
    )
    padded_mask = F.pad(
        mask, (0, pad_w, 0, pad_h), mode="constant", value=MASK_IGNORE_INDEX
    )

    # ------------------------------------------------------------------
    # Extract patches at identical (top, left) coordinates -> aligned.
    # ------------------------------------------------------------------
    image_patches: List[torch.Tensor] = []
    mask_patches: List[torch.Tensor] = []
    metadata: List[Dict[str, Any]] = []

    patch_index = 0
    for top in tops:
        for left in lefts:
            image_patch = padded_image[
                :, top : top + patch_size, left : left + patch_size
            ]
            mask_patch = padded_mask[
                top : top + patch_size, left : left + patch_size
            ]

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
            metadata.append(
                {
                    "sample_id": sample_id,
                    "patch_index": patch_index,
                    "top": top,
                    "left": left,
                    "original_height": orig_h,
                    "original_width": orig_w,
                    "patch_height": patch_size,
                    "patch_width": patch_size,
                }
            )
            patch_index += 1

    image_patches = torch.stack(image_patches)  # [N, C, P, P]
    mask_patches = torch.stack(mask_patches)    # [N, P, P]

    return image_patches, mask_patches, metadata
