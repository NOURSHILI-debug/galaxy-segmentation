"""Pure PyTorch functional spatial transforms for astronomical segmentation.

All spatial transforms operate synchronously on image, mask, and optional
validity-mask tensors to guarantee pixel-level alignment.  No interpolation
is used — only exact index permutations (flip, rot90) and constant-value
padding.

Tensor conventions
------------------
- **image** : ``torch.FloatTensor`` with shape ``[C, H, W]``.
- **mask**  : ``torch.LongTensor``  with shape ``[H, W]``.
  Values must be categorical training IDs (``{0, 1, 2, 3}`` or
  ``IGNORE_INDEX = 255``).
- **valid_mask** : ``torch.BoolTensor`` with shape ``[H, W]``, optional.
  ``True`` = finite/valid pixel, ``False`` = invalid.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Re-export the project-wide ignore index so downstream code can import it
# from this module without a circular dependency on mask_decoder.
IGNORE_INDEX: int = 255


# ---------------------------------------------------------------------------
# A1.  Horizontal flip
# ---------------------------------------------------------------------------

def functional_hflip(
    image: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Flip image, mask, and validity mask horizontally (left ↔ right).

    Args:
        image: Float tensor ``[C, H, W]``.
        mask:  Long tensor ``[H, W]``.
        valid_mask: Optional bool tensor ``[H, W]``.

    Returns:
        ``(flipped_image, flipped_mask, flipped_valid_mask)``
    """
    _validate_spatial_inputs(image, mask, valid_mask)
    out_image = torch.flip(image, dims=[-1])
    out_mask = torch.flip(mask, dims=[-1])
    out_valid = torch.flip(valid_mask, dims=[-1]) if valid_mask is not None else None
    return out_image, out_mask, out_valid


# ---------------------------------------------------------------------------
# A2.  Vertical flip
# ---------------------------------------------------------------------------

def functional_vflip(
    image: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Flip image, mask, and validity mask vertically (top ↔ bottom).

    Args:
        image: Float tensor ``[C, H, W]``.
        mask:  Long tensor ``[H, W]``.
        valid_mask: Optional bool tensor ``[H, W]``.

    Returns:
        ``(flipped_image, flipped_mask, flipped_valid_mask)``
    """
    _validate_spatial_inputs(image, mask, valid_mask)
    out_image = torch.flip(image, dims=[-2])
    out_mask = torch.flip(mask, dims=[-2])
    out_valid = torch.flip(valid_mask, dims=[-2]) if valid_mask is not None else None
    return out_image, out_mask, out_valid


# ---------------------------------------------------------------------------
# A3.  90-degree rotations (k × 90°, counter-clockwise)
# ---------------------------------------------------------------------------

def functional_rot90(
    image: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Rotate image, mask, and validity mask by *k* × 90° counter-clockwise.

    Uses ``torch.rot90`` which performs exact index permutation — no
    interpolation, no rounding, no sub-pixel artifacts.

    Args:
        image: Float tensor ``[C, H, W]``.
        mask:  Long tensor ``[H, W]``.
        k:     Number of 90° rotations.  Must be in ``{0, 1, 2, 3}``.
        valid_mask: Optional bool tensor ``[H, W]``.

    Returns:
        ``(rotated_image, rotated_mask, rotated_valid_mask)``

    Raises:
        ValueError: If *k* is not in ``{0, 1, 2, 3}``.
    """
    if k not in (0, 1, 2, 3):
        raise ValueError(f"k must be 0, 1, 2 or 3; received {k}.")
    _validate_spatial_inputs(image, mask, valid_mask)
    if k == 0:
        return image, mask, valid_mask
    out_image = torch.rot90(image, k, dims=[-2, -1])
    out_mask = torch.rot90(mask, k, dims=[-2, -1])
    out_valid = (
        torch.rot90(valid_mask, k, dims=[-2, -1])
        if valid_mask is not None
        else None
    )
    return out_image, out_mask, out_valid


# ---------------------------------------------------------------------------
# A4.  Symmetric constant padding
# ---------------------------------------------------------------------------

def functional_pad(
    image: torch.Tensor,
    mask: torch.Tensor,
    target_h: int,
    target_w: int,
    valid_mask: Optional[torch.Tensor] = None,
    image_fill: float = 0.0,
    mask_fill: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Symmetrically pad image, mask, and validity mask to a target size.

    Padding is applied symmetrically (centred).  When the required padding
    is odd, the extra pixel is placed on the right / bottom.

    Args:
        image: Float tensor ``[C, H, W]``.
        mask:  Long tensor ``[H, W]``.
        target_h: Target height (must be ≥ current height).
        target_w: Target width  (must be ≥ current width).
        valid_mask: Optional bool tensor ``[H, W]``.
        image_fill: Fill value for image padding (default ``0.0``).
        mask_fill:  Fill value for mask padding (default ``255``).

    Returns:
        ``(padded_image, padded_mask, padded_valid_mask)``

    Raises:
        ValueError: If the target dimensions are smaller than the current
            dimensions.
    """
    _validate_spatial_inputs(image, mask, valid_mask)
    _, cur_h, cur_w = image.shape
    if target_h < cur_h or target_w < cur_w:
        raise ValueError(
            f"Target size ({target_h}, {target_w}) must be ≥ current size "
            f"({cur_h}, {cur_w}).  Cropping is not supported."
        )
    if target_h == cur_h and target_w == cur_w:
        return image, mask, valid_mask

    pad_h = target_h - cur_h
    pad_w = target_w - cur_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad uses (left, right, top, bottom) ordering.
    pad_spec = (pad_left, pad_right, pad_top, pad_bottom)

    out_image = F.pad(image, pad_spec, mode="constant", value=image_fill)
    out_mask = F.pad(mask, pad_spec, mode="constant", value=mask_fill)
    out_valid = (
        F.pad(valid_mask, pad_spec, mode="constant", value=False)
        if valid_mask is not None
        else None
    )
    return out_image, out_mask, out_valid


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------

def _validate_spatial_inputs(
    image: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    """Check that image/mask/valid_mask shapes and dtypes are compatible."""
    if image.ndim != 3:
        raise ValueError(
            f"Image must have 3 dimensions [C, H, W]; got ndim={image.ndim}."
        )
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must have 2 dimensions [H, W]; got ndim={mask.ndim}."
        )
    if not image.is_floating_point():
        raise TypeError(
            f"Image must be a floating-point tensor; got dtype={image.dtype}."
        )
    if mask.dtype != torch.int64:
        raise TypeError(
            f"Mask must be torch.int64 (LongTensor); got dtype={mask.dtype}."
        )
    _, h, w = image.shape
    if mask.shape != (h, w):
        raise ValueError(
            f"Mask spatial shape {tuple(mask.shape)} does not match "
            f"image spatial shape ({h}, {w})."
        )
    if valid_mask is not None:
        if valid_mask.ndim != 2:
            raise ValueError(
                f"Validity mask must have 2 dimensions [H, W]; "
                f"got ndim={valid_mask.ndim}."
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"Validity mask must be torch.bool; got dtype={valid_mask.dtype}."
            )
        if valid_mask.shape != (h, w):
            raise ValueError(
                f"Validity mask shape {tuple(valid_mask.shape)} does not "
                f"match image spatial shape ({h}, {w})."
            )
