"""Synchronized image/mask augmentation pipeline for galaxy segmentation.

Every transform in this module applies identical geometric operations to
the image, segmentation mask **and** optional validity mask so that
pixel-level alignment is maintained at all times.  Photometric operations
modify only the image.

Tensor conventions (per-sample, unbatched)
------------------------------------------
- **image** : ``torch.float32`` with shape ``[C, H, W]``.
- **mask**  : ``torch.int64``   with shape ``[H, W]``.
  Categorical training IDs ``{0, 1, 2, 3}`` plus ``IGNORE_INDEX = 255``.
- **valid_mask** : ``torch.bool`` with shape ``[H, W]``, or ``None``.

Reproducibility
---------------
Stochastic transforms accept an optional ``torch.Generator`` so that
two pipelines sharing the same seed produce **identical** results.
When no generator is supplied the transforms fall back to the default
PyTorch RNG.

Usage
-----
>>> from src.preprocessing.augmentation import build_augmentation_pipeline
>>> pipeline = build_augmentation_pipeline("configs/augmentation.yaml", split="train")
>>> aug_img, aug_mask, aug_valid = pipeline(image, mask, valid_mask)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import yaml

from src.preprocessing.transforms import (
    IGNORE_INDEX,
    _validate_spatial_inputs,
    functional_hflip,
    functional_pad,
    functional_rot90,
    functional_vflip,
)


# ===================================================================
# C1.  DualCompose — compose a sequence of dual transforms
# ===================================================================

class DualCompose:
    """Compose several synchronised image/mask transforms.

    Parameters
    ----------
    transforms : sequence of callables
        Each must accept ``(image, mask, valid_mask)`` and return the
        same triple.
    """

    def __init__(self, transforms: Sequence) -> None:
        self._transforms = list(transforms)

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        for t in self._transforms:
            image, mask, valid_mask = t(image, mask, valid_mask)
        return image, mask, valid_mask

    def __repr__(self) -> str:
        lines = [f"  {t!r}" for t in self._transforms]
        body = ",\n".join(lines)
        return f"{self.__class__.__name__}([\n{body}\n])"


# ===================================================================
# C2.  DualRandomHorizontalFlip
# ===================================================================

class DualRandomHorizontalFlip:
    """Randomly flip image, mask, and validity mask horizontally.

    Parameters
    ----------
    p : float
        Probability of applying the flip.  Default ``0.5``.
    generator : torch.Generator, optional
        RNG for reproducibility.
    """

    def __init__(
        self,
        p: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        _validate_probability(p, "p")
        self._p = p
        self._generator = generator

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if _should_apply(self._p, self._generator):
            return functional_hflip(image, mask, valid_mask)
        return image, mask, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self._p})"


# ===================================================================
# C3.  DualRandomVerticalFlip
# ===================================================================

class DualRandomVerticalFlip:
    """Randomly flip image, mask, and validity mask vertically.

    Parameters
    ----------
    p : float
        Probability of applying the flip.  Default ``0.5``.
    generator : torch.Generator, optional
        RNG for reproducibility.
    """

    def __init__(
        self,
        p: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        _validate_probability(p, "p")
        self._p = p
        self._generator = generator

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if _should_apply(self._p, self._generator):
            return functional_vflip(image, mask, valid_mask)
        return image, mask, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self._p})"


# ===================================================================
# C4.  DualRandomRotate90
# ===================================================================

class DualRandomRotate90:
    """Randomly rotate image, mask, and validity mask by a 90° multiple.

    When activated, chooses *k* uniformly from ``{1, 2, 3}`` (i.e. 90°,
    180°, or 270° counter-clockwise).  ``k = 0`` (no-op) is excluded to
    ensure the rotation always changes the data when activated.

    Parameters
    ----------
    p : float
        Probability of applying a rotation.  Default ``0.5``.
    generator : torch.Generator, optional
        RNG for reproducibility.
    """

    def __init__(
        self,
        p: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        _validate_probability(p, "p")
        self._p = p
        self._generator = generator

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if _should_apply(self._p, self._generator):
            # k ∈ {1, 2, 3}
            gen_device = self._generator.device if self._generator is not None else "cpu"
            k = int(torch.randint(1, 4, (1,), generator=self._generator, device=gen_device).item())
            return functional_rot90(image, mask, k, valid_mask)
        return image, mask, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self._p})"


# ===================================================================
# C5.  RandomPhotometricJitter (image-only)
# ===================================================================

class RandomPhotometricJitter:
    """Apply random brightness and contrast jitter to the image only.

    The mask and validity mask are **never modified**.

    Brightness is applied as a multiplicative factor, and contrast as a
    multiplicative factor relative to the per-channel mean.  The same
    factor is used for **all channels** so that relative band calibration
    is preserved.

    After the jitter, the image is clamped to ``[0, 1]`` and pixels
    marked invalid by the validity mask are reset to ``0.0``.

    Parameters
    ----------
    brightness_range : tuple of (float, float)
        ``(low, high)`` inclusive range for the brightness factor.
        Default ``(0.9, 1.1)``.
    contrast_range : tuple of (float, float)
        ``(low, high)`` inclusive range for the contrast factor.
        Default ``(0.9, 1.1)``.
    p : float
        Probability of applying the jitter.  Default ``0.3``.
    generator : torch.Generator, optional
        RNG for reproducibility.
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.3,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        _validate_probability(p, "p")
        _validate_range(brightness_range, "brightness_range")
        _validate_range(contrast_range, "contrast_range")
        self._brightness_range = brightness_range
        self._contrast_range = contrast_range
        self._p = p
        self._generator = generator

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _validate_spatial_inputs(image, mask, valid_mask)
        if not torch.isfinite(image).all():
            raise ValueError(
                "Image contains non-finite values (NaN or Inf); invalid pixels "
                "must be masked and zeroed before jitter."
            )
        eps = 1e-6
        if (image < -eps).any() or (image > 1.0 + eps).any():
            raise ValueError(
                f"Image values must be normalized in [0, 1]; "
                f"got min={image.min().item():.4f}, max={image.max().item():.4f}."
            )
        if not _should_apply(self._p, self._generator):
            return image, mask, valid_mask

        # Sample a single brightness and contrast factor (shared across
        # all channels to preserve relative band calibration).
        brightness = _uniform(
            self._brightness_range[0],
            self._brightness_range[1],
            self._generator,
        )
        contrast = _uniform(
            self._contrast_range[0],
            self._contrast_range[1],
            self._generator,
        )

        out = image.clone()

        # Contrast: scale around the per-channel spatial mean.
        # When a validity mask is provided, compute the mean over valid
        # pixels only so that masked/invalid zero-regions do not distort
        # contrast statistics.
        if valid_mask is not None:
            valid_count = valid_mask.sum()
            if valid_count > 0:
                mean = (out * valid_mask.unsqueeze(0)).sum(dim=(-2, -1), keepdim=True) / valid_count
            else:
                mean = torch.zeros((out.shape[0], 1, 1), dtype=out.dtype, device=out.device)
        else:
            mean = out.mean(dim=(-2, -1), keepdim=True)

        out = contrast * (out - mean) + mean

        # Brightness: multiplicative.
        out = out * brightness

        # Clamp to valid range.
        out = out.clamp(0.0, 1.0)

        # Reset invalid pixels to 0.0 when a validity mask is provided.
        if valid_mask is not None:
            out[:, ~valid_mask] = 0.0

        return out, mask, valid_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_range={self._brightness_range}, "
            f"contrast_range={self._contrast_range}, "
            f"p={self._p})"
        )


# ===================================================================
# C6.  DualPadToSize — optional deterministic padding
# ===================================================================

class DualPadToSize:
    """Pad image, mask, and validity mask to a fixed target size.

    This is a **deterministic** transform (no randomness) and is safe
    to include in validation / test pipelines.

    Parameters
    ----------
    target_h : int
        Target height.
    target_w : int
        Target width.
    image_fill : float
        Fill value for image padding.  Default ``0.0``.
    mask_fill : int
        Fill value for mask padding.  Default ``255`` (ignore index).
    """

    def __init__(
        self,
        target_h: int,
        target_w: int,
        image_fill: float = 0.0,
        mask_fill: int = IGNORE_INDEX,
    ) -> None:
        if target_h < 1 or target_w < 1:
            raise ValueError(
                f"Target dimensions must be positive; got ({target_h}, {target_w})."
            )
        self._target_h = target_h
        self._target_w = target_w
        self._image_fill = image_fill
        self._mask_fill = mask_fill

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return functional_pad(
            image, mask,
            target_h=self._target_h,
            target_w=self._target_w,
            valid_mask=valid_mask,
            image_fill=self._image_fill,
            mask_fill=self._mask_fill,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_h={self._target_h}, target_w={self._target_w}, "
            f"mask_fill={self._mask_fill})"
        )


# ===================================================================
# Pipeline builder
# ===================================================================

def build_augmentation_pipeline(
    config: Union[str, Path, Dict[str, Any]],
    split: str = "train",
    generator: Optional[torch.Generator] = None,
) -> DualCompose:
    """Build an augmentation pipeline from a YAML file or dictionary.

    Parameters
    ----------
    config : str, Path, or dict
        Path to a YAML configuration file, or an already-loaded dict.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    generator : torch.Generator, optional
        RNG for reproducible stochastic transforms.  Ignored for
        deterministic ``val`` / ``test`` splits.

    Returns
    -------
    DualCompose
        A composed pipeline of dual transforms.

    Raises
    ------
    ValueError
        If *split* is invalid or required configuration keys are missing.
    FileNotFoundError
        If *config* is a path that does not exist.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'.")

    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise TypeError(
            f"config must be a dict or a path to a YAML file; "
            f"got {type(config).__name__}."
        )

    split_cfg = config.get(split)
    if split_cfg is None:
        raise ValueError(
            f"Configuration has no section for split '{split}'.  "
            f"Available keys: {list(config.keys())}."
        )

    transforms: List = []

    # --- Geometric (spatial) augmentations ---
    if split_cfg.get("horizontal_flip", False):
        p = float(split_cfg.get("horizontal_flip_p", 0.5))
        _validate_probability(p, "horizontal_flip_p")
        transforms.append(DualRandomHorizontalFlip(p=p, generator=generator))

    if split_cfg.get("vertical_flip", False):
        p = float(split_cfg.get("vertical_flip_p", 0.5))
        _validate_probability(p, "vertical_flip_p")
        transforms.append(DualRandomVerticalFlip(p=p, generator=generator))

    if split_cfg.get("random_rotate90", False):
        p = float(split_cfg.get("random_rotate90_p", 0.5))
        _validate_probability(p, "random_rotate90_p")
        transforms.append(DualRandomRotate90(p=p, generator=generator))

    # --- Photometric augmentation (image only) ---
    if split_cfg.get("photometric_jitter", False):
        p = float(split_cfg.get("photometric_jitter_p", 0.3))
        b_lo = float(split_cfg.get("brightness_min", 0.9))
        b_hi = float(split_cfg.get("brightness_max", 1.1))
        c_lo = float(split_cfg.get("contrast_min", 0.9))
        c_hi = float(split_cfg.get("contrast_max", 1.1))
        _validate_probability(p, "photometric_jitter_p")
        _validate_range((b_lo, b_hi), "brightness_range")
        _validate_range((c_lo, c_hi), "contrast_range")
        transforms.append(
            RandomPhotometricJitter(
                brightness_range=(b_lo, b_hi),
                contrast_range=(c_lo, c_hi),
                p=p,
                generator=generator,
            )
        )

    # --- Optional deterministic padding ---
    if split_cfg.get("pad", False):
        th = int(split_cfg.get("pad_target_h", 256))
        tw = int(split_cfg.get("pad_target_w", 256))
        transforms.append(DualPadToSize(target_h=th, target_w=tw))

    return DualCompose(transforms)


# ===================================================================
# Internal helpers
# ===================================================================

def _should_apply(p: float, generator: Optional[torch.Generator]) -> bool:
    """Return True with probability *p* using *generator*."""
    gen_device = generator.device if generator is not None else "cpu"
    return torch.rand(1, generator=generator, device=gen_device).item() < p


def _uniform(
    low: float,
    high: float,
    generator: Optional[torch.Generator],
) -> float:
    """Sample a uniform random float in ``[low, high]``."""
    gen_device = generator.device if generator is not None else "cpu"
    return low + (high - low) * torch.rand(1, generator=generator, device=gen_device).item()


def _validate_probability(p: float, name: str) -> None:
    """Assert that *p* is a valid probability in ``[0, 1]``."""
    if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
        raise ValueError(
            f"{name} must be a float in [0, 1]; got {p!r}."
        )


def _validate_range(
    rng: Tuple[float, float],
    name: str,
) -> None:
    """Assert that *rng* is a valid ``(low, high)`` pair."""
    if (
        not isinstance(rng, (tuple, list))
        or len(rng) != 2
        or not all(isinstance(v, (int, float)) for v in rng)
    ):
        raise ValueError(
            f"{name} must be a (low, high) tuple of two numbers; got {rng!r}."
        )
    low, high = rng
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{name} values must be finite; got ({low}, {high}).")
    if low > high:
        raise ValueError(
            f"{name} low ({low}) must be ≤ high ({high})."
        )
