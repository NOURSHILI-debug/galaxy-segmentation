"""Astronomical image enhancement for galaxy segmentation.

This module provides **optional, deterministic** nonlinear intensity
transforms designed for astronomical imagery.  They operate on
already-normalised ``[0, 1]`` float tensors and return ``[0, 1]`` float
tensors of the same shape and dtype.

Currently implemented
---------------------
- ``AsinhStretch`` — inverse hyperbolic sine intensity mapping.

Future controlled experiments (not yet implemented)
---------------------------------------------------
- CLAHE (Contrast-Limited Adaptive Histogram Equalisation):
  Requires either ``opencv-python`` (``cv2.createCLAHE``) or a non-trivial
  pure-PyTorch / NumPy implementation.  Planned as experiment **E3** once
  the project dependency list is extended.  Do **not** add a custom CLAHE
  implementation merely to avoid installing OpenCV.

Tensor conventions
------------------
- Input and output: ``torch.float32`` in ``[0, 1]``.
- Supported shapes: ``[C, H, W]`` or ``[B, C, H, W]``.
- The input tensor is **never mutated**.

Integration note
----------------
Enhancement is applied **after** Person A's global normalisation
(which maps raw FITS flux to ``[0, 1]``) and **before** the
augmentation pipeline (``src/preprocessing/augmentation.py``).
"""

from __future__ import annotations

from typing import Union

import torch


class AsinhStretch:
    r"""Inverse hyperbolic sine intensity stretch.

    Maps normalised pixel intensities ``x ∈ [0, 1]`` through:

    .. math::

        f(x) = \frac{\operatorname{asinh}(x / \beta)}
                     {\operatorname{asinh}(1 / \beta)}

    This compresses high dynamic range while preserving linearity near
    the noise floor — the standard nonlinear stretch used in professional
    astronomical imaging (Lupton et al. 2004).

    Parameters
    ----------
    beta : float
        Softness / knee parameter.  Smaller values produce a more
        aggressive stretch (higher contrast for faint structures).
        Must be strictly positive.  Recommended range: ``0.01``–``0.5``.
        Default: ``0.1``.

    Raises
    ------
    ValueError
        If *beta* is not a finite positive number.

    Examples
    --------
    >>> stretch = AsinhStretch(beta=0.1)
    >>> img = torch.rand(5, 208, 208)
    >>> enhanced = stretch(img)
    >>> assert enhanced.shape == img.shape
    >>> assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    """

    def __init__(self, beta: float = 0.1) -> None:
        if not isinstance(beta, (int, float)):
            raise TypeError(
                f"beta must be a number; received {type(beta).__name__}."
            )
        beta = float(beta)
        if not (beta > 0 and _isfinite(beta)):
            raise ValueError(
                f"beta must be a finite positive number; received {beta}."
            )
        self._beta: float = beta
        # Pre-compute the normalisation denominator (a Python float).
        self._norm: float = float(torch.asinh(torch.tensor(1.0 / beta)))

    @property
    def beta(self) -> float:
        """The softness parameter."""
        return self._beta

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the asinh stretch to *image*.

        Args:
            image: Float tensor with values in ``[0, 1]``.
                   Shape ``[C, H, W]`` or ``[B, C, H, W]``.

        Returns:
            Enhanced float32 tensor in ``[0, 1]``, same shape.

        Raises:
            TypeError:  If *image* is not a floating-point tensor.
            ValueError: If *image* has fewer than 3 dimensions.
        """
        _validate_image(image)
        # Operate on a copy — never mutate the input.
        stretched = torch.asinh(image / self._beta) / self._norm
        # Clamp to [0, 1] for numerical safety (e.g. from tiny
        # floating-point overshoot at the boundaries).
        stretched = stretched.clamp(0.0, 1.0)
        return stretched.to(dtype=torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta={self._beta})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isfinite(x: float) -> bool:
    """Return True if *x* is finite (not NaN, not ±inf)."""
    import math
    return math.isfinite(x)


def _validate_image(image: torch.Tensor) -> None:
    """Shared validation for enhancement inputs."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor; received {type(image).__name__}."
        )
    if not image.is_floating_point():
        raise TypeError(
            f"Image must be a floating-point tensor; got dtype={image.dtype}."
        )
    if image.ndim < 3:
        raise ValueError(
            f"Image must have at least 3 dimensions [C, H, W] or "
            f"[B, C, H, W]; got ndim={image.ndim}."
        )
    if not torch.isfinite(image).all():
        raise ValueError(
            "Image contains non-finite values (NaN or Inf); invalid pixels "
            "must be masked or zeroed before enhancement."
        )
    eps = 1e-6
    if (image < -eps).any() or (image > 1.0 + eps).any():
        raise ValueError(
            f"Image values must be normalized in [0, 1]; "
            f"got min={image.min().item():.4f}, max={image.max().item():.4f}."
        )
