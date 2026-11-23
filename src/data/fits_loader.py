"""Read-only FITS ingestion for the NASA Galaxy Segmentation dataset.

This module is deliberately limited to *loading*.  It converts FITS files into
PyTorch tensors while preserving the raw scientific values; it does **not**
normalize, background-subtract, enhance, augment, resample, or extract patches.
Those operations belong to :mod:`src.preprocessing` and must remain separable
from ingestion so that the effect of astronomy-aware preprocessing can be
measured experimentally.

Image files
-----------
The local benchmark files contain five two-dimensional image HDUs, one per
Pan-STARRS filter.  The documented filter order is::

    HDU 0 -> g
    HDU 1 -> r
    HDU 2 -> i
    HDU 3 -> z
    HDU 4 -> y

**Band identity is positional.  It is not verified by FITS headers.**  The
audited files have no ``FILTER`` card and no ``BUNIT`` value, and HDUs 1-4 have
empty ``EXTNAME`` values.  The order above therefore comes from the benchmark
documentation and must be treated as a documented assumption, not a
header-verified fact.  :data:`BAND_ORDER_IS_HEADER_VERIFIED` records this
explicitly.

Mask files
----------
Mask files contain one two-dimensional, integer-valued primary HDU.  Raw
categorical values are decoded with the frozen policy in
:mod:`src.data.mask_decoder`; this module does not redefine class semantics or
the ignore index.

Non-finite pixels
-----------------
Image pixels may contain NaN (the audit found no Inf, but the loader treats both
uniformly).  Non-finite values are **never** replaced or repaired here.  The
caller receives a per-pixel validity mask and per-band non-finite counts so it
can decide how to handle them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from astropy.io import fits

from src.data.mask_decoder import MaskDecodeError, decode_mask

__all__ = [
    "BAND_ORDER_IS_HEADER_VERIFIED",
    "IMAGE_BAND_COUNT",
    "IMAGE_BAND_NAMES",
    "FitsLoadError",
    "load_image_fits",
    "load_mask_fits",
]

#: Documented Pan-STARRS filter order of the five image HDUs.
IMAGE_BAND_NAMES: tuple[str, ...] = ("g", "r", "i", "z", "y")

#: Number of image HDUs in a conformant benchmark image file.
IMAGE_BAND_COUNT: int = len(IMAGE_BAND_NAMES)

#: False because the filter order is documented, not encoded in FITS headers.
BAND_ORDER_IS_HEADER_VERIFIED: bool = False

_IMAGE_NAME = re.compile(r"^imagedata_(.+)\.fits(?:\.gz)?$", re.IGNORECASE)
_MASK_NAME = re.compile(r"^mask_(.+)\.fits(?:\.gz)?$", re.IGNORECASE)


class FitsLoadError(ValueError):
    """Raised when a FITS file does not match the expected ingestion contract."""


def _resolve_existing_file(fits_path: str | Path) -> Path:
    """Return an existing regular file or raise a clear exception."""
    path = Path(fits_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")
    if not path.is_file():
        raise FitsLoadError(f"FITS path is not a regular file: {path}")
    return path


def _filename_stem(path: Path) -> str:
    """Strip ``.fits`` / ``.fits.gz`` suffixes from a filename."""
    name = path.name
    lowered = name.lower()
    if lowered.endswith(".gz"):
        name = name[:-3]
        lowered = name.lower()
    if lowered.endswith(".fits"):
        name = name[:-5]
    return name


def _resolve_sample_id(
    path: Path,
    sample_id: str | None,
    pattern: re.Pattern[str],
) -> str:
    """Return an explicit or filename-derived sample identifier."""
    if sample_id is not None:
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise FitsLoadError(
                f"sample_id must be a non-empty string when provided; received {sample_id!r}."
            )
        return sample_id
    match = pattern.match(path.name)
    if match:
        return match.group(1)
    return _filename_stem(path)


def _normalize_band_indices(
    band_indices: Iterable[int] | None,
) -> tuple[int, ...]:
    """Validate and normalize an explicit image-HDU selection.

    ``None`` selects all five documented bands.  Otherwise the selection must
    be a non-empty iterable of unique integer indices in ``0..4``.  Order is
    preserved, so ``(2, 0)`` yields channels in i, g order.
    """
    if band_indices is None:
        return tuple(range(IMAGE_BAND_COUNT))
    if isinstance(band_indices, (str, bytes)):
        raise FitsLoadError(
            "band_indices must be an iterable of integers or None, not a string."
        )
    try:
        candidates = tuple(band_indices)
    except TypeError as exc:
        raise FitsLoadError(
            "band_indices must be an iterable of integers or None; "
            f"received {type(band_indices).__name__}."
        ) from exc
    if not candidates:
        raise FitsLoadError("band_indices must not be empty.")

    normalized: list[int] = []
    for value in candidates:
        # bool is an int subclass but is not a meaningful HDU index.
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise FitsLoadError(
                "band_indices must contain integers; "
                f"received {value!r} of type {type(value).__name__}."
            )
        index = int(value)
        if not 0 <= index < IMAGE_BAND_COUNT:
            raise FitsLoadError(
                f"band index {index} is out of range; expected 0..{IMAGE_BAND_COUNT - 1}."
            )
        normalized.append(index)

    if len(set(normalized)) != len(normalized):
        raise FitsLoadError(
            f"band_indices contains duplicate indices: {normalized}."
        )
    return tuple(normalized)


def _band_names(band_indices: Sequence[int]) -> tuple[str, ...]:
    """Map selected HDU indices to documented band names."""
    return tuple(IMAGE_BAND_NAMES[index] for index in band_indices)


def load_image_fits(
    fits_path: str | Path,
    band_indices: Iterable[int] | None = None,
    sample_id: str | None = None,
) -> dict:
    """Load a five-band image FITS file into raw, model-ready tensors.

    Args:
        fits_path: Path to an ``imagedata_<sample_id>.fits`` file.
        band_indices: Explicit image-HDU selection.  ``None`` loads all five
            documented bands in order.  Indices must be unique integers in
            ``0..4``; the selection order is preserved in the output channels.
        sample_id: Optional explicit sample identifier.  When omitted it is
            derived from the filename (``imagedata_<id>.fits``).

    Returns:
        A dictionary with the following keys:

        ``image``
            ``torch.float32`` tensor of shape ``[C, H, W]`` in native byte
            order.  Values are the raw FITS flux; negative values and values
            outside ``[0, 1]`` are preserved exactly.  Non-finite pixels are
            preserved (not replaced).
        ``valid_mask``
            ``torch.bool`` tensor of shape ``[H, W]``.  ``True`` only where
            every selected band is finite.
        ``sample_id``
            Sample identifier.
        ``source_path``
            Path of the FITS file as a string.
        ``band_indices``
            Selected image-HDU indices, in output-channel order.
        ``band_names``
            Documented filter names corresponding to ``band_indices``.
        ``shape``
            ``(H, W)`` spatial shape.
        ``non_finite_counts``
            Mapping ``{hdu_index: count}`` of non-finite pixel counts for each
            selected band.

    Raises:
        FileNotFoundError: If ``fits_path`` does not exist.
        FitsLoadError: If the file is not a regular file, does not have exactly
            five data-bearing image HDUs, has a non-2D or non-floating HDU, has
            inconsistent band shapes, or if ``band_indices`` is invalid.
    """
    path = _resolve_existing_file(fits_path)
    selected = _normalize_band_indices(band_indices)
    selected_set = set(selected)
    resolved_id = _resolve_sample_id(path, sample_id, _IMAGE_NAME)

    try:
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            if len(hdul) != IMAGE_BAND_COUNT:
                raise FitsLoadError(
                    f"{path.name}: expected exactly {IMAGE_BAND_COUNT} image HDUs, "
                    f"found {len(hdul)}."
                )

            reference_shape: tuple[int, int] | None = None
            band_arrays: dict[int, np.ndarray] = {}
            non_finite_counts: dict[int, int] = {}

            for index in range(IMAGE_BAND_COUNT):
                data = hdul[index].data
                if data is None:
                    raise FitsLoadError(
                        f"{path.name}: image HDU {index} ({IMAGE_BAND_NAMES[index]}) "
                        "contains no data."
                    )
                array = np.asarray(data)
                if array.ndim != 2:
                    raise FitsLoadError(
                        f"{path.name}: image HDU {index} ({IMAGE_BAND_NAMES[index]}) "
                        f"must be two-dimensional, found {array.ndim} dimensions "
                        f"with shape {array.shape}."
                    )
                if array.shape[0] < 1 or array.shape[1] < 1:
                    raise FitsLoadError(
                        f"{path.name}: image HDU {index} has an empty spatial shape "
                        f"{array.shape}."
                    )
                if not np.issubdtype(array.dtype, np.floating):
                    raise FitsLoadError(
                        f"{path.name}: image HDU {index} ({IMAGE_BAND_NAMES[index]}) "
                        f"has non-floating dtype {array.dtype}; expected float32 data."
                    )

                shape = (int(array.shape[0]), int(array.shape[1]))
                if reference_shape is None:
                    reference_shape = shape
                elif shape != reference_shape:
                    raise FitsLoadError(
                        f"{path.name}: image HDU {index} shape {shape} does not match "
                        f"the first band shape {reference_shape}."
                    )

                if index in selected_set:
                    # Copy inside the open context: values must survive file close.
                    # ``np.float32`` is native-endian, so this converts the FITS
                    # big-endian ``>f4`` layout without altering finite values.
                    native = np.array(array, dtype=np.float32, copy=True)
                    finite = np.isfinite(native)
                    non_finite_counts[index] = int(native.size - np.count_nonzero(finite))
                    band_arrays[index] = native
    except FitsLoadError:
        raise
    except Exception as exc:  # astropy read/verification failure at the boundary
        raise FitsLoadError(f"Failed to read image FITS file {path}: {exc}") from exc

    if reference_shape is None:  # defensive; unreachable after HDU validation
        raise FitsLoadError(f"{path.name}: no image bands were validated.")
    stacked = np.stack([band_arrays[index] for index in selected], axis=0)
    valid = np.ones(reference_shape, dtype=bool)
    for index in selected:
        valid &= np.isfinite(band_arrays[index])

    return {
        "image": torch.from_numpy(stacked),
        "valid_mask": torch.from_numpy(valid),
        "sample_id": resolved_id,
        "source_path": str(path),
        "band_indices": selected,
        "band_names": _band_names(selected),
        "shape": reference_shape,
        "non_finite_counts": {
            index: non_finite_counts[index] for index in selected
        },
    }


def load_mask_fits(
    fits_path: str | Path,
    sample_id: str | None = None,
) -> dict:
    """Load a mask FITS file and decode it with the frozen class mapping.

    The primary HDU must contain a two-dimensional, finite, integer-valued
    categorical mask.  Decoding is delegated to
    :func:`src.data.mask_decoder.decode_mask`, so raw value 0 becomes
    ``IGNORE_INDEX`` (255) and raw values 1-4 become training IDs 0-3.

    Args:
        fits_path: Path to a ``mask_<sample_id>.fits`` file.
        sample_id: Optional explicit sample identifier.  When omitted it is
            derived from the filename (``mask_<id>.fits``).

    Returns:
        A dictionary with the following keys:

        ``mask``
            ``torch.int64`` tensor of shape ``[H, W]`` with values in
            ``{0, 1, 2, 3, IGNORE_INDEX}``.
        ``sample_id``
            Sample identifier.
        ``source_path``
            Path of the FITS file as a string.
        ``shape``
            ``(H, W)`` spatial shape.
        ``raw_unique``
            Sorted raw categorical values observed before decoding.

    Raises:
        FileNotFoundError: If ``fits_path`` does not exist.
        FitsLoadError: If the file is not a regular file, does not contain
            exactly one data-bearing primary HDU, or the mask is not 2D.
        MaskDecodeError: If the mask contains non-finite, non-integer-like, or
            out-of-range raw values (raised by :func:`decode_mask`).
    """
    path = _resolve_existing_file(fits_path)
    resolved_id = _resolve_sample_id(path, sample_id, _MASK_NAME)

    try:
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            data_hdu_indices = [
                index for index, hdu in enumerate(hdul) if hdu.data is not None
            ]
            if not data_hdu_indices:
                raise FitsLoadError(f"{path.name}: mask file contains no data-bearing HDU.")
            if data_hdu_indices != [0]:
                raise FitsLoadError(
                    f"{path.name}: expected exactly one data-bearing primary HDU, "
                    f"found data in HDUs {data_hdu_indices}."
                )

            array = np.asarray(hdul[0].data)
            if array.ndim != 2:
                raise FitsLoadError(
                    f"{path.name}: mask must be two-dimensional, found "
                    f"{array.ndim} dimensions with shape {array.shape}."
                )
            if array.size == 0:
                raise FitsLoadError(f"{path.name}: mask has an empty spatial shape {array.shape}.")

            shape = (int(array.shape[0]), int(array.shape[1]))
            # decode_mask validates finiteness, integer-like values, and the
            # allowed raw range 0..4 before mapping.
            decoded = decode_mask(array)
            raw_unique = [int(value) for value in np.unique(array)]
    except FitsLoadError:
        raise
    except MaskDecodeError:
        # Semantic mask errors propagate unchanged so callers can distinguish
        # invalid class values from structural FITS-read failures.
        raise
    except Exception as exc:
        raise FitsLoadError(f"Failed to read mask FITS file {path}: {exc}") from exc

    return {
        "mask": decoded,
        "sample_id": resolved_id,
        "source_path": str(path),
        "shape": shape,
        "raw_unique": raw_unique,
    }
