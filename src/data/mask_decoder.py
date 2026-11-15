"""Strict conversion from benchmark raw mask values to training targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


IGNORE_INDEX = 255
NUM_CLASSES = 4
RAW_TO_TRAINING_ID = {
    0: IGNORE_INDEX,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}
TRAINING_ID_TO_CLASS_NAME = {
    0: "Galaxy center or core",
    1: "Galaxy bar",
    2: "Galaxy arms",
    3: "Star",
}


class MaskDecodeError(ValueError):
    """Raised when a raw mask contains unsupported or invalid values."""


def decode_mask(mask: Any) -> torch.LongTensor:
    """Decode a 2D raw categorical mask to PyTorch training IDs.

    Raw value 0 becomes ``IGNORE_INDEX`` (255). Raw values 1 through 4 become
    contiguous training IDs 0 through 3. Values are validated before mapping;
    the input array is never modified.

    Args:
        mask: Integer or floating-point NumPy-compatible mask. Floating-point
            arrays are accepted only when every value is finite and exactly
            integer-like.

    Returns:
        A CPU ``torch.LongTensor`` with shape ``[H, W]`` and dtype
        ``torch.int64``.

    Raises:
        MaskDecodeError: If the input dtype is unsupported, any value is
            non-finite or non-integer-like, or a raw value is not in 0..4.
    """
    raw = np.asarray(mask)
    if raw.ndim != 2:
        raise MaskDecodeError(
            f"Mask must be two-dimensional with shape [H, W]; received {raw.ndim} dimensions."
        )
    if raw.dtype.kind not in "iuf":
        raise MaskDecodeError(
            "Mask dtype must be an integer or floating-point numeric type; "
            f"received {raw.dtype}."
        )

    if raw.size == 0:
        return torch.empty(raw.shape, dtype=torch.int64)

    if raw.dtype.kind == "f":
        finite = np.isfinite(raw)
        if not bool(np.all(finite)):
            invalid_count = int(raw.size - np.count_nonzero(finite))
            raise MaskDecodeError(
                f"Mask contains {invalid_count} non-finite value(s); "
                "NaN and infinity cannot be decoded."
            )
        integer_like = np.equal(raw, np.floor(raw))
        if not bool(np.all(integer_like)):
            invalid_values = np.unique(raw[~integer_like]).tolist()
            raise MaskDecodeError(
                "Mask contains non-integer-like value(s): "
                f"{invalid_values}. Raw mask values must be whole numbers."
            )

    known = np.isin(raw, tuple(RAW_TO_TRAINING_ID))
    if not bool(np.all(known)):
        unknown_values = np.unique(raw[~known]).tolist()
        raise MaskDecodeError(
            f"Mask contains unexpected raw value(s): {unknown_values}. "
            f"Expected only {sorted(RAW_TO_TRAINING_ID)}."
        )

    decoded = np.empty(raw.shape, dtype=np.int64)
    for raw_value, training_id in RAW_TO_TRAINING_ID.items():
        decoded[raw == raw_value] = training_id
    return torch.from_numpy(decoded)
