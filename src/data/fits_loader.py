"""FITS I/O helpers for dataset indexing and loading.

Read-only utilities. Never modify files under dataset/raw/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.io import fits


def open_fits_info(path: Path, deep: bool = False) -> Dict[str, Any]:
    """Inspect a FITS file and return structural information.

    Returns a dict with keys:
      - readable: bool
      - error: str or None
      - n_hdus: int or None
      - hdu_shapes: list of shape tuples (or None per HDU)
      - hdu_dtypes: list of str dtypes
      - image_shape: tuple or None (shape of primary/first image-like HDU data summary)
      - nonfinite_count: int or None (deep only, over all HDUs with array data)
      - raw_mask_values: sorted list of int (deep only, for mask files)
    """
    info: Dict[str, Any] = {
        "readable": False,
        "error": None,
        "n_hdus": None,
        "hdu_shapes": [],
        "hdu_dtypes": [],
        "image_shape": None,
        "nonfinite_count": None,
        "raw_mask_values": None,
    }
    try:
        with fits.open(path, memmap=False) as hdul:
            info["readable"] = True
            info["n_hdus"] = len(hdul)
            shapes = []
            dtypes = []
            nonfinite = 0
            mask_values = set()
            for hdu in hdul:
                data = hdu.data
                if data is None:
                    shapes.append(None)
                    dtypes.append(None)
                    continue
                arr = np.asarray(data)
                shapes.append(tuple(int(x) for x in arr.shape))
                dtypes.append(str(arr.dtype))
                if deep:
                    try:
                        nf = int(np.count_nonzero(~np.isfinite(arr)))
                    except Exception:
                        nf = None
                    if nf is not None:
                        nonfinite += nf
                    # raw mask values (integer-like)
                    try:
                        flat = arr.ravel()
                        # sample/check all if small; for 208x208 trivial
                        if np.issubdtype(arr.dtype, np.integer):
                            mask_values.update(int(v) for v in np.unique(flat))
                        elif np.issubdtype(arr.dtype, np.floating):
                            finite_flat = flat[np.isfinite(flat)]
                            if finite_flat.size:
                                # record integer-like values
                                mask_values.update(int(v) for v in np.unique(finite_flat))
                    except Exception:
                        pass
            info["hdu_shapes"] = shapes
            info["hdu_dtypes"] = dtypes
            # image_shape: first non-None shape
            for s in shapes:
                if s is not None:
                    info["image_shape"] = s
                    break
            if deep:
                info["nonfinite_count"] = nonfinite
                if mask_values:
                    info["raw_mask_values"] = sorted(mask_values)
                else:
                    info["raw_mask_values"] = []
        return info
    except Exception as e:  # noqa: BLE001 - must not crash on unreadable files
        info["error"] = f"{type(e).__name__}: {e}"
        return info
