"""Reusable PyTorch Dataset adapter for image / mask patch extraction.

Connects in-memory image / mask samples to the B6.1 patch extraction
utility (``src.data.patches.extract_patches``) and exposes deterministic,
patch-level indexing compatible with ``torch.utils.data.DataLoader``.

Memory strategy
---------------
The dataset does NOT materialize every extracted patch during
initialization.  For each sample it calls ``extract_patches`` once to
obtain the lightweight patch *metadata* (coordinates, counts, sample id)
and then discards the patch tensors, keeping only:

* a reference to the original image tensor ``[C, H, W]``,
* a reference to the original mask tensor ``[H, W]``,
* the per-patch metadata list (top, left, ...).

At ``__getitem__`` time the required patch is sliced from the original
tensors and padded to ``patch_size`` if it touches the image boundary.
Steady-state memory is therefore proportional to the original samples
plus small metadata, rather than to the total number of patches.

The patch *coordinates* come entirely from ``extract_patches`` (B6.1);
this module does not duplicate B6.1's position logic and does not modify
B6.1.

Note
----
Calling ``extract_patches`` during initialization transiently allocates
one sample's worth of patches at a time (freed immediately after the
metadata is kept).  For extremely large samples this is a brief peak;
steady-state memory remains bounded by the originals plus metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.patches import MASK_IGNORE_INDEX, extract_patches


def _as_sample(sample: Any, index: int) -> Dict[str, Any]:
    """Normalize a sample into a dict with keys ``sample_id``, ``image``,
    ``mask``.

    Accepted forms:
        * ``{"sample_id": ..., "image": ..., "mask": ...}`` (mapping)
        * ``(sample_id, image, mask)`` (3-tuple / list)
    """
    if isinstance(sample, Mapping):
        missing = [k for k in ("sample_id", "image", "mask") if k not in sample]
        if missing:
            raise ValueError(
                f"Sample at index {index} is a mapping missing keys: {missing}."
            )
        return {
            "sample_id": sample["sample_id"],
            "image": sample["image"],
            "mask": sample["mask"],
        }
    if isinstance(sample, (tuple, list)) and len(sample) == 3:
        return {
            "sample_id": sample[0],
            "image": sample[1],
            "mask": sample[2],
        }
    raise TypeError(
        f"Sample at index {index} must be a mapping with keys "
        f"'sample_id', 'image', 'mask' or a 3-tuple "
        f"(sample_id, image, mask); received {type(sample).__name__}."
    )


class PatchDataset(Dataset):
    """PyTorch Dataset yielding image / mask patches with metadata.

    Parameters
    ----------
    samples : sequence of sample-like
        Each sample provides ``sample_id``, ``image`` ``[C, H, W]`` and
        ``mask`` ``[H, W]``.  Synthetic tensors are fine; no FITS or
        Person A manifest is required.
    patch_size : int, default 256
        Patch height / width in pixels.
    stride : int, default 256
        Stride between consecutive patch starts.
    image_pad_value : float, default 0.0
        Fill value used when padding an image patch that touches the
        image boundary.  Mask padding always uses the ignore index 255.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        patch_size: int = 256,
        stride: int = 256,
        image_pad_value: float = 0.0,
    ) -> None:
        if not isinstance(patch_size, int) or isinstance(patch_size, bool):
            raise ValueError(
                f"patch_size must be a positive integer; got {patch_size!r}."
            )
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1; got {patch_size}.")
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise ValueError(f"stride must be a positive integer; got {stride!r}.")
        if stride < 1:
            raise ValueError(f"stride must be >= 1; got {stride}.")

        self.patch_size = patch_size
        self.stride = stride
        self.image_pad_value = float(image_pad_value)

        self._samples: List[Dict[str, Any]] = []
        # Flat, deterministic index: (sample_idx, patch_idx_within_sample).
        self._patch_index: List[Tuple[int, int]] = []

        for sample_idx, raw_sample in enumerate(samples):
            sample = _as_sample(raw_sample, sample_idx)
            image = sample["image"]
            mask = sample["mask"]

            # Validate the sample up front with a clear, index-scoped error.
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    f"Sample {sample_idx} image must be a torch.Tensor; "
                    f"received {type(image).__name__}."
                )
            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    f"Sample {sample_idx} mask must be a torch.Tensor; "
                    f"received {type(mask).__name__}."
                )
            if image.ndim != 3:
                raise ValueError(
                    f"Sample {sample_idx} image must be 3D [C, H, W]; "
                    f"got {image.ndim}D {tuple(image.shape)}."
                )
            if mask.ndim != 2:
                raise ValueError(
                    f"Sample {sample_idx} mask must be 2D [H, W]; "
                    f"got {mask.ndim}D {tuple(mask.shape)}."
                )
            if image.shape[1:] != mask.shape:
                raise ValueError(
                    f"Sample {sample_idx} image spatial shape "
                    f"{tuple(image.shape[1:])} does not match mask shape "
                    f"{tuple(mask.shape)}."
                )

            # Reuse B6.1 for coordinate logic + validation; keep only the
            # lightweight metadata and references to the original tensors.
            _, _, metadata = extract_patches(
                image,
                mask,
                sample["sample_id"],
                patch_size=patch_size,
                stride=stride,
                image_pad_value=image_pad_value,
            )

            self._samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "image": image,
                    "mask": mask,
                    "metadata": metadata,
                    "num_patches": len(metadata),
                }
            )
            for patch_idx in range(len(metadata)):
                self._patch_index.append((sample_idx, patch_idx))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not isinstance(idx, int):
            raise TypeError(
                f"PatchDataset indices must be integers; "
                f"got {type(idx).__name__}."
            )
        if idx < 0:
            idx += len(self._patch_index)
        if idx < 0 or idx >= len(self._patch_index):
            raise IndexError(
                f"PatchDataset index {idx} out of range "
                f"[0, {len(self._patch_index)})."
            )

        sample_idx, patch_idx = self._patch_index[idx]
        entry = self._samples[sample_idx]
        meta = entry["metadata"][patch_idx]

        top = meta["top"]
        left = meta["left"]
        p = self.patch_size

        image = entry["image"]
        mask = entry["mask"]

        # Slice from the original tensors (Python slicing clamps to bounds);
        # then pad to patch_size when the slice touches the image edge.
        img_patch = image[:, top : top + p, left : left + p]
        mask_patch = mask[top : top + p, left : left + p]

        pad_h = p - img_patch.shape[1]
        pad_w = p - img_patch.shape[2]
        if pad_h > 0 or pad_w > 0:
            img_patch = F.pad(
                img_patch, (0, pad_w, 0, pad_h), value=self.image_pad_value
            )
            mask_patch = F.pad(
                mask_patch, (0, pad_w, 0, pad_h), value=MASK_IGNORE_INDEX
            )

        return {
            "image": img_patch,
            "mask": mask_patch,
            "metadata": dict(meta),
        }


def patch_dataset_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of PatchDataset items into a batch.

    Stacks ``image`` and ``mask`` into batched tensors and keeps
    ``metadata`` as a list of per-patch dicts (one entry per patch in the
    batch).  This keeps metadata associated with the correct sample /
    patch without forcing string fields such as ``sample_id`` into tensors.

    Returns:
        dict with keys:
            ``image``   : ``torch.Tensor`` ``[B, C, P, P]``
            ``mask``    : ``torch.Tensor`` ``[B, P, P]``
            ``metadata``: ``list`` of ``B`` dicts
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    metadata = [item["metadata"] for item in batch]
    return {"image": images, "mask": masks, "metadata": metadata}
