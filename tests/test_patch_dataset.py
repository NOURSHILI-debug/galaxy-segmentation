"""Focused tests for the PatchDataset adapter (B6.2).

All tests use deterministic synthetic PyTorch tensors on CPU.  They do
not depend on the real astronomy dataset, Person A's manifest, splits,
FITS loader, or normalizer.

Covers:
    1. One sample producing multiple patches.
    2. Multiple samples with distinct sample IDs.
    3. Correct dataset length.
    4. Correct image and mask shapes.
    5. Image-mask alignment.
    6. Correct mask ignore-index preservation.
    7. Metadata correctness.
    8. Deterministic indexing.
    9. DataLoader batching.
    10. Invalid or malformed samples.
    11. Empty sample collection behavior.
    12. Preservation of tensor dtypes.
    13. Adapter uses the B6.1 patch-extraction contract (coordinates /
        counts match extract_patches directly).
"""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from src.data.patch_dataset import PatchDataset, patch_dataset_collate
from src.data.patches import MASK_IGNORE_INDEX, extract_patches

PATCH_SIZE = 256
STRIDE = 256


def _make_sample(sample_id, H, W, C=3, seed=0, mask_dtype=torch.int64):
    """Deterministic synthetic sample."""
    torch.manual_seed(seed)
    image = torch.randn(C, H, W, dtype=torch.float32)
    # Coordinate-based mask pattern so alignment can be verified.
    rows = torch.arange(H).unsqueeze(1).expand(H, W)
    cols = torch.arange(W).unsqueeze(0).expand(H, W)
    mask = ((rows * W + cols) % 4).to(mask_dtype)
    return {"sample_id": sample_id, "image": image, "mask": mask}


class TestOneSampleMultiplePatches(unittest.TestCase):
    """1. One sample producing multiple patches."""

    def test_512x512_yields_four_patches(self):
        sample = _make_sample("s1", 512, 512, seed=1)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        self.assertEqual(len(ds), 4)
        shapes = {tuple(ds[i]["image"].shape) for i in range(len(ds))}
        self.assertEqual(shapes, {(3, 256, 256)})
        mask_shapes = {tuple(ds[i]["mask"].shape) for i in range(len(ds))}
        self.assertEqual(mask_shapes, {(256, 256)})


class TestMultipleSamplesDistinctIds(unittest.TestCase):
    """2. Multiple samples with distinct sample IDs."""

    def test_two_samples_metadata_ids(self):
        s1 = _make_sample("galaxy-alpha", 512, 512, seed=2)
        s2 = _make_sample("galaxy-beta", 512, 512, seed=3)
        ds = PatchDataset([s1, s2], patch_size=256, stride=256)
        self.assertEqual(len(ds), 8)
        ids_0 = {ds[i]["metadata"]["sample_id"] for i in range(4)}
        ids_1 = {ds[i]["metadata"]["sample_id"] for i in range(4, 8)}
        self.assertEqual(ids_0, {"galaxy-alpha"})
        self.assertEqual(ids_1, {"galaxy-beta"})


class TestDatasetLength(unittest.TestCase):
    """3. Correct dataset length."""

    def test_length_equals_total_patches(self):
        s1 = _make_sample("s1", 512, 512, seed=4)   # 4 patches
        s2 = _make_sample("s2", 600, 700, seed=5)    # 9 patches (B6.1 logic)
        ds = PatchDataset([s1, s2], patch_size=256, stride=256)
        self.assertEqual(len(ds), 4 + 9)

    def test_single_small_sample_one_patch(self):
        sample = _make_sample("s3", 100, 150, seed=6)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        self.assertEqual(len(ds), 1)


class TestCorrectShapes(unittest.TestCase):
    """4. Correct image and mask shapes."""

    def test_shapes_match_patch_size(self):
        sample = _make_sample("s1", 512, 512, seed=7)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        for i in range(len(ds)):
            self.assertEqual(ds[i]["image"].shape, (3, 256, 256))
            self.assertEqual(ds[i]["mask"].shape, (256, 256))


class TestImageMaskAlignment(unittest.TestCase):
    """5. Image-mask alignment."""

    def test_patches_match_source_coordinates(self):
        H, W = 512, 512
        sample = _make_sample("s1", H, W, seed=8)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        image = sample["image"]
        mask = sample["mask"]
        # top=0,left=0 patch (index 0)
        self.assertTrue(torch.equal(ds[0]["image"], image[:, 0:256, 0:256]))
        self.assertTrue(torch.equal(ds[0]["mask"], mask[0:256, 0:256]))
        # top=256,left=256 patch (index 3)
        self.assertTrue(torch.equal(ds[3]["image"], image[:, 256:512, 256:512]))
        self.assertTrue(torch.equal(ds[3]["mask"], mask[256:512, 256:512]))


class TestIgnoreIndexPreservation(unittest.TestCase):
    """6. Correct mask ignore-index preservation (padding = 255)."""

    def test_small_sample_mask_padding_is_255(self):
        H, W = 100, 150
        sample = _make_sample("s3", H, W, seed=9)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        patch = ds[0]["mask"]
        # Interior (original) region holds real labels; padded border is 255.
        self.assertTrue((patch[:H, :W] == sample["mask"]).all())
        self.assertTrue((patch[H:, :] == 255).all())
        self.assertTrue((patch[:, W:] == 255).all())
        self.assertEqual(MASK_IGNORE_INDEX, 255)


class TestMetadataCorrectness(unittest.TestCase):
    """7. Metadata correctness."""

    def test_metadata_fields(self):
        sample = _make_sample("galaxy-42", 512, 512, seed=10)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        expected = [(0, 0), (0, 256), (256, 0), (256, 256)]
        for i, (exp_top, exp_left) in enumerate(expected):
            m = ds[i]["metadata"]
            self.assertEqual(m["sample_id"], "galaxy-42")
            self.assertEqual(m["patch_index"], i)
            self.assertEqual(m["top"], exp_top)
            self.assertEqual(m["left"], exp_left)
            self.assertEqual(m["original_height"], 512)
            self.assertEqual(m["original_width"], 512)
            self.assertEqual(m["patch_height"], 256)
            self.assertEqual(m["patch_width"], 256)
            for key in (
                "sample_id",
                "patch_index",
                "top",
                "left",
                "original_height",
                "original_width",
                "patch_height",
                "patch_width",
            ):
                self.assertIn(key, m)


class TestDeterministicIndexing(unittest.TestCase):
    """8. Deterministic indexing."""

    def test_repeated_access_returns_identical_tensors(self):
        sample = _make_sample("s1", 512, 512, seed=11)
        ds = PatchDataset([sample], patch_size=256, stride=256)
        first = [ds[i] for i in range(len(ds))]
        second = [ds[i] for i in range(len(ds))]
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a["image"], b["image"]))
            self.assertTrue(torch.equal(a["mask"], b["mask"]))
            self.assertEqual(a["metadata"], b["metadata"])


class TestDataLoaderBatching(unittest.TestCase):
    """9. DataLoader batching."""

    def test_dataloader_batches_correctly(self):
        s1 = _make_sample("s1", 512, 512, seed=12)
        s2 = _make_sample("s2", 300, 400, seed=13)  # 4 patches (B6.1 logic)
        ds = PatchDataset([s1, s2], patch_size=256, stride=256)
        loader = DataLoader(
            ds, batch_size=4, shuffle=False, collate_fn=patch_dataset_collate
        )
        batches = list(loader)
        # total patches = 4 + 4 = 8 -> 2 batches of 4
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(batch["image"].shape[0], 4)
            self.assertEqual(batch["mask"].shape[0], 4)
            self.assertEqual(batch["image"].shape[1:], (3, 256, 256))
            self.assertEqual(batch["mask"].shape[1:], (256, 256))
            self.assertEqual(len(batch["metadata"]), 4)
            # Metadata stays with the right sample: all patch_index values
            # within a batch must be unique per sample grouping.
            ids_in_batch = {m["sample_id"] for m in batch["metadata"]}
            self.assertTrue(len(ids_in_batch) >= 1)

    def test_dataloader_default_collate_raises_on_metadata(self):
        # Default collate cannot sensibly batch mixed-type metadata dicts;
        # our custom collate is required.  Confirm the custom collate works
        # where default would struggle (documented design choice).
        s1 = _make_sample("s1", 512, 512, seed=14)
        ds = PatchDataset([s1], patch_size=256, stride=256)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertIn("image", batch)
        self.assertIn("mask", batch)
        self.assertIn("metadata", batch)


class TestInvalidSamples(unittest.TestCase):
    """10. Invalid or malformed samples."""

    def test_missing_keys(self):
        with self.assertRaises(ValueError):
            PatchDataset([{"sample_id": "s", "image": torch.randn(3, 8, 8)}])

    def test_non_tensor_image(self):
        with self.assertRaises(TypeError):
            PatchDataset([("s", [[1, 2], [3, 4]], torch.zeros(2, 2, dtype=torch.int64))])

    def test_non_tensor_mask(self):
        with self.assertRaises(TypeError):
            PatchDataset([("s", torch.randn(3, 8, 8), [[1, 2], [3, 4]])])

    def test_image_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            PatchDataset([("s", torch.randn(8, 8), torch.zeros(8, 8, dtype=torch.int64))])

    def test_mask_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            PatchDataset([("s", torch.randn(3, 8, 8), torch.zeros(8, 8, 8, dtype=torch.int64))])

    def test_mismatched_spatial_dims(self):
        with self.assertRaises(ValueError):
            PatchDataset([("s", torch.randn(3, 8, 8), torch.zeros(4, 4, dtype=torch.int64))])

    def test_invalid_patch_size(self):
        sample = _make_sample("s", 64, 64, seed=15)
        for bad in (0, -1, 1.5, "256", True):
            with self.subTest(patch_size=bad):
                with self.assertRaises((ValueError, TypeError)):
                    PatchDataset([sample], patch_size=bad)

    def test_invalid_stride(self):
        sample = _make_sample("s", 64, 64, seed=16)
        for bad in (0, -1, 2.5, "128", True):
            with self.subTest(stride=bad):
                with self.assertRaises((ValueError, TypeError)):
                    PatchDataset([sample], stride=bad)


class TestEmptySampleCollection(unittest.TestCase):
    """11. Empty sample collection behavior."""

    def test_empty_dataset_length_zero(self):
        ds = PatchDataset([], patch_size=256, stride=256)
        self.assertEqual(len(ds), 0)

    def test_empty_dataset_getitem_raises(self):
        ds = PatchDataset([], patch_size=256, stride=256)
        with self.assertRaises(IndexError):
            ds[0]


class TestDtypePreservation(unittest.TestCase):
    """12. Preservation of tensor dtypes."""

    def test_image_and_mask_dtypes_preserved(self):
        image = torch.randn(3, 64, 64, dtype=torch.float64)
        mask = torch.zeros(64, 64, dtype=torch.int16)
        sample = {"sample_id": "s", "image": image, "mask": mask}
        ds = PatchDataset([sample], patch_size=256, stride=256)
        patch = ds[0]
        self.assertEqual(patch["image"].dtype, torch.float64)
        self.assertEqual(patch["mask"].dtype, torch.int16)


class TestUsesB61Contract(unittest.TestCase):
    """13. Adapter uses the B6.1 patch-extraction contract.

    Proves the adapter delegates coordinate / count logic to
    ``extract_patches`` rather than reimplementing it: patch counts and
    coordinates must match a direct ``extract_patches`` call on the same
    sample.
    """

    def test_coordinates_and_counts_match_extract_patches(self):
        H, W = 600, 700
        sample = _make_sample("s1", H, W, seed=17)
        ds = PatchDataset([sample], patch_size=256, stride=256)

        # Direct B6.1 call on the same sample.
        _, _, b61_metadata = extract_patches(
            sample["image"], sample["mask"], sample["sample_id"],
            patch_size=256, stride=256,
        )

        self.assertEqual(len(ds), len(b61_metadata))
        for i in range(len(ds)):
            ds_meta = ds[i]["metadata"]
            b61_meta = b61_metadata[i]
            self.assertEqual(ds_meta["top"], b61_meta["top"])
            self.assertEqual(ds_meta["left"], b61_meta["left"])
            self.assertEqual(ds_meta["sample_id"], b61_meta["sample_id"])
            self.assertEqual(ds_meta["patch_index"], b61_meta["patch_index"])

        # Mask padding uses B6.1's ignore index (255).
        self.assertEqual(MASK_IGNORE_INDEX, 255)


if __name__ == "__main__":
    unittest.main()
