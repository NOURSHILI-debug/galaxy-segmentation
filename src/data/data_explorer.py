"""Step 3 exploration: survey dataset/raw and inspect a few FITS files.

Read-only. Makes no assumption about folder layout, HDU structure, shapes
or mask encoding. Run from the repo root:

    python explore_dataset.py --root dataset/raw --n 3

Paste the printed output back to the team to design the loader/manifest.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")
KEY_HEADER_KEYS = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
                   "BUNIT", "OBJECT", "FILTER", "BAND", "EXPTIME", "CTYPE1",
                   "CTYPE2", "CDELT1", "CDELT2", "CD1_1", "CD2_2")


def is_fits(p: Path) -> bool:
    return p.name.lower().endswith(FITS_SUFFIXES)


def survey(root: Path):
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"Total files under {root}: {len(files)}")
    ext = Counter("".join(p.suffixes).lower() or "<none>" for p in files)
    print("Extensions:", dict(ext))

    by_dir = defaultdict(list)
    for p in files:
        if is_fits(p):
            by_dir[p.parent].append(p)
    print(f"\nFolders containing FITS files: {len(by_dir)}")
    for d, fs in sorted(by_dir.items()):
        fs.sort()
        print(f"  {d}  ({len(fs)} files)")
        print("    first names:", [f.name for f in fs[:3]])
        print("    last names: ", [f.name for f in fs[-2:]])
    return by_dir


def inspect_file(path: Path):
    print(f"\n{'=' * 78}\nFILE: {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    try:
        with fits.open(path, memmap=True) as hdul:
            hdul.info()
            for i, hdu in enumerate(hdul):
                print(f"\n  -- HDU {i}: name={hdu.name!r} type={type(hdu).__name__}")
                hdr = hdu.header
                keys = {k: hdr[k] for k in KEY_HEADER_KEYS if k in hdr}
                print("     key header values:", keys)
                print(f"     total header cards: {len(hdr)}")
                if hdu.data is None:
                    print("     (no data)")
                    continue
                d = np.asarray(hdu.data)
                print(f"     shape={d.shape} dtype={d.dtype}")
                if d.dtype.names:  # table HDU
                    print("     table columns:", d.dtype.names)
                    continue
                if d.dtype.kind == "f":
                    nan, inf = int(np.isnan(d).sum()), int(np.isinf(d).sum())
                else:
                    nan, inf = 0, 0
                print(f"     NaN={nan} Inf={inf}")
                f = d[np.isfinite(d)] if d.dtype.kind == "f" else d.ravel()
                if f.size:
                    print(f"     min={f.min()} max={f.max()} "
                          f"mean={f.mean():.6g} std={f.std():.6g}")
                    if d.dtype.kind in "iu" or np.unique(f[:200000]).size < 30:
                        vals, cnt = np.unique(f, return_counts=True)
                        if vals.size <= 30:
                            print("     unique values (value: count):",
                                  dict(zip(vals.tolist(), cnt.tolist())))
    except Exception as e:  # report, never crash: unreadable files are data
        print(f"  !! UNREADABLE: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/raw")
    ap.add_argument("--n", type=int, default=3, help="files to inspect per folder")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    by_dir = survey(root)
    for d, fs in sorted(by_dir.items()):
        print(f"\n\n##### Inspecting {args.n} file(s) from {d}")
        for p in fs[: args.n]:
            inspect_file(p)


if __name__ == "__main__":
    main()