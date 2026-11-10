# Environment

Recorded on 2026-11-10 for branch `feature/person-a-data`.

## Platform

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.22631) |
| Architecture | x86_64 |
| Python | 3.13.5 |
| Virtual environment | `.venv` (project root, git-ignored) |

## Installed packages

| Package | Version |
| --- | --- |
| astropy | 8.0.2 |
| astropy-iers-data | 0.2026.11.9.0.35.9 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| torch | 2.15.0+cpu |
| CUDA available | No (CPU build) |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --retries 10 --timeout 60 "astropy==8.0.2" pandas torch
```

`astropy` is pinned to `8.0.2`: the unpinned resolver returned no candidate
versions in this environment, while the pinned wheel resolves correctly.

## FITS smoke test

Opened a bundled Astropy test file (no dataset download) with:

```python
from pathlib import Path
from astropy.io import fits
import astropy.io.fits.tests as tests

path = Path(tests.__file__).parent / "data" / "test0.fits"
with fits.open(path) as hdul:
    hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
    print(hdu.data.shape, hdu.data.dtype)
```

Result: `test0.fits`, HDU 1 (`SCI`), shape `(40, 40)`, dtype `>i2`,
values in `[309, 474]`. FITS reading works with the installed stack.
