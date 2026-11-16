# Image statistics (A3 audit)

Built from real measured outputs:
- `dataset/metadata/image_stats.csv` (per-sample, per-HDU, raw values; **not staged/committed**, gitignored)
- `dataset/metadata/image_audit_summary.json` (committed)

Audit command: `python -m src.data.dataset_audit --manifest dataset/metadata/manifest.csv --root dataset/raw --out-dir dataset/metadata`  
Scope: 2,908 manifest samples (all `status=ok`), sorted by `sample_id`, read-only over `dataset/raw/`, no normalization/background subtraction/NaN filling.  
Runtime/size (this run): full audit completed via chunked execution and merged (single-process runs were terminated in this environment); total wall time ~20–25 min (approx). `image_stats.csv` = 4,135,351 bytes, 14,541 lines (header + 14,540 rows = 2,908 samples × 5 HDUs). Read errors: 0.

## Observed

### Dataset size, dtype/shape/HDU
| Property | Observed |
|---|---|
| Image files audited | 2,908 |
| HDUs per image | 5 |
| Rows in `image_stats.csv` | 14,540 (2,908×5) |
| On-disk dtype | `>f4` (14,540/14,540 HDU rows) |
| BITPIX | `-32` (14,540/14,540) |
| Shape | `208x208` (14,540/14,540) |
| Constant HDUs | 0 |
| All-zero HDUs | 0 |
| All-NaN HDUs | 0 |

### NaN / non-finite findings
| Metric | Observed |
|---|---:|
| Image files with any non-finite pixel | 127 |
| HDUs containing NaN | 162 |
| Total NaN pixels | 58,044 |
| Total +Inf pixels | 0 |
| Total -Inf pixels | 0 |

Per HDU index (NaNs):  
| HDU | HDUs with NaN | Max NaN fraction (in an HDU) | Total NaN pixels |
|---:|---:|---:|---:|
| 0 | 78 | 0.410202 | 38,442 |
| 1 | 39 | 0.021750 | 6,698 |
| 2 | 20 | 0.061159 | 8,166 |
| 3 | 16 | 0.009662 | 1,639 |
| 4 | 9 | 0.064950 | 3,099 |

Spatial NaN summary (distance to nearest image border, 2D): NaNs are **mostly not on the immediate border**; majority at distance > 1 (scattered/interior in aggregate), though a small fraction touches/near border.
| HDU | Total NaN px | dist=0 | dist≤1 | frac dist=0 | frac dist≤1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 38,442 | 774 | 1,530 | 0.0201 | 0.0398 |
| 1 | 6,698 | 169 | 340 | 0.0252 | 0.0508 |
| 2 | 8,166 | 107 | 208 | 0.0131 | 0.0255 |
| 3 | 1,639 | 64 | 130 | 0.0390 | 0.0793 |
| 4 | 3,099 | 86 | 161 | 0.0278 | 0.0520 |

### Per-HDU statistics (finite pixels only; distribution across images)
Values below are summaries across images of per-HDU statistics: median/sigma_mad/max/fraction_nonpositive (min, p5, median, p95, max; n=2908).

| HDU | median (med) | sigma_mad (med) | max (med) | frac_nonpositive (med) |
|---:|---:|---:|---:|---:|
| 0 | 35.587 (min -5.555, p5 9.895, p95 138.519, max 1242.838) | 101.243 (46.001–518.147) | 7612.374 (551.967–1,211,210.5) | 0.349 (0.0–0.528) |
| 1 | 58.424 (-3.396–2667.100) | 144.926 (48.914–1148.805) | 18331.105 (940.048–1,377,168.375) | 0.323 (0.0–0.508) |
| 2 | 108.012 (-28.418–3871.825) | 265.880 (93.524–2466.823) | 43960.926 (1655.190–3,850,093.75) | 0.317 (0.0–0.532) |
| 3 | 83.863 (4.947–3558.875) | 247.983 (116.427–1672.889) | 30204.166 (1152.566–1,787,450.375) | 0.353 (0.0–0.490) |
| 4 | 162.656 (-54.663–3835.234) | 585.130 (312.158–1970.803) | 39069.598 (2050.954–4,058,796.75) | 0.385 (0.0003–0.535) |

(Full per-HDU distributions in `dataset/metadata/image_audit_summary.json`.)

### Header census
- Distinct header keys seen: 293. Keys present in **every HDU of every image** include (not exhaustive): `BITPIX`, `NAXIS*`, `CTYPE1/2`, `CDELT1/2`, `CRVAL*`, `CRPIX*`, `EXPTIME`, `FPA.FILTER`, `FPA.FILTERID`, etc. Many keys vary across images/HDUs (e.g. per-exposure/cell metadata such as `CELL.*`, `HIERARCH CELL.*`, `ZP`, comments/history-like varying fields); see `header_census` in `image_audit_summary.json` for per-HDU `keys_in_every_hdu` vs `keys_that_vary_across_images`.
- FILTER/BAND-like keys per HDU index (constant across all images, by header values):
  - HDU 0: `FPA.FILTER=g.00000`, `FPA.FILTERID=g.00000`
  - HDU 1: `FPA.FILTER=r.00000`, `FPA.FILTERID=r.00000`
  - HDU 2: `FPA.FILTER=i.00000`, `FPA.FILTERID=i.00000`
  - HDU 3: `FPA.FILTER=z.00000`, `FPA.FILTERID=z.00000`
  - HDU 4: `FPA.FILTER=y.00000`, `FPA.FILTERID=y.00000`
- **Conclusion:** headers confirm HDU order **g, r, i, z, y** (consistent with Person B’s claim).

## Not verified yet
- Physical calibration/units beyond header presence (e.g. flux units, zeropoints consistency) not interpreted.
- Whether NaN spatial patterns correspond to specific sensor artifacts per-object beyond aggregate border-distance statistics.
- Per-class/mask-aligned image statistics (mask semantics owned by Person B).
- Any training-time normalization/background/asinh parameters (explicitly out of scope for this raw audit).

## Display stretch vs training preprocessing
- **Display stretch (`notebooks/03_image_exploration.ipynb`, `display_stretch`)**: percentile clipping + asinh for visualization only, operates on copies, never writes data back, and is **not** training preprocessing.
- **Training preprocessing**: not performed here. No normalization, no background subtraction, no NaN filling in this audit or its outputs; all statistics are on raw on-disk values.

## Comparison vs expected / Person B reported values
| Metric | Expected / Person B | Measured (A3) | Difference |
|---|---|---:|---|
| Image files | 2,908 (expected); B reported on 2,879 images | 2,908 | matches expected; **+29 vs B’s reported image count** (B copy likely incomplete) |
| HDUs per image, shape, dtype | 5 HDUs, 208×208, big-endian float32 `>f4` | 5, 208×208, `>f4` (BITPIX -32) | none |
| Image files with any non-finite pixel | 127 (manifest/A2); B reported 125 | 127 | matches 127; **+2 vs B** |
| HDUs containing NaN | B reported 160 (mostly HDU 0) | 162 total (HDU0:78,1:39,2:20,3:16,4:9) | **+2 HDUs vs B**; NaNs mostly in HDU 0 but present in all HDUs |
| Total NaN pixels | — | 58,044 | (not previously specified) |
| Infinite (±Inf) pixels | 0 | 0 | none |
| HDU band order g,r,i,z,y | B claims g,r,i,z,y | confirmed by `FPA.FILTER/FILTERID` per HDU (g,r,i,z,y) | **confirmed by headers** |

Differences vs Person B are consistent with Person B’s audit running on a smaller/incomplete image set (2,879) than the 2,908 images present/audited here. No checks were weakened.