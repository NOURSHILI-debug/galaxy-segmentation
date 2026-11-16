# Mask Semantics and Data Provenance

## 1. Dataset identity and provenance

The dataset is the **Astrophysics Division Galaxy Segmentation Benchmark Dataset**. The AWS registry describes its images as Pan-STARRS imaging and its labels as derived from the Galaxy Zoo: 3D project. The benchmark documentation identifies Pan-STARRS as the imaging source and Galaxy Zoo: 3D citizen-science annotations as the label origin.

- [Official AWS registry entry](https://registry.opendata.aws/apd_galaxysegmentation/)
- [Official benchmark data documentation](https://github.com/erinleeryan/nasa_astro_aiml/blob/main/galaxysegmentation/galaxy_segmentation_data_documentation.md)

The benchmark documentation says the Pan-STARRS images are stacked exposures and 60-arcsecond cutouts made through the Pan-STARRS 1 Cutout Service. It says the Galaxy Zoo: 3D label masks were projected to the Pan-STARRS world-coordinate orientation and spatial resolution using Reproject. The documentation describes galaxies used for bar or spiral mask generation as having at least five users tasked with those markings.

## 2. Image FITS structure

The benchmark documentation describes five image HDUs, in a fixed filter order:

| Image HDU index | Pan-STARRS filter |
| ---: | :--- |
| 0 | g |
| 1 | r |
| 2 | i |
| 3 | z |
| 4 | y |

These HDUs are image channels in the Pan-STARRS filters. They are **not segmentation classes**. This order is documented in the official benchmark documentation under “Data types.”

The official documentation names the paired mask file `maskdata_<objectidentifier>.fits`; the local files audited for this project are named `mask_<sample_id>.fits`. The documentation does not explain this filename difference. The local audit found one data-bearing mask HDU per local mask file; that is a local observation, while the value meanings below come from the benchmark documentation.

## 3. Mask FITS structure and verified raw-value mapping

The benchmark documentation describes the released mask as a single array containing the combined label categories. Its documented raw-value mapping is:

| Raw value | Documented meaning    |
| --------- | --------------------- |
| 0         | No relevant data      |
| 1         | Galaxy center or core |
| 2         | Galaxy bar            |
| 3         | Galaxy arms           |
| 4         | Star                  |

The benchmark defines raw value 0 as **No relevant data** The benchmark documentation does not define value 0 as a training ignore index. The project has separately selected raw value 0 as an ignore label, represented by `255`; raw values 1-4 map respectively to contiguous training IDs 0-3. This project training policy is distinct from the dataset semantics documented above.

## 4. Mask construction and overlap priority

According to the official benchmark documentation, the four label types are first represented as separate 2D masks, then combined into one categorical mask. When annotations conflict at a pixel, the combined mask receives the **lowest numerical mask value** among the conflicting labels. The resulting pixel stores one selected category, even though the source annotations may overlap.

The benchmark documentation describes retaining a pixel for a label when its vote count is at least the mean of nonzero votes per pixel minus one. This is the documented threshold used to identify reliable regions when producing the benchmark mask; it is not a training-time threshold policy for this project.

The combined benchmark mask is distinct from the underlying Galaxy Zoo: 3D annotation products, which preserve separate annotation/count products and other data described in the original paper. The benchmark documentation is the primary source for the combined mask, its priority rule, threshold, and values.

## 5. Distinction from original Galaxy Zoo: 3D FITS products

Masters et al. (2021), Section 4 (“How to use GZ:3D masks”), describes a different product: one FITS file per galaxy with 11 HDUs. It lists an SDSS cutout image in HDU 0; separate center, star, spiral, and bar masks in HDUs 1–4; then metadata and raw/clustered classification tables in HDUs 5–10. The paper describes original center/star products as clustered points and their regions, and bar/spiral products as counts of volunteer drawings.

This benchmark instead documents separate five-HDU Pan-STARRS image files and a single combined categorical mask file. Therefore, the original paper’s HDU indices and count-mask values must **not** be applied directly to this benchmark mask. The original paper provides annotation context and the original product format; it does not independently establish the benchmark’s raw-value mapping.

## 6. Observed local dataset facts

The following are observations from this repository’s completed audit of `dataset/raw/` and its saved report, [`dataset/metadata/mask_audit.json`](../dataset/metadata/mask_audit.json). They describe the local files at audit time and are not guaranteed properties of every copy or release of the benchmark.

| Local audit finding | Result |
| :--- | ---: |
| Image files | 2,879 |
| Mask files | 2,908 |
| Unique matched image–mask pairs | 2,879 |
| Unmatched masks | 29 |
| Unmatched images | 0 |
| Duplicate image IDs | 0 |
| Duplicate mask IDs | 0 |
| Mask FITS read errors | 0 |
| Distinct observed raw mask values | 5 (0, 1, 2, 3, 4) |
| Image files containing non-finite values | 125 |
| Masks with non-finite values | 0 |
| Masks with non-integer-like finite values | 0 |
| Matched image/mask dimension comparisons passed | 14,395 of 14,395 |

The audit also reported no image FITS read errors. The non-finite image finding is a data-quality observation; this document does not prescribe a repair or filtering strategy.

## 7. Training implications and current project policy

The benchmark documents the five raw meanings in the table above. Separately, the current project policy treats raw value 0 (No relevant data) as an ignore label with `ignore_index=255`; it is not treated as a confirmed background class. Raw values 1, 2, 3, and 4 map respectively to contiguous training IDs 0, 1, 2, and 3, representing galaxy center or core, galaxy bar, galaxy arms, and star. The model has four trainable classes. The raw meanings come from the official benchmark documentation; the ignore treatment and training IDs are project policy and are not claims about the dataset documentation.

Raw FITS files are source data and must remain unchanged. Geometric transformations applied during model preparation must keep each image and mask spatially aligned. Because the mask is categorical, use nearest-neighbor interpolation for mask resampling so interpolation does not create new numeric labels. Image interpolation choices are a separate preprocessing decision.

## 8. Sources and evidence

1. **Primary source for benchmark format, value mapping, and construction:** [Galaxy Segmentation benchmark data documentation](https://github.com/erinleeryan/nasa_astro_aiml/blob/main/galaxysegmentation/galaxy_segmentation_data_documentation.md), especially “Data types” and “Data origins and preprocessing.” It documents the five Pan-STARRS filters, the separate image/mask files, meanings for values 0–4, source-mask combination and overlap priority, vote threshold, and reprojection.
2. **Dataset identity and high-level provenance:** [Astrophysics Division Galaxy Segmentation Benchmark Dataset registry entry](https://registry.opendata.aws/apd_galaxysegmentation/). The registry identifies the dataset and describes Pan-STARRS imaging with labels derived from Galaxy Zoo: 3D.
3. **Original Galaxy Zoo: 3D annotation context and FITS product format:** Masters et al. (2021), [“Galaxy Zoo: 3D – crowdsourced bar, spiral, and foreground star masks for MaNGA target galaxies,” *Monthly Notices of the Royal Astronomical Society*, 507, 3923–3935](https://academic.oup.com/mnras/article/507/3/3923/6347355), DOI [10.1093/mnras/stab2282](https://doi.org/10.1093/mnras/stab2282), especially Sections 2.4 and 4. The paper supports claims about original Galaxy Zoo: 3D annotations and its 11-HDU product; it is **not** cited as the source for the benchmark’s 0–4 mapping.
