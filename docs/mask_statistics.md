# Mask Statistics and Training Policy

## Scope and sources

This document summarizes the local audit saved in [`dataset/metadata/mask_audit.json`](../dataset/metadata/mask_audit.json). Counts and proportions below describe the audited local files, not every copy or release of the benchmark. Raw meanings are from the official [Galaxy Segmentation benchmark documentation](https://github.com/erinleeryan/nasa_astro_aiml/blob/main/galaxysegmentation/galaxy_segmentation_data_documentation.md), which identifies the benchmark values as “No relevant data,” galaxy center/core, bar, arms, and star. The official documentation is the source for those meanings; local frequencies are not semantic evidence.

The approved project policy treats raw value 0 (“No relevant data”) as an ignore label and trains on raw values 1–4. The model therefore has four trainable classes with `ignore_index=255`. The exact decoding is recorded in [`dataset/metadata/class_mapping.json`](../dataset/metadata/class_mapping.json) and implemented by [`src/data/mask_decoder.py`](../src/data/mask_decoder.py).

## Local audit summary

The audit inspected 2,879 image files and 2,908 mask files, with 2,879 unique matched pairs and 29 unmatched masks. It found five raw mask values, no mask read errors, no mask non-finite pixels, and no non-integer-like finite mask values. It recorded 125 image files containing non-finite values. These local findings do not define the dataset's universal properties.

Raw mask-value counts and all-pixel percentages below include **all 2,908 mask files, including the 29 unmatched masks**. Trainable-class counts also use those same 2,908 masks, retaining raw values 1–4 only; trainable-class percentages use the resulting raw-value-1–4 pixel total as their denominator. Image-mask dimension comparisons apply only to the **2,879 matched pairs**, not to all 2,908 masks.

The audit counted 125,811,712 total mask pixels. All were finite in this audit, so the total-pixel and finite-pixel denominators are equal. The tables below nevertheless distinguish the all-pixel distribution from the training distribution after ignoring raw value 0.

## Raw distribution over all mask pixels

Percentages in this table use **all 125,811,712 raw mask pixels** as the denominator. The raw meanings are documented by the official benchmark source; “mask files containing value” is a local audit statistic.

| Raw value | Documented meaning | Pixels | Percent of all raw mask pixels | Mask files containing value |
| ---: | :--- | ---: | ---: | ---: |
| 0 | No relevant data | 103,962,425 | 82.6333% | 2,907 |
| 1 | Galaxy center or core | 24,812 | 0.0197% | 2,902 |
| 2 | Galaxy bar | 5,242,726 | 4.1671% | 2,520 |
| 3 | Galaxy arms | 16,556,365 | 13.1596% | 2,638 |
| 4 | Star | 25,384 | 0.0202% | 1,866 |
| **Total** |  | **125,811,712** | **100.0000%** |  |

## Trainable distribution after ignoring raw value 0

Raw value 0 contributes 103,962,425 pixels, or 82.6333% of all audited mask pixels. Under the approved policy, those pixels receive the ignore index and do not contribute to the segmentation loss or evaluation metrics. The remaining **21,849,287 finite pixels** (17.3667% of all raw mask pixels) are trainable-label pixels before applying the image-validity mask.

Percentages in this table use **21,849,287 trainable pixels (raw values 1–4)** as the denominator. The counts are unchanged from the raw distribution table; only the denominator differs.

Percentages are rounded to four decimal places, so displayed class percentages may sum to 100.0001% rather than exactly 100%.

| Training ID | Raw value | Class name | Pixels | Percent of trainable pixels |
| ---: | ---: | :--- | ---: | ---: |
| 0 | 1 | Galaxy center or core | 24,812 | 0.1136% |
| 1 | 2 | Galaxy bar | 5,242,726 | 23.9950% |
| 2 | 3 | Galaxy arms | 16,556,365 | 75.7753% |
| 3 | 4 | Star | 25,384 | 0.1162% |
| **Total trainable** |  |  | **21,849,287** | **100.0000%** |

## Empty masks and class presence

This analysis uses all **2,908 mask FITS files**, including the **29 unmatched masks**. The denominator for the mask-count percentages below is 2,908. An **empty mask** is defined operationally as a mask containing no pixels with raw values 1-4. A mask containing **only raw value 0** is counted separately, meaning every pixel in that mask equals 0. These definitions describe label occupancy; raw value 0 retains its documented meaning, "No relevant data," and is not described as dataset-defined background.

There were **0 empty masks** under the raw-values-1-4 definition and **0 masks containing only raw value 0**. These counts happen to be equal for this audited dataset, but the definitions are distinct: a future mask containing another unexpected value and no 1-4 pixels would be empty under the first definition but not the second.

| Raw value | Documented meaning | Masks where value is absent | Percent of all 2,908 masks | Absent from entire analyzed dataset? |
| ---: | :--- | ---: | ---: | :--- |
| 1 | Galaxy center or core | 6 | 0.2063% | No; 24,812 pixels occur |
| 2 | Galaxy bar | 388 | 13.3425% | No; 5,242,726 pixels occur |
| 3 | Galaxy arms | 270 | 9.2847% | No; 16,556,365 pixels occur |
| 4 | Star | 1,042 | 35.8322% | No; 25,384 pixels occur |

Per-mask absence means that a mask has zero pixels with that raw value. Dataset-level absence is assessed separately across all 2,908 masks; none of raw values 1-4 is absent from the overall dataset. These counts include unmatched masks.

## Spatial connected-component and small-region statistics

Connected components were computed independently for each raw value 1-4 in each of the 2,908 masks. The binary class mask was labeled with **8-connectivity** (diagonal neighbors connect). A component's area is its number of pixels of that raw value. The size bins are inclusive: 1 pixel, 2-5 pixels, 6-10 pixels, 11-25 pixels, and more than 25 pixels. No component was removed, merged across classes, or relabeled in the source masks.

For this descriptive analysis, a **small region** is an 8-connected component of **10 pixels or fewer** (the first three bins). This is an operational pixel-count threshold only; it is not an astrophysical definition. The tables report component counts and the total pixel area contributed by components in each bin. The small-region summary also reports the total pixels in components of 10 pixels or fewer and the number of masks containing at least one such component for each class.

**Component counts by component-area bin**

| Raw value / documented meaning | 1 pixel | 2-5 pixels | 6-10 pixels | 11-25 pixels | >25 pixels | Total components |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 - Galaxy center or core | 33 | 801 | 1,373 | 727 | 17 | 2,951 |
| 2 - Galaxy bar | 15 | 11 | 5 | 1 | 2,578 | 2,610 |
| 3 - Galaxy arms | 389 | 502 | 262 | 259 | 3,999 | 5,411 |
| 4 - Star | 90 | 777 | 1,348 | 844 | 3 | 3,062 |

**Component pixel totals by component-area bin**

| Raw value / documented meaning | 1 pixel | 2-5 pixels | 6-10 pixels | 11-25 pixels | >25 pixels | Total class pixels |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 - Galaxy center or core | 33 | 3,198 | 10,666 | 10,405 | 510 | 24,812 |
| 2 - Galaxy bar | 15 | 30 | 39 | 18 | 5,242,624 | 5,242,726 |
| 3 - Galaxy arms | 389 | 1,544 | 2,020 | 4,282 | 16,548,130 | 16,556,365 |
| 4 - Star | 90 | 2,959 | 10,580 | 11,671 | 84 | 25,384 |

**Small-region totals (components of 10 pixels or fewer)**

| Raw value / documented meaning | Small components | Pixels in small components | Masks with at least one small component (of 2,908) |
| :--- | ---: | ---: | ---: |
| 1 - Galaxy center or core | 2,207 | 13,897 | 2,175 |
| 2 - Galaxy bar | 31 | 84 | 17 |
| 3 - Galaxy arms | 1,153 | 3,953 | 765 |
| 4 - Star | 2,215 | 13,629 | 1,508 |

### Connected-component validation and coverage

All **2,908 of 2,908** discovered masks were analyzed, including all 29 unmatched masks. There were no excluded or unreadable files, no HDU-structure issues, and no non-finite, non-integer-like, or unexpected mask values. Each file contained one readable 2D data-bearing mask HDU.

For every class, the component counts summed across the five bins equal the total component count. The pixel areas summed across bins equal both the component-labeled pixel total and the direct raw-value pixel count from the FITS masks. They also match the corresponding aggregate raw-value count in `mask_audit.json`:

| Raw value | Components across bins = total components | Component pixels = direct FITS pixels = saved audit pixels |
| ---: | :--- | :--- |
| 1 | 2,951 = 2,951 | 24,812 = 24,812 = 24,812 |
| 2 | 2,610 = 2,610 | 5,242,726 = 5,242,726 = 5,242,726 |
| 3 | 5,411 = 5,411 | 16,556,365 = 16,556,365 = 16,556,365 |
| 4 | 3,062 = 3,062 | 25,384 = 25,384 = 25,384 |

FITS files were opened read-only and processed sequentially. Astropy in the project Python environment read the FITS arrays; OpenCV 4.11.0 in the available system Python performed the 8-connected component labeling through an in-memory pipe. No analysis script or intermediate data file was saved, and the raw FITS files were not modified.

## Class imbalance

The trainable distribution is strongly imbalanced: galaxy-arm pixels make up about 75.78% of trainable pixels, while bar pixels make up about 23.99%. Center/core and star pixels each make up about 0.12%. The center/core class is roughly 668 times less frequent than the arms class by pixel count; the star class is roughly 652 times less frequent. These are descriptive audit observations, not a prescription for class weighting, resampling, or other balancing methods. Such training choices require separate approval and evaluation.

## Non-finite image policy

The local audit found non-finite values in 125 image files. The later image-preprocessing pipeline must detect non-finite image pixels **before normalization**. After normalization, it must set the corresponding invalid normalized values to `0` and produce a spatial validity mask. Those locations must be excluded from both the loss and evaluation metrics. This policy does not alter the raw FITS files. This document records the approved requirement; it does not implement the preprocessing pipeline.

Image validity is an additional gate on the trainable mask pixels: locations marked with raw mask value 0 are ignored regardless, and locations at non-finite image pixels are also excluded from training and evaluation by the validity mask.

## References

- [Official benchmark data documentation](https://github.com/erinleeryan/nasa_astro_aiml/blob/main/galaxysegmentation/galaxy_segmentation_data_documentation.md) — raw mask meanings and benchmark label construction.
- [Official AWS dataset registry](https://registry.opendata.aws/apd_galaxysegmentation/) — dataset identity and high-level provenance.
- Local audit: [`dataset/metadata/mask_audit.json`](../dataset/metadata/mask_audit.json) — counts, pixel frequencies, and file-quality findings for this repository's dataset snapshot.
- Approved mapping: [`dataset/metadata/class_mapping.json`](../dataset/metadata/class_mapping.json).
