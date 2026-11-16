"""Generate Person B4 classical computer vision visualizations and reports.

Produces:
1. experiments/b4_classical_results/five_band_comparison.png
2. experiments/b4_classical_results/classical_pipeline_stages.png
3. experiments/b4_classical_results/comparison_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.classical_cv.band_comparison import (
    BAND_NAMES,
    compare_five_bands,
    load_band_fits,
    recommend_reference_band,
)
from src.classical_cv.baseline import run_classical_pipeline
from src.data.mask_decoder import decode_mask


def generate_five_band_visualization(
    clean_fits: Path,
    nan_fits: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate 5-band side-by-side diagnostic visualization for clean & problematic samples."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    fig.patch.set_facecolor("#0f1117")

    samples = [("Clean Sample (1-100523)", clean_fits), ("Problematic Sample (1-103458)", nan_fits)]
    all_metrics = {}

    for row_idx, (sample_title, fpath) in enumerate(samples):
        band_comp_res = compare_five_bands(fpath)
        all_metrics[sample_title] = band_comp_res
        band_comp = band_comp_res["bands"]

        for col_idx, band_name in enumerate(BAND_NAMES):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#161822")

            data, val_mask, _ = load_band_fits(fpath, band=band_name)
            metrics = band_comp[band_name]

            # Stretch valid pixels for clear contrast
            val_pixels = data[val_mask]
            if val_pixels.size > 0:
                p01 = float(np.percentile(val_pixels, 1.0))
                p99 = float(np.percentile(val_pixels, 99.0))
                display_img = np.clip((data - p01) / max(p99 - p01, 1e-6), 0.0, 1.0)
            else:
                display_img = np.zeros_like(data)

            # Red tint for invalid/NaN pixels
            cmap = plt.cm.inferno.copy()
            cmap.set_bad(color="#ff3333")

            masked_display = np.ma.masked_where(~val_mask, display_img)
            im = ax.imshow(masked_display, cmap=cmap, origin="lower")

            # Overlay NaN / invalid pixels in cyan/magenta if any
            invalid_cnt = int(metrics["invalid_count"])
            snr = metrics["peak_snr"]
            snr_str = f"SNR: {snr:.1f}" if snr is not None else "SNR: N/A"

            title_color = "#4ade80" if band_name in ("r", "i") else "#f1f5f9"
            ax.set_title(
                f"{band_name}-band (HDU {col_idx})\n{snr_str} | Inval: {invalid_cnt}",
                color=title_color,
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")

    fig.suptitle(
        "Pan-STARRS Five-Band Diagnostic Comparison (g, r, i, z, y)\n"
        "Provisional r-band reference recommendation based on diagnostics from two representative samples. Broader validation required.",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return all_metrics


def generate_pipeline_stages_visualization(
    fits_path: Path,
    mask_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate 9-panel stage-by-stage visualization of classical CV baseline."""
    # 1. Load r-band (HDU 1)
    r_band, val_mask, _ = load_band_fits(fits_path, band="r")

    # 2. Provisional normalization to [0, 1]
    p05 = float(np.percentile(r_band[val_mask], 0.5))
    p995 = float(np.percentile(r_band[val_mask], 99.5))
    norm_img = np.clip((r_band - p05) / max(p995 - p05, 1e-8), 0.0, 1.0)

    # 3. Run complete baseline pipeline
    results = run_classical_pipeline(
        norm_img,
        valid_mask=val_mask,
        threshold_method="otsu",
        morph_shape="disk",
        morph_radius=1,
        connectivity=8,
        watershed_min_distance=5,
        watershed_threshold=2.0,
    )

    # Also compute adaptive threshold for comparison
    from src.classical_cv.thresholding import adaptive_threshold
    adapt_mask, _, _ = adaptive_threshold(norm_img, valid_mask=val_mask, window_size=25, c_offset=-0.05)

    # 4. Load & decode ground-truth mask
    with fits.open(mask_path, memmap=False) as hdul:
        raw_mask_data = hdul[0].data
    gt_tensor = decode_mask(raw_mask_data)
    gt_np = gt_tensor.numpy()

    # Create 3x3 stage visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.patch.set_facecolor("#0b0f19")

    panels = [
        ("1. Pan-STARRS r-band Flux", r_band, "inferno", "raw"),
        ("2. Normalized Flux [0, 1]", norm_img, "magma", "norm"),
        ("3. Global Otsu Threshold Mask", results["threshold_mask"], "binary", "bool"),
        ("4. Adaptive Local Threshold Mask", adapt_mask, "binary", "bool"),
        ("5. Opened & Closed Morph Mask", results["morph_mask"], "Blues_r", "bool"),
        ("6. Euclidean Distance Transform", results["distance_map"], "turbo", "dist"),
        ("7. Regional Marker Seeds", results["markers"], "nipy_spectral", "markers"),
        ("8. Watershed Basins + Boundaries", results["labeled_basins"], "tab20", "watershed"),
        ("9. Semantic Ground Truth (Benchmark)", gt_np, "tab10", "gt"),
    ]

    for ax, (title, data, cmap_name, mode) in zip(axes.flat, panels):
        ax.set_facecolor("#161b26")
        ax.axis("off")
        ax.set_title(title, color="#f1f5f9", fontsize=11, fontweight="bold", pad=8)

        if mode in ("raw", "norm", "dist"):
            im = ax.imshow(data, cmap=cmap_name, origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif mode == "bool":
            display = np.where(data, 1.0, 0.0)
            ax.imshow(display, cmap="Blues_r", vmin=0, vmax=1, origin="lower")
        elif mode == "markers":
            ax.imshow(norm_img, cmap="gray", origin="lower")
            marker_pts = np.argwhere(data > 0)
            if marker_pts.size > 0:
                ax.scatter(marker_pts[:, 1], marker_pts[:, 0], c="#f43f5e", s=40, edgecolors="white", linewidth=1.2, label="Seeds")
                ax.legend(loc="upper right", fontsize=8)
        elif mode == "watershed":
            # Display colored basins with white boundary overlays
            ax.imshow(data, cmap="tab20", origin="lower")
            bound = results["watershed_boundaries"]
            if np.any(bound):
                bound_pts = np.argwhere(bound)
                ax.scatter(bound_pts[:, 1], bound_pts[:, 0], c="yellow", s=1, alpha=0.8)
        elif mode == "gt":
            # Ground truth 4 classes: 0=core, 1=bar, 2=arm, 3=star, 255=ignore
            # Map ignore to dark slate, and 4 classes to distinct vibrant colors
            h, w = data.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[data == 255] = [0.12, 0.14, 0.20]  # dark slate (ignore / benchmark sky)
            rgb[data == 0] = [0.95, 0.25, 0.25]    # red (core)
            rgb[data == 1] = [0.95, 0.75, 0.10]    # orange-yellow (bar)
            rgb[data == 2] = [0.20, 0.65, 0.95]    # sky blue (spiral arms)
            rgb[data == 3] = [0.30, 0.90, 0.40]    # green (star)
            ax.imshow(rgb, origin="lower")

    fig.suptitle(
        "Person B4 Classical Computer Vision Baseline — Step-by-Step Processing\n"
        "Sample 1-100523 (Pan-STARRS r-band) | Threshold -> Morphology -> Distance -> Watershed",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Compute quantitative comparison metrics
    # Note: 255 in ground truth is ignore index (no benchmark annotation).
    gt_annotated = gt_np != 255
    gt_foreground = gt_annotated & (gt_np >= 0) & (gt_np <= 3)
    classical_fg = results["morph_mask"]

    total_pixels = int(gt_np.size)
    annotated_pixels = int(np.count_nonzero(gt_annotated))
    ignored_pixels = int(np.count_nonzero(~gt_annotated))
    classical_predictions = int(np.count_nonzero(classical_fg))

    # Overlap with annotated regions
    predictions_overlapping_annotated = int(np.count_nonzero(classical_fg & gt_foreground))
    predictions_outside_annotated = int(np.count_nonzero(classical_fg & ~gt_annotated))
    missed_annotated_pixels = int(np.count_nonzero(~classical_fg & gt_foreground))

    # Sensitivity (Recall) within annotated regions: 1005 / 14091 = 0.0713 (7.13%)
    annotated_sensitivity = float(predictions_overlapping_annotated / max(annotated_pixels, 1))

    # Annotated-domain IoU: TP / (TP + FN) = 1005 / (1005 + 13086) = 0.0713
    annotated_domain_iou = float(predictions_overlapping_annotated / max(predictions_overlapping_annotated + missed_annotated_pixels, 1))

    # Annotated-domain Dice: 2 * TP / (2 * TP + FN) = 2010 / (2010 + 13086) = 0.1331
    annotated_domain_dice = float(2 * predictions_overlapping_annotated / max(2 * predictions_overlapping_annotated + missed_annotated_pixels, 1))

    return {
        "sample_id": "1-100523",
        "reference_band": "r",
        "threshold_method": "otsu",
        "threshold_value": float(results["metadata"]["threshold"]["threshold"]),
        "valid_pixels": int(np.count_nonzero(val_mask)),
        "pixel_breakdown": {
            "total_pixels": total_pixels,
            "annotated_pixels": annotated_pixels,
            "ignored_pixels": ignored_pixels,
            "classical_predictions": classical_predictions,
            "predictions_overlapping_annotated": predictions_overlapping_annotated,
            "predictions_outside_annotated": predictions_outside_annotated,
            "missed_annotated_pixels": missed_annotated_pixels,
        },
        "annotated_domain_metrics": {
            "annotated_domain_sensitivity": round(annotated_sensitivity, 4),
            "annotated_domain_iou": round(annotated_domain_iou, 4),
            "annotated_domain_dice": round(annotated_domain_dice, 4),
            "predictions_outside_annotated_unverified": predictions_outside_annotated,
            "precision_status": "not_meaningfully_measurable_with_available_annotations",
        },
        "morphology_components": results["metadata"]["connected_components"]["num_components"],
        "watershed_basins": results["metadata"]["watershed"]["num_basins"],
        "scientific_disclaimer": (
            "Ground-truth annotations cover only astronomical structures (classes 0-3), with all other pixels "
            "designated as unannotated (ignore index 255) rather than confirmed sky background. Consequently, "
            "false positives and precision are not meaningfully measurable. Predictions outside annotated regions "
            "(186 pixels) represent unverified detections in the benchmark field, not confirmed errors. "
            "Classical computer vision produces binary structural detections and regional basins that cannot be "
            "mapped to the four semantic astronomical classes."
        ),
    }


def main():
    clean_fits = Path("dataset/raw/imagedata_1-100523.fits")
    nan_fits = Path("dataset/raw/imagedata_1-103458.fits")
    clean_mask = Path("dataset/raw/mask_1-100523.fits")

    out_dir = Path("experiments/b4_classical_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating five-band diagnostic comparison visualization...")
    band_metrics = generate_five_band_visualization(
        clean_fits=clean_fits,
        nan_fits=nan_fits,
        output_path=out_dir / "five_band_comparison.png",
    )

    print("Generating stage-by-stage classical baseline visualization & evaluation...")
    pipeline_report = generate_pipeline_stages_visualization(
        fits_path=clean_fits,
        mask_path=clean_mask,
        output_path=out_dir / "classical_pipeline_stages.png",
    )

    # Combine into comprehensive JSON report
    report = {
        "experiment": "Person B4 — Classical Computer Vision Baseline",
        "recommended_reference_band": "r",
        "reference_band_status": "provisional",
        "reference_band_rationale": (
            "Provisional r-band reference recommendation based on diagnostics from two representative samples. "
            "Broader validation across the dataset is required. Pan-STARRS r-band (HDU 1) and i-band (HDU 2) "
            "offer the highest robust SNR (~15.6 - 16.7), lowest Sky-MAD noise (~106 - 402), and zero non-finite pixels "
            "in the audited samples. In contrast, g-band (HDU 0) contains 1,280 NaNs in sample 1-103458, and y-band (HDU 4) "
            "suffers from atmospheric and instrument thermal background noise (Sky-MAD ~698)."
        ),
        "band_comparison_diagnostics": band_metrics,
        "pipeline_stages_evaluation": pipeline_report,
    }

    report_path = out_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
