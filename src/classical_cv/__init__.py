"""Classical computer vision baseline module for galaxy segmentation."""

from src.classical_cv.band_comparison import (
    BAND_NAMES,
    BAND_TO_HDU,
    HDU_TO_BAND,
    compare_five_bands,
    compute_band_metrics,
    load_band_fits,
    recommend_reference_band,
)
from src.classical_cv.baseline import run_classical_pipeline
from src.classical_cv.connected_components import label_connected_components
from src.classical_cv.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    get_structuring_element,
)
from src.classical_cv.thresholding import adaptive_threshold, otsu_threshold
from src.classical_cv.watershed import (
    distance_transform_edt,
    extract_watershed_boundaries,
    generate_markers_from_peaks,
    watershed_segmentation,
)

__all__ = [
    "BAND_NAMES",
    "BAND_TO_HDU",
    "HDU_TO_BAND",
    "load_band_fits",
    "compute_band_metrics",
    "compare_five_bands",
    "recommend_reference_band",
    "otsu_threshold",
    "adaptive_threshold",
    "get_structuring_element",
    "binary_erosion",
    "binary_dilation",
    "binary_opening",
    "binary_closing",
    "label_connected_components",
    "distance_transform_edt",
    "generate_markers_from_peaks",
    "watershed_segmentation",
    "extract_watershed_boundaries",
    "run_classical_pipeline",
]
