"""Severity Assessment and Leaf/Lesion Segmentation Package.

This package implements Phases 2, 3, and 4 of the Apple Leaf Disease Detection and Severity Assessment System:
- Phase 2: Complete Leaf Segmentation (severity.leaf_segmentation)
- Phase 3: Disease-Specific Lesion Segmentation (severity.lesion_segmentation)
- Phase 4: Severity Calculation (severity.severity_calculator)
- Visualization: 4-Panel Diagnostic Figure & Artifacts (severity.visualization)
- End-to-End Pipeline: Integrated Classification & Severity (severity.pipeline)
"""

from severity.leaf_segmentation import (
    segment_complete_leaf,
    fill_mask_holes,
    LEAF_LOWER_GREEN,
    LEAF_UPPER_GREEN,
    LEAF_LOWER_BROWN_RUST,
    LEAF_UPPER_BROWN_RUST,
    LEAF_LOWER_DARK,
    LEAF_UPPER_DARK,
)
from severity.lesion_segmentation import (
    segment_disease_lesions,
    create_lesion_overlay,
    highlight_lesion_boundaries,
    detect_apple_scab_lesions,
    detect_black_rot_lesions,
    detect_cedar_apple_rust_lesions,
    detect_healthy_lesions,
    DEFAULT_LESION_CONFIG,
)
from severity.severity_calculator import (
    calculate_severity,
    classify_severity_level,
    compute_severity_from_image,
    SEVERITY_THRESHOLDS,
    SEVERITY_DESCRIPTIONS,
)
from severity.visualization import (
    create_diagnostic_panel,
    generate_diagnostic_summary_dict,
    SEVERITY_COLORS,
)
from severity.pipeline import (
    IntegratedDiseaseSeverityPipeline,
    predict_and_assess_severity,
)

__all__ = [
    "segment_complete_leaf",
    "fill_mask_holes",
    "LEAF_LOWER_GREEN",
    "LEAF_UPPER_GREEN",
    "LEAF_LOWER_BROWN_RUST",
    "LEAF_UPPER_BROWN_RUST",
    "LEAF_LOWER_DARK",
    "LEAF_UPPER_DARK",
    "segment_disease_lesions",
    "create_lesion_overlay",
    "highlight_lesion_boundaries",
    "detect_apple_scab_lesions",
    "detect_black_rot_lesions",
    "detect_cedar_apple_rust_lesions",
    "detect_healthy_lesions",
    "DEFAULT_LESION_CONFIG",
    "calculate_severity",
    "classify_severity_level",
    "compute_severity_from_image",
    "SEVERITY_THRESHOLDS",
    "SEVERITY_DESCRIPTIONS",
    "create_diagnostic_panel",
    "generate_diagnostic_summary_dict",
    "SEVERITY_COLORS",
    "IntegratedDiseaseSeverityPipeline",
    "predict_and_assess_severity",
]

