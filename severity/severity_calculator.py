"""Severity Calculation Module (Phase 4).

Calculates foliar disease severity by comparing the segmented lesion area against
the total segmented leaf surface area:

Formula:
    Severity Percentage = (Diseased Lesion Area / Total Leaf Surface Area) * 100

Where:
    - Diseased Lesion Area = count of non-zero pixels in (lesion_mask & leaf_mask).
    - Total Leaf Surface Area = count of non-zero pixels in leaf_mask.

Severity Categories:
    - Healthy:  0.0% (asymptomatic foliar tissue)
    - Mild:     < 15.0% affected leaf area
    - Moderate: 15.0% to 40.0% affected leaf area
    - Severe:   > 40.0% affected leaf area

Edge Case Handling:
    - Empty leaf mask (0 pixels): Division by zero prevented, returns 0.0% and warning.
    - Empty lesion mask (0 pixels): Returns 0.0% and category "Healthy" or "Mild".
    - Non-intersected lesion pixels: Mask is strictly intersected with leaf mask.
    - Invalid inputs: Graceful error response with descriptive message.
"""

from typing import Dict, Any, Optional, Union
import os
import sys
import argparse
import numpy as np
import cv2

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from severity.leaf_segmentation import segment_complete_leaf
from severity.lesion_segmentation import segment_disease_lesions


# Severity Classification Cutoffs
SEVERITY_THRESHOLDS = {
    "MILD_MAX": 15.0,        # < 15% is Mild
    "MODERATE_MAX": 40.0,    # 15% to 40% is Moderate
}

# Agronomic severity descriptions
SEVERITY_DESCRIPTIONS = {
    "Healthy": "No visible foliar symptoms detected. Foliage exhibits full chlorophyll integrity.",
    "Mild": "Localized early-stage infection (< 15% leaf area). Preventative treatment and monitoring recommended.",
    "Moderate": "Established infection (15% - 40% leaf area). Curative fungicide and sanitation measures advised.",
    "Severe": "Extensive foliar damage (> 40% leaf area). High risk of premature defoliation and yield loss. Immediate intervention required.",
    "Indeterminate": "Unable to determine severity due to empty leaf mask or segmentation failure.",
}


def classify_severity_level(severity_percentage: float, disease_name: str = "") -> str:
    """Map a numerical severity percentage to its standard phytopathological category.

    Args:
        severity_percentage: Calculated severity percentage (0.0 to 100.0).
        disease_name: Disease class name.

    Returns:
        str: "Healthy", "Mild", "Moderate", or "Severe".
    """
    norm_name = disease_name.strip().lower()
    if "healthy" in norm_name:
        return "Healthy"

    if severity_percentage <= 0.001:
        # If disease was diagnosed as diseased but 0 lesion pixels found, mark as Mild/Trace
        return "Mild"
    elif severity_percentage < SEVERITY_THRESHOLDS["MILD_MAX"]:
        return "Mild"
    elif severity_percentage <= SEVERITY_THRESHOLDS["MODERATE_MAX"]:
        return "Moderate"
    else:
        return "Severe"


def calculate_severity(
    leaf_mask: Optional[np.ndarray],
    lesion_mask: Optional[np.ndarray],
    disease_name: str = "Unknown",
) -> Dict[str, Any]:
    """Calculate percentage of leaf surface affected by disease and determine severity grade.

    Ensures mathematical rigor:
    1. Validates presence and compatibility of both binary masks.
    2. Enforces intersection: (lesion_mask & leaf_mask) before computing pixel counts.
    3. Handles zero leaf area division by zero gracefully.
    4. Categorizes into Healthy, Mild, Moderate, or Severe.

    Args:
        leaf_mask: Binary uint8 mask representing complete leaf surface (Phase 2).
        lesion_mask: Binary uint8 mask representing disease lesions (Phase 3).
        disease_name: Name of diagnosed condition.

    Returns:
        Dict[str, Any] matching the required project specification:
        {
            "disease": str,
            "leaf_area_pixels": int,
            "lesion_area_pixels": int,
            "severity_percentage": float,
            "severity_level": str,
            "description": str,
            "status": str,
        }
    """
    norm_disease = disease_name.strip()
    if norm_disease in ["Apple___Apple_scab", "scab", "apple_scab"]:
        norm_disease = "Apple Scab"
    elif norm_disease in ["Apple___Black_rot", "blackrot", "black_rot"]:
        norm_disease = "Black Rot"
    elif norm_disease in ["Apple___Cedar_apple_rust", "rust", "cedar_apple_rust"]:
        norm_disease = "Cedar Apple Rust"
    elif norm_disease in ["Apple___healthy", "healthy"]:
        norm_disease = "Healthy"

    # Edge Case: Both masks None or invalid
    if leaf_mask is None:
        return {
            "disease": norm_disease,
            "leaf_area_pixels": 0,
            "lesion_area_pixels": 0,
            "severity_percentage": 0.0,
            "severity_level": "Indeterminate",
            "description": SEVERITY_DESCRIPTIONS["Indeterminate"],
            "status": "error_null_leaf_mask",
        }

    # Count leaf surface area
    leaf_area_pixels = int(np.count_nonzero(leaf_mask))

    # Edge Case: Empty leaf mask (0 pixels) -> prevents DivisionByZero
    if leaf_area_pixels == 0:
        return {
            "disease": norm_disease,
            "leaf_area_pixels": 0,
            "lesion_area_pixels": 0,
            "severity_percentage": 0.0,
            "severity_level": "Indeterminate",
            "description": "Empty leaf mask: no leaf pixels were detected.",
            "status": "empty_leaf_mask",
        }

    # For Healthy class: severity is mathematically and biologically 0%
    if norm_disease == "Healthy":
        return {
            "disease": "Healthy",
            "leaf_area_pixels": leaf_area_pixels,
            "lesion_area_pixels": 0,
            "severity_percentage": 0.0,
            "severity_level": "Healthy",
            "description": SEVERITY_DESCRIPTIONS["Healthy"],
            "status": "success",
        }

    # If lesion_mask is None or empty
    if lesion_mask is None or np.count_nonzero(lesion_mask) == 0:
        return {
            "disease": norm_disease,
            "leaf_area_pixels": leaf_area_pixels,
            "lesion_area_pixels": 0,
            "severity_percentage": 0.0,
            "severity_level": "Mild",
            "description": "No active lesions segmented. Leaf exhibits trace or negligible symptoms.",
            "status": "success",
        }

    # Enforce strict intersection of lesion mask with leaf mask
    # Background pixels are guaranteed never to be included in lesion area
    intersected_lesion = cv2.bitwise_and(lesion_mask, leaf_mask)
    lesion_area_pixels = int(np.count_nonzero(intersected_lesion))

    # Calculate severity percentage
    severity_pct = (lesion_area_pixels / leaf_area_pixels) * 100.0
    severity_pct = round(min(100.0, max(0.0, severity_pct)), 2)

    # Assign categorical severity level
    severity_level = classify_severity_level(severity_pct, disease_name=norm_disease)

    return {
        "disease": norm_disease,
        "leaf_area_pixels": leaf_area_pixels,
        "lesion_area_pixels": lesion_area_pixels,
        "severity_percentage": severity_pct,
        "severity_level": severity_level,
        "description": SEVERITY_DESCRIPTIONS.get(severity_level, ""),
        "status": "success",
    }


def compute_severity_from_image(
    image_input: Union[str, np.ndarray],
    disease_name: str,
    target_size: Optional[tuple] = (500, 500),
    is_rgb: bool = False,
) -> Dict[str, Any]:
    """End-to-end convenience function: Image -> Leaf Mask -> Lesion Mask -> Severity Calculation.

    Args:
        image_input: Filepath or numpy image array.
        disease_name: Diagnosed disease.
        target_size: Dimensions for standardized processing.
        is_rgb: Whether input array is RGB.

    Returns:
        Dict[str, Any] with combined severity metrics and intermediate masks.
    """
    leaf_res = segment_complete_leaf(image_input, target_size=target_size, is_rgb=is_rgb)
    if not leaf_res["success"]:
        return {
            "severity": {
                "disease": disease_name,
                "leaf_area_pixels": 0,
                "lesion_area_pixels": 0,
                "severity_percentage": 0.0,
                "severity_level": "Indeterminate",
                "description": f"Segmentation failed: {leaf_res['error_message']}",
                "status": "segmentation_failure",
            },
            "leaf_artifacts": leaf_res,
            "lesion_artifacts": None,
        }

    lesion_res = segment_disease_lesions(
        rgb_image=leaf_res["original_image"],
        leaf_mask=leaf_res["leaf_mask"],
        disease_name=disease_name,
        hsv_image=leaf_res["hsv_image"],
    )

    sev_res = calculate_severity(
        leaf_mask=leaf_res["leaf_mask"],
        lesion_mask=lesion_res["lesion_mask"],
        disease_name=disease_name,
    )

    return {
        "severity": sev_res,
        "leaf_artifacts": leaf_res,
        "lesion_artifacts": lesion_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Foliar Disease Severity Calculator")
    parser.add_argument("--image", required=True, help="Path to input apple leaf image")
    parser.add_argument(
        "--disease",
        required=True,
        choices=["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
        help="Disease class to evaluate",
    )
    args = parser.parse_args()

    result = compute_severity_from_image(args.image, disease_name=args.disease)
    sev = result["severity"]

    print("\n========================================================")
    print(" PHASE 4: DISEASE SEVERITY QUANTIFICATION RESULT")
    print("========================================================")
    print(f"Disease Class:       {sev['disease']}")
    print(f"Leaf Area Pixels:    {sev['leaf_area_pixels']:,} px")
    print(f"Lesion Area Pixels:  {sev['lesion_area_pixels']:,} px")
    print(f"Severity Percentage: {sev['severity_percentage']:.2f}%")
    print(f"Severity Level:      {sev['severity_level']}")
    print(f"Clinical Guidance:   {sev['description']}")
    print("========================================================\n")

