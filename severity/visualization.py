"""Visualization Module for Apple Leaf Disease and Severity Analysis.

Generates comprehensive visual diagnostic artifacts:
1. Original RGB Leaf Photograph
2. Complete Leaf Segmentation Mask (Phase 2)
3. Disease-Specific Lesion Segmentation Mask (Phase 3)
4. Final Overlay with Infected Regions and Boundary Contours (Phase 4)
5. Comprehensive multi-panel figure with severity metrics and clinical status badges.
"""

from typing import Dict, Any, Optional, Tuple, Union
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from severity.lesion_segmentation import create_lesion_overlay, highlight_lesion_boundaries


SEVERITY_COLORS = {
    "Healthy": "#10b981",     # Emerald green
    "Mild": "#f59e0b",        # Amber
    "Moderate": "#f97316",    # Vivid orange
    "Severe": "#ef4444",      # Crimson red
    "Indeterminate": "#6b7280"# Gray
}


def create_diagnostic_panel(
    original_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    disease_name: str,
    severity_percentage: float,
    severity_level: str,
    leaf_area_pixels: int,
    lesion_area_pixels: int,
    confidence: Optional[float] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Generate a high-resolution 4-panel diagnostic figure visualizing all segmentation phases.

    Panels:
        (1) Original Input Leaf Photograph
        (2) Complete Leaf Mask (Phase 2)
        (3) Disease Lesion Mask (Phase 3)
        (4) Lesion Overlay & Contours on Leaf (Phase 4)

    Args:
        original_rgb: Standardized RGB image array.
        leaf_mask: Binary complete leaf mask (0 or 255).
        lesion_mask: Binary lesion mask (0 or 255).
        disease_name: Predicted or true disease name.
        severity_percentage: Calculated foliar severity percentage.
        severity_level: Categorical severity level ("Healthy", "Mild", "Moderate", "Severe").
        leaf_area_pixels: Total count of pixels in leaf mask.
        lesion_area_pixels: Total count of pixels in lesion mask.
        confidence: Optional classification confidence probability (0.0 to 1.0).
        save_path: Optional file path to export the resulting figure.
        dpi: Dots per inch for saved figure.

    Returns:
        plt.Figure: The generated Matplotlib Figure.
    """
    # Create composite overlay
    overlay_img = create_lesion_overlay(original_rgb, lesion_mask, color=(220, 20, 60), alpha=0.45)
    composite_vis = highlight_lesion_boundaries(overlay_img, lesion_mask, color=(255, 230, 0), thickness=2)

    # Segmented leaf on black background
    segmented_leaf = cv2.bitwise_and(original_rgb, original_rgb, mask=leaf_mask)

    # Initialize figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), facecolor="#f8fafc")

    badge_color = SEVERITY_COLORS.get(severity_level, "#6b7280")

    # Overall Supertitle
    conf_str = f" | Confidence: {confidence*100:.1f}%" if confidence is not None else ""
    fig.suptitle(
        f"Apple Leaf Disease & Severity Diagnostic Report\n"
        f"Condition: {disease_name}{conf_str} | Severity: {severity_percentage:.2f}% [{severity_level.upper()}]",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    # Panel 1: Original Image
    axes[0].imshow(original_rgb)
    axes[0].set_title("1. Original Leaf Image", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: Complete Leaf Segmentation Mask (Phase 2)
    axes[1].imshow(leaf_mask, cmap="gray")
    axes[1].set_title(
        f"2. Complete Leaf Mask\n({leaf_area_pixels:,} px)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[1].axis("off")

    # Panel 3: Disease Lesion Mask (Phase 3)
    axes[2].imshow(lesion_mask, cmap="hot")
    axes[2].set_title(
        f"3. Disease Lesions\n({lesion_area_pixels:,} px)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[2].axis("off")

    # Panel 4: Composite Lesion Overlay & Contours (Phase 4)
    axes[3].imshow(composite_vis)
    axes[3].set_title(
        f"4. Infected Overlay Map\n(Severity: {severity_percentage:.1f}%)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[3].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.90])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[INFO] Diagnostic panel exported to: {save_path}")

    return fig


def generate_diagnostic_summary_dict(
    original_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    disease_name: str,
    severity_percentage: float,
    severity_level: str,
    leaf_area_pixels: int,
    lesion_area_pixels: int,
) -> Dict[str, Any]:
    """Generate all visualization image artifacts as in-memory numpy RGB arrays."""
    overlay_img = create_lesion_overlay(original_rgb, lesion_mask, color=(220, 20, 60), alpha=0.45)
    composite_vis = highlight_lesion_boundaries(overlay_img, lesion_mask, color=(255, 230, 0), thickness=2)
    segmented_leaf = cv2.bitwise_and(original_rgb, original_rgb, mask=leaf_mask)

    return {
        "original_image": original_rgb,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
        "segmented_leaf": segmented_leaf,
        "overlay_image": overlay_img,
        "composite_visualization": composite_vis,
        "badge_color": SEVERITY_COLORS.get(severity_level, "#6b7280"),
    }

