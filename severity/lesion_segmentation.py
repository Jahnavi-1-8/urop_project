"""Disease Lesion Segmentation Module (Phase 3).

Implements disease-specific lesion segmentation restricted strictly to the leaf tissue:
1. Converts RGB leaf image to HSV color space.
2. Applies disease-specific color thresholding and morphological filtering tailored to:
   - Apple Scab: Dark olive, brown, and velvety black necrotic lesions.
   - Black Rot: Dark necrotic centers and reddish-brown / purple circular frog-eye margins.
   - Cedar Apple Rust: Bright orange, yellow-orange pustules and chlorotic halos.
   - Healthy: Asymptomatic tissue; returns zero lesion mask and 0 pixels.
3. Restricts candidate lesions strictly to the segmented leaf mask (cv2.bitwise_and).
4. Cleans noise via morphological opening/closing and filters isolated spurious contours.
5. Produces binary lesion masks, boundary-highlighted images, and semi-transparent overlays.
"""

from typing import Tuple, Dict, Any, Optional, Union
import os
import sys
import argparse
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import rgb_to_hsv, load_image, resize_image, bgr_to_rgb, DEFAULT_IMAGE_SIZE
from severity.leaf_segmentation import segment_complete_leaf


# Disease-Specific Configurable Color Thresholds (HSV)
DEFAULT_LESION_CONFIG: Dict[str, Dict[str, Any]] = {
    "Apple Scab": {
        # Olive-green to dark velvety brown scab spots
        "olive_brown_lower": np.array([12, 30, 20], dtype=np.uint8),
        "olive_brown_upper": np.array([42, 255, 140], dtype=np.uint8),
        # Dark necrotic velvety spots (low luminance, non-green)
        "dark_scab_lower": np.array([0, 15, 10], dtype=np.uint8),
        "dark_scab_upper": np.array([50, 255, 75], dtype=np.uint8),
        "min_contour_area": 8,
        "overlay_color": (220, 20, 60),      # Crimson red overlay for scab
    },
    "Black Rot": {
        # Dark necrotic central core
        "necrotic_dark_lower": np.array([0, 15, 10], dtype=np.uint8),
        "necrotic_dark_upper": np.array([180, 255, 80], dtype=np.uint8),
        # Distinct frog-eye brown / reddish margin
        "frogeye_margin_lower": np.array([5, 35, 25], dtype=np.uint8),
        "frogeye_margin_upper": np.array([28, 255, 160], dtype=np.uint8),
        "min_contour_area": 10,
        "overlay_color": (178, 34, 34),       # Firebrick red overlay for black rot
    },
    "Cedar Apple Rust": {
        # Vibrant orange / rust-colored aecial pustules
        "rust_pustule_lower": np.array([8, 55, 50], dtype=np.uint8),
        "rust_pustule_upper": np.array([28, 255, 255], dtype=np.uint8),
        # Yellowish chlorotic rings / halos surrounding pustules
        "chlorotic_halo_lower": np.array([28, 65, 65], dtype=np.uint8),
        "chlorotic_halo_upper": np.array([38, 255, 255], dtype=np.uint8),
        "min_contour_area": 6,
        "overlay_color": (255, 140, 0),       # Dark orange overlay for rust
    },
    "Healthy": {
        "min_contour_area": 0,
        "overlay_color": (46, 139, 87),       # Sea green placeholder
    },
}


def create_lesion_overlay(
    rgb_image: np.ndarray,
    lesion_mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Create a semi-transparent colored overlay over detected lesion areas.

    Args:
        rgb_image: Standardized RGB image.
        lesion_mask: Binary uint8 mask of detected lesions (0 or 255).
        color: RGB tuple for the lesion tint.
        alpha: Blending weight for the overlay (0.0 = original, 1.0 = solid color).

    Returns:
        np.ndarray: Blended RGB image with lesions tinted.
    """
    if lesion_mask is None or np.count_nonzero(lesion_mask) == 0:
        return rgb_image.copy()

    overlay = rgb_image.copy()
    color_plane = np.zeros_like(rgb_image)
    color_plane[lesion_mask > 0] = color
    blended = cv2.addWeighted(rgb_image, 1.0 - alpha, color_plane, alpha, 0)
    overlay[lesion_mask > 0] = blended[lesion_mask > 0]
    return overlay


def highlight_lesion_boundaries(
    rgb_image: np.ndarray,
    lesion_mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw vibrant boundary outlines around detected lesion regions.

    Args:
        rgb_image: Standardized RGB image.
        lesion_mask: Binary uint8 mask of detected lesions (0 or 255).
        color: RGB tuple for the contour boundary.
        thickness: Line thickness in pixels.

    Returns:
        np.ndarray: RGB image with lesion boundaries outlined.
    """
    highlighted = rgb_image.copy()
    if lesion_mask is None or np.count_nonzero(lesion_mask) == 0:
        return highlighted

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(highlighted, contours, -1, color, thickness)
    return highlighted


def detect_apple_scab_lesions(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    leaf_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Segment Apple Scab (Venturia inaequalis) lesions within the leaf region."""
    cfg = config or DEFAULT_LESION_CONFIG["Apple Scab"]
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    # Olive / brown velvety lesions
    lower_ob = cfg["olive_brown_lower"]
    upper_ob = cfg["olive_brown_upper"]
    mask_olive_brown = (
        (h >= lower_ob[0]) & (h <= upper_ob[0]) &
        (s >= lower_ob[1]) & (s <= upper_ob[1]) &
        (v >= lower_ob[2]) & (v <= upper_ob[2])
    )

    # Dark necrotic spots (low luminance scab lesions)
    lower_dark = cfg["dark_scab_lower"]
    upper_dark = cfg["dark_scab_upper"]
    mask_dark = (
        (h >= lower_dark[0]) & (h <= upper_dark[0]) &
        (s >= lower_dark[1]) & (s <= upper_dark[1]) &
        (v >= lower_dark[2]) & (v <= upper_dark[2])
    )

    raw_lesion = (mask_olive_brown | mask_dark).astype(np.uint8) * 255
    return raw_lesion


def detect_black_rot_lesions(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    leaf_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Segment Black Rot (Botryosphaeria obtusa) frog-eye lesions within the leaf region."""
    cfg = config or DEFAULT_LESION_CONFIG["Black Rot"]
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    # Dark necrotic center
    lower_dk = cfg["necrotic_dark_lower"]
    upper_dk = cfg["necrotic_dark_upper"]
    mask_dark = (
        (h >= lower_dk[0]) & (h <= upper_dk[0]) &
        (s >= lower_dk[1]) & (s <= upper_dk[1]) &
        (v >= lower_dk[2]) & (v <= upper_dk[2])
    )

    # Distinct brown/purple frog-eye margin
    lower_mg = cfg["frogeye_margin_lower"]
    upper_mg = cfg["frogeye_margin_upper"]
    mask_margin = (
        (h >= lower_mg[0]) & (h <= upper_mg[0]) &
        (s >= lower_mg[1]) & (s <= upper_mg[1]) &
        (v >= lower_mg[2]) & (v <= upper_mg[2])
    )

    raw_lesion = (mask_dark | mask_margin).astype(np.uint8) * 255
    return raw_lesion


def detect_cedar_apple_rust_lesions(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    leaf_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Segment Cedar Apple Rust (Gymnosporangium juniperi-virginianae) pustules within the leaf region."""
    cfg = config or DEFAULT_LESION_CONFIG["Cedar Apple Rust"]
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    # Bright orange/rust-colored pustules
    lower_rust = cfg["rust_pustule_lower"]
    upper_rust = cfg["rust_pustule_upper"]
    mask_rust = (
        (h >= lower_rust[0]) & (h <= upper_rust[0]) &
        (s >= lower_rust[1]) & (s <= upper_rust[1]) &
        (v >= lower_rust[2]) & (v <= upper_rust[2])
    )

    # Chlorotic yellow haloes surrounding rust centers
    lower_halo = cfg["chlorotic_halo_lower"]
    upper_halo = cfg["chlorotic_halo_upper"]
    mask_halo = (
        (h >= lower_halo[0]) & (h <= upper_halo[0]) &
        (s >= lower_halo[1]) & (s <= upper_halo[1]) &
        (v >= lower_halo[2]) & (v <= upper_halo[2])
    )

    raw_lesion = (mask_rust | mask_halo).astype(np.uint8) * 255
    return raw_lesion


def detect_healthy_lesions(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    leaf_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return an empty lesion mask for asymptomatic healthy leaves."""
    return np.zeros_like(leaf_mask)


def segment_disease_lesions(
    rgb_image: np.ndarray,
    leaf_mask: np.ndarray,
    disease_name: str,
    hsv_image: Optional[np.ndarray] = None,
    custom_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Segment foliar lesions according to the specified disease diagnosis.

    Guarantees:
    - Lesion segmentation is strictly intersected with the leaf mask (cv2.bitwise_and).
    - Background pixels are 100% excluded.
    - Healthy condition returns zero lesion pixels.
    - Morphological cleanup removes single-pixel noise and bridges lesion clusters.

    Args:
        rgb_image: Standardized RGB leaf image.
        leaf_mask: Binary uint8 complete leaf mask from Phase 2 (0 or 255).
        disease_name: Predicted disease ("Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy").
        hsv_image: Optional precomputed HSV leaf image.
        custom_config: Optional custom dictionary of HSV thresholds for tuning.

    Returns:
        Dict[str, Any] containing:
            - "disease": Standardized disease name.
            - "lesion_mask": Binary uint8 mask (255 for lesion, 0 elsewhere).
            - "lesion_area_pixels": Total number of lesion pixels inside leaf mask.
            - "highlighted_image": RGB image with lesion boundaries highlighted.
            - "overlay_image": RGB image with semi-transparent lesion overlay.
            - "status": str execution status.
    """
    if rgb_image is None or leaf_mask is None or np.count_nonzero(leaf_mask) == 0:
        h, w = (500, 500) if rgb_image is None else rgb_image.shape[:2]
        blank_mask = np.zeros((h, w), dtype=np.uint8)
        blank_rgb = np.zeros((h, w, 3), dtype=np.uint8) if rgb_image is None else rgb_image.copy()
        return {
            "disease": disease_name,
            "lesion_mask": blank_mask,
            "lesion_area_pixels": 0,
            "highlighted_image": blank_rgb,
            "overlay_image": blank_rgb,
            "status": "empty_leaf_mask",
        }

    # Normalize disease label
    norm_name = disease_name.strip()
    if norm_name in ["Apple___Apple_scab", "scab", "apple_scab"]:
        norm_name = "Apple Scab"
    elif norm_name in ["Apple___Black_rot", "blackrot", "black_rot"]:
        norm_name = "Black Rot"
    elif norm_name in ["Apple___Cedar_apple_rust", "rust", "cedar_apple_rust"]:
        norm_name = "Cedar Apple Rust"
    elif norm_name in ["Apple___healthy", "healthy"]:
        norm_name = "Healthy"

    # Healthy leaves return strictly zero lesion mask
    if norm_name == "Healthy":
        empty_mask = np.zeros_like(leaf_mask)
        return {
            "disease": "Healthy",
            "lesion_mask": empty_mask,
            "lesion_area_pixels": 0,
            "highlighted_image": rgb_image.copy(),
            "overlay_image": rgb_image.copy(),
            "status": "healthy_asymptomatic",
        }

    # Precompute HSV if not supplied
    if hsv_image is None:
        hsv_image = rgb_to_hsv(rgb_image)

    # Retrieve disease configuration
    cfg = custom_config or DEFAULT_LESION_CONFIG.get(norm_name, DEFAULT_LESION_CONFIG["Apple Scab"])
    min_contour_area = cfg.get("min_contour_area", 8)
    overlay_color = cfg.get("overlay_color", (220, 20, 60))

    # Dispatch disease-specific detector
    if norm_name == "Apple Scab":
        raw_lesions = detect_apple_scab_lesions(rgb_image, hsv_image, leaf_mask, config=cfg)
    elif norm_name == "Black Rot":
        raw_lesions = detect_black_rot_lesions(rgb_image, hsv_image, leaf_mask, config=cfg)
    elif norm_name == "Cedar Apple Rust":
        raw_lesions = detect_cedar_apple_rust_lesions(rgb_image, hsv_image, leaf_mask, config=cfg)
    else:
        # Fallback to Scab detector if unrecognized disease string
        raw_lesions = detect_apple_scab_lesions(rgb_image, hsv_image, leaf_mask, config=cfg)

    # 1. Strictly intersect lesion detection with complete leaf mask
    # Background pixels are mathematically prevented from being counted
    lesion_mask_bounded = cv2.bitwise_and(raw_lesions, leaf_mask)

    # 2. Morphological cleanup: remove single-pixel artifacts and close lesion boundaries
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(lesion_mask_bounded, cv2.MORPH_OPEN, kernel_clean)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_clean)

    # 3. Contour area filtering: suppress spurious isolated speckles (< min_contour_area)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_lesion_mask = np.zeros_like(cleaned)
    for c in contours:
        if cv2.contourArea(c) >= min_contour_area:
            cv2.drawContours(final_lesion_mask, [c], -1, 255, thickness=cv2.FILLED)

    # Final intersection verification
    final_lesion_mask = cv2.bitwise_and(final_lesion_mask, leaf_mask)
    lesion_area_pixels = int(np.count_nonzero(final_lesion_mask))

    # Generate visualizations
    overlay_img = create_lesion_overlay(rgb_image, final_lesion_mask, color=overlay_color, alpha=0.45)
    highlight_img = highlight_lesion_boundaries(rgb_image, final_lesion_mask, color=(255, 230, 0), thickness=2)

    return {
        "disease": norm_name,
        "lesion_mask": final_lesion_mask,
        "lesion_area_pixels": lesion_area_pixels,
        "highlighted_image": highlight_img,
        "overlay_image": overlay_img,
        "status": "success",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Disease-Specific Lesion Segmentation")
    parser.add_argument("--image", required=True, help="Path to input apple leaf image")
    parser.add_argument(
        "--disease",
        required=True,
        choices=["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
        help="Disease class to guide lesion segmentation",
    )
    parser.add_argument("--save-overlay", default=None, help="Optional path to save overlay visualization")
    args = parser.parse_args()

    # Step 1: Complete Leaf Segmentation (Phase 2)
    leaf_res = segment_complete_leaf(args.image)
    if not leaf_res["success"]:
        print(f"[ERROR] Leaf segmentation failed: {leaf_res['error_message']}")
        sys.exit(1)

    # Step 2: Lesion Segmentation (Phase 3)
    lesion_res = segment_disease_lesions(
        rgb_image=leaf_res["original_image"],
        leaf_mask=leaf_res["leaf_mask"],
        disease_name=args.disease,
        hsv_image=leaf_res["hsv_image"],
    )

    print("\n========================================================")
    print(" PHASE 3: DISEASE LESION SEGMENTATION RESULT")
    print("========================================================")
    print(f"Target Disease:      {lesion_res['disease']}")
    print(f"Total Leaf Pixels:   {leaf_res['leaf_area_pixels']:,} px")
    print(f"Lesion Pixels:       {lesion_res['lesion_area_pixels']:,} px")
    pct = (lesion_res['lesion_area_pixels'] / leaf_res['leaf_area_pixels']) * 100.0
    print(f"Lesion Percentage:   {pct:.2f}%")
    print(f"Status:              {lesion_res['status']}")

    if args.save_overlay:
        bgr_overlay = cv2.cvtColor(lesion_res["overlay_image"], cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_overlay)), exist_ok=True)
        cv2.imwrite(args.save_overlay, bgr_overlay)
        print(f"Saved Overlay Image: {args.save_overlay}")
    print("========================================================\n")

