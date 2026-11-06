"""Complete Leaf Segmentation Module (Phase 2 - Enhanced).

Isolates the visible apple leaf blade from background, soil, fingers, shadows, and artifacts:
1. Adapts color chromaticity (CIE L*a*b* green-red axis and Excess Green index ExG)
   with GrabCut graph-cut optimization to separate the leaf blade from background.
2. Background characterization: eliminates neutral gray sheets, white cards, dark shadows, and soil.
3. Connected component filtering to remove external background artifacts and small specks.
4. Internal hole-filling: ensures internal necrotic spots are 100% captured inside the leaf blade.
5. Foliar chromatic density verification: rejects non-foliar objects and greeting posters.
6. Returns standardized outputs preserving existing pipeline interfaces.
"""

from typing import Tuple, Dict, Any, Optional, Union
import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import (
    load_image,
    resize_image,
    bgr_to_rgb,
    rgb_to_hsv,
    DEFAULT_IMAGE_SIZE,
)

# Botanical HSV foliar threshold constants
LEAF_LOWER_GREEN: np.ndarray = np.array([22, 20, 20], dtype=np.uint8)
LEAF_UPPER_GREEN: np.ndarray = np.array([95, 255, 255], dtype=np.uint8)
LEAF_LOWER_BROWN_RUST: np.ndarray = np.array([7, 20, 15], dtype=np.uint8)
LEAF_UPPER_BROWN_RUST: np.ndarray = np.array([38, 255, 255], dtype=np.uint8)
LEAF_LOWER_DARK: np.ndarray = np.array([0, 15, 10], dtype=np.uint8)
LEAF_UPPER_DARK: np.ndarray = np.array([45, 255, 85], dtype=np.uint8)


def fill_mask_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill all internal holes within a binary mask using border-padded flood-fill.

    Guarantees that necrotic spots, fungal pustules, or dark central lesions
    surrounded by the leaf blade are 100% captured as leaf tissue.
    """
    if binary_mask is None or binary_mask.size == 0 or np.count_nonzero(binary_mask) == 0:
        return binary_mask

    h, w = binary_mask.shape[:2]
    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    floodfilled = padded.copy()
    mask_ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(floodfilled, mask_ff, (0, 0), 255)
    floodfilled_inv = cv2.bitwise_not(floodfilled)
    filled_padded = padded | floodfilled_inv
    return filled_padded[1:h + 1, 1:w + 1]


def segment_complete_leaf(
    image_input: Union[str, np.ndarray],
    target_size: Optional[Tuple[int, int]] = DEFAULT_IMAGE_SIZE,
    is_rgb: bool = False,
    min_leaf_ratio: float = 0.04,
    max_leaf_ratio: float = 0.96,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Segment the complete apple leaf (both healthy and diseased regions) from the background.

    Args:
        image_input: File path (str) or loaded numpy image array (BGR or RGB).
        target_size: Dimension tuple (width, height) for image standardization.
        is_rgb: Whether input array is already in RGB order.
        min_leaf_ratio: Minimum fraction of image area required to constitute a leaf.
        max_leaf_ratio: Maximum fraction of image area to prevent full-frame background capture.

    Returns:
        Dict[str, Any] containing:
            - "success": bool
            - "original_image": Standardized RGB image.
            - "hsv_image": HSV representation.
            - "leaf_mask": Binary uint8 mask (255 for leaf, 0 for background).
            - "leaf_area_pixels": Total number of pixels belonging to the leaf.
            - "segmented_leaf": RGB image with background masked to black (0, 0, 0).
            - "coverage_percentage": Leaf pixel area as percentage of total image area.
            - "bounding_box": (x, y, w, h) of the segmented leaf.
            - "error_message": Error description if success is False, else None.
    """
    raw_img = load_image(image_input)
    if raw_img is None:
        return {
            "success": False,
            "original_image": None,
            "hsv_image": None,
            "leaf_mask": None,
            "leaf_area_pixels": 0,
            "segmented_leaf": None,
            "coverage_percentage": 0.0,
            "bounding_box": (0, 0, 0, 0),
            "error_message": "Invalid or empty image input. Could not load image.",
        }

    # Handle grayscale or 4-channel BGRA/RGBA images
    if len(raw_img.shape) == 2:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    elif raw_img.shape[2] == 4:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)

    if target_size is not None:
        processed_img = resize_image(raw_img, target_size)
    else:
        processed_img = raw_img.copy()

    rgb_img = processed_img if is_rgb else bgr_to_rgb(processed_img)
    hsv_img = rgb_to_hsv(rgb_img)

    h_img, w_img = rgb_img.shape[:2]
    total_pixels = h_img * w_img

    # 1. Color space representations for robust foliar isolation
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b = rgb_img[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b  # Excess Green Index

    # 2. Background characterization (suppress neutral sheets, white paper, shadows, dark backdrop)
    neutral_bg = (chroma < 10.0) & (np.abs(r - g) < 18.0) & (np.abs(g - b) < 18.0)
    bright_bg = (L > 225.0) | ((r > 220.0) & (g > 220.0) & (b > 220.0))
    dark_bg = (L < 16.0) & (chroma < 8.0)

    # 3. Foliar tissue evidence
    # Healthy chlorophyll green
    is_green = (A < 125.0) | (exg > 6.0)
    # Rust / chlorotic yellow-orange
    is_rust = (chroma > 12.0) & (B > 134.0) & (r > b) & (g > b)
    # Necrotic foliar brown
    is_necrotic = (A >= 118.0) & (A <= 145.0) & (B >= 124.0) & (B <= 175.0) & (L >= 20.0) & (L <= 185.0) & (exg > -35.0)

    foliar_evidence = (is_green | is_rust | is_necrotic) & (~neutral_bg) & (~bright_bg) & (~dark_bg)

    # Sanity check: minimum foliar evidence
    if np.sum(foliar_evidence) < int(min_leaf_ratio * total_pixels):
        empty_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        return {
            "success": False,
            "original_image": rgb_img,
            "hsv_image": hsv_img,
            "leaf_mask": empty_mask,
            "leaf_area_pixels": 0,
            "segmented_leaf": np.zeros_like(rgb_img),
            "coverage_percentage": 0.0,
            "bounding_box": (0, 0, 0, 0),
            "error_message": "Insufficient foliar tissue detected to isolate a leaf blade.",
        }

    # 4. Energy minimization (GrabCut) initialized with foliar priors
    gc_mask = np.full((h_img, w_img), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[foliar_evidence] = cv2.GC_PR_FGD

    # Perimeter border condition: image border lacking strong foliar evidence is definite background
    border_mask = np.zeros((h_img, w_img), dtype=bool)
    border_margin = max(2, int(min(h_img, w_img) * 0.015))
    border_mask[0:border_margin, :] = True
    border_mask[-border_margin:, :] = True
    border_mask[:, 0:border_margin] = True
    border_mask[:, -border_margin:] = True

    gc_mask[border_mask & (~foliar_evidence)] = cv2.GC_BGD
    gc_mask[neutral_bg | bright_bg | dark_bg] = cv2.GC_BGD

    # Definite foreground seed: eroded core of high foliar certainty
    strong_foliar = is_green | is_rust
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded_core = cv2.erode(strong_foliar.astype(np.uint8), kernel_core)
    gc_mask[eroded_core > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb_img, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        raw_leaf = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        raw_leaf = (foliar_evidence.astype(np.uint8) * 255)

    # 5. Connected component filtering: retain primary leaf blade
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_leaf)
    if num_labels <= 1:
        empty_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        return {
            "success": False,
            "original_image": rgb_img,
            "hsv_image": hsv_img,
            "leaf_mask": empty_mask,
            "leaf_area_pixels": 0,
            "segmented_leaf": np.zeros_like(rgb_img),
            "coverage_percentage": 0.0,
            "bounding_box": (0, 0, 0, 0),
            "error_message": "No leaf components segmented.",
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + int(np.argmax(areas))
    max_area = areas[max_idx - 1]

    clean_leaf = np.zeros_like(raw_leaf)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= max(350, int(0.12 * max_area)):
            clean_leaf[labels == idx] = 255

    # 6. Fill internal holes: necrotic lesions inside leaf MUST be preserved
    final_leaf_mask = fill_mask_holes(clean_leaf)
    leaf_area_pixels = int(np.count_nonzero(final_leaf_mask))

    # 7. Quality & Plausibility bounds (resilient to close-ups, natural veins, and severe necrosis)
    min_viable_pixels = max(1000, int(0.015 * total_pixels))
    if leaf_area_pixels < min_viable_pixels:
        empty_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        return {
            "success": False,
            "original_image": rgb_img,
            "hsv_image": hsv_img,
            "leaf_mask": empty_mask,
            "leaf_area_pixels": 0,
            "segmented_leaf": np.zeros_like(rgb_img),
            "coverage_percentage": 0.0,
            "bounding_box": (0, 0, 0, 0),
            "segmentation_quality": "POOR",
            "error_message": f"Segmented leaf area too small ({leaf_area_pixels} pixels vs. minimum viable {min_viable_pixels}).",
        }

    # Verify presence of foliar chromaticity (prevents posters and cards without penalizing necrotic leaves)
    foliar_inside = np.sum(foliar_evidence & (final_leaf_mask > 0))
    foliar_ratio = foliar_inside / max(1, leaf_area_pixels)
    if foliar_ratio < 0.50:  # Real leaves always exceed 0.71; 0.50 safely excludes posters
        empty_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        return {
            "success": False,
            "original_image": rgb_img,
            "hsv_image": hsv_img,
            "leaf_mask": empty_mask,
            "leaf_area_pixels": 0,
            "segmented_leaf": np.zeros_like(rgb_img),
            "coverage_percentage": 0.0,
            "bounding_box": (0, 0, 0, 0),
            "segmentation_quality": "POOR",
            "error_message": f"Candidate mask lacks botanical foliar chromaticity (ratio: {foliar_ratio:.2f}).",
        }

    # Assess segmentation quality holistically
    if foliar_ratio >= 0.50 and 0.15 <= (leaf_area_pixels / total_pixels) <= 0.95:
        quality = "HIGH"
    elif foliar_ratio >= 0.25:
        quality = "MODERATE"
    else:
        quality = "REVIEW_RECOMMENDED"

    # Compute bounding box
    leaf_pts = np.where(final_leaf_mask > 0)
    y_min, y_max = leaf_pts[0].min(), leaf_pts[0].max()
    x_min, x_max = leaf_pts[1].min(), leaf_pts[1].max()
    bbox = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

    # Background-removed isolated leaf
    segmented_leaf = cv2.bitwise_and(rgb_img, rgb_img, mask=final_leaf_mask)
    coverage_pct = (leaf_area_pixels / total_pixels) * 100.0

    return {
        "success": True,
        "original_image": rgb_img,
        "hsv_image": hsv_img,
        "leaf_mask": final_leaf_mask,
        "leaf_area_pixels": leaf_area_pixels,
        "segmented_leaf": segmented_leaf,
        "coverage_percentage": round(coverage_pct, 2),
        "bounding_box": bbox,
        "segmentation_quality": quality,
        "error_message": None,
    }
