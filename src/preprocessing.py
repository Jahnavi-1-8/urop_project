"""Image Preprocessing Module for Apple Leaf Disease Classification.

This module provides standard image loading, resizing, color space conversions,
and color segmentation designed to isolate leaf tissue (healthy green and diseased brown/rust)
from background artifacts according to the reference methodology.
"""

from typing import Optional, Tuple, Dict, Any, Union
import os
import cv2
import numpy as np


# Default image dimensions (consistent with reference methodology)
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (500, 500)

# HSV Color range thresholds for segmenting plant tissue
# Healthy foliage: green spectrum (minimum saturation 20 to reject neutral grey/white backgrounds)
LOWER_GREEN: np.ndarray = np.array([22, 20, 20], dtype=np.uint8)
UPPER_GREEN: np.ndarray = np.array([100, 255, 255], dtype=np.uint8)

# Diseased lesions: brown / rust spectrum
LOWER_BROWN: np.ndarray = np.array([8, 20, 15], dtype=np.uint8)
UPPER_BROWN: np.ndarray = np.array([32, 255, 255], dtype=np.uint8)

# Dark necrotic spots (black rot / severe scab lesions)
LOWER_DARK: np.ndarray = np.array([0, 15, 10], dtype=np.uint8)
UPPER_DARK: np.ndarray = np.array([35, 255, 90], dtype=np.uint8)


def load_image(image_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
    """Load an image from a file path or validate an existing numpy image array.

    Args:
        image_input: File path (str) or loaded numpy image array (BGR or RGB).

    Returns:
        np.ndarray: Loaded image in BGR format if from file, or existing array,
                    or None if loading fails.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return None
        # cv2.imread loads images in BGR format
        image = cv2.imread(image_input)
        if image is None or image.size == 0:
            return None
        return image
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return None
        return image_input.copy()
    return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize an image to target dimensions using bilinear interpolation.

    Args:
        image: Input image array.
        target_size: Desired (width, height) tuple.

    Returns:
        np.ndarray: Resized image array.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert an image from BGR color space to RGB color space."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an image from RGB color space to HSV color space."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


def segment_leaf(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    lower_green: np.ndarray = LOWER_GREEN,
    upper_green: np.ndarray = UPPER_GREEN,
    lower_brown: np.ndarray = LOWER_BROWN,
    upper_brown: np.ndarray = UPPER_BROWN,
    lower_dark: np.ndarray = LOWER_DARK,
    upper_dark: np.ndarray = UPPER_DARK,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment leaf tissue (healthy green and diseased brown/rust) from background.

    Uses color-based thresholding in HSV color space to create:
    1. Healthy tissue mask (green hues)
    2. Disease lesion mask (brown/rust hues)
    3. Dark necrotic lesion mask (black rot / scab)
    The masks are merged via bitwise OR with morphological closing to preserve both
    symptomatic lesions and leaf blades while suppressing background artifacts.

    Args:
        rgb_image: RGB leaf image.
        hsv_image: HSV leaf image.
        lower_green: Lower HSV bound for healthy tissue.
        upper_green: Upper HSV bound for healthy tissue.
        lower_brown: Lower HSV bound for diseased lesions.
        upper_brown: Upper HSV bound for diseased lesions.
        lower_dark: Lower HSV bound for necrotic dark spots.
        upper_dark: Upper HSV bound for necrotic dark spots.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - segmented_rgb: RGB image with background masked to black (0).
            - final_mask: Binary mask (255 where leaf tissue is present).
    """
    # Create healthy tissue mask
    healthy_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Create diseased lesion mask
    disease_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Create dark necrotic spot mask
    dark_mask = cv2.inRange(hsv_image, lower_dark, upper_dark)

    # Combine masks (bitwise OR to prevent byte overflow)
    final_mask = cv2.bitwise_or(healthy_mask, cv2.bitwise_or(disease_mask, dark_mask))

    # Morphological noise cleanup and lesion hole filling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Guard against empty mask
    if final_mask.sum() < 500:
        final_mask = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8) * 255

    # Apply mask to RGB image
    segmented_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=final_mask)

    return segmented_rgb, final_mask


def preprocess_image(
    image_input: Union[str, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    is_rgb: bool = False,
) -> Optional[Dict[str, Any]]:
    """Complete image preprocessing pipeline for apple leaf disease analysis.

    Steps:
    1. Load image (gracefully handle missing/corrupt inputs).
    2. Standardize dimensions via resizing.
    3. Convert to RGB (if input is BGR from cv2.imread).
    4. Convert to HSV for robust color thresholding.
    5. Perform HSV segmentation to isolate leaf tissue.
    6. Convert segmented leaf to grayscale for texture/shape descriptor extraction.

    Args:
        image_input: File path or numpy image array.
        target_size: Output resolution (width, height).
        is_rgb: Flag indicating whether input array is already in RGB format.

    Returns:
        Optional[Dict[str, Any]]: Preprocessed artifacts dictionary or None if failed:
            - 'original_rgb': Standardized RGB input image
            - 'hsv': HSV converted image
            - 'mask': Binary segmentation mask
            - 'segmented_rgb': Segmented RGB leaf
            - 'gray_segmented': Grayscale version of segmented leaf
    """
    raw_img = load_image(image_input)
    if raw_img is None:
        return None

    # Handle grayscale or 4-channel alpha images
    if len(raw_img.shape) == 2:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    elif raw_img.shape[2] == 4:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)

    # Resize to standardized dimensions
    resized = resize_image(raw_img, target_size)

    # Convert to RGB if needed
    if is_rgb:
        rgb_img = resized
    else:
        rgb_img = bgr_to_rgb(resized)

    # Convert to HSV
    hsv_img = rgb_to_hsv(rgb_img)

    # Color segmentation
    segmented_rgb, final_mask = segment_leaf(rgb_img, hsv_img)

    # Grayscale conversion of segmented leaf
    gray_segmented = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2GRAY)

    return {
        "original_rgb": rgb_img,
        "hsv": hsv_img,
        "mask": final_mask,
        "segmented_rgb": segmented_rgb,
        "gray_segmented": gray_segmented,
    }


if __name__ == "__main__":
    import sys

    sample_dir = os.path.join("data", "raw", "Apple___Apple_scab")
    if os.path.exists(sample_dir):
        sample_files = os.listdir(sample_dir)
        if sample_files:
            test_path = os.path.join(sample_dir, sample_files[0])
            result = preprocess_image(test_path)
            if result:
                print(f"[SUCCESS] Preprocessed: {test_path}")
                print(f" - Segmented RGB shape: {result['segmented_rgb'].shape}")
                print(f" - Grayscale shape: {result['gray_segmented'].shape}")
                print(f" - Mask coverage: {(result['mask'] > 0).mean() * 100:.2f}%")
            else:
                print("[ERROR] Preprocessing failed.")
    else:
        print("[INFO] Run from project root with data/raw populated.")
