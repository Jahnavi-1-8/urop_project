"""Feature Extraction Module for Apple Leaf Disease Classification.

Extracts three complementary global feature descriptors from segmented leaf images:
1. Hu Moments (Shape descriptors - 7 dimensions)
2. Haralick Texture Features (GLCM statistical descriptors - 13 dimensions)
3. HSV 3D Color Histogram (Color distribution - 512 dimensions with 8 bins)

Total concatenated feature vector dimension: dynamically computed (532 dimensions).
"""

from typing import Optional, Tuple, Dict, Any, List, Union
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is in sys.path when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import mahotas
import numpy as np
from tqdm import tqdm

from src.preprocessing import preprocess_image, DEFAULT_IMAGE_SIZE


# Default histogram bins per channel (H, S, V)
DEFAULT_HIST_BINS: int = 8

# Target class directory mapping
CLASS_MAP: Dict[str, str] = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy",
}


def extract_hu_moments(gray_image: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """Extract Hu Moments from a grayscale image to capture shape information.

    Calculates 7 orthogonal moment invariants that are invariant to translation,
    scale, and rotation, with log-transformation for numerical stability.

    Args:
        gray_image: Grayscale segmented leaf image.

    Returns:
        np.ndarray: 1D array of 7 log-transformed Hu Moments.
    """
    moments = cv2.moments(gray_image)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform for numerical stability across orders of magnitude
    hu_moments = -np.sign(hu) * np.log10(np.abs(hu) + 1e-15)
    return hu_moments


def extract_haralick_textures(
    gray_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Extract Haralick texture features from a Gray-Level Co-occurrence Matrix (GLCM).

    Computes 13 second-order statistical texture features across 4 spatial directions
    (0, 45, 90, 135 degrees) and averages them to achieve rotational invariance.
    Calculated on the leaf bounding box to avoid boundary artifacts.

    Args:
        gray_image: Grayscale leaf image.
        mask: Optional binary leaf mask.

    Returns:
        np.ndarray: 1D array of 13 averaged Haralick texture features.
    """
    if mask is None:
        mask = kwargs.get("mask", None)

    if mask is not None and mask.sum() > 0:
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) > 20:
            crop = gray_image[y_idx.min():y_idx.max()+1, x_idx.min():x_idx.max()+1]
            small_crop = cv2.resize(crop, (128, 128))
            return mahotas.features.haralick(small_crop).mean(axis=0)

    small_gray = cv2.resize(gray_image, (128, 128))
    haralick = mahotas.features.haralick(small_gray).mean(axis=0)
    return haralick


def extract_color_histogram(
    hsv_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bins: int = DEFAULT_HIST_BINS,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Extract a 3D HSV color histogram from the segmented leaf image.

    Quantizes the Hue, Saturation, and Value channels into `bins` intervals each.
    When a mask is provided, counts ONLY the segmented leaf pixels and normalizes
    by leaf area (L1 norm), ensuring true scale invariance against background area.

    Args:
        hsv_image: HSV leaf image (or segmented leaf).
        mask: Binary mask of leaf tissue.
        bins: Number of bins per color channel (default 8 -> 8x8x8 = 512 features).

    Returns:
        np.ndarray: Normalized and flattened 1D color histogram (bins^3 dimensions).
    """
    if mask is not None and mask.sum() > 0:
        hist = cv2.calcHist(
            [hsv_image],
            [0, 1, 2],
            mask,
            [bins, bins, bins],
            [0, 256, 0, 256, 0, 256],
        )
        total_pixels = hist.sum()
        if total_pixels > 0:
            hist = hist.astype(np.float64) / total_pixels
        else:
            hist = np.zeros(bins ** 3, dtype=np.float64)
    else:
        hist = cv2.calcHist(
            [hsv_image],
            [0, 1, 2],
            None,
            [bins, bins, bins],
            [0, 256, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
    return hist.flatten()


def extract_image_features(
    image_input: Union[str, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    bins: int = DEFAULT_HIST_BINS,
    is_rgb: bool = False,
) -> Optional[np.ndarray]:
    """Extract full concatenated feature vector from a single leaf image.

    Pipeline:
    1. Preprocessing & HSV Segmentation
    2. Hu Moments from grayscale segmented leaf (7 features)
    3. Haralick Texture from leaf bounding box (13 features)
    4. HSV Color Histogram from masked leaf tissue (512 features)
    5. Horizontal concatenation -> dynamically sized feature vector (532 dims)

    Args:
        image_input: Image filepath or numpy array.
        target_size: Target image dimensions.
        bins: Histogram bins per channel.
        is_rgb: Flag if numpy array is already RGB.

    Returns:
        Optional[np.ndarray]: Combined 1D feature vector or None if preprocessing fails.
    """
    prep = preprocess_image(image_input, target_size=target_size, is_rgb=is_rgb)
    if prep is None:
        return None

    gray = prep["gray_segmented"]
    hsv = prep["hsv"]
    mask = prep["mask"]

    fv_hu = extract_hu_moments(gray)
    fv_haralick = extract_haralick_textures(gray, mask=mask)
    fv_hist = extract_color_histogram(hsv, mask=mask, bins=bins)

    # Order matches reference methodology: [histogram, haralick, hu]
    combined_features = np.hstack([fv_hist, fv_haralick, fv_hu])
    return combined_features


def _process_single_image(args: Tuple[str, str, int]) -> Optional[Tuple[np.ndarray, str, str]]:
    """Worker function for multithreaded dataset processing."""
    file_path, class_name, bins = args
    fv = extract_image_features(file_path, bins=bins)
    if fv is not None and not np.isnan(fv).any():
        return fv, class_name, file_path
    return None


def extract_features_from_dataset(
    raw_data_dir: str = os.path.join("data", "raw"),
    output_path: str = os.path.join("data", "processed", "apple_leaf_features.npz"),
    bins: int = DEFAULT_HIST_BINS,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Batch-extract features from all dataset images across all 4 classes.

    Args:
        raw_data_dir: Path to directory containing class subdirectories.
        output_path: Destination .npz file path for processed features.
        bins: Histogram bins per channel.
        max_workers: Number of concurrent threads for I/O and processing.

    Returns:
        Dict[str, Any]: Extraction summary statistics.
    """
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    # Gather tasks
    tasks: List[Tuple[str, str, int]] = []
    class_dirs = sorted([d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))])

    print(f"\n========================================================")
    print(f" FEATURE EXTRACTION PIPELINE")
    print(f"========================================================")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Found classes: {class_dirs}")

    for c_dir in class_dirs:
        full_dir = os.path.join(raw_data_dir, c_dir)
        files = [
            os.path.join(full_dir, f)
            for f in os.listdir(full_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for f in files:
            tasks.append((f, c_dir, bins))

    print(f"Total valid images discovered: {len(tasks)}")

    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    paths_list: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(_process_single_image, tasks), total=len(tasks), desc="Extracting features"):
            if res is not None:
                fv, label, p = res
                features_list.append(fv)
                labels_list.append(label)
                paths_list.append(p)

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list)
    paths = np.array(paths_list)

    # Class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique_classes, [int(c) for c in counts]))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to compressed NumPy archive
    np.savez_compressed(
        output_path,
        features=X,
        labels=y,
        file_paths=paths,
        classes=unique_classes,
    )

    num_samples, num_features = X.shape

    summary = {
        "num_samples": int(num_samples),
        "num_features": int(num_features),
        "feature_vector_shape": list(X.shape),
        "class_distribution": class_distribution,
        "feature_breakdown": {
            "hsv_histogram_bins_per_channel": bins,
            "hsv_histogram_dims": bins ** 3,
            "haralick_texture_dims": 13,
            "hu_moments_dims": 7,
            "total_calculated_dimension": (bins ** 3) + 13 + 7,
        },
        "saved_to": output_path,
    }

    # Save metadata JSON
    meta_path = os.path.join(os.path.dirname(output_path), "dataset_summary.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[SUMMARY] Feature extraction complete:")
    print(f" - Number of samples: {num_samples}")
    print(f" - Number of features: {num_features} (Dynamically verified: {bins**3} + 13 + 7 = {(bins**3) + 13 + 7})")
    print(f" - Feature vector shape: {X.shape}")
    print(f" - Class distribution:")
    for cls, count in class_distribution.items():
        print(f"     * {cls} ({CLASS_MAP.get(cls, cls)}): {count} samples")
    print(f" - Saved dataset archive: {output_path}")
    print(f" - Saved metadata summary: {meta_path}")

    return summary


if __name__ == "__main__":
    extract_features_from_dataset()
