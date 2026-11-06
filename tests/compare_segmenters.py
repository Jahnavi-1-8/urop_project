"""Benchmark Comparison: Baseline HSV vs Enhanced GrabCut + L*a*b* Leaf and Lesion Segmentation.

Compares:
1. Leaf blade isolation quality and background removal.
2. Foliar disease lesion segmentation quality.
3. Strict subset enforcement (lesion_mask <= leaf_mask).
4. Greeting poster / non-leaf rejection and artifact suppression.
"""

import os
import sys
import glob
import cv2
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import segment_leaf as legacy_segment_leaf, rgb_to_hsv, resize_image, load_image, DEFAULT_IMAGE_SIZE
from severity.leaf_segmentation import segment_complete_leaf
from severity.lesion_segmentation import segment_disease_lesions
from severity.severity_calculator import calculate_severity


def run_legacy_pipeline(image_path: str, disease_hint: str):
    """Run baseline legacy HSV dual-threshold segmentation."""
    img = load_image(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = resize_image(img_rgb, DEFAULT_IMAGE_SIZE)
    hsv = rgb_to_hsv(img_resized)
    seg_leaf, leaf_mask = legacy_segment_leaf(img_resized, hsv)
    
    leaf_px = int(np.count_nonzero(leaf_mask))
    total_px = leaf_mask.size
    leaf_pct = (leaf_px / total_px) * 100.0
    
    # Legacy lesion detection (basic HSV thresholds)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    if disease_hint == "Healthy":
        lesion_px = 0
        lesion_pct = 0.0
    elif disease_hint == "Cedar Apple Rust":
        mask_les = ((h >= 8) & (h <= 28) & (s >= 55) & (v >= 50) & (leaf_mask > 0))
        lesion_px = int(np.count_nonzero(mask_les))
        lesion_pct = (lesion_px / max(1, leaf_px)) * 100.0
    elif disease_hint == "Black Rot":
        mask_les = (((v <= 80) | ((h >= 5) & (h <= 28) & (s >= 35))) & (leaf_mask > 0))
        lesion_px = int(np.count_nonzero(mask_les))
        lesion_pct = (lesion_px / max(1, leaf_px)) * 100.0
    else: # Apple Scab
        mask_les = (((v <= 75) | ((h >= 12) & (h <= 42) & (s >= 30))) & (leaf_mask > 0))
        lesion_px = int(np.count_nonzero(mask_les))
        lesion_pct = (lesion_px / max(1, leaf_px)) * 100.0
        
    return {
        "leaf_px": leaf_px,
        "leaf_pct": leaf_pct,
        "lesion_px": lesion_px,
        "lesion_pct": lesion_pct,
    }


def run_enhanced_pipeline(image_path: str, disease_hint: str):
    """Run enhanced GrabCut + CIE L*a*b* + ExG segmentation."""
    leaf_res = segment_complete_leaf(image_path)
    if not leaf_res["success"]:
        return {
            "success": False,
            "leaf_px": 0,
            "leaf_pct": 0.0,
            "lesion_px": 0,
            "lesion_pct": 0.0,
            "error": leaf_res.get("error_message", "Segmentation failed"),
        }
        
    leaf_mask = leaf_res["leaf_mask"]
    leaf_px = leaf_res["leaf_area_pixels"]
    leaf_pct = leaf_res["coverage_percentage"]
    
    lesion_res = segment_disease_lesions(
        rgb_image=leaf_res["original_image"],
        leaf_mask=leaf_mask,
        disease_name=disease_hint,
        hsv_image=leaf_res["hsv_image"],
    )
    lesion_mask = lesion_res["lesion_mask"]
    lesion_px = lesion_res["lesion_area_pixels"]
    
    # Strict subset check: lesion_mask <= leaf_mask
    is_subset = np.all((lesion_mask > 0) <= (leaf_mask > 0))
    
    sev_res = calculate_severity(leaf_mask, lesion_mask, disease_hint)
    lesion_pct = sev_res["severity_percentage"]
    
    return {
        "success": True,
        "leaf_px": leaf_px,
        "leaf_pct": leaf_pct,
        "lesion_px": lesion_px,
        "lesion_pct": lesion_pct,
        "is_subset": is_subset,
        "severity_level": sev_res["severity_level"],
    }


def main():
    test_samples = [
        ("Apple___Apple_scab", "Apple Scab"),
        ("Apple___Black_rot", "Black Rot"),
        ("Apple___Cedar_apple_rust", "Cedar Apple Rust"),
        ("Apple___healthy", "Healthy"),
    ]
    
    print("\n" + "=" * 95)
    print("      BENCHMARK SEGMENTATION COMPARISON: BASELINE (HSV) vs ENHANCED (GRABCUT + LAB)")
    print("=" * 95)
    print(f"{'CATEGORY / FILE':28} | {'BASE LEAF%':11} | {'ENH LEAF%':10} | {'BASE LESION%':13} | {'ENH LESION%':12} | {'SUBSET':6}")
    print("-" * 95)
    
    for cat_dir, hint in test_samples:
        folder = os.path.join(project_root, "data", "raw", cat_dir)
        files = glob.glob(os.path.join(folder, "*.JPG"))
        if not files:
            continue
        # Test 2 representative files per category
        for fpath in files[:2]:
            fname = os.path.basename(fpath)
            leg = run_legacy_pipeline(fpath, hint)
            enh = run_enhanced_pipeline(fpath, hint)
            
            label = f"{hint[:12]}: {fname[:14]}"
            subset_str = "YES" if enh.get("is_subset", False) else "NO"
            print(
                f"{label:28} | {leg['leaf_pct']:10.1f}% | {enh['leaf_pct']:9.1f}% | "
                f"{leg['lesion_pct']:12.2f}% | {enh['lesion_pct']:11.2f}% | {subset_str:6}"
            )
            
    print("-" * 95)
    print("NON-LEAF / GREETING POSTER ROBUSTNESS TEST:")
    print("-" * 95)
    # Create greeting poster
    poster = np.full((500, 500, 3), 235, dtype=np.uint8)
    cv2.rectangle(poster, (25, 25), (475, 475), (0, 190, 60), 20)
    cv2.putText(poster, "SPECIAL GREETINGS", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 160), 3)
    tmp_poster_path = os.path.join(project_root, "scratch_poster.jpg")
    cv2.imwrite(tmp_poster_path, poster)
    
    leg_post = run_legacy_pipeline(tmp_poster_path, "Apple Scab")
    enh_post = run_enhanced_pipeline(tmp_poster_path, "Apple Scab")
    if os.path.exists(tmp_poster_path):
        os.remove(tmp_poster_path)
        
    print(f"{'Greeting Poster':28} | {leg_post['leaf_pct']:10.1f}% | {enh_post['leaf_pct']:9.1f}% | {leg_post['lesion_pct']:12.2f}% | {enh_post['lesion_pct']:11.2f}% | "
          f"{'REJECTED' if not enh_post['success'] else 'ACCEPTED'}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()

