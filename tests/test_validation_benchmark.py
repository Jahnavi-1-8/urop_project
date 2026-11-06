"""Validation Benchmark: Tests Genuine Apple Leaves and Non-Leaf Images Separately.

Reports:
1. False Rejection Rate (FRR) on genuine apple leaves across all 4 disease/healthy classes.
2. False Acceptance Rate (FAR) on non-leaf images: posters, documents, faces, cars, greeting cards, blanks.
"""

import os
import sys
import glob
import cv2
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input_validation import AppleLeafValidator


def create_synthetic_non_leaves():
    """Create a suite of realistic non-leaf images (posters, docs, cards, random objects)."""
    samples = []
    
    # 1. Blank white
    blank_white = np.full((400, 400, 3), 255, dtype=np.uint8)
    samples.append(("Blank White Canvas", blank_white))
    
    # 2. Blank black
    blank_black = np.zeros((400, 400, 3), dtype=np.uint8)
    samples.append(("Blank Black Canvas", blank_black))
    
    # 3. Document / Text table
    doc = np.full((500, 500, 3), 250, dtype=np.uint8)
    for y in range(50, 450, 30):
        cv2.line(doc, (40, y), (460, y), (50, 50, 50), 1)
        cv2.putText(doc, f"Financial Report Line Item {y}", (50, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    for x in range(40, 470, 70):
        cv2.line(doc, (x, 50), (x, 440), (80, 80, 80), 1)
    samples.append(("Financial Spreadsheet Document", doc))
    
    # 4. Greeting Card / Poster
    poster = np.full((500, 500, 3), 240, dtype=np.uint8)
    cv2.rectangle(poster, (20, 20), (480, 480), (180, 30, 30), 15)
    cv2.rectangle(poster, (35, 35), (465, 465), (220, 180, 40), 4)
    cv2.putText(poster, "SEASONS GREETINGS", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 20, 20), 3)
    cv2.putText(poster, "AND HAPPY NEW YEAR", (65, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 150), 2)
    samples.append(("Seasons Greetings Poster", poster))
    
    # 5. Blue geometric object (car/appliance proxy)
    blue_obj = np.full((400, 400, 3), 200, dtype=np.uint8)
    cv2.circle(blue_obj, (200, 200), 140, (230, 80, 30), -1)
    cv2.rectangle(blue_obj, (120, 160), (280, 240), (255, 255, 255), -1)
    samples.append(("Synthetic Geometric / Car Proxy", blue_obj))
    
    # 6. Blurry image
    blurry = cv2.GaussianBlur(doc, (51, 51), 0)
    samples.append(("Heavily Blurry Image", blurry))
    
    return samples


def main():
    validator = AppleLeafValidator()
    
    print("\n" + "=" * 80)
    print("           APPLE LEAF VALIDATION BENCHMARK & RELIABILITY AUDIT")
    print("=" * 80)
    
    # 1. Test Genuine Apple Leaves
    classes = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
    genuine_results = {}
    total_genuine = 0
    false_rejections = 0
    
    for cls in classes:
        folder = os.path.join(project_root, "data", "raw", cls)
        files = glob.glob(os.path.join(folder, "*.JPG"))
        # Test 25 leaves per class (100 total genuine leaves)
        tested = files[:25]
        cls_fr = 0
        for f in tested:
            total_genuine += 1
            res = validator.validate(f)
            if not res["is_valid"]:
                cls_fr += 1
                false_rejections += 1
        genuine_results[cls] = (len(tested), cls_fr)
        
    print("\n1. GENUINE APPLE LEAVES TEST (100 Samples Across 4 Classes):")
    print("-" * 80)
    for cls, (cnt, fr) in genuine_results.items():
        acc = ((cnt - fr) / cnt) * 100.0
        print(f"  {cls:28}: {cnt - fr}/{cnt} Accepted ({acc:.1f}%), False Rejections: {fr}")
    
    frr = (false_rejections / total_genuine) * 100.0
    print(f"\n  TOTAL GENUINE TESTED   : {total_genuine}")
    print(f"  FALSE REJECTIONS (FR)  : {false_rejections}")
    print(f"  FALSE REJECTION RATE   : {frr:.2f}%")
    
    # 2. Test Non-Leaf Images
    non_leaves = create_synthetic_non_leaves()
    false_acceptances = 0
    print("\n2. NON-LEAF & UNRELATED IMAGES TEST:")
    print("-" * 80)
    for name, img in non_leaves:
        res = validator.validate(img)
        status = res["status"]
        accepted = res["is_valid"]
        if accepted:
            false_acceptances += 1
        verdict = "REJECTED (Correct)" if not accepted else "ACCEPTED (ERROR)"
        reason = res.get("reason", "N/A")
        print(f"  {name:32}: {verdict:18} | Status: {status:8} | Reason: {reason}")
        
    far = (false_acceptances / len(non_leaves)) * 100.0
    print(f"\n  TOTAL NON-LEAF TESTED  : {len(non_leaves)}")
    print(f"  FALSE ACCEPTANCES (FA) : {false_acceptances}")
    print(f"  FALSE ACCEPTANCE RATE  : {far:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

