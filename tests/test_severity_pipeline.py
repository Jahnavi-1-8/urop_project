"""Comprehensive Test Suite for Leaf Segmentation, Lesion Segmentation, and Severity Estimation (Phases 2-4).
"""

import os
import sys
import unittest
import numpy as np
import cv2

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from severity.leaf_segmentation import segment_complete_leaf, fill_mask_holes
from severity.lesion_segmentation import segment_disease_lesions, create_lesion_overlay, highlight_lesion_boundaries
from severity.severity_calculator import calculate_severity, classify_severity_level
from severity.pipeline import predict_and_assess_severity
from src.predict import predict_leaf, AppleLeafPredictor


class TestPhase2LeafSegmentation(unittest.TestCase):
    """Test suite for Phase 2 Complete Leaf Segmentation."""

    def setUp(self):
        self.sample_image = os.path.join(
            project_root, "data", "raw", "Apple___healthy", "0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG"
        )

    def test_segment_valid_leaf(self):
        res = segment_complete_leaf(self.sample_image)
        self.assertTrue(res["success"])
        self.assertGreater(res["leaf_area_pixels"], 10000)
        self.assertLessEqual(res["leaf_area_pixels"], 500 * 500)
        self.assertGreater(res["coverage_percentage"], 20.0)
        self.assertLess(res["coverage_percentage"], 95.0)
        self.assertEqual(res["leaf_mask"].shape, (500, 500))
        self.assertEqual(res["segmented_leaf"].shape, (500, 500, 3))

    def test_fill_mask_holes(self):
        # Create a hollow donut mask
        donut = np.zeros((100, 100), dtype=np.uint8)
        donut[20:80, 20:80] = 255
        donut[40:60, 40:60] = 0 # internal hole
        self.assertEqual(np.count_nonzero(donut), (60 * 60) - (20 * 20))

        filled = fill_mask_holes(donut)
        self.assertEqual(np.count_nonzero(filled), 60 * 60)
        self.assertTrue(np.all(filled[40:60, 40:60] == 255))

    def test_edge_cases(self):
        # Invalid file path
        res_invalid = segment_complete_leaf("non_existent_file.jpg")
        self.assertFalse(res_invalid["success"])
        self.assertEqual(res_invalid["leaf_area_pixels"], 0)

        # Blank/empty array
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        res_blank = segment_complete_leaf(blank)
        self.assertFalse(res_blank["success"])
        self.assertEqual(res_blank["leaf_area_pixels"], 0)


class TestPhase3LesionSegmentation(unittest.TestCase):
    """Test suite for Phase 3 Disease Lesion Segmentation."""

    def test_healthy_yields_zero_lesions(self):
        dummy_rgb = np.full((100, 100, 3), 100, dtype=np.uint8)
        leaf_mask = np.full((100, 100), 255, dtype=np.uint8)
        res = segment_disease_lesions(dummy_rgb, leaf_mask, "Healthy")
        self.assertEqual(res["lesion_area_pixels"], 0)
        self.assertEqual(np.count_nonzero(res["lesion_mask"]), 0)
        self.assertEqual(res["status"], "healthy_asymptomatic")

    def test_lesions_strictly_inside_leaf_mask(self):
        # Synthetic leaf with disease colors
        rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        leaf_mask = np.zeros((200, 200), dtype=np.uint8)

        # Define leaf region in center
        leaf_mask[50:150, 50:150] = 255
        # Healthy green everywhere in leaf
        rgb[50:150, 50:150] = [34, 139, 34]
        # Put simulated rust lesion in center of leaf
        rgb[80:120, 80:120] = [230, 115, 0] # orange rust
        # Put simulated rust OUTSIDE the leaf mask (in background)
        rgb[0:30, 0:30] = [230, 115, 0]

        res = segment_disease_lesions(rgb, leaf_mask, "Cedar Apple Rust")
        lesion_mask = res["lesion_mask"]

        # Ensure background rust at (0:30, 0:30) is NOT captured
        self.assertEqual(np.count_nonzero(lesion_mask[0:30, 0:30]), 0)
        # Ensure interior rust is captured
        self.assertGreater(np.count_nonzero(lesion_mask[80:120, 80:120]), 100)

        # Enforce strict subset: lesion_mask & leaf_mask == lesion_mask
        intersection = cv2.bitwise_and(lesion_mask, leaf_mask)
        self.assertTrue(np.array_equal(intersection, lesion_mask))


class TestPhase4SeverityCalculator(unittest.TestCase):
    """Test suite for Phase 4 Severity Calculation."""

    def test_severity_categories(self):
        leaf_mask = np.full((100, 100), 255, dtype=np.uint8) # 10,000 px

        # 1. Healthy
        res_h = calculate_severity(leaf_mask, np.zeros_like(leaf_mask), "Healthy")
        self.assertEqual(res_h["severity_percentage"], 0.0)
        self.assertEqual(res_h["severity_level"], "Healthy")

        # 2. Mild (< 15%)
        lesion_mild = np.zeros((100, 100), dtype=np.uint8)
        lesion_mild[:8, :10] = 255 # 80 / 10000 = 0.8%
        res_m = calculate_severity(leaf_mask, lesion_mild, "Apple Scab")
        self.assertEqual(res_m["severity_level"], "Mild")
        self.assertEqual(res_m["severity_percentage"], 0.8)

        # 3. Moderate (15% to 40%)
        lesion_mod = np.zeros((100, 100), dtype=np.uint8)
        lesion_mod[:25, :] = 255 # 2500 / 10000 = 25.0%
        res_mod = calculate_severity(leaf_mask, lesion_mod, "Black Rot")
        self.assertEqual(res_mod["severity_level"], "Moderate")
        self.assertEqual(res_mod["severity_percentage"], 25.0)

        # 4. Severe (> 40%)
        lesion_sev = np.zeros((100, 100), dtype=np.uint8)
        lesion_sev[:60, :] = 255 # 6000 / 10000 = 60.0%
        res_sev = calculate_severity(leaf_mask, lesion_sev, "Cedar Apple Rust")
        self.assertEqual(res_sev["severity_level"], "Severe")
        self.assertEqual(res_sev["severity_percentage"], 60.0)

    def test_division_by_zero_prevention(self):
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        res = calculate_severity(empty_mask, empty_mask, "Apple Scab")
        self.assertEqual(res["leaf_area_pixels"], 0)
        self.assertEqual(res["severity_percentage"], 0.0)
        self.assertEqual(res["severity_level"], "Indeterminate")
        self.assertEqual(res["status"], "empty_leaf_mask")


class TestIntegratedPipeline(unittest.TestCase):
    """Test suite for integrated classification and severity estimation pipeline."""

    def test_predict_and_assess_severity_schema(self):
        sample = os.path.join(
            project_root, "data", "raw", "Apple___Apple_scab", "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
        )
        res = predict_and_assess_severity(sample, model_name="random_forest")

        # Verify classification schema
        self.assertIn("classification", res)
        cls = res["classification"]
        self.assertIn("predicted_disease", cls)
        self.assertIn("confidence", cls)
        self.assertIn("model_used", cls)
        self.assertIn("probabilities", cls)

        # Verify severity schema
        self.assertIn("severity", res)
        sev = res["severity"]
        self.assertIn("disease", sev)
        self.assertIn("leaf_area_pixels", sev)
        self.assertIn("lesion_area_pixels", sev)
        self.assertIn("severity_percentage", sev)
        self.assertIn("severity_level", sev)
        self.assertIn("description", sev)

        # Verify visual artifacts schema
        self.assertIn("visualizations", res)
        vis = res["visualizations"]
        self.assertIn("original_image", vis)
        self.assertIn("leaf_mask", vis)
        self.assertIn("lesion_mask", vis)
        self.assertIn("composite_visualization", vis)

        # Verify mathematical consistency
        self.assertGreater(sev["leaf_area_pixels"], 0)
        self.assertGreaterEqual(sev["lesion_area_pixels"], 0)
        self.assertLessEqual(sev["lesion_area_pixels"], sev["leaf_area_pixels"])
        expected_pct = round((sev["lesion_area_pixels"] / sev["leaf_area_pixels"]) * 100.0, 2)
        self.assertEqual(sev["severity_percentage"], expected_pct)

    def test_original_predict_leaf_compatibility(self):
        sample = os.path.join(
            project_root, "data", "raw", "Apple___Apple_scab", "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
        )
        # Calling original function should still succeed without changes
        res = predict_leaf(sample, model_name="random_forest")
        self.assertEqual(res["predicted_class"], "Apple Scab")
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)


if __name__ == "__main__":
    unittest.main()

