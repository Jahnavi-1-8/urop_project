"""Comprehensive Test Suite for Evidence-Consistency Decision Logic (Cases A, B, C, D)
and Resilient Leaf Segmentation.
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.orchestrator import AgenticDecisionOrchestrator
from severity.leaf_segmentation import segment_complete_leaf


class TestEvidenceConsistencyDecisionLogic(unittest.TestCase):
    """Verify LangGraph Orchestrator evidence-integration decision logic."""

    def setUp(self):
        self.orchestrator = AgenticDecisionOrchestrator()

    def test_case_a_classifier_disease_zero_lesions(self):
        """Case A: Classifier predicts Apple Rust/Scab, but 0 lesions are detected on the segmented blade.
        Expected: Does NOT blindly diagnose disease; assesses as likely healthy on measured blade,
        explicitly highlighting the classifier disagreement for field monitoring.
        """
        severity_data = {
            "disease": "Cedar Apple Rust",
            "leaf_area_pixels": 85000,
            "lesion_area_pixels": 0,
            "severity_percentage": 0.0,
            "severity_level": "Mild", # Edge case: level string might say mild from prior step, but lesion px is 0
            "segmentation_quality": "HIGH",
            "is_valid_segmentation": True,
        }
        res = self.orchestrator.run(
            predicted_disease="Cedar Apple Rust",
            confidence=94.5,
            model_used="Random Forest",
            severity_data=severity_data,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["uncertainty_flag"])
        self.assertIn("Case A", res["decision_case"])
        exec_sum = res["executive_summary"]
        self.assertIn("Likely Healthy", exec_sum["confirmed_diagnosis"])
        self.assertIn("disagrees with Cedar Apple Rust", exec_sum["final_assessment"])
        self.assertEqual(res["leaf_analysis_metrics"]["lesion_area_pixels"], 0)

    def test_case_b_corroborated_disease(self):
        """Case B: Classifier predicts Apple Scab and lesions are detected.
        Expected: Corroborated diagnosis with verified severity.
        """
        severity_data = {
            "disease": "Apple Scab",
            "leaf_area_pixels": 80000,
            "lesion_area_pixels": 12000,
            "severity_percentage": 15.0,
            "severity_level": "Moderate",
            "segmentation_quality": "HIGH",
            "is_valid_segmentation": True,
        }
        res = self.orchestrator.run(
            predicted_disease="Apple Scab",
            confidence=96.0,
            model_used="Random Forest",
            severity_data=severity_data,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertFalse(res["uncertainty_flag"])
        self.assertIn("Case B", res["decision_case"])
        exec_sum = res["executive_summary"]
        self.assertIn("Apple Scab", exec_sum["condition"])
        self.assertEqual(exec_sum["severity_level"], "Moderate")
        self.assertEqual(exec_sum["affected_foliar_area"], 15.0)

    def test_case_c_segmentation_failed(self):
        """Case C: Classifier predicts disease, but leaf segmentation failed.
        Expected: Reports classifier prediction separately, marks severity as Indeterminate,
        requests clearer photograph, and NEVER assumes health.
        """
        severity_data = {
            "disease": "Black Rot",
            "leaf_area_pixels": 0,
            "lesion_area_pixels": 0,
            "severity_percentage": 0.0,
            "severity_level": "Indeterminate",
            "segmentation_quality": "FAILED",
            "is_valid_segmentation": False,
        }
        res = self.orchestrator.run(
            predicted_disease="Black Rot",
            confidence=91.0,
            model_used="SVM",
            severity_data=severity_data,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["uncertainty_flag"])
        self.assertIn("Case C", res["decision_case"])
        exec_sum = res["executive_summary"]
        self.assertEqual(exec_sum["severity_level"], "Indeterminate")
        self.assertIn("Segmentation Failed", exec_sum["confirmed_diagnosis"])
        self.assertIn("Black Rot", exec_sum["model_prediction"])

    def test_case_d_healthy_classifier_with_suspected_lesions(self):
        """Case D: Classifier predicts Healthy, but lesion analysis detects substantial lesions.
        Expected: Flags discrepancy, refuses to declare leaf healthy, requests physical inspection.
        """
        severity_data = {
            "disease": "Healthy",
            "leaf_area_pixels": 90000,
            "lesion_area_pixels": 4500,
            "severity_percentage": 5.0,
            "severity_level": "Mild",
            "segmentation_quality": "HIGH",
            "is_valid_segmentation": True,
        }
        res = self.orchestrator.run(
            predicted_disease="Healthy",
            confidence=88.0,
            model_used="SVM",
            severity_data=severity_data,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["uncertainty_flag"])
        self.assertIn("Case D", res["decision_case"])
        exec_sum = res["executive_summary"]
        self.assertIn("Review Recommended", exec_sum["confirmed_diagnosis"])
        self.assertIn("Physical Diagnostic Review", exec_sum["action_priority"])


class TestLeafSegmentationResilience(unittest.TestCase):
    """Verify leaf segmentation does not arbitrarily reject large or close-up leaves."""

    def test_close_up_leaf_not_rejected(self):
        """Close-up leaf filling 97% of the frame should NOT be discarded with empty mask."""
        synthetic_closeup = np.full((500, 500, 3), [34, 139, 34], dtype=np.uint8) # Dark green
        # Leave a tiny 3px border
        synthetic_closeup[:3, :] = [240, 240, 240]
        synthetic_closeup[-3:, :] = [240, 240, 240]
        synthetic_closeup[:, :3] = [240, 240, 240]
        synthetic_closeup[:, -3:] = [240, 240, 240]

        res = segment_complete_leaf(synthetic_closeup, is_rgb=True)
        self.assertTrue(res["success"])
        self.assertGreater(res["leaf_area_pixels"], 200000)
        self.assertGreater(res["coverage_percentage"], 90.0)
        self.assertIsNotNone(res["leaf_mask"])
        self.assertIn(res["segmentation_quality"], ["HIGH", "MODERATE"])


if __name__ == "__main__":
    unittest.main()
