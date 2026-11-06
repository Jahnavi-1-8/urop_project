"""Comprehensive Test Suite for Input Validation, 3 Specialist Agents, and LangGraph Orchestrator.
"""

import os
import sys
import glob
import unittest
import numpy as np
import cv2
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input_validation import validate_apple_leaf_input, AppleLeafValidator
from agents.disease_expert import DiseaseExpertAgent
from agents.risk_severity import RiskSeverityAgent
from agents.recommendation import RecommendationAgent
from agents.orchestrator import AgenticDecisionOrchestrator


class TestInputValidation(unittest.TestCase):
    """Test Suite for Apple Leaf Input Validation Stage."""

    def setUp(self):
        self.validator = AppleLeafValidator()
        self.scab_samples = glob.glob(os.path.join(project_root, "data", "raw", "Apple___Apple_scab", "*.JPG"))
        self.rot_samples = glob.glob(os.path.join(project_root, "data", "raw", "Apple___Black_rot", "*.JPG"))
        self.rust_samples = glob.glob(os.path.join(project_root, "data", "raw", "Apple___Cedar_apple_rust", "*.JPG"))
        self.healthy_samples = glob.glob(os.path.join(project_root, "data", "raw", "Apple___healthy", "*.JPG"))

    def test_genuine_apple_leaves_with_prominent_veins(self):
        """Verify genuine apple leaves with prominent veins are NOT rejected as documents."""
        # Test across 15 real apple leaves known to have strong venation
        for sample_path in self.scab_samples[:10] + self.healthy_samples[:5]:
            res = self.validator.validate(sample_path)
            self.assertTrue(
                res["is_valid"],
                f"False rejection on real apple leaf {os.path.basename(sample_path)}: {res.get('reason')}"
            )
            self.assertIn(res["status"], ["APPROVED", "REVIEW_RECOMMENDED"])

    def test_genuine_apple_leaves_plain_and_complex_backgrounds(self):
        """Verify leaves with plain or natural orchard backgrounds are approved."""
        samples = self.rot_samples[:5] + self.rust_samples[:5]
        for sample_path in samples:
            res = self.validator.validate(sample_path)
            self.assertTrue(res["is_valid"], f"Failed on sample {sample_path}: {res['reason']}")

    def test_leaf_with_central_stem_and_branching_veins(self):
        """Verify leaf with prominent central stem & branching veins passes validation."""
        sample_path = self.healthy_samples[0]
        res = self.validator.validate(sample_path)
        self.assertTrue(res["is_valid"])
        self.assertGreater(res["details"]["foliar_ratio"], 0.10)

    def test_reject_posters_and_greeting_cards(self):
        """Verify posters / greeting cards with text and borders are rejected."""
        poster = np.full((400, 400, 3), 240, dtype=np.uint8)
        # Add decorative border
        cv2.rectangle(poster, (20, 20), (380, 380), (180, 40, 40), 6)
        # Add greeting text lines
        for y in range(80, 320, 40):
            cv2.putText(poster, "SPECIAL GREETINGS & POSTER", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        res = self.validator.validate(poster)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["status"], "REJECTED")

    def test_reject_screenshots_and_documents(self):
        """Verify text document / table grid screenshots are rejected."""
        doc = np.full((400, 400, 3), 255, dtype=np.uint8)
        # Table grid
        for y in range(40, 360, 25):
            cv2.line(doc, (30, y), (370, y), (0, 0, 0), 2)
        for x in range(30, 380, 50):
            cv2.line(doc, (x, 40), (x, 340), (0, 0, 0), 2)
        res = self.validator.validate(doc)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["status"], "REJECTED")

    def test_reject_blurry_and_corrupted(self):
        """Verify blurry or corrupted images are rejected."""
        # 1. Corrupted path
        res_corr = self.validator.validate("corrupted_file.jpg")
        self.assertFalse(res_corr["is_valid"])
        self.assertEqual(res_corr["status"], "REJECTED")

        # 2. Heavily blurred uniform canvas
        blurred = np.full((300, 300, 3), 128, dtype=np.uint8)
        res_blur = self.validator.validate(blurred)
        self.assertFalse(res_blur["is_valid"])
        self.assertEqual(res_blur["status"], "REJECTED")

    def test_reject_unrelated_synthetic_objects(self):
        """Verify non-leaf objects (cars/furniture/synthetic colors) are rejected."""
        # Pure blue synthetic object
        blue_obj = np.zeros((300, 300, 3), dtype=np.uint8)
        blue_obj[:, :] = (20, 50, 220) # Blue in RGB
        noise = np.random.randint(0, 30, (300, 300, 3), dtype=np.uint8)
        blue_obj = cv2.add(blue_obj, noise)
        res = self.validator.validate(blue_obj)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["status"], "REJECTED")

    def test_face_detection_does_not_reject_leaves(self):
        """Verify spurious face detections on leaves are filtered by foliar/skin check."""
        sample_path = self.scab_samples[0]
        img = cv2.imread(sample_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        foliar_mask, _, _ = self.validator._extract_foliar_pigmentation(img_rgb)
        # Even if a candidate is placed over a leaf, it must be ignored
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        confirmed, count, _ = self.validator._verify_human_face(img_rgb, gray, foliar_mask)
        self.assertFalse(confirmed)
        self.assertEqual(count, 0)


class TestAgenticSystem(unittest.TestCase):
    """Test Suite for 3 Specialist Agents and LangGraph Coordinator."""

    def test_agent_1_disease_expert(self):
        agent = DiseaseExpertAgent()
        res = agent.run(disease_class="Apple Scab", confidence=95.2, model_used="Random Forest")
        self.assertEqual(res["agent"], "Disease Expert Agent")
        self.assertEqual(res["predicted_disease"], "Apple Scab")
        self.assertIn("Venturia", res["causal_agent"])
        self.assertIn("spots", res["primary_symptoms"].lower())

    def test_agent_2_risk_severity_with_metrics(self):
        agent = RiskSeverityAgent()
        sev_data = {
            "severity_percentage": 22.5,
            "leaf_area_pixels": 150000,
            "lesion_area_pixels": 33750,
            "severity_level": "Moderate",
        }
        res = agent.run(predicted_disease="Apple Scab", severity_data=sev_data)
        self.assertEqual(res["agent"], "Risk and Severity Agent")
        self.assertEqual(res["severity_level"], "Moderate")
        self.assertEqual(res["affected_area_percent"], 22.5)
        self.assertEqual(res["measurement_status"], "VERIFIED")

    def test_agent_2_risk_severity_missing_metrics(self):
        """Verify Agent 2 reports uncertainty and avoids fabricating missing metrics."""
        agent = RiskSeverityAgent()
        res = agent.run(predicted_disease="Cedar Apple Rust", severity_data=None)
        self.assertEqual(res["severity_level"], "Undetermined")
        self.assertIsNone(res["affected_area_percent"])
        self.assertEqual(res["measurement_status"], "UNAVAILABLE / UNCERTAIN")
        self.assertIn("uncertainty_notes", res)

    def test_agent_3_recommendation(self):
        agent = RecommendationAgent()
        expert_data = {"predicted_disease": "Black Rot"}
        risk_data = {"severity_level": "Severe", "affected_area_percent": 45.0}
        res = agent.run(expert_data=expert_data, risk_data=risk_data)
        self.assertEqual(res["agent"], "Recommendation Agent")
        self.assertGreater(len(res["cultural_management"]), 0)
        self.assertGreater(len(res["monitoring_protocol"]), 0)
        self.assertIn("pesticide", res["chemical_safety_disclaimer"].lower())

    def test_langgraph_orchestrator_coordination(self):
        """Verify LangGraph coordinates exactly the 3 agents into unified structured JSON."""
        orchestrator = AgenticDecisionOrchestrator()
        sev_data = {
            "severity_percentage": 12.0,
            "leaf_area_pixels": 110000,
            "lesion_area_pixels": 13200,
            "severity_level": "Mild",
        }
        result = orchestrator.run(
            predicted_disease="Cedar Apple Rust",
            confidence=92.8,
            model_used="SVM",
            severity_data=sev_data,
        )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["orchestrator"], "LangGraph Coordinator")
        agents = result["specialist_agent_outputs"]
        self.assertIn("agent_1_disease_expert", agents)
        self.assertIn("agent_2_risk_severity", agents)
        self.assertIn("agent_3_recommendation", agents)
        self.assertEqual(len(agents), 3) # Exactly 3 agents
        self.assertEqual(result["executive_summary"]["condition"], "Cedar Apple Rust")


if __name__ == "__main__":
    unittest.main()
