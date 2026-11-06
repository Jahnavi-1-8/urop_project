"""Integrated Disease Classification and Severity Estimation Pipeline.

Connects the existing Phase 1 classification model with Phase 2, 3, and 4:
1. Receives input apple leaf photograph (file path or numpy array).
2. Executes existing classification inference to obtain predicted disease and confidence.
3. Passes original leaf and predicted disease to Phase 2 (Complete Leaf Segmentation).
4. Executes Phase 3 (Disease-Specific Lesion Segmentation) within the segmented leaf mask.
5. Executes Phase 4 (Severity Quantification) to compute percentage and assign severity level.
6. Returns integrated classification and severity results accompanied by visual artifacts.
"""

from typing import Dict, Any, Optional, Union, Tuple
import os
import sys
import argparse
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict import AppleLeafPredictor, DEFAULT_MODELS_DIR
from severity.leaf_segmentation import segment_complete_leaf
from severity.lesion_segmentation import segment_disease_lesions
from severity.severity_calculator import calculate_severity
from severity.visualization import create_diagnostic_panel, generate_diagnostic_summary_dict


class IntegratedDiseaseSeverityPipeline:
    """Integrated engine executing classification, leaf extraction, lesion detection, and severity estimation."""

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        self.predictor = AppleLeafPredictor(models_dir=models_dir)

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        model_name: str = "random_forest",
        is_rgb: bool = False,
        image_name: Optional[str] = None,
        target_size: Tuple[int, int] = (500, 500),
        save_panel_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end classification and severity assessment.

        Args:
            image_input: File path or numpy image array (BGR or RGB).
            model_name: "random_forest" or "svm".
            is_rgb: Flag indicating if image array is already RGB.
            image_name: Optional image filename for diagnostic context.
            target_size: Dimension tuple (width, height) for image standardization.
            save_panel_path: Optional path to save visual 4-panel diagnostic figure.

        Returns:
            Dict[str, Any] matching the required integrated output schema:
            {
                "classification": {
                    "predicted_disease": str,
                    "confidence": float,
                    "model_used": str,
                    "probabilities": dict,
                    "explanation": str,
                },
                "severity": {
                    "disease": str,
                    "leaf_area_pixels": int,
                    "lesion_area_pixels": int,
                    "severity_percentage": float,
                    "severity_level": str,
                    "description": str,
                },
                "visualizations": dict of images/masks,
            }
        """
        # Step 1 & 2 & 3: Run existing classification model
        pred_res = self.predictor.predict(
            image_input,
            model_name=model_name,
            is_rgb=is_rgb,
            image_name=image_name,
        )

        predicted_disease = pred_res["predicted_class"]
        confidence = float(pred_res["confidence"])

        # Step 4 & 5: Phase 2 Complete Leaf Segmentation
        leaf_res = segment_complete_leaf(image_input, target_size=target_size, is_rgb=is_rgb)

        if not leaf_res["success"]:
            # Handle segmentation failure gracefully
            return {
                "classification": {
                    "predicted_disease": predicted_disease,
                    "confidence": confidence,
                    "confidence_percent": confidence * 100.0,
                    "model_used": pred_res["model_used"],
                    "probabilities": pred_res["probabilities"],
                    "explanation": pred_res["explanation"],
                },
                "severity": {
                    "disease": predicted_disease,
                    "leaf_area_pixels": 0,
                    "lesion_area_pixels": 0,
                    "severity_percentage": 0.0,
                    "severity_level": "Indeterminate",
                    "segmentation_quality": "FAILED",
                    "is_valid_segmentation": False,
                    "description": f"Leaf segmentation failed: {leaf_res['error_message']}",
                },
                "visualizations": None,
            }

        original_rgb = leaf_res["original_image"]
        leaf_mask = leaf_res["leaf_mask"]
        hsv_img = leaf_res["hsv_image"]

        # Step 6: Phase 3 Disease-Specific Lesion Segmentation
        lesion_res = segment_disease_lesions(
            rgb_image=original_rgb,
            leaf_mask=leaf_mask,
            disease_name=predicted_disease,
            hsv_image=hsv_img,
        )
        lesion_mask = lesion_res["lesion_mask"]

        # Step 7 & 8: Phase 4 Severity Calculation & Category Assignment
        sev_res = calculate_severity(
            leaf_mask=leaf_mask,
            lesion_mask=lesion_mask,
            disease_name=predicted_disease,
        )

        # Generate Visual Artifacts
        vis_dict = generate_diagnostic_summary_dict(
            original_rgb=original_rgb,
            leaf_mask=leaf_mask,
            lesion_mask=lesion_mask,
            disease_name=predicted_disease,
            severity_percentage=sev_res["severity_percentage"],
            severity_level=sev_res["severity_level"],
            leaf_area_pixels=sev_res["leaf_area_pixels"],
            lesion_area_pixels=sev_res["lesion_area_pixels"],
        )

        # Optional save of 4-panel diagnostic figure
        if save_panel_path:
            create_diagnostic_panel(
                original_rgb=original_rgb,
                leaf_mask=leaf_mask,
                lesion_mask=lesion_mask,
                disease_name=predicted_disease,
                severity_percentage=sev_res["severity_percentage"],
                severity_level=sev_res["severity_level"],
                leaf_area_pixels=sev_res["leaf_area_pixels"],
                lesion_area_pixels=sev_res["lesion_area_pixels"],
                confidence=confidence,
                save_path=save_panel_path,
            )

        # Build clean integrated output adhering strictly to requirements
        return {
            "classification": {
                "predicted_disease": predicted_disease,
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100.0, 2),
                "model_used": pred_res["model_used"],
                "probabilities": pred_res["probabilities"],
                "explanation": pred_res["explanation"],
            },
            "severity": {
                "disease": sev_res["disease"],
                "leaf_area_pixels": sev_res["leaf_area_pixels"],
                "lesion_area_pixels": sev_res["lesion_area_pixels"],
                "severity_percentage": sev_res["severity_percentage"],
                "severity_level": sev_res["severity_level"],
                "segmentation_quality": leaf_res.get("segmentation_quality", "HIGH"),
                "is_valid_segmentation": True,
                "description": sev_res["description"],
            },
            "visualizations": vis_dict,
        }


def predict_and_assess_severity(
    image_input: Union[str, np.ndarray],
    model_name: str = "random_forest",
    models_dir: str = DEFAULT_MODELS_DIR,
    save_panel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for complete end-to-end analysis."""
    pipeline = IntegratedDiseaseSeverityPipeline(models_dir=models_dir)
    return pipeline.process_image(image_input, model_name=model_name, save_panel_path=save_panel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apple Leaf Disease Classification & Severity Pipeline")
    parser.add_argument("--image", required=True, help="Path to input apple leaf image")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "rf", "svm"], help="ML model")
    parser.add_argument("--save-panel", default=None, help="Optional output path for 4-panel diagnostic figure")
    args = parser.parse_args()

    result = predict_and_assess_severity(args.image, model_name=args.model, save_panel_path=args.save_panel)

    cls_res = result["classification"]
    sev_res = result["severity"]

    print("\n========================================================")
    print(" INTEGRATED APPLE LEAF DIAGNOSTIC RESULT")
    print("========================================================")
    print(f"Classification Model: {cls_res['model_used']}")
    print(f"Predicted Disease:    {cls_res['predicted_disease']}")
    print(f"Confidence:           {cls_res['confidence_percent']:.2f}%")
    print("--------------------------------------------------------")
    print(f"Total Leaf Area:      {sev_res['leaf_area_pixels']:,} px")
    print(f"Infected Lesion Area: {sev_res['lesion_area_pixels']:,} px")
    print(f"Severity Percentage:  {sev_res['severity_percentage']:.2f}%")
    print(f"Severity Category:    {sev_res['severity_level'].upper()}")
    print(f"Clinical Guidance:    {sev_res['description']}")
    print("========================================================\n")

