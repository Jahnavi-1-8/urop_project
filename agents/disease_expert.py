"""Agent 1: Disease Expert Agent.

Role:
- Receives the existing model's predicted disease and confidence.
- Explains the disease biology, symptoms, causal pathogen, and characteristics.
- Strictly adheres to the classifier's prediction (never mutates or reclassifies).
- Evidence-based, phytopathological domain knowledge.
"""

from typing import Dict, Any, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Scientific pathology reference database for deterministic grounding
GROUNDED_DISEASE_DB = {
    "Apple Scab": {
        "pathogen": "Venturia inaequalis (Ascomycete fungus)",
        "symptoms": (
            "Dull olive-green to dark velvety circular spots on leaf upper surface, "
            "later turning brown, corky, or scabby. Severe infection leads to premature leaf drop."
        ),
        "favorable_conditions": "Prolonged wet foliage, high relative humidity (>85%), temperatures between 15°C and 24°C.",
        "biological_cycle": "Overwinters as pseudothecia in dead fallen apple leaves on the orchard floor; ascospores eject in spring.",
    },
    "Black Rot": {
        "pathogen": "Diplocarpa mali / Botryosphaeria obtusa (Fungus)",
        "symptoms": (
            "Distinct small circular 'frog-eye' leaf spots with light tan/brown centers and dark purple or reddish margins. "
            "Can cause cankers on twigs and black rot mummies on fruit."
        ),
        "favorable_conditions": "Warm (20°C–27°C), wet weather following rain periods; opportunistic entry through pruning wounds.",
        "biological_cycle": "Overwinters in dead wood, branch cankers, and mummified fruit hanging from previous seasons.",
    },
    "Cedar Apple Rust": {
        "pathogen": "Gymnosporangium juniperi-virginianae (Heteroecious rust fungus)",
        "symptoms": (
            "Bright yellow-orange or reddish-orange circular spots on upper leaf surfaces. "
            "Small dark pycnia appear at the center, followed by tube-like aecia structures on the lower surface."
        ),
        "favorable_conditions": "Spring rains and temperatures of 12°C–24°C coinciding with telial horn gelatinization on Eastern Red Cedar.",
        "biological_cycle": "Obligate heteroecious lifecycle alternating strictly between Juniperus virginiana (cedar/juniper) and Malus domestica (apple).",
    },
    "Healthy": {
        "pathogen": "None (Asymptomatic foliage)",
        "symptoms": "Uniform chlorophyll pigmentation, intact cuticular integrity, vigorous foliar lamina, no necrotic or chlorotic lesions.",
        "favorable_conditions": "Optimal agronomic conditions, balanced nutrition, adequate sunlight, and sound moisture management.",
        "biological_cycle": "Normal photosynthetic vegetative leaf development.",
    },
}


class DiseaseExpertAgent:
    """Specialist phytopathology agent providing evidence-based disease biology."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def run(
        self,
        disease_class: str,
        confidence: float,
        model_used: str = "Classifier",
        leaf_analysis: Optional[Dict[str, Any]] = None,
        validation_status: Optional[str] = "APPROVED",
    ) -> Dict[str, Any]:
        """Analyze disease biology and cross-examine evidence against quantitative leaf segmentation."""
        # Normalize class name
        normalized_name = "Healthy" if "Healthy" in disease_class else (
            "Apple Scab" if "Scab" in disease_class else (
                "Black Rot" if "Black" in disease_class else (
                    "Cedar Apple Rust" if "Rust" in disease_class else disease_class
                )
            )
        )

        db_entry = GROUNDED_DISEASE_DB.get(normalized_name, GROUNDED_DISEASE_DB["Healthy"])

        # Evaluate evidence consistency with quantitative leaf analysis
        leaf_info = leaf_analysis or {}
        seg_success = leaf_info.get("is_valid_segmentation", True)
        seg_quality = leaf_info.get("segmentation_quality", "HIGH")
        lesion_px = leaf_info.get("lesion_area_pixels", 0)
        affected_pct = leaf_info.get("severity_percentage", 0.0)

        if not seg_success or seg_quality == "FAILED":
            consistency = "INCONCLUSIVE_SEGMENTATION_FAILED"
            consistency_note = (
                f"Specimen classified as {normalized_name} ({confidence:.1f}%), but leaf blade segmentation failed. "
                "Physical blade measurements are unavailable to corroborate the prediction."
            )
        elif normalized_name != "Healthy" and lesion_px == 0:
            consistency = "DISCORDANT_ZERO_LESIONS"
            consistency_note = (
                f"Specimen classified as {normalized_name} ({confidence:.1f}%), but quantitative lesion segmentation "
                "detected 0 lesion pixels across the leaf surface. Foliage appears visually asymptomatic; "
                "disagreement between classifier and segmentation suggests possible classifier false positive or pre-symptomatic stage."
            )
        elif normalized_name == "Healthy" and lesion_px > 0 and affected_pct >= 1.0:
            consistency = "DISCORDANT_SUSPECTED_LESIONS"
            consistency_note = (
                f"Specimen classified as Healthy ({confidence:.1f}%), but image analysis detected {lesion_px:,} px "
                f"({affected_pct:.1f}% affected area) of symptomatic foliar anomalies. Further inspection recommended."
            )
        elif normalized_name != "Healthy" and lesion_px > 0:
            consistency = "CONSISTENT"
            consistency_note = (
                f"Specimen classified as {normalized_name} ({confidence:.1f}%) is corroborated by "
                f"{lesion_px:,} pixels ({affected_pct:.2f}% affected area) of visible foliar lesions."
            )
        else:
            consistency = "CONSISTENT_HEALTHY"
            consistency_note = (
                f"Specimen classified as Healthy ({confidence:.1f}%) with 0 detected lesions, "
                "consistent with healthy, intact foliar cuticle."
            )

        expert_output = {
            "agent": "Disease Expert Agent",
            "predicted_disease": normalized_name,
            "confidence_percent": round(confidence * 100.0, 2) if confidence <= 1.0 else round(confidence, 2),
            "model_source": model_used,
            "causal_agent": db_entry["pathogen"],
            "primary_symptoms": db_entry["symptoms"],
            "epidemiological_conditions": db_entry["favorable_conditions"],
            "disease_lifecycle": db_entry["biological_cycle"],
            "evidence_consistency": consistency,
            "consistency_analysis": consistency_note,
            "expert_summary": (
                f"The foliar specimen was predicted as {normalized_name} by {model_used} "
                f"({confidence:.1f}% confidence). Evidence Evaluation: {consistency_note}"
            ),
        }

        # If LLM is available, enrich with structured dynamic formulation while preserving factual boundaries
        if self.llm:
            try:
                system_prompt = (
                    "You are the phytopathology Disease Expert Agent for apple orchards. "
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Evaluate the classifier prediction alongside the physical leaf segmentation findings.\n"
                    "2. Explain symptoms, biology, and pathology factually without inventing symptoms.\n"
                    "3. If classifier predicts disease but 0 lesions were detected, clearly note the discrepancy.\n"
                    "4. Respond ONLY with valid JSON having the exact keys: 'expert_summary', 'pathological_details'."
                )
                user_prompt = (
                    f"Predicted Disease: {normalized_name}\n"
                    f"Confidence: {confidence:.2f}%\n"
                    f"Evidence Consistency: {consistency}\n"
                    f"Evidence Notes: {consistency_note}\n"
                    f"Causal Pathogen: {db_entry['pathogen']}\n"
                    f"Synthesize an evidence-based clinical pathology summary reflecting both prediction and measured blade data."
                )

                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
                expert_output["expert_summary"] = data.get("expert_summary", expert_output["expert_summary"])
                expert_output["pathological_details"] = data.get("pathological_details", db_entry["symptoms"])
            except Exception:
                pass

        return expert_output

