"""Agent 2: Risk and Severity Agent.

Role:
- Receives the disease prediction and existing leaf-analysis measurements.
- Interprets affected-area percentage and available foliar severity metrics.
- Evaluates epidemic risk and severity cautiously.
- Reports uncertainty if measurements are unavailable, partial, or unreliable.
- Strictly avoids fabricating measurements or severity.
"""

from typing import Dict, Any, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel


class RiskSeverityAgent:
    """Specialist foliar risk and severity assessment agent."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def run(
        self,
        predicted_disease: str,
        severity_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess foliar risk and severity based strictly on provided measurements."""
        # Check if measurements are available
        has_metrics = (
            severity_data is not None
            and "severity_percentage" in severity_data
            and severity_data["severity_percentage"] is not None
        )

        is_healthy = "healthy" in predicted_disease.lower()

        seg_success = severity_data.get("is_valid_segmentation", True) if severity_data else False
        seg_quality = severity_data.get("segmentation_quality", "HIGH") if severity_data else "FAILED"

        if not has_metrics:
            risk_output = {
                "agent": "Risk and Severity Agent",
                "severity_level": "Undetermined",
                "affected_area_percent": None,
                "measurement_status": "UNAVAILABLE / UNCERTAIN",
                "risk_rating": "Uncertain (Pending Measurement)",
                "epidemic_threat": "Indeterminate",
                "risk_assessment": (
                    "Foliar segmentation measurements are unavailable for this specimen. "
                    "Cannot quantify lesion area or assign severity category without verified measurements. "
                    "Cautious manual physical inspection of orchard canopy is advised."
                ),
                "uncertainty_notes": "Lesion area percentage not provided; no measurements fabricated.",
            }
            return risk_output

        if not seg_success or seg_quality == "FAILED":
            risk_output = {
                "agent": "Risk and Severity Agent",
                "severity_level": "Indeterminate",
                "affected_area_percent": None,
                "measurement_status": "SEGMENTATION_FAILED",
                "risk_rating": "Indeterminate (Segmentation Failed)",
                "epidemic_threat": "Indeterminate",
                "risk_assessment": (
                    "Foliar segmentation failed or produced an unreliable mask. "
                    "Cannot quantify lesion area or assign severity category without verified measurements. "
                    "Failed segmentation must not be interpreted as a healthy leaf. A clearer photograph is required."
                ),
                "uncertainty_notes": "Segmentation failed; zero measurements fabricated.",
            }
            return risk_output

        # Extract verified measurements
        affected_pct = float(severity_data.get("severity_percentage", 0.0))
        leaf_px = severity_data.get("leaf_area_pixels", 0)
        lesion_px = severity_data.get("lesion_area_pixels", 0)
        computed_level = severity_data.get("severity_level", "Unknown")

        # Cautious epidemic threat assessment based strictly on biological thresholds
        if affected_pct == 0.0 or lesion_px == 0:
            if is_healthy:
                risk_rating = "Negligible"
                threat = "None - Specimen exhibits asymptomatic foliar tissue."
                assessment = "No discernible foliar lesion tissue detected. Epidemic spread risk from this leaf is negligible."
            else:
                risk_rating = "Low / Trace (Uncorroborated by visible necrosis)"
                threat = "Classifier predicted disease, but 0 lesion pixels were measured on the segmented leaf."
                assessment = (
                    f"The ML classifier predicted '{predicted_disease}', but quantitative analysis measured 0 lesion pixels (0.0%). "
                    "Visible foliar damage is absent on this specimen. Recommend visual re-inspection before initiating treatment."
                )
        elif affected_pct < 15.0:
            risk_rating = "Low to Moderate"
            threat = "Early localized infection. Potential for secondary spore cycling under humid microclimates."
            assessment = (
                f"Localized early-stage damage occupying {affected_pct:.2f}% of foliar surface. "
                "Risk of rapid orchard contagion is currently manageable if contained promptly."
            )
        elif 15.0 <= affected_pct <= 40.0:
            risk_rating = "Substantial (Moderate)"
            threat = "Active inoculum pressure. Foliar photosynthetic disruption and expanding lesion margins."
            assessment = (
                f"Significant foliar compromise ({affected_pct:.2f}% infected area). "
                "Active spore dispersal threat to adjacent vegetative canopy and developing fruit clusters."
            )
        else:
            risk_rating = "Critical (Severe)"
            threat = "Extensive foliar necrosis, defoliation hazard, and aggressive secondary sporulation."
            assessment = (
                f"Severe foliar breakdown affecting {affected_pct:.2f}% of the leaf surface. "
                "High risk of premature abscission, canopy defoliation, and orchard-wide inoculum amplification."
            )

        risk_output = {
            "agent": "Risk and Severity Agent",
            "severity_level": computed_level,
            "affected_area_percent": round(affected_pct, 2),
            "leaf_area_pixels": leaf_px,
            "lesion_area_pixels": lesion_px,
            "measurement_status": "VERIFIED",
            "risk_rating": risk_rating,
            "epidemic_threat": threat,
            "risk_assessment": assessment,
            "uncertainty_notes": (
                "Assessment strictly bounded to the submitted 2D leaf image; entire tree or orchard canopy "
                "severity requires representative multi-point sampling."
            ),
        }

        # LLM refinement if available
        if self.llm:
            try:
                system_prompt = (
                    "You are the Risk and Severity Agent for orchard pathology. "
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Strictly use the provided leaf measurements. DO NOT invent or alter pixel counts or percentages.\n"
                    "2. Assess risk cautiously and report any field uncertainty.\n"
                    "3. Respond ONLY with valid JSON having keys: 'risk_assessment', 'canopy_impact'."
                )
                user_prompt = (
                    f"Disease: {predicted_disease}\n"
                    f"Severity Level: {computed_level}\n"
                    f"Affected Foliar Area: {affected_pct:.2f}%\n"
                    f"Leaf Pixels: {leaf_px}\n"
                    f"Lesion Pixels: {lesion_px}\n"
                    f"Synthesize a cautious epidemiological risk evaluation."
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
                risk_output["risk_assessment"] = data.get("risk_assessment", risk_output["risk_assessment"])
                if "canopy_impact" in data:
                    risk_output["canopy_impact"] = data["canopy_impact"]
            except Exception:
                pass

        return risk_output

