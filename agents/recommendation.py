"""Agent 3: Recommendation Agent.

Role:
- Receives Disease Expert and Risk and Severity Agent outputs.
- Synthesizes practical, evidence-based agronomic recommendations.
- Prescribes sanitation, cultural management, and field monitoring.
- Explicitly guides when certified extension specialist consultation is necessary.
- STRICTLY avoids unauthorized/unsupported pesticide dosage calculations or chemical claims.
"""

from typing import Dict, Any, List, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel


class RecommendationAgent:
    """Specialist agronomic advisory agent providing integrated management strategies."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def run(
        self,
        expert_data: Dict[str, Any],
        risk_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Formulate targeted agronomic recommendations."""
        disease = expert_data.get("predicted_disease", "Unknown")
        severity_lvl = risk_data.get("severity_level", "Unknown")
        affected_pct = risk_data.get("affected_area_percent")
        is_healthy = "healthy" in disease.lower()

        # Build grounded cultural, sanitation, and monitoring actions
        cultural_practices: List[str] = []
        monitoring_protocol: List[str] = []
        specialist_guidance: str = ""
        safety_notice = (
            "NOTICE: Chemical pesticide application must strictly adhere to local agricultural regulatory guidelines "
            "and manufacturer label directions. Dosages must be determined by a certified regional crop advisor."
        )

        if is_healthy:
            cultural_practices = [
                "Maintain balanced nitrogen fertilization; avoid excessive succulent shoot growth.",
                "Ensure proper annual canopy pruning to maximize sunlight penetration and air circulation.",
                "Implement proactive orchard floor weed control and moisture management.",
            ]
            monitoring_protocol = [
                "Routine bi-weekly scouting across randomized orchard rows.",
                "Inspect lower and interior canopy leaves after rain events for early spore signs.",
            ]
            specialist_guidance = "No immediate extension intervention needed. Continue standard seasonal maintenance."

        elif "scab" in disease.lower():
            cultural_practices = [
                "Prune trees to open canopy and accelerate foliar drying after rain or morning dew.",
                "Shred or remove fallen apple leaves in autumn/winter to interrupt overwintering pseudothecia.",
                "Flail mow orchard floor or apply agricultural urea in late autumn to accelerate leaf decomposition.",
            ]
            monitoring_protocol = [
                "Track Mills Infection Periods based on ambient temperature and hours of leaf wetness.",
                "Scout cluster leaves and fruitlets 10–14 days following warm rain events.",
            ]
            specialist_guidance = (
                "Consult local university agricultural extension service for current fungicide resistance profiles "
                "and approved protectant spray schedules for your regional apple cultivar."
            )

        elif "black" in disease.lower():
            cultural_practices = [
                "Prune out dead wood, fire blight strikes, and visible branch cankers during dry weather.",
                "Remove and destroy all mummified fruit hanging on branches or fallen to the ground.",
                "Disinfect pruning shears with 70% ethanol or 10% bleach solution between cuts.",
            ]
            monitoring_protocol = [
                "Regularly inspect tree scaffold limbs and trunk for sunken bark or oozing cankers.",
                "Inspect fruit development at petal fall and cover spray intervals for rot initiation.",
            ]
            specialist_guidance = (
                "Seek certified pomologist or plant diagnostic laboratory consultation if branch cankers expand "
                "or trunk girdling symptoms are observed."
            )

        elif "rust" in disease.lower():
            cultural_practices = [
                "Remove nearby Eastern Red Cedar or decorative juniper hosts within a 1–2 mile radius where feasible.",
                "Prune and destroy cedar galls from adjacent ornamental junipers before orange spore horns gelatinize in spring.",
                "Plant rust-resistant apple cultivars (e.g., Liberty, Freedom, Enterprise) for future orchard plantings.",
            ]
            monitoring_protocol = [
                "Monitor juniper trees in early spring for brown galls and gelatinous orange spore tendrils during warm rains.",
                "Scout young apple foliage from tight cluster through bloom and early fruit development.",
            ]
            specialist_guidance = (
                "Contact local extension specialist regarding regional rust infection timing models and rust-specific "
                "fungicide protectants permitted in your jurisdiction."
            )

        else:
            cultural_practices = ["General canopy aeration and foliar moisture reduction."]
            monitoring_protocol = ["Regular orchard foliar monitoring."]
            specialist_guidance = "Submit physical leaf sample to an accredited agricultural diagnostic clinic."

        meas_status = risk_data.get("measurement_status", "VERIFIED")
        consistency = expert_data.get("evidence_consistency", "CONSISTENT")

        if meas_status == "SEGMENTATION_FAILED":
            action_priority = "Resubmission / Clearer Image Required"
            specialist_guidance = (
                "Leaf blade segmentation was unsuccessful or produced an incomplete mask. "
                "Retake a clear photograph with the leaf centered on a neutral background before applying any interventions."
            )
        elif consistency == "DISCORDANT_ZERO_LESIONS":
            action_priority = "Reassessment / Field Monitoring"
            specialist_guidance = (
                f"The classifier flagged {disease}, but quantitative lesion detection measured 0 lesion pixels across the leaf surface. "
                "Do not apply curative fungicides on asymptomatic leaves. Monitor the tree closely for emerging symptoms and re-sample if suspicious spots appear."
            )
        elif consistency == "DISCORDANT_SUSPECTED_LESIONS":
            action_priority = "Physical Diagnostic Review"
            specialist_guidance = (
                "Classifier predicted Healthy foliage, but image segmentation identified visible lesion areas. "
                "Consult a regional extension pomologist or send leaf tissue to a plant pathology lab to confirm pathogen presence."
            )
        elif is_healthy:
            action_priority = "Routine"
        elif severity_lvl == "Severe":
            action_priority = "Urgent / Priority Action"
        elif severity_lvl == "Moderate":
            action_priority = "Moderate / Active Management"
        else:
            action_priority = "Preventative / Monitoring"

        rec_output = {
            "agent": "Recommendation Agent",
            "targeted_disease": disease,
            "severity_context": severity_lvl,
            "cultural_management": cultural_practices,
            "monitoring_protocol": monitoring_protocol,
            "specialist_consultation": specialist_guidance,
            "chemical_safety_disclaimer": safety_notice,
            "action_priority": action_priority,
        }

        # LLM dynamic enrichment if available
        if self.llm:
            try:
                system_prompt = (
                    "You are the Agronomic Recommendation Agent for apple orchard disease management. "
                    "CRITICAL CONSTRAINTS:\n"
                    "1. Never invent chemical dosages or make unsupported pesticide claims.\n"
                    "2. Emphasize cultural practices, sanitation, monitoring, and expert consultation.\n"
                    "3. Respond ONLY with valid JSON with keys: 'agronomic_advisory', 'immediate_action'."
                )
                user_prompt = (
                    f"Disease: {disease}\n"
                    f"Severity Level: {severity_lvl} ({affected_pct} % infected area)\n"
                    f"Draft practical advisory guidance."
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
                if "agronomic_advisory" in data:
                    rec_output["agronomic_advisory"] = data["agronomic_advisory"]
                if "immediate_action" in data:
                    rec_output["immediate_action"] = data["immediate_action"]
            except Exception:
                pass

        return rec_output

