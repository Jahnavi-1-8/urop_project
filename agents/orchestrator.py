"""LangGraph Orchestrator for Multi-Agent Decision Support.

Coordinates EXACTLY three specialist agents:
1. Disease Expert Agent
2. Risk and Severity Agent
3. Recommendation Agent

Workflow:
[START]
   |
   +---> Disease Expert Agent (interprets disease & symptoms)
   |
   +---> Risk & Severity Agent (quantifies foliar risk & severity)
   |
   v
[Recommendation Agent] (synthesizes practical agronomic management)
   |
   v
[Structured Final Result]
   |
 [END]
"""

from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from agents.llm_factory import get_configured_llm
from agents.disease_expert import DiseaseExpertAgent
from agents.risk_severity import RiskSeverityAgent
from agents.recommendation import RecommendationAgent


class AgenticWorkflowState(TypedDict):
    """LangGraph shared workflow state."""
    # Inputs from existing classification and severity pipeline
    predicted_disease: str
    confidence: float
    model_used: str
    severity_data: Optional[Dict[str, Any]]

    # Specialist Agent Outputs
    disease_expert_output: Optional[Dict[str, Any]]
    risk_severity_output: Optional[Dict[str, Any]]
    recommendation_output: Optional[Dict[str, Any]]

    # Coordinated final structured payload
    final_result: Optional[Dict[str, Any]]


class AgenticDecisionOrchestrator:
    """LangGraph-powered coordinator for the 3 specialist agents."""

    def __init__(self, temperature: float = 0.2):
        self.llm = get_configured_llm(temperature=temperature)
        self.disease_expert_agent = DiseaseExpertAgent(llm=self.llm)
        self.risk_severity_agent = RiskSeverityAgent(llm=self.llm)
        self.recommendation_agent = RecommendationAgent(llm=self.llm)
        self.graph = self._build_graph()

    def _disease_expert_node(self, state: AgenticWorkflowState) -> Dict[str, Any]:
        """Node 1: Execute Disease Expert Agent."""
        out = self.disease_expert_agent.run(
            disease_class=state["predicted_disease"],
            confidence=state["confidence"],
            model_used=state.get("model_used", "Classifier"),
            leaf_analysis=state.get("severity_data"),
        )
        return {"disease_expert_output": out}

    def _risk_severity_node(self, state: AgenticWorkflowState) -> Dict[str, Any]:
        """Node 2: Execute Risk and Severity Agent."""
        out = self.risk_severity_agent.run(
            predicted_disease=state["predicted_disease"],
            severity_data=state.get("severity_data"),
        )
        return {"risk_severity_output": out}

    def _recommendation_node(self, state: AgenticWorkflowState) -> Dict[str, Any]:
        """Node 3: Execute Recommendation Agent using Agent 1 & 2 findings."""
        expert_data = state.get("disease_expert_output") or {}
        risk_data = state.get("risk_severity_output") or {}
        out = self.recommendation_agent.run(
            expert_data=expert_data,
            risk_data=risk_data,
        )
        return {"recommendation_output": out}

    def _compile_result_node(self, state: AgenticWorkflowState) -> Dict[str, Any]:
        """Coordinator node assembling unified structured JSON with evidence-consistency decision logic."""
        disease = state["predicted_disease"]
        conf = state["confidence"]
        expert_res = state.get("disease_expert_output", {})
        risk_res = state.get("risk_severity_output", {})
        rec_res = state.get("recommendation_output", {})

        # Decision Case Resolution Logic (Cases A, B, C, D)
        sev_data = state.get("severity_data") or {}
        seg_success = sev_data.get("is_valid_segmentation", True)
        seg_quality = sev_data.get("segmentation_quality", "HIGH")
        lesion_px = sev_data.get("lesion_area_pixels", 0)
        affected_pct = sev_data.get("severity_percentage", 0.0)
        sev_level = risk_res.get("severity_level", "Unknown")

        is_pred_healthy = "healthy" in disease.lower()

        if not seg_success or seg_quality == "FAILED" or sev_level == "Indeterminate":
            # Case C: Classifier predicts a condition, but leaf segmentation fails
            decision_case = "Case C: Unverified Diagnosis (Segmentation Failed)"
            final_assessment = f"Model indicated {disease} ({conf:.1f}%), but leaf blade could not be isolated reliably."
            confirmed_status = "Unverified - Segmentation Failed"
            final_severity = "Indeterminate"
            uncertainty_flag = True
            decision_summary = (
                f"Classification model predicted {disease}, but foliar blade segmentation failed. "
                "Severity cannot be calculated from failed segmentation, and failed segmentation must not be interpreted as healthy. "
                "A clearer photograph is required."
            )

        elif not is_pred_healthy and (lesion_px == 0 or affected_pct == 0.0):
            # Case A: Classifier predicts disease, but reliable segmentation detects 0 lesions
            decision_case = "Case A: Classifier Disagreement (Zero Measured Lesions)"
            final_assessment = f"Likely Healthy on analyzed blade (disagrees with {disease} classifier prediction)"
            confirmed_status = "Likely Healthy / Asymptomatic (Discrepancy Noted)"
            final_severity = "None Detected (0.0% affected area)"
            uncertainty_flag = True
            decision_summary = (
                f"The classifier flagged {disease} ({conf:.1f}%), but verified foliar segmentation detected 0 lesion pixels. "
                "The analyzed leaf blade is visibly asymptomatic. In accordance with clinical evidence rules, "
                "the leaf is assessed as likely healthy, with the classifier discrepancy highlighted for field monitoring."
            )

        elif is_pred_healthy and lesion_px > 0 and affected_pct >= 1.0:
            # Case D: Classifier predicts Healthy, but lesion analysis detects substantial lesions
            decision_case = "Case D: Classifier Disagreement (Suspected Lesions on Healthy Leaf)"
            final_assessment = f"Uncertain - Visible foliar lesions detected ({affected_pct:.2f}%) despite Healthy classifier prediction"
            confirmed_status = "Suspicious / Review Recommended"
            final_severity = sev_level
            uncertainty_flag = True
            decision_summary = (
                f"Classifier predicted Healthy ({conf:.1f}%), but segmentation detected {lesion_px:,} pixels "
                f"({affected_pct:.2f}% affected area) of foliar lesions. The leaf cannot be confirmed healthy; "
                "physical expert inspection is recommended."
            )

        elif not is_pred_healthy and lesion_px > 0:
            # Case B: Classifier predicts disease, lesions detected and consistent
            decision_case = "Case B: Corroborated Disease Diagnosis"
            final_assessment = f"{disease} confirmed with visible lesions ({affected_pct:.2f}% affected foliar area)"
            confirmed_status = f"{disease} (Corroborated)"
            final_severity = sev_level
            uncertainty_flag = False
            decision_summary = (
                f"Diagnosis of {disease} is corroborated by quantitative foliar segmentation measuring "
                f"{lesion_px:,} lesion pixels ({affected_pct:.2f}% affected leaf surface, graded as {sev_level})."
            )

        else:
            # Normal Baseline: Confirmed Asymptomatic Foliage
            decision_case = "Normal Baseline: Confirmed Asymptomatic Foliage"
            final_assessment = "Healthy (Asymptomatic foliage with 0 detected lesions)"
            confirmed_status = "Healthy"
            final_severity = "Healthy"
            uncertainty_flag = False
            decision_summary = "Specimen exhibits healthy chlorophyll pigmentation and intact leaf cuticle with zero disease lesions."

        final_structure = {
            "status": "SUCCESS",
            "orchestrator": "LangGraph Coordinator",
            "decision_case": decision_case,
            "uncertainty_flag": uncertainty_flag,
            "model_classification": {
                "predicted_disease": disease,
                "confidence_percent": conf,
                "model_used": state.get("model_used", "Classifier"),
            },
            "leaf_analysis_metrics": {
                "leaf_area_pixels": sev_data.get("leaf_area_pixels", 0),
                "lesion_area_pixels": lesion_px,
                "severity_percentage": affected_pct,
                "severity_level": final_severity,
                "segmentation_quality": seg_quality,
            },
            "specialist_agent_outputs": {
                "agent_1_disease_expert": expert_res,
                "agent_2_risk_severity": risk_res,
                "agent_3_recommendation": rec_res,
            },
            "executive_summary": {
                "condition": disease,
                "confirmed_diagnosis": confirmed_status,
                "model_prediction": disease,
                "final_assessment": final_assessment,
                "severity_level": final_severity,
                "affected_foliar_area": affected_pct,
                "action_priority": rec_res.get("action_priority", "Moderate"),
                "primary_guidance": rec_res.get("specialist_consultation", ""),
                "decision_case": decision_case,
                "decision_summary": decision_summary,
            },
        }
        return {"final_result": final_structure}

    def _build_graph(self):
        """Construct the LangGraph StateGraph connecting exactly the 3 agents."""
        workflow = StateGraph(AgenticWorkflowState)

        # Register nodes for the 3 agents + compiler
        workflow.add_node("disease_expert_agent", self._disease_expert_node)
        workflow.add_node("risk_severity_agent", self._risk_severity_node)
        workflow.add_node("recommendation_agent", self._recommendation_node)
        workflow.add_node("compile_result", self._compile_result_node)

        # Wire execution flow:
        # START -> disease_expert_agent -> risk_severity_agent -> recommendation_agent -> compile_result -> END
        workflow.add_edge(START, "disease_expert_agent")
        workflow.add_edge("disease_expert_agent", "risk_severity_agent")
        workflow.add_edge("risk_severity_agent", "recommendation_agent")
        workflow.add_edge("recommendation_agent", "compile_result")
        workflow.add_edge("compile_result", END)

        return workflow.compile()

    def run(
        self,
        predicted_disease: str,
        confidence: float,
        model_used: str = "Classifier",
        severity_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end multi-agent orchestration."""
        initial_state: AgenticWorkflowState = {
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "model_used": model_used,
            "severity_data": severity_data,
            "disease_expert_output": None,
            "risk_severity_output": None,
            "recommendation_output": None,
            "final_result": None,
        }

        output_state = self.graph.invoke(initial_state)
        return output_state["final_result"]

