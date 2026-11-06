"""Streamlit Web Application: Apple Leaf Disease Detection & Agentic AI Decision Support.

Professional Dark-Themed Dashboard with:
- Left Sidebar: Input image upload, sample leaves, model selection.
- Main Navigation (Exactly 3 Sections):
    1. Disease Classification
    2. Leaf & Lesion Analysis
    3. Agentic AI (Disease Expert, Risk & Severity, Recommendation, Final Synthesized Decision)
"""

from typing import Optional, Dict, Any
import os
import sys
import glob
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict import AppleLeafPredictor
from src.feature_extraction import CLASS_MAP
from src.input_validation import validate_apple_leaf_input
from agents.orchestrator import AgenticDecisionOrchestrator


st.set_page_config(
    page_title="Apple Leaf Disease Detection & Agentic AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional dark theme styling
st.markdown(
    """
    <style>
    /* Dark theme core styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 0.15rem;
        letter-spacing: -0.3px;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 1.2rem;
    }
    .status-badge-healthy {
        background-color: #0d381e;
        color: #3fb950;
        border: 1px solid #238636;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .status-badge-infected {
        background-color: #490202;
        color: #f85149;
        border: 1px solid #da3633;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .severity-badge {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    .agent-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
    }
    .agent-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.4rem;
        margin-bottom: 0.6rem;
    }
    .decision-box {
        background-color: #161b22;
        border: 1px solid #388bfd33;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_predictor():
    """Load predictor instance with trained models."""
    if "apple_leaf_predictor" not in st.session_state:
        st.cache_resource.clear()
        models_dir = os.path.join(project_root, "models")
        st.session_state.apple_leaf_predictor = AppleLeafPredictor(models_dir=models_dir)
    return st.session_state.apple_leaf_predictor


def load_orchestrator():
    """Load LangGraph agentic orchestrator instance."""
    if "agentic_orchestrator" not in st.session_state:
        st.session_state.agentic_orchestrator = AgenticDecisionOrchestrator()
    return st.session_state.agentic_orchestrator


def get_sample_images() -> Dict[str, str]:
    """Retrieve sample images from each of the 4 classes."""
    samples = {}
    for raw_cls, readable in CLASS_MAP.items():
        pattern = os.path.join(project_root, "data", "raw", raw_cls, "*.JPG")
        matches = glob.glob(pattern)
        if matches:
            samples[f"{readable} (Sample)"] = matches[0]
    return samples


def main():
    st.markdown('<div class="main-header">🍎 Apple Leaf Disease Detection & Decision Support</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Classical Machine Learning (RF / SVM), Foliar Lesion Quantification, and LangGraph Multi-Agent AI.</div>',
        unsafe_allow_html=True,
    )

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Failed to load machine learning models: {e}. Please ensure models exist in `models/`.")
        return

    # --- LEFT SIDEBAR: IMAGE INPUT & SAMPLES ---
    with st.sidebar:
        st.markdown("### 📷 Input Image")
        uploaded_file = st.file_uploader(
            "Upload an apple leaf photograph:",
            type=["jpg", "jpeg", "png"],
            help="Supports JPG, JPEG, and PNG formats.",
        )

        st.markdown("### 🌿 Samples")
        sample_images = get_sample_images()
        selected_sample = st.selectbox(
            "Or choose a pre-loaded sample:",
            options=["None (Upload your own)"] + list(sample_images.keys()),
            index=0,
        )

        # Clear / reset image option
        if uploaded_file is not None or selected_sample != "None (Upload your own)":
            if st.button("🔄 Reset / Remove Image", use_container_width=True):
                st.session_state.pop("apple_leaf_predictor", None)
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Model Selection")
        model_choice = st.radio(
            "Active Classification Model:",
            options=["Random Forest", "Support Vector Machine (SVM)"],
            index=0,
        )
        model_key = "random_forest" if "Random" in model_choice else "svm"

        st.markdown("---")
        # Sidebar image preview
        leaf_image: Optional[Image.Image] = None
        leaf_source_name = ""

        if uploaded_file is not None:
            leaf_image = Image.open(uploaded_file)
            leaf_source_name = uploaded_file.name
        elif selected_sample != "None (Upload your own)" and selected_sample in sample_images:
            sample_path = sample_images[selected_sample]
            leaf_image = Image.open(sample_path)
            leaf_source_name = os.path.basename(sample_path)

        if leaf_image is not None:
            st.markdown("#### Selected Preview")
            st.image(leaf_image, caption=leaf_source_name, use_container_width=True)

    # If no image provided, prompt user cleanly
    if leaf_image is None:
        st.info("👈 Please upload an apple leaf image or choose a pre-loaded sample from the left sidebar to begin.")
        return

    # Convert PIL Image to RGB numpy array
    img_np = np.array(leaf_image.convert("RGB"))

    # --- INPUT VALIDATION (Non-leaf restriction & quality verification) ---
    val_res = validate_apple_leaf_input(img_np, is_rgb=True)

    if not val_res["is_valid"]:
        st.error(f"⚠️ **Image Validation Notice:** {val_res['reason']}")
        with st.expander("Inspection Telemetry", expanded=False):
            st.write(f"**Status:** `{val_res['status']}`")
            st.json(val_res.get("details", {}))
        st.warning("Please upload a clear, focused photograph containing a genuine apple leaf blade.")
        return

    if val_res["status"] == "REVIEW_RECOMMENDED":
        st.warning(f"ℹ️ **Advisory:** {val_res['reason']}")

    # Run inference & severity
    try:
        sev_res = predictor.predict_with_severity(
            img_np,
            model_name=model_key,
            is_rgb=True,
            image_name=leaf_source_name,
        )
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return

    cls_info = sev_res["classification"]
    sev_info = sev_res["severity"]
    sev_vis = sev_res.get("visualizations") or {}

    predicted_class = cls_info["predicted_disease"]
    confidence = cls_info["confidence_percent"]
    is_healthy = "healthy" in predicted_class.lower()

    # Preprocessed artifacts for isolated leaf
    prep_res = predictor.predict(img_np, model_name=model_key, is_rgb=True, image_name=leaf_source_name)
    artifacts = prep_res.get("preprocessed_artifacts", {})

    # Run LangGraph Orchestrator (3 Specialist Agents)
    orchestrator = load_orchestrator()
    agentic_res = orchestrator.run(
        predicted_disease=predicted_class,
        confidence=confidence,
        model_used=cls_info["model_used"],
        severity_data=sev_info,
    )
    agent_outputs = agentic_res.get("specialist_agent_outputs", {})
    a1 = agent_outputs.get("agent_1_disease_expert", {})
    a2 = agent_outputs.get("agent_2_risk_severity", {})
    a3 = agent_outputs.get("agent_3_recommendation", {})
    exec_summary = agentic_res.get("executive_summary", {})

    # --- MAIN NAVIGATION: EXACTLY 3 SECTIONS ---
    tab_classification, tab_analysis, tab_agentic = st.tabs([
        "Disease Classification",
        "Leaf & Lesion Analysis",
        "Agentic AI",
    ])

    # ==================== 1. DISEASE CLASSIFICATION ====================
    with tab_classification:
        col_c1, col_c2 = st.columns([1, 1], gap="large")

        with col_c1:
            st.markdown("### Diagnosis Result")
            if is_healthy:
                st.markdown(
                    f'<div class="status-badge-healthy">🌿 Healthy — {predicted_class}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="status-badge-infected">⚠️ Infected — {predicted_class}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Predicted Disease:** `{predicted_class}`")
            st.markdown(f"**Active Model:** `{cls_info['model_used']}`")
            st.markdown(f"**Prediction Confidence:** `{confidence:.2f}%`")
            st.progress(min(1.0, float(confidence / 100.0)))

            st.markdown("#### Diagnostic Note")
            st.info(cls_info.get("explanation", ""))

        with col_c2:
            st.markdown("### Probability Distribution")
            probs = cls_info.get("probabilities", {})
            if probs:
                probs_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
                probs_df["Confidence %"] = probs_df["Probability"] * 100.0
                st.bar_chart(
                    data=probs_df.set_index("Class")["Confidence %"],
                    use_container_width=True,
                )

            with st.expander("View Numerical Probability Table"):
                if probs:
                    display_df = probs_df[["Class", "Confidence %"]].copy()
                    display_df["Confidence %"] = display_df["Confidence %"].map(lambda x: f"{x:.2f}%")
                    st.table(display_df.set_index("Class"))

    # ==================== 2. LEAF & LESION ANALYSIS ====================
    with tab_analysis:
        st.markdown("### Foliar & Lesion Diagnostics")

        # Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        leaf_px = sev_info.get("leaf_area_pixels", 0)
        lesion_px = sev_info.get("lesion_area_pixels", 0)
        sev_pct = sev_info.get("severity_percentage", 0.0)
        sev_lvl = sev_info.get("severity_level", "Unknown")

        with m1:
            st.metric("Total Leaf Area", f"{leaf_px:,} px")
        with m2:
            st.metric("Diseased Lesion Area", f"{lesion_px:,} px")
        with m3:
            st.metric("Affected Area", f"{sev_pct:.2f}%")
        with m4:
            st.markdown(f'<div class="severity-badge">Severity: {sev_lvl.upper()}</div>', unsafe_allow_html=True)
            st.caption("Standard foliar cutoff metric")

        st.progress(min(1.0, float(sev_pct / 100.0)))

        # 5 Required Image Displays: Original, Binary Leaf Mask, Isolated Leaf, Lesion Mask, Diagnostic Overlay
        st.markdown("#### Visual Segmentation Gallery")
        v1, v2, v3, v4, v5 = st.columns(5)

        with v1:
            st.image(
                sev_vis.get("original_image", img_np),
                caption="1. Original Image",
                use_container_width=True,
            )
        with v2:
            if "leaf_mask" in sev_vis and sev_vis["leaf_mask"] is not None:
                st.image(sev_vis["leaf_mask"], caption="2. Binary Leaf Mask", use_container_width=True)
            else:
                st.caption("Leaf mask unavailable")
        with v3:
            if "segmented_rgb" in artifacts:
                st.image(artifacts["segmented_rgb"], caption="3. Isolated Leaf", use_container_width=True)
            else:
                st.caption("Isolated leaf unavailable")
        with v4:
            if "lesion_mask" in sev_vis and sev_vis["lesion_mask"] is not None:
                st.image(sev_vis["lesion_mask"], caption="4. Lesion Mask", use_container_width=True)
            else:
                st.caption("Lesion mask unavailable")
        with v5:
            if "composite_visualization" in sev_vis and sev_vis["composite_visualization"] is not None:
                st.image(
                    sev_vis["composite_visualization"],
                    caption="5. Diagnostic Overlay",
                    use_container_width=True,
                )
            else:
                st.caption("Overlay unavailable")

        st.caption(f"**Agronomic Guidance:** {sev_info.get('description', '')}")

    # ==================== 3. AGENTIC AI ====================
    with tab_agentic:
        st.markdown("### Specialist AI Agents")
        st.caption("Collaborative decision support coordinated via LangGraph orchestrator.")

        # 3 Specialist Agent Cards
        col_ag1, col_ag2, col_ag3 = st.columns(3)

        with col_ag1:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.markdown('<div class="agent-header">🔬 Disease Expert Agent</div>', unsafe_allow_html=True)
            st.markdown(f"**Causal Pathogen:** {a1.get('causal_agent', 'N/A')}")
            st.markdown(f"**Symptoms:** {a1.get('primary_symptoms', 'N/A')}")
            st.markdown(f"**Epidemiology:** {a1.get('epidemiological_conditions', 'N/A')}")
            st.markdown(f"**Lifecycle:** {a1.get('disease_lifecycle', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_ag2:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.markdown('<div class="agent-header">⚖️ Risk & Severity Agent</div>', unsafe_allow_html=True)
            st.markdown(f"**Severity Level:** `{a2.get('severity_level', 'N/A')}`")
            st.markdown(f"**Affected Area:** `{a2.get('affected_area_percent', 'N/A')}%`")
            st.markdown(f"**Epidemic Threat:** {a2.get('epidemic_threat', 'N/A')}")
            st.markdown(f"**Risk Evaluation:** {a2.get('risk_assessment', 'N/A')}")
            if a2.get("uncertainty_notes"):
                st.caption(f"*{a2.get('uncertainty_notes')}*")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_ag3:
            st.markdown('<div class="agent-card">', unsafe_allow_html=True)
            st.markdown('<div class="agent-header">📋 Recommendation Agent</div>', unsafe_allow_html=True)
            st.markdown(f"**Action Priority:** `{a3.get('action_priority', 'Standard')}`")
            st.markdown("**Cultural Practices:**")
            for cp in a3.get("cultural_management", []):
                st.markdown(f"- {cp}")
            st.markdown("**Monitoring Protocol:**")
            for mp in a3.get("monitoring_protocol", []):
                st.markdown(f"- {mp}")
            st.markdown(f"**Consultation:** {a3.get('specialist_consultation', 'Standard maintenance.')}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Final Synthesized Decision Summary Box
        case_badge = exec_summary.get("decision_case", "")
        summary_text = exec_summary.get("decision_summary", "")
        condition_title = exec_summary.get("condition", predicted_class)
        final_assessment = exec_summary.get("final_assessment", f"{predicted_class} ({confidence:.1f}%)")

        st.markdown(
            f"""
            <div class="decision-box">
                <div style="font-weight: 700; font-size: 1.15rem; color: #58a6ff; margin-bottom: 0.4rem;">
                    🎯 Final Synthesized Decision Summary
                </div>
                <div style="margin-bottom: 0.4rem;">
                    <span style="background-color: #21262d; border: 1px solid #388bfd; border-radius: 4px; padding: 2px 8px; font-size: 0.82rem; font-weight: 600; color: #58a6ff;">
                        {case_badge}
                    </span>
                </div>
                <div style="margin-bottom: 0.35rem; font-size: 0.95rem;">
                    <strong>Evidence-Synthesized Assessment:</strong> {final_assessment}
                </div>
                <div style="margin-bottom: 0.3rem;">
                    <strong>Classifier Output:</strong> {predicted_class} ({confidence:.2f}% confidence via {cls_info['model_used']})
                </div>
                <div style="margin-bottom: 0.3rem;">
                    <strong>Foliar Severity:</strong> {sev_lvl} ({sev_pct:.2f}% affected leaf area)
                </div>
                <div style="margin-bottom: 0.3rem;">
                    <strong>Action Priority:</strong> {a3.get('action_priority', 'Standard')}
                </div>
                <div style="margin-bottom: 0.35rem; color: #e6edf3; background: #0d1117; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #3fb950; margin-top: 6px;">
                    <strong>Decision Protocol:</strong> {summary_text}
                </div>
                <div style="margin-top: 0.4rem;">
                    <strong>Primary Recommendation:</strong> {a3.get('specialist_consultation', '')}
                </div>
                <div style="font-size: 0.82rem; color: #8b949e; margin-top: 0.6rem;">
                    <em>{a3.get('chemical_safety_disclaimer', '')}</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View Coordinated Agent JSON Payload"):
            st.json(agentic_res)


if __name__ == "__main__":
    main()
