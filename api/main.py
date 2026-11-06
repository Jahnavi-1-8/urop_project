"""FastAPI Service for Apple Leaf Disease Detection & Multi-Agent Decision Support.

Exposes structured JSON endpoints:
- GET  /health: System and model status
- POST /validate: Non-leaf and image quality verification
- POST /classify: Core model prediction and severity quantification
- POST /analyze: Complete end-to-end pipeline (Validation -> Classification -> Severity -> LangGraph 3-Agent Decision Support)
"""

from typing import Optional, Dict, Any
import os
import sys
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input_validation import validate_apple_leaf_input
from src.predict import AppleLeafPredictor
from agents.orchestrator import AgenticDecisionOrchestrator

app = FastAPI(
    title="Apple Leaf Disease & Agentic AI Decision Support API",
    version="2.0.0",
    description="Combines input validation, classical ML classification, foliar severity assessment, and LangGraph 3-agent advisory.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_predictor: Optional[AppleLeafPredictor] = None
_orchestrator: Optional[AgenticDecisionOrchestrator] = None


def get_predictor() -> AppleLeafPredictor:
    global _predictor
    if _predictor is None:
        models_dir = os.path.join(project_root, "models")
        _predictor = AppleLeafPredictor(models_dir=models_dir)
    return _predictor


def get_orchestrator() -> AgenticDecisionOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgenticDecisionOrchestrator()
    return _orchestrator


@app.get("/health")
def health_check():
    """Health check endpoint."""
    predictor = get_predictor()
    return {
        "status": "healthy",
        "models_loaded": {
            "random_forest": predictor.rf_model is not None,
            "svm": predictor.svm_model is not None,
            "scaler": predictor.scaler is not None,
            "label_encoder": predictor.label_encoder is not None,
        },
        "agents": ["Disease Expert Agent", "Risk and Severity Agent", "Recommendation Agent"],
        "orchestrator": "LangGraph",
    }


@app.post("/validate")
def validate_image(file: UploadFile = File(...)):
    """Stage 1: Validate input image to reject non-leaf, corrupted, blurry, human face, or synthetic images."""
    try:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "reason": f"Corrupted or invalid image file: {str(e)}",
            "details": {"error": str(e)},
        }

    val_res = validate_apple_leaf_input(img_np, is_rgb=True)
    val_res.pop("image_rgb", None)
    return val_res


@app.post("/classify")
def classify_leaf(
    file: UploadFile = File(...),
    model_name: str = "random_forest",
):
    """Stage 2: Execute existing classification and severity pipeline (preceded by validation)."""
    try:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {str(e)}")

    val_res = validate_apple_leaf_input(img_np, is_rgb=True)
    if not val_res["is_valid"]:
        raise HTTPException(
            status_code=422,
            detail={"error": "Input rejected by validator", "reason": val_res["reason"], "details": val_res["details"]},
        )

    predictor = get_predictor()
    sev_res = predictor.predict_with_severity(
        img_np,
        model_name=model_name,
        is_rgb=True,
        image_name=file.filename,
    )

    cls_info = sev_res["classification"]
    return {
        "predicted_class": cls_info["predicted_disease"],
        "confidence": cls_info["confidence"],
        "confidence_percent": cls_info["confidence_percent"],
        "is_healthy": "healthy" in cls_info["predicted_disease"].lower(),
        "model_used": cls_info["model_used"],
        "probabilities": cls_info["probabilities"],
        "severity": sev_res["severity"],
    }


@app.post("/analyze")
def analyze_end_to_end(
    file: UploadFile = File(...),
    model_name: str = "random_forest",
):
    """Complete Pipeline: Validation -> Classification & Severity -> LangGraph 3-Agent Decision Support."""
    try:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {str(e)}")

    # 1. Validation
    val_res = validate_apple_leaf_input(img_np, is_rgb=True)
    val_res.pop("image_rgb", None)
    if not val_res["is_valid"]:
        return {
            "status": "REJECTED",
            "validation": val_res,
            "message": f"Image processing halted: {val_res['reason']}",
        }

    # 2. Existing Classification & Severity
    predictor = get_predictor()
    sev_res = predictor.predict_with_severity(
        img_np,
        model_name=model_name,
        is_rgb=True,
        image_name=file.filename,
    )

    cls_info = sev_res["classification"]
    sev_info = sev_res["severity"]

    # 3. LangGraph Orchestrator (Exactly 3 Agents)
    orchestrator = get_orchestrator()
    agentic_result = orchestrator.run(
        predicted_disease=cls_info["predicted_disease"],
        confidence=cls_info["confidence_percent"],
        model_used=cls_info["model_used"],
        severity_data=sev_info,
    )

    return {
        "status": "APPROVED_AND_ANALYZED",
        "validation": val_res,
        "classification": {
            "predicted_class": cls_info["predicted_disease"],
            "confidence_percent": cls_info["confidence_percent"],
            "model_used": cls_info["model_used"],
            "is_healthy": "healthy" in cls_info["predicted_disease"].lower(),
            "probabilities": cls_info["probabilities"],
        },
        "severity_analysis": sev_info,
        "agentic_decision_support": agentic_result,
    }
