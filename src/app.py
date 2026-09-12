import os
import io
import time
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import cv2
import numpy as np

from src.detector import PCBDetector
from src.reasoning import IntentRouter, PCBReasoningEngine, LocalLLMReasoningEngine

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PCB-API")

app = FastAPI(
    title="Smart PCB Component Inspection & Reasoning API",
    description="Production RT-DETR Detection & Real Local LLM Reasoning Layer for PCB Defect & Assembly Inspection",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector and reasoning engine
MODEL_WEIGHTS = os.environ.get("PCB_MODEL_PATH", "runs/detect/pcb_rtdetr/weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("PCB_CONF_THRESHOLD", "0.35"))

detector = PCBDetector(model_path=MODEL_WEIGHTS, conf_threshold=CONFIDENCE_THRESHOLD)
reasoning_engine = PCBReasoningEngine(min_confidence_guardrail=0.30)
local_llm_engine = LocalLLMReasoningEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct", min_confidence_guardrail=0.30)


# Models for API schemas
class DetectionBox(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: list[float]
    box_normalized: list[float]
    centroid: list[float]
    centroid_normalized: list[float]
    dimensions: list[float]
    area_pixels: float

class DetectionResponse(BaseModel):
    status: str
    inference_time_ms: float
    image_dimensions: dict[str, int]
    total_detections: int
    class_counts: dict[str, int]
    detections: list[DetectionBox]
    quality_metrics: dict[str, float]

class ReasoningResponse(BaseModel):
    status: str
    question: str
    intent: str
    requires_detection: bool
    answer: str
    confidence_guardrail_triggered: bool
    guardrail_reason: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    inference_time_ms: float


@app.get("/health", tags=["System"])
async def health_check():
    """System health status, device info, and model weight availability."""
    weights_exist = Path(MODEL_WEIGHTS).exists()
    return {
        "status": "healthy",
        "service": "Smart PCB Inspection API",
        "model_architecture": "RT-DETR-Large",
        "model_weights_loaded": weights_exist,
        "weights_path": MODEL_WEIGHTS,
        "device": detector.device,
        "confidence_threshold": detector.conf_threshold
    }


@app.post("/detect", response_model=DetectionResponse, tags=["Part A - Detection"])
async def detect_components(
    file: UploadFile = File(..., description="PCB Image file (JPG/PNG)"),
    conf: Optional[float] = Query(None, ge=0.05, le=1.0, description="Override detection confidence threshold")
):
    """
    Part A Detection Endpoint:
    Accepts a PCB image and returns detected component bounding boxes, labels,
    centroid coordinates, dimensions, and detection confidences.
    """
    start_t = time.time()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")
            
        result = detector.detect(image_bytes, conf=conf)
        elapsed_ms = round((time.time() - start_t) * 1000.0, 2)
        
        return {
            "status": "success",
            "inference_time_ms": elapsed_ms,
            "image_dimensions": result["image_dimensions"],
            "total_detections": result["total_detections"],
            "class_counts": result["class_counts"],
            "detections": result["detections"],
            "quality_metrics": result["quality_metrics"]
        }
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/reason", response_model=ReasoningResponse, tags=["Part B - Reasoning Layer"])
async def reason_about_pcb(
    file: Optional[UploadFile] = File(None, description="PCB Image file (required if query is visual)"),
    question: str = Form(..., description="Natural language question about the PCB (e.g. 'How many capacitors exist?', 'Which component is closest to the transformer?')")
):
    """
    Part B Reasoning Layer Endpoint (Hand-Written, Framework-Free):
    1. Intent Routing: Decides if the question requires detection or is general/out-of-scope.
    2. Structured Spatial Reasoning: Executes nearest-neighbor math, missing component checks, count analyses.
    3. Confidence Guardrail: Triggers explicit 'Insufficient Information' if quality is below safe threshold.
    """
    start_t = time.time()
    
    # 1. Intent Routing
    route_result = IntentRouter.route(question)
    intent = route_result["intent"]
    requires_detection = route_result.get("requires_detection", False)
    
    if not requires_detection:
        elapsed_ms = round((time.time() - start_t) * 1000.0, 2)
        return {
            "status": "success",
            "question": question,
            "intent": intent,
            "requires_detection": False,
            "answer": route_result.get("direct_answer", "No visual detection needed."),
            "confidence_guardrail_triggered": False,
            "guardrail_reason": None,
            "data": None,
            "inference_time_ms": elapsed_ms
        }
        
    # Visual question requires image
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="The question requires visual inspection of a PCB image, but no image file was uploaded."
        )
        
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")
            
        # 2. Run RT-DETR Detection
        det_result = detector.detect(image_bytes)
        
        # 3. Run Real Local LLM Reasoning with Confidence Guardrails & Geometric Grounding
        reasoning_res = local_llm_engine.reason(question, det_result)
        
        elapsed_ms = round((time.time() - start_t) * 1000.0, 2)
        
        return {
            "status": reasoning_res.get("status", "SUCCESS"),
            "question": question,
            "intent": intent,
            "requires_detection": True,
            "answer": reasoning_res["answer"],
            "confidence_guardrail_triggered": reasoning_res.get("guardrail_triggered", False),
            "guardrail_reason": reasoning_res.get("guardrail_reason"),
            "data": reasoning_res.get("data"),
            "inference_time_ms": elapsed_ms
        }
    except Exception as e:
        logger.error(f"Reasoning failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reasoning error: {str(e)}")


# Serve static web dashboard
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", tags=["Dashboard"])
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Smart PCB Inspection API is live. Visit /docs for interactive Swagger API documentation."
    }
