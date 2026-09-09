import pytest
from src.reasoning import IntentRouter, PCBReasoningEngine, REFERENCE_PCB_RECIPE

def test_intent_router_general_question():
    res = IntentRouter.route("What is the weather in Tokyo today?")
    assert res["requires_detection"] is False
    assert res["intent"] == "GENERAL_KNOWLEDGE"
    assert "PCB image" in res["direct_answer"]

def test_intent_router_greeting():
    res = IntentRouter.route("Hello, who are you?")
    assert res["requires_detection"] is False
    assert res["intent"] == "GREETING_HELP"
    assert "Smart PCB Inspection" in res["direct_answer"]

def test_intent_router_visual_queries():
    queries = [
        "How many capacitors exist?",
        "Which component is closest to the transformer?",
        "Are any capacitors missing?",
        "Is the board fully assembled?",
        "Where is the MOSFET located?"
    ]
    for q in queries:
        res = IntentRouter.route(q)
        assert res["requires_detection"] is True
        assert res["intent"] == "VISUAL_INSPECTION"

def test_reasoning_confidence_guardrail():
    engine = PCBReasoningEngine(min_confidence_guardrail=0.35)
    
    # Test zero detections
    empty_det = {
        "total_detections": 0,
        "detections": [],
        "quality_metrics": {"mean_confidence": 0.0}
    }
    res = engine.reason_over_detections("How many capacitors exist?", empty_det)
    assert res["guardrail_triggered"] is True
    assert "Insufficient information" in res["answer"]
    
    # Test low mean confidence below guardrail threshold
    low_conf_det = {
        "total_detections": 2,
        "detections": [
            {"id": 1, "class_name": "Cap1", "confidence": 0.20, "centroid": [100, 100]},
            {"id": 2, "class_name": "Resistor", "confidence": 0.22, "centroid": [200, 200]}
        ],
        "quality_metrics": {"mean_confidence": 0.21}
    }
    res2 = engine.reason_over_detections("How many capacitors exist?", low_conf_det)
    assert res2["guardrail_triggered"] is True
    assert "below the minimum reliability threshold" in res2["answer"]

def test_reasoning_counts_and_subtypes():
    engine = PCBReasoningEngine(min_confidence_guardrail=0.30)
    
    mock_dets = {
        "total_detections": 6,
        "class_counts": {"Cap1": 2, "Cap2": 1, "Cap3": 1, "Cap4": 0, "Transformer": 1, "MOSFET": 1},
        "quality_metrics": {"mean_confidence": 0.88},
        "detections": [
            {"id": 1, "class_name": "Cap1", "confidence": 0.90, "centroid": [100, 100]},
            {"id": 2, "class_name": "Cap1", "confidence": 0.85, "centroid": [150, 100]},
            {"id": 3, "class_name": "Cap2", "confidence": 0.91, "centroid": [200, 100]},
            {"id": 4, "class_name": "Cap3", "confidence": 0.89, "centroid": [250, 100]},
            {"id": 5, "class_name": "Transformer", "confidence": 0.95, "centroid": [500, 500]},
            {"id": 6, "class_name": "MOSFET", "confidence": 0.80, "centroid": [300, 400]}
        ]
    }
    
    res = engine.reason_over_detections("How many capacitors are present?", mock_dets)
    assert res["guardrail_triggered"] is False
    assert res["data"]["total_count"] == 4
    assert "4 capacitors detected" in res["answer"]

def test_reasoning_nearest_neighbor():
    engine = PCBReasoningEngine(min_confidence_guardrail=0.30)
    
    mock_dets = {
        "total_detections": 3,
        "class_counts": {"Transformer": 1, "MOSFET": 1, "Cap1": 1},
        "quality_metrics": {"mean_confidence": 0.90},
        "detections": [
            {"id": 1, "class_name": "Transformer", "confidence": 0.95, "centroid": [500, 500]},
            {"id": 2, "class_name": "MOSFET", "confidence": 0.92, "centroid": [520, 520]}, # Distance sqrt(400+400) = 28.28
            {"id": 3, "class_name": "Cap1", "confidence": 0.88, "centroid": [100, 100]}     # Distance >> 500
        ]
    }
    
    res = engine.reason_over_detections("Which component is closest to the transformer?", mock_dets)
    assert res["guardrail_triggered"] is False
    assert res["data"]["closest_object"]["class_name"] == "MOSFET"
    assert "MOSFET" in res["answer"]
    assert res["data"]["distance_pixels"] == pytest.approx(28.28, rel=1e-2)

def test_reasoning_assembly_completeness():
    engine = PCBReasoningEngine(min_confidence_guardrail=0.30)
    
    # Incomplete board (missing MOV and Resistors)
    incomplete_dets = {
        "total_detections": 5,
        "class_counts": {"Cap1": 2, "Cap2": 1, "Cap3": 1, "Transformer": 1},
        "quality_metrics": {"mean_confidence": 0.85},
        "detections": [
            {"id": 1, "class_name": "Cap1", "confidence": 0.9, "centroid": [10, 10]},
            {"id": 2, "class_name": "Cap1", "confidence": 0.9, "centroid": [20, 10]},
            {"id": 3, "class_name": "Cap2", "confidence": 0.9, "centroid": [30, 10]},
            {"id": 4, "class_name": "Cap3", "confidence": 0.9, "centroid": [40, 10]},
            {"id": 5, "class_name": "Transformer", "confidence": 0.9, "centroid": [100, 100]}
        ]
    }
    
    res = engine.reason_over_detections("Is the board fully assembled?", incomplete_dets)
    assert res["guardrail_triggered"] is False
    assert res["data"]["is_fully_assembled"] is False
    assert "NOT fully assembled" in res["answer"]
