import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Smart PCB Inspection API"
    assert "model_architecture" in data

def test_reason_endpoint_greeting():
    response = client.post(
        "/reason",
        data={"question": "Hello, what can you do?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "GREETING_HELP"
    assert data["requires_detection"] is False
    assert "Smart PCB Inspection" in data["answer"]

def test_reason_endpoint_general_question():
    response = client.post(
        "/reason",
        data={"question": "What is the capital of France?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "GENERAL_KNOWLEDGE"
    assert data["requires_detection"] is False
    assert "does not appear to relate" in data["answer"]

def test_reason_endpoint_missing_image_for_visual_query():
    response = client.post(
        "/reason",
        data={"question": "How many capacitors are in this image?"}
    )
    # Since no image file was provided for a visual inspection query, it should return 400
    assert response.status_code == 400
    assert "requires visual inspection" in response.json()["detail"]
