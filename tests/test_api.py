"""
Integration & Unit Tests for FastAPI endpoints and Structured Logging Middleware.
"""

import json
import logging
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    """Verify that the health check endpoint responds with 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_analyze_stance_success():
    """Verify NLP stance scoring endpoint with valid policy text."""
    payload = {
        "statement_text": "The committee noted inflation risks and agreed on rate hike necessity.",
        "negation_window": 3,
    }
    response = client.post("/api/v1/analyze-stance", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "net_stance_score" in data
    assert "stance_label" in data
    assert data["stance_label"] in ["Hawkish", "Dovish", "Neutral"]
    assert "inflation risk" in data["hawkish_signals_found"]


def test_analyze_stance_empty_text():
    """Verify 400 Bad Request handling when statement text is empty/whitespace."""
    payload = {"statement_text": "   "}
    response = client.post("/api/v1/analyze-stance", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "statement_text cannot be empty."


def test_run_scenario_success():
    """Verify integrated rate shock scenario endpoint execution."""
    payload = {
        "statement_text": "Inflation remains elevated due to food price volatility.",
        "repo_rate_change_bps": 50.0,
        "borrowing_transmission_factor": 0.75,
        "deposit_transmission_factor": 0.68,
    }
    response = client.post("/api/v1/run-scenario", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "scenario_summary" in data
    assert "profile_impacts" in data
    # Values returned in scenario_summary are rounded to 1 decimal in the code; compute expected
    assert data["scenario_summary"]["borrowing_rate_shock_bps"] == 37.5  # 50 * 0.75
    assert data["scenario_summary"]["deposit_rate_shock_bps"] == 34.0   # 50 * 0.68


def test_middleware_structured_logging(caplog):
    """Verify that middleware logs HTTP requests in structured JSON format."""
    with caplog.at_level(logging.INFO):
        response = client.get("/health")
        assert response.status_code == 200

    # Locate middleware log entry
    log_messages = [rec.getMessage() for rec in caplog.records if rec.name == "api.main"]
    assert any("HTTP Request Processed" in msg for msg in log_messages)

    # Inspect record extra fields formatted by JSONFormatter
    for record in caplog.records:
        if record.name == "api.main" and "HTTP Request Processed" in record.getMessage():
            assert hasattr(record, "extra")
            extra = record.extra.get("extra", {})
            assert extra.get("method") == "GET"
            assert extra.get("url") == "/health"
            assert extra.get("status_code") == 200
            assert "duration_ms" in extra
