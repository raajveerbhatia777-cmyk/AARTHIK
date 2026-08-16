"""
FastAPI Application for Monetary Policy & Financial Resilience Engine.

Exposes REST API endpoints for standalone policy stance analysis,
integrated scenario simulation, and economic profile querying.
This version offloads persistence to FastAPI BackgroundTasks and initializes
the SQLAlchemy-backed storage on startup.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai.policy_analyser import PolicyAnalyser
from analysis.scenario_analysis import ScenarioEngine
from models.resilience_model import ResilienceEngine

from storage.db import init_db, SessionLocal
from storage.models import PolicyAnalysisLog, ScenarioRunLog
from sqlalchemy.orm import Session

app = FastAPI(
    title="Monetary Policy & Resilience API",
    version="1.0.0",
    description="REST API for RBI policy stance NLP scoring, interest rate transmission, and household resilience analysis.",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engines
policy_analyser = PolicyAnalyser()
scenario_engine = ScenarioEngine()
resilience_engine = ResilienceEngine()


# ------------------ Request & Response Schemas ------------------

class PolicyStanceRequest(BaseModel):
    statement_text: str = Field(
        ...,
        description="Raw text of the central bank policy statement, speech, or meeting minutes.",
        example="Inflation risk remains elevated due to food price volatility, requiring a rate hike.",
    )
    negation_window: Optional[int] = Field(
        default=3,
        description="Number of preceding words to check for negation modifiers.",
    )


class ScenarioRequest(BaseModel):
    statement_text: str = Field(
        ...,
        description="Text of the policy statement for NLP stance scoring.",
        example="The MPC noted persistent inflation risks, requiring sustained vigilance.",
    )
    repo_rate_change_bps: float = Field(
        ...,
        description="Policy rate adjustment in basis points (e.g., +50.0 or -25.0).",
        example=50.0,
    )
    borrowing_transmission_factor: float = Field(
        default=0.75,
        description="Pass-through rate to lending products (0.0 to 1.0).",
        example=0.75,
    )
    deposit_transmission_factor: float = Field(
        default=0.68,
        description="Pass-through rate to deposit products (0.0 to 1.0).",
        example=0.68,
    )


# ------------------ Startup & Background Tasks ------------------

@app.on_event("startup")
def startup_event():
    init_db()


def log_analysis_task(statement_text: str, result: Dict[str, Any]):
    """Background task to persist policy stance analysis."""
    db: Session = SessionLocal()
    try:
        analysis = PolicyAnalysisLog(
            raw_text=statement_text,
            stance_label=result.get("stance_label", "Neutral"),
            net_stance_score=result.get("net_stance_score", 50.0),
            weighted_hawkish_score=result.get("weighted_hawkish_score", 0.0),
            weighted_dovish_score=result.get("weighted_dovish_score", 0.0),
            hawkish_signals=result.get("hawkish_signals_found", []),
            dovish_signals=result.get("dovish_signals_found", []),
        )
        db.add(analysis)
        db.commit()
    finally:
        db.close()


def log_scenario_task(statement_text: str, payload_data: Dict[str, Any], results: Dict[str, Any]):
    """Background task to persist both analysis and scenario run."""
    db: Session = SessionLocal()
    try:
        analyser = PolicyAnalyser()
        policy_result = analyser.analyze_statement(statement_text)

        analysis = PolicyAnalysisLog(
            raw_text=statement_text,
            stance_label=policy_result.get("stance_label", "Neutral"),
            net_stance_score=policy_result.get("net_stance_score", 50.0),
            weighted_hawkish_score=policy_result.get("weighted_hawkish_score", 0.0),
            weighted_dovish_score=policy_result.get("weighted_dovish_score", 0.0),
            hawkish_signals=policy_result.get("hawkish_signals_found", []),
            dovish_signals=policy_result.get("dovish_signals_found", []),
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        scenario_record = ScenarioRunLog(
            policy_analysis_id=analysis.id,
            repo_rate_change_bps=payload_data["repo_rate_change_bps"],
            borrowing_transmission_factor=payload_data["borrowing_transmission_factor"],
            deposit_transmission_factor=payload_data["deposit_transmission_factor"],
            profile_impacts_summary=results.get("profile_impacts", {}),
        )
        db.add(scenario_record)
        db.commit()
    finally:
        db.close()


# ------------------ API Endpoints ------------------

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify service availability."""
    return {"status": "healthy", "service": "Monetary Policy & Resilience API"}


@app.post("/api/v1/analyze-stance", tags=["NLP Policy Analyser"]) 
def analyze_policy_stance(payload: PolicyStanceRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Parses a policy statement, detects hawkish/dovish signals, handles negation,
    and returns a weighted stance score (0-100). Persists analysis in background.
    """
    if not payload.statement_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="statement_text cannot be empty.",
        )

    analyser = PolicyAnalyser(negation_window=payload.negation_window)
    result = analyser.analyze_statement(payload.statement_text)

    # Offload DB write to background
    background_tasks.add_task(log_analysis_task, payload.statement_text, result)
    return result


@app.post("/api/v1/run-scenario", tags=["Scenario Engine"]) 
def run_integrated_scenario(payload: ScenarioRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Executes an end-to-end simulation combining policy stance scoring, interest rate
    transmission shocks, and financial resilience evaluation across economic profiles.
    Persists results asynchronously.
    """
    if not payload.statement_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="statement_text cannot be empty.",
        )

    results = scenario_engine.run_scenario(
        statement_text=payload.statement_text,
        repo_rate_change_bps=payload.repo_rate_change_bps,
        borrowing_transmission_factor=payload.borrowing_transmission_factor,
        deposit_transmission_factor=payload.deposit_transmission_factor,
    )

    # Offload DB write to background
    background_tasks.add_task(
        log_scenario_task,
        payload.statement_text,
        payload.dict(),
        results,
    )
    return results


@app.get("/api/v1/profiles", tags=["Resilience Engine"]) 
def list_default_profiles() -> Dict[str, Any]:
    """Returns baseline financial configurations for built-in economic profiles."""
    profiles = resilience_engine.default_profiles
    return {
        key: {
            "name": prof.name,
            "category": prof.category,
            "monthly_income": prof.monthly_income,
            "liquid_savings": prof.liquid_savings,
            "floating_debt_principal": prof.floating_debt_principal,
            "fixed_income_assets": prof.fixed_income_assets,
            "current_borrowing_rate": prof.current_borrowing_rate,
            "current_deposit_rate": prof.current_deposit_rate,
        }
        for key, prof in profiles.items()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
