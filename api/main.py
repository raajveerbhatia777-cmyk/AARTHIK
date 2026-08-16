import sys
import traceback
import logging
from fastapi import FastAPI, HTTPException, status, Security, Depends, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("api.imports")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

analysis_engine = None
_import_errors = []
for candidate in ("analysis.analysis_engine", "analysis_engine", "services.analysis.analysis_engine", "engines.analysis_engine"):
    try:
        mod = __import__(candidate, fromlist=["analysis_engine"])
        if hasattr(mod, "analysis_engine"):
            analysis_engine = getattr(mod, "analysis_engine")
            logger.info("Loaded analysis engine from '%s'", candidate)
            break
        if hasattr(mod, "analyze"):
            analysis_engine = mod
            logger.info("Loaded analysis engine module '%s' exposing analyze()", candidate)
            break
    except Exception as e:
        _import_errors.append((candidate, traceback.format_exc()))

if analysis_engine is None:
    for candidate, tb in _import_errors:
        logger.warning("Failed to import %s: %s", candidate, tb)

    class _LocalStubEngine:
        @staticmethod
        def analyze(text: str):
            return {
                "score": 50,
                "tone": "Neutral",
                "summary": "Placeholder analysis (no production engine available).",
                "key_findings": ["engine:stub"],
                "details": {"stub": True, "input_preview": text[:120]},
            }

    analysis_engine = _LocalStubEngine()
    logger.info("Using local stub analysis_engine for fallback.")

app = FastAPI(title="Policy Resilience API", version="1.0.0")

class StanceRequest(BaseModel):
    statement_text: str

@app.post("/api/v1/analyze-stance", response_class=PlainTextResponse)
def analyze_stance(req: StanceRequest):
    res = analysis_engine.analyze(req.statement_text)
    return f"Score: {res.get('score')}\nTone: {res.get('tone')}\nSummary: {res.get('summary')}"

@app.post("/api/v1/analyze-stance/json", response_class=JSONResponse)
def analyze_stance_json(req: StanceRequest):
    return analysis_engine.analyze(req.statement_text)
