"""
================================================================================
FastAPI Application — Loan Eligibility Prediction API
================================================================================

Endpoints:
    GET  /              → Serves the web UI
    GET  /health        → Health check
    POST /predict       → Predict loan eligibility
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import LoanApplication, PredictionResponse, HealthResponse
from src.loan_ml.inference.predictor import predict_single, load_model
from config.settings import DEFAULT_MODEL

# ── App setup ──
app = FastAPI(
    title="LoanTrack API",
    description="LoanTrack — Machine Learning Loan Eligibility & Bank Recommendation System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static files (CSS, JS) ──
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Global state ──
_model_loaded = False


@app.on_event("startup")
async def startup():
    """Validate model file exists on startup."""
    global _model_loaded
    if os.path.exists(DEFAULT_MODEL):
        try:
            load_model(DEFAULT_MODEL)
            _model_loaded = True
            print(f"[INFO] Model loaded from '{DEFAULT_MODEL}'")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
    else:
        print(
            f"[WARN] Model file not found at '{DEFAULT_MODEL}'. "
            "Run `python cli.py train` first."
        )


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=_model_loaded)


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """Predict loan eligibility from raw applicant details."""
    if not os.path.exists(DEFAULT_MODEL):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python cli.py train` first.",
        )

    raw = application.model_dump()

    try:
        result = predict_single(raw, model_filename=DEFAULT_MODEL, apply_log=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        status=result["status"],
        probability=result["probability"],
        probability_pct=f"{result['probability'] * 100:.1f}%",
    )
