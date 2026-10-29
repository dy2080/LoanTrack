"""
================================================================================
Pydantic Schemas for the FastAPI Loan Prediction API
================================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional


class LoanApplication(BaseModel):
    """Request schema — raw applicant details (human-readable values)."""

    Gender: str = Field(..., example="Male", description="Male / Female")
    Married: str = Field(..., example="Yes", description="Yes / No")
    Dependents: str = Field(..., example="1", description="0, 1, 2, 3+")
    Education: str = Field(..., example="Graduate", description="Graduate / Not Graduate")
    Self_Employed: str = Field(..., example="No", description="Yes / No")
    ApplicantIncome: float = Field(..., example=5000, ge=0)
    CoapplicantIncome: float = Field(..., example=2000, ge=0)
    LoanAmount: float = Field(..., example=150, ge=0)
    Loan_Amount_Term: float = Field(..., example=360, ge=0)
    Credit_History: float = Field(..., example=1.0, ge=0, le=1)
    Property_Area: str = Field(..., example="Urban", description="Urban / Semiurban / Rural")


class PredictionResponse(BaseModel):
    """Response schema — prediction result."""

    status: str = Field(..., description="APPROVED or REJECTED")
    probability: float = Field(..., description="Approval probability 0-1")
    probability_pct: str = Field(..., description="Approval probability as formatted %")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_loaded: bool = False
