"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoanApplicationCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    applicant_name: str = Field(min_length=1, max_length=200)
    applicant_email: str = Field(
        min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )
    loan_amount: float = Field(gt=0, le=1e12, allow_inf_nan=False)
    loan_term_months: int = Field(default=36, ge=1, le=480)
    loan_purpose: Optional[str] = Field(default=None, max_length=200)

    @field_validator("applicant_email")
    @classmethod
    def _lowercase_email(cls, value: str) -> str:
        return value.lower()


class LoanApplicationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    applicant_name: str
    applicant_email: str
    loan_amount: float
    loan_term_months: int
    loan_purpose: Optional[str]
    status: str
    credit_score: Optional[float]
    risk_level: Optional[str]
    created_at: datetime
    updated_at: datetime


class StatementUploadResponse(BaseModel):
    statement_id: int
    application_id: int
    transactions_found: int
    period_start: Optional[str]
    period_end: Optional[str]
    total_deposits: float
    total_withdrawals: float
    quality: Dict[str, Any]
    warnings: List[str]


class StatementResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    statement_type: str
    original_filename: Optional[str]
    transaction_count: int
    total_deposits: float
    total_withdrawals: float
    period_start: Optional[str]
    period_end: Optional[str]
    created_at: datetime


class LoanDecisionResponse(BaseModel):
    decision_id: int
    application_id: int
    outcome: str
    approved: bool
    credit_score: float
    risk_level: str
    reasoning: List[str]
    red_flags: List[str]
    affordability: Dict[str, Any]
    scorecard: Dict[str, Any]
    features: Dict[str, Any]
    statement_quality: Dict[str, Any]
    llm: Dict[str, Any]
    engine_version: str
    created_at: datetime


class DecisionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    outcome: str
    credit_score: float
    risk_level: str
    model_version: str
    llm_provider: str
    llm_model: Optional[str]
    created_at: datetime
