"""
Schemas for LLM output.

Model output is never trusted as-is. Each stage's reply must validate against
one of these schemas, or the stage fails and the application goes to manual
review. In particular ``approved`` must be a real JSON boolean: the string
"false" is rejected rather than being read as truthy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

CATEGORIES = (
    "Income",
    "Expense",
    "Investment",
    "Transfer",
    "Cash Withdrawal",
    "Loan Payment",
)
UNKNOWN_CATEGORY = "Unknown"
_CATEGORY_LOOKUP = {name.lower(): name for name in CATEGORIES}

Level = Literal["low", "medium", "high"]
Recommendation = Literal["approve", "reject", "review"]


class _Output(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _clip_strings(values: Any, limit: int = 20, length: int = 300) -> Any:
    if not isinstance(values, list):
        return values
    return [str(item)[:length] for item in values[:limit]]


# ------------------------------------------------------------------ Stage 1
class StructuredTransaction(_Output):
    index: int = Field(ge=0)
    merchant: str = Field(min_length=1, max_length=200)
    amount: float = Field(ge=0)
    transaction_type: Literal["credit", "debit"]

    _normalise_type = field_validator("transaction_type", mode="before")(_lower)


class StructuringOutput(_Output):
    transactions: List[StructuredTransaction]


# ------------------------------------------------------------------ Stage 2
class ClassifiedTransaction(_Output):
    index: int = Field(ge=0)
    category: str
    merchant_type: str = Field(default="", max_length=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("category", mode="before")
    @classmethod
    def _known_category(cls, value: Any) -> str:
        if isinstance(value, str):
            return _CATEGORY_LOOKUP.get(value.strip().lower(), UNKNOWN_CATEGORY)
        return UNKNOWN_CATEGORY


class ClassificationOutput(_Output):
    transactions: List[ClassifiedTransaction]


# ------------------------------------------------------------------ Stage 3
class BehaviourAnalysis(_Output):
    income_analysis: Dict[str, Any]
    spending_analysis: Dict[str, Any]
    investment_behaviour: Dict[str, Any] = Field(default_factory=dict)
    cash_management: Dict[str, Any] = Field(default_factory=dict)
    debt_obligations: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ Stage 4
class RiskSignal(_Output):
    signal: str = Field(min_length=1, max_length=200)
    severity: Level
    evidence: str = Field(default="", max_length=500)

    _normalise_severity = field_validator("severity", mode="before")(_lower)


class RiskDetection(_Output):
    risk_signals: List[RiskSignal] = Field(default_factory=list, max_length=30)
    risk_level: Level
    risk_score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(default="", max_length=2000)

    _normalise_level = field_validator("risk_level", mode="before")(_lower)


# ------------------------------------------------------------------ Stage 5
class RiskProfile(_Output):
    score: float = Field(ge=0.0, le=1.0)
    level: Level
    key_concerns: List[str] = Field(default_factory=list)
    key_strengths: List[str] = Field(default_factory=list)

    _normalise_level = field_validator("level", mode="before")(_lower)
    _clip = field_validator("key_concerns", "key_strengths", mode="before")(_clip_strings)


class CreditDecision(_Output):
    approved: StrictBool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: List[str] = Field(default_factory=list)

    _clip = field_validator("reasoning", mode="before")(_clip_strings)


class AuditSummary(_Output):
    red_flags: List[str] = Field(default_factory=list)
    green_flags: List[str] = Field(default_factory=list)
    recommendation: Recommendation

    _normalise = field_validator("recommendation", mode="before")(_lower)
    _clip = field_validator("red_flags", "green_flags", mode="before")(_clip_strings)


class IntelligenceSummary(_Output):
    risk_profile: RiskProfile
    credit_decision: CreditDecision
    audit_summary: AuditSummary
    income_profile: Dict[str, Any] = Field(default_factory=dict)
    spending_profile: Dict[str, Any] = Field(default_factory=dict)
    investment_profile: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "IntelligenceSummary":
        approved = self.credit_decision.approved
        recommendation = self.audit_summary.recommendation
        if approved and recommendation == "reject":
            raise ValueError("approved=true contradicts recommendation=reject")
        if not approved and recommendation == "approve":
            raise ValueError("approved=false contradicts recommendation=approve")
        return self

    @property
    def supports_approval(self) -> bool:
        """True only for an unambiguous approval that is not rated high risk."""
        return (
            self.credit_decision.approved
            and self.audit_summary.recommendation == "approve"
            and self.risk_profile.level != "high"
        )
