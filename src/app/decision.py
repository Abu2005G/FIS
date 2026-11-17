"""
Decision engine: combines the scorecard with the optional LLM pipeline.

Policy
------
The deterministic scorecard is the decision of record. The LLM is advisory and
can only make the outcome more cautious:

    scorecard   LLM result                    outcome
    ---------   ---------------------------   --------------
    reject      (anything)                    rejected
    review      (anything)                    manual_review
    approve     switched off                  approved
    approve     clear approval, not high risk approved
    approve     concerns, reject or review    manual_review
    approve     failed or invalid output      manual_review

So a model failure, a hallucination, or text planted in a statement to steer
the model can never produce an approval on its own.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.app.ml.features import extract_features_from_transactions, features_to_dict
from src.app.ml.model import APPROVE, REJECT, CreditRiskModel, get_credit_model
from src.app.prompts.pipeline import FinancialIntelligencePipeline

logger = logging.getLogger(__name__)

ENGINE_VERSION = "decision-2.0"

# Same values as src.app.models.LoanStatus; kept as plain strings so this
# module can be used without a database.
APPROVED = "approved"
REJECTED = "rejected"
MANUAL_REVIEW = "manual_review"


def merge_transactions(statements: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Combine statements, dropping rows repeated across overlapping statements."""
    seen = set()
    merged: List[Dict[str, Any]] = []
    for transactions in statements:
        for txn in transactions:
            if not isinstance(txn, dict):
                continue
            key = (
                txn.get("date"),
                txn.get("description"),
                txn.get("withdrawal"),
                txn.get("deposit"),
                txn.get("balance"),
            )
            # Only rows carrying a balance are safely identifiable as duplicates
            if txn.get("balance") is not None:
                if key in seen:
                    continue
                seen.add(key)
            merged.append(txn)
    merged.sort(key=lambda txn: str(txn.get("date", "")))
    return merged


def make_decision(
    *,
    transactions: Sequence[Dict[str, Any]],
    loan_amount: float,
    loan_term_months: int,
    loan_purpose: str = "",
    reconciliation_rate: Optional[float] = None,
    pipeline: Optional[FinancialIntelligencePipeline] = None,
    model: Optional[CreditRiskModel] = None,
) -> Dict[str, Any]:
    """
    Decide one loan application.

    Returns a JSON-serialisable dict that is both the API response body and
    the stored audit record.
    """
    scorecard_model = model or get_credit_model()
    features = extract_features_from_transactions(transactions)
    assessment = scorecard_model.assess(
        features, loan_amount, loan_term_months, reconciliation_rate
    )
    feature_dict = features_to_dict(features)

    reasoning = list(assessment.reasoning)
    red_flags = list(assessment.red_flags)

    llm: Dict[str, Any] = {"status": "disabled", "provider": "none", "model": None}
    if assessment.recommendation == REJECT:
        outcome = REJECTED
        if pipeline is not None:
            llm = {
                "status": "skipped",
                "provider": pipeline.client.provider,
                "model": pipeline.client.model,
                "note": "Not run: the scorecard rejected the application",
            }
    elif pipeline is None:
        outcome = (
            APPROVED
            if assessment.recommendation == APPROVE
            else MANUAL_REVIEW
        )
    else:
        result = pipeline.run(
            transactions=transactions,
            features=feature_dict,
            scorecard={
                "recommendation": assessment.recommendation,
                "credit_score": assessment.credit_score,
                "risk_level": assessment.risk_level,
                "red_flags": assessment.red_flags,
                "affordability": assessment.affordability,
            },
            loan_amount=loan_amount,
            loan_term_months=loan_term_months,
            loan_purpose=loan_purpose,
        )
        llm = result.to_dict()
        summary = result.summary or {}

        if not result.completed:
            outcome = MANUAL_REVIEW
            reasoning.append("LLM analysis did not complete; a person must review this application")
            logger.warning("LLM pipeline failed: %s", "; ".join(result.errors)[:500])
        elif assessment.recommendation != APPROVE:
            outcome = MANUAL_REVIEW
        elif summary.get("supports_approval"):
            outcome = APPROVED
            reasoning.append("LLM analysis agrees with the scorecard")
        else:
            outcome = MANUAL_REVIEW
            reasoning.append("LLM analysis raised concerns the scorecard did not")

        if result.completed:
            for signal in (result.risk or {}).get("risk_signals", []):
                if signal.get("severity") in ("medium", "high"):
                    red_flags.append(f"LLM: {signal.get('signal')} ({signal.get('severity')})")

    return {
        "outcome": outcome,
        "approved": outcome == APPROVED,
        "credit_score": assessment.credit_score,
        "risk_level": assessment.risk_level,
        "reasoning": reasoning,
        "red_flags": red_flags,
        "affordability": assessment.affordability,
        "scorecard": {
            "recommendation": assessment.recommendation,
            "score_breakdown": assessment.score_breakdown,
            "model_version": assessment.model_version,
        },
        "features": feature_dict,
        "statement_quality": {"reconciliation_rate": reconciliation_rate},
        "llm": llm,
        "engine_version": ENGINE_VERSION,
    }
