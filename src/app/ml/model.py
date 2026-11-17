"""
Credit risk scorecard.

A transparent, deterministic points-based scorecard. Every point awarded or
deducted is reported, so a decision can be explained line by line and replayed
later from the stored features.

This is the decision of record. The LLM pipeline is advisory: it can escalate
an application to manual review but can never approve one the scorecard does
not support (see ``src.app.decision``).

The scorecard is hand-built from standard underwriting ratios; it is not a
statistically fitted model. ``scripts/validate_berka.py`` measures how well it
ranks real loans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.app.ml.features import FinancialFeatures

MODEL_VERSION = "scorecard-2.0"

APPROVE = "approve"
REVIEW = "review"
REJECT = "reject"


@dataclass(frozen=True)
class RiskPolicy:
    """Underwriting thresholds. Change these, not the scoring code."""

    # Used only to estimate the monthly instalment of the requested loan
    assumed_annual_rate: float = 0.14
    default_term_months: int = 36

    # Minimum evidence needed before the scorecard is trusted
    min_history_days: int = 60
    min_transactions: int = 15

    # (existing debt payments + new instalment) / monthly income
    max_debt_service_ratio: float = 0.60
    target_debt_service_ratio: float = 0.40

    approve_score: float = 65.0
    reject_score: float = 40.0

    # A statement whose balances do not add up is not scored automatically
    min_reconciliation_rate: float = 0.98


@dataclass
class RiskAssessment:
    """Outcome of scoring one application."""

    recommendation: str
    credit_score: float  # 0-100, higher is safer
    risk_level: str  # low | medium | high
    reasoning: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    affordability: Dict[str, Optional[float]] = field(default_factory=dict)
    model_version: str = MODEL_VERSION

    @property
    def approved(self) -> bool:
        return self.recommendation == APPROVE

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["approved"] = self.approved
        return data


def monthly_instalment(principal: float, annual_rate: float, term_months: int) -> float:
    """Standard amortised monthly payment."""
    if principal <= 0 or term_months <= 0:
        return 0.0
    rate = annual_rate / 12.0
    if rate <= 0:
        return principal / term_months
    return principal * rate / (1.0 - (1.0 + rate) ** -term_months)


def principal_for_instalment(payment: float, annual_rate: float, term_months: int) -> float:
    """Largest principal that a monthly payment can service."""
    if payment <= 0 or term_months <= 0:
        return 0.0
    rate = annual_rate / 12.0
    if rate <= 0:
        return payment * term_months
    return payment * (1.0 - (1.0 + rate) ** -term_months) / rate


def _scale(value: float, worst: float, best: float) -> float:
    """Map value onto 0..1 where `worst` scores 0 and `best` scores 1."""
    if best == worst:
        return 0.0
    return max(0.0, min(1.0, (value - worst) / (best - worst)))


class CreditRiskModel:
    """Points-based credit scorecard."""

    def __init__(self, policy: Optional[RiskPolicy] = None):
        self.policy = policy or RiskPolicy()

    def assess(
        self,
        features: FinancialFeatures,
        loan_amount: float,
        loan_term_months: Optional[int] = None,
        reconciliation_rate: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Score an application.

        Args:
            features: Output of ``extract_features_from_transactions``.
            loan_amount: Amount requested, in the statement's currency.
            loan_term_months: Repayment period; defaults to the policy term.
            reconciliation_rate: Statement quality from the parser, if known.
        """
        policy = self.policy
        term = loan_term_months or policy.default_term_months
        income = features.avg_monthly_income
        reasoning: List[str] = []
        red_flags: List[str] = []
        breakdown: Dict[str, float] = {}

        instalment = monthly_instalment(loan_amount, policy.assumed_annual_rate, term)
        affordability: Dict[str, Optional[float]] = {
            "estimated_monthly_instalment": round(instalment, 2),
            "assumed_annual_rate": policy.assumed_annual_rate,
            "loan_term_months": float(term),
            "debt_service_ratio": None,
            "max_affordable_amount": None,
        }

        # ---- Hard stop: nothing to lend against -------------------------
        if features.transaction_count == 0 or income <= 0:
            return RiskAssessment(
                recommendation=REJECT,
                credit_score=0.0,
                risk_level="high",
                reasoning=["No income could be identified in the statements"],
                red_flags=["No income detected"],
                affordability=affordability,
            )

        debt_service_ratio = (features.monthly_debt_payments + instalment) / income
        headroom = policy.target_debt_service_ratio * income - features.monthly_debt_payments
        max_affordable = principal_for_instalment(headroom, policy.assumed_annual_rate, term)
        affordability["debt_service_ratio"] = round(debt_service_ratio, 4)
        affordability["max_affordable_amount"] = round(max(0.0, max_affordable), 2)

        # ---- 1. Affordability of this loan (25) -------------------------
        breakdown["affordability"] = round(25 * _scale(debt_service_ratio, 0.55, 0.15), 2)
        if debt_service_ratio <= 0.25:
            reasoning.append(
                f"Loan repayments would take {debt_service_ratio:.0%} of monthly income"
            )
        elif debt_service_ratio > policy.target_debt_service_ratio:
            red_flags.append(
                f"Repayments would take {debt_service_ratio:.0%} of monthly income "
                f"(target is under {policy.target_debt_service_ratio:.0%})"
            )

        # ---- 2. Income stability (15) -----------------------------------
        if features.income_stability is None:
            breakdown["income_stability"] = 7.0
        else:
            breakdown["income_stability"] = round(15 * features.income_stability, 2)
            if features.income_stability >= 0.7:
                reasoning.append("Income is steady from month to month")
            elif features.income_stability < 0.4:
                red_flags.append("Income varies widely from month to month")

        # ---- 3. Savings rate (15) ---------------------------------------
        breakdown["savings_rate"] = round(15 * _scale(features.savings_rate, 0.0, 0.25), 2)
        if features.savings_rate >= 0.2:
            reasoning.append(f"Saves {features.savings_rate:.0%} of income")
        elif features.savings_rate < 0:
            red_flags.append("Spending exceeds income over the statement period")

        # ---- 4. Liquidity buffer (15) -----------------------------------
        if features.avg_balance is None:
            breakdown["liquidity"] = 5.0
        else:
            monthly_expense = max(features.avg_monthly_expense, 1.0)
            buffer_months = features.avg_balance / monthly_expense
            breakdown["liquidity"] = round(15 * _scale(buffer_months, 0.0, 2.0), 2)
            if buffer_months >= 1.5:
                reasoning.append(
                    f"Average balance covers {buffer_months:.1f} months of spending"
                )
            elif features.avg_balance <= 0:
                red_flags.append("Average balance is zero or negative")

        # ---- 5. Overdraft behaviour (15) --------------------------------
        overdrawn = features.negative_balance_days
        if features.min_balance is None:
            breakdown["overdrafts"] = 7.0
        elif overdrawn == 0:
            breakdown["overdrafts"] = 15.0
            reasoning.append("Account was never overdrawn")
        else:
            breakdown["overdrafts"] = 8.0 if overdrawn <= 2 else 3.0 if overdrawn <= 5 else 0.0
            red_flags.append(f"Account was overdrawn on {overdrawn} day(s)")

        # ---- 6. Existing debt (10) --------------------------------------
        breakdown["existing_debt"] = round(10 * _scale(features.debt_to_income, 0.40, 0.10), 2)
        if features.debt_to_income > 0.30:
            red_flags.append(
                f"Existing debt payments already take {features.debt_to_income:.0%} of income"
            )
        elif features.debt_to_income <= 0.10:
            reasoning.append("Little or no existing debt")

        # ---- 7. Balance trend (5) ---------------------------------------
        if features.balance_trend is None:
            breakdown["balance_trend"] = 2.5
        else:
            breakdown["balance_trend"] = round(5 * _scale(features.balance_trend, -0.20, 0.0), 2)
            if features.balance_trend < -0.20:
                red_flags.append("Balance is falling steadily")

        # ---- Penalties ---------------------------------------------------
        if features.returned_payments:
            breakdown["returned_payments"] = -float(min(8 * features.returned_payments, 24))
            red_flags.append(
                f"{features.returned_payments} returned payment(s) or penalty charge(s)"
            )
        if features.gambling_transactions:
            breakdown["gambling"] = -10.0 if features.gambling_transactions >= 5 else -5.0
            red_flags.append(f"{features.gambling_transactions} gambling transaction(s)")

        score = round(max(0.0, min(100.0, sum(breakdown.values()))), 2)
        risk_level = "low" if score >= 70 else "medium" if score >= 50 else "high"

        # ---- Decision ----------------------------------------------------
        thin_file = (
            features.history_days < policy.min_history_days
            or features.transaction_count < policy.min_transactions
        )
        unreliable = (
            reconciliation_rate is not None
            and reconciliation_rate < policy.min_reconciliation_rate
        )

        if debt_service_ratio > policy.max_debt_service_ratio:
            recommendation = REJECT
            reasoning.insert(
                0,
                "Requested loan is not affordable: repayments would take "
                f"{debt_service_ratio:.0%} of monthly income",
            )
        elif unreliable:
            recommendation = REVIEW
            red_flags.append(
                f"Statement balances reconcile for only {reconciliation_rate:.0%} of "
                "transactions; the statement may be misread or altered"
            )
        elif score < policy.reject_score:
            recommendation = REJECT
        elif thin_file:
            recommendation = REVIEW
            red_flags.append(
                f"Only {features.history_days} day(s) and {features.transaction_count} "
                "transaction(s) of history; too little to decide automatically"
            )
        elif score >= policy.approve_score and not red_flags:
            recommendation = APPROVE
        else:
            recommendation = REVIEW

        summary = {
            APPROVE: "Strong financial profile",
            REVIEW: "Mixed financial profile; needs a manual review",
            REJECT: "Financial profile does not support this loan",
        }[recommendation]
        reasoning.insert(0, f"{summary} (score {score:.0f}/100)")

        return RiskAssessment(
            recommendation=recommendation,
            credit_score=score,
            risk_level=risk_level,
            reasoning=reasoning,
            red_flags=red_flags,
            score_breakdown=breakdown,
            affordability=affordability,
        )


_credit_model: Optional[CreditRiskModel] = None


def get_credit_model() -> CreditRiskModel:
    """Shared scorecard instance. The model holds no per-request state."""
    global _credit_model
    if _credit_model is None:
        _credit_model = CreditRiskModel()
    return _credit_model
