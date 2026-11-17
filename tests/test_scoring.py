"""Feature extraction and scorecard behaviour."""

import pytest

from src.app.ml.features import extract_features_from_transactions, features_to_dict
from src.app.ml.model import (
    APPROVE,
    REJECT,
    REVIEW,
    CreditRiskModel,
    monthly_instalment,
    principal_for_instalment,
)
from tests.conftest import statement_rows

model = CreditRiskModel()


def distressed_rows():
    """Irregular income, spending above income, overdrafts and bounced payments."""
    rows = []
    balance = 3000.0
    plan = [
        (1, 5, "CASH DEPOSIT", None, 22000.0),
        (1, 9, "RENT PAYMENT", 15000.0, None),
        (1, 12, "DREAM11 GAMING", 6000.0, None),
        (1, 20, "ATM CASH WITHDRAWAL", 9000.0, None),
        (2, 3, "EMI RETURN CHARGES INSUFFICIENT FUNDS", 600.0, None),
        (2, 14, "PERSONAL LOAN EMI", 9000.0, None),
        (2, 25, "FREELANCE TRANSFER", None, 9000.0),
        (3, 2, "RENT PAYMENT", 15000.0, None),
        (3, 8, "CHQ RETURN CHARGES", 600.0, None),
        (3, 18, "CASH DEPOSIT", None, 41000.0),
        (3, 21, "PERSONAL LOAN EMI", 9000.0, None),
        (3, 27, "ATM CASH WITHDRAWAL", 12000.0, None),
        (4, 4, "RENT PAYMENT", 15000.0, None),
        (4, 9, "DREAM11 GAMING", 4000.0, None),
        (4, 15, "PERSONAL LOAN EMI", 9000.0, None),
        (4, 22, "SMALL TRANSFER IN", None, 6000.0),
    ]
    for month, day, description, withdrawal, deposit in plan:
        balance += (deposit or 0) - (withdrawal or 0)
        rows.append(
            {
                "date": f"2026-{month:02d}-{day:02d}",
                "description": description,
                "withdrawal": withdrawal,
                "deposit": deposit,
                "balance": balance,
            }
        )
    return rows


# ---------------------------------------------------------------- features
def test_income_and_expenses_are_averaged_per_month():
    features = extract_features_from_transactions(statement_rows(months=4, salary=60000.0))

    assert features.transaction_count == 28
    assert features.total_income == 240000.0
    assert features.avg_monthly_income == pytest.approx(60000.0, rel=0.06)
    assert features.income_stability == 1.0
    assert features.negative_balance_days == 0
    assert 0.4 < features.savings_rate < 0.6


def test_transactions_are_ordered_by_real_date_not_by_text():
    rows = [
        {"date": "05/02/26", "description": "B", "withdrawal": 50, "balance": 950},
        {"date": "20/01/26", "description": "A", "deposit": 1000, "balance": 1000},
    ]
    features = extract_features_from_transactions(rows)

    assert features.history_days == 17
    assert features.balance_trend is not None and features.balance_trend < 0


def test_refunds_and_loan_disbursals_are_not_income():
    rows = statement_rows(months=2)
    rows.append({"date": "2026-02-26", "description": "REFUND AMAZON", "deposit": 9000.0,
                 "balance": rows[-1]["balance"] + 9000.0})

    assert extract_features_from_transactions(rows).total_income == 120000.0


def test_risk_signals_are_detected():
    features = extract_features_from_transactions(distressed_rows())

    assert features.gambling_transactions == 2
    assert features.returned_payments == 2
    assert features.negative_balance_days > 0
    assert features.monthly_debt_payments > 0
    assert features.savings_rate < 0
    assert features.cash_withdrawal_ratio > 0.15


def test_empty_and_malformed_input_does_not_crash():
    assert extract_features_from_transactions([]).transaction_count == 0
    junk = [{"date": "not a date", "deposit": 5}, "text", {"date": "2026-01-01"}, None]
    assert extract_features_from_transactions(junk).transaction_count == 0


def test_features_are_json_serialisable():
    import json

    json.dumps(features_to_dict(extract_features_from_transactions(statement_rows())))


# --------------------------------------------------------------- scorecard
def test_strong_profile_is_approved():
    features = extract_features_from_transactions(statement_rows(months=4))
    result = model.assess(features, loan_amount=200000.0, loan_term_months=36)

    assert result.recommendation == APPROVE
    assert result.risk_level == "low"
    assert not result.red_flags
    assert result.credit_score == pytest.approx(sum(result.score_breakdown.values()))


def test_distressed_profile_is_rejected_with_reasons():
    features = extract_features_from_transactions(distressed_rows())
    result = model.assess(features, loan_amount=200000.0, loan_term_months=36)

    assert result.recommendation == REJECT
    assert result.risk_level == "high"
    assert any("gambling" in flag for flag in result.red_flags)
    assert any("returned" in flag for flag in result.red_flags)


def test_decision_depends_on_the_amount_requested():
    features = extract_features_from_transactions(statement_rows(months=4))

    small = model.assess(features, 100000.0, 36)
    huge = model.assess(features, 50000000.0, 36)

    assert small.recommendation == APPROVE
    assert huge.recommendation == REJECT
    assert huge.credit_score < small.credit_score
    assert any("not affordable" in reason for reason in huge.reasoning)
    assert small.affordability["max_affordable_amount"] < 50000000.0


def test_score_never_rises_as_the_loan_grows():
    features = extract_features_from_transactions(statement_rows(months=4))
    scores = [model.assess(features, amount, 36).credit_score
              for amount in (1e4, 1e5, 5e5, 1e6, 5e6)]

    assert scores == sorted(scores, reverse=True)


def test_no_income_is_rejected():
    rows = [{"date": f"2026-01-{day:02d}", "description": "SHOP", "withdrawal": 100.0,
             "balance": 5000.0 - day * 100} for day in range(1, 20)]
    result = model.assess(extract_features_from_transactions(rows), 10000.0)

    assert result.recommendation == REJECT
    assert result.credit_score == 0.0


def test_thin_history_goes_to_manual_review():
    features = extract_features_from_transactions(statement_rows(months=1))
    result = model.assess(features, 100000.0, 36)

    assert result.recommendation == REVIEW
    assert any("too little" in flag for flag in result.red_flags)


def test_unreconciled_statement_goes_to_manual_review():
    features = extract_features_from_transactions(statement_rows(months=4))
    result = model.assess(features, 100000.0, 36, reconciliation_rate=0.5)

    assert result.recommendation == REVIEW
    assert any("reconcile" in flag for flag in result.red_flags)


def test_instalment_maths_round_trips():
    payment = monthly_instalment(100000.0, 0.12, 12)

    assert payment == pytest.approx(8884.88, abs=0.01)
    assert principal_for_instalment(payment, 0.12, 12) == pytest.approx(100000.0)
    assert monthly_instalment(1200.0, 0.0, 12) == 100.0
