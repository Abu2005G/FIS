"""
Feature extraction from bank transactions for credit risk assessment.

Input is the normalised transaction list produced by
``src.app.ml.statement_parser``. All ratios are unit-free, so the features work
for any currency.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.app.ml.statement_parser import parse_date, to_number

DAYS_PER_MONTH = 30.4375

_SALARY = re.compile(r"salary|payroll|wages?\b|stipend", re.IGNORECASE)
_BENEFIT = re.compile(r"pension|benefit|annuity|social\s*security", re.IGNORECASE)
_TRANSFER = re.compile(r"transfer|neft|imps|rtgs|upi|tpt", re.IGNORECASE)
_CASH_IN = re.compile(r"cash\s*dep|cash\s*deposit", re.IGNORECASE)

# Money in that is not earned income
_NOT_INCOME = re.compile(
    r"refund|reversal|\brev\b|cashback|charge\s*back|loan\s*disb|disbursement"
    r"|interest\s*(credit|paid|pd)|int\.?\s*pd",
    re.IGNORECASE,
)

_ESSENTIAL = re.compile(
    r"rent\b|mortgage|utilit|electric|water\s*bill|\bgas\b|grocer|supermarket|food"
    r"|transport|fuel|petrol|medical|pharmacy|hospital|insurance|school|tuition"
    r"|household",
    re.IGNORECASE,
)

_DEBT = re.compile(
    r"\bloan\b|\bemi\b|mortgage|repayment|credit\s*card|\bcc\s*payment|instal?ment",
    re.IGNORECASE,
)

_CASH_OUT = re.compile(
    r"\batm\b|cash\s*withdraw|cash\s*wdl|\bcwdr\b|\bnwd\b|\bawb\b|card\s*withdraw",
    re.IGNORECASE,
)

_GAMBLING = re.compile(
    r"casino|gambl|betting|\bbet365|betway|dream11|my11circle|rummy|poker|lottery"
    r"|winzo|\bmpl\b|sportsbook|\bstake\b",
    re.IGNORECASE,
)

_RETURNED = re.compile(
    r"(chq|cheque|ecs|ach|nach|emi|payment)\s*(ret|rtn|return|bounce)"
    r"|bounce|dishono|insufficient\s*fund|penal|sanction\s*interest"
    r"|overdraft\s*(fee|charge|interest)|\bod\s*(fee|charge|interest)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FinancialFeatures:
    """Summary of an applicant's account behaviour."""

    # History
    transaction_count: int
    history_days: int
    months_covered: float

    # Income
    total_income: float
    avg_monthly_income: float
    income_stability: Optional[float]  # 0..1, higher is steadier; None if too little history
    income_sources: int

    # Spending
    total_expenses: float
    avg_monthly_expense: float
    essential_expense_ratio: float
    avg_transaction_size: float
    large_transactions: int

    # Balances (None when the statement carries no running balance)
    avg_balance: Optional[float]
    min_balance: Optional[float]
    max_balance: Optional[float]
    balance_trend: Optional[float]  # monthly balance change as a share of monthly income
    negative_balance_days: int

    # Obligations and ratios
    monthly_debt_payments: float
    debt_to_income: float
    savings_rate: float
    expense_to_income: float

    # Behavioural signals
    cash_withdrawal_ratio: float
    gambling_transactions: int
    returned_payments: int


@dataclass(frozen=True)
class _Txn:
    when: date
    description: str
    withdrawal: float
    deposit: float
    balance: Optional[float]


def _prepare(transactions: Sequence[Dict[str, Any]]) -> List[_Txn]:
    """Validate, convert and date-order raw transaction dicts."""
    prepared: List[Tuple[date, int, _Txn]] = []
    for index, raw in enumerate(transactions):
        if not isinstance(raw, dict):
            continue
        when = parse_date(raw.get("date"))
        if when is None:
            continue
        withdrawal = abs(to_number(raw.get("withdrawal")) or 0.0)
        deposit = abs(to_number(raw.get("deposit")) or 0.0)
        if withdrawal == 0.0 and deposit == 0.0:
            continue
        txn = _Txn(
            when=when,
            description=str(raw.get("description") or ""),
            withdrawal=withdrawal,
            deposit=deposit,
            balance=to_number(raw.get("balance")),
        )
        prepared.append((when, index, txn))
    # Stable sort: keeps the statement's own order within a day
    prepared.sort(key=lambda item: (item[0], item[1]))
    return [txn for _, _, txn in prepared]


def _month_keys(first: date, last: date) -> List[Tuple[int, int]]:
    keys = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        keys.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return keys


def _income_stability(
    income_by_month: Dict[Tuple[int, int], float], first: date, last: date
) -> Optional[float]:
    """1 minus the coefficient of variation of monthly income, floored at 0."""
    keys = _month_keys(first, last)
    # Drop partial months at the edges when enough full months remain
    if len(keys) >= 4:
        if first.day > 5:
            keys = keys[1:]
        if last.day < 25:
            keys = keys[:-1]
    if len(keys) < 2:
        return None
    values = [income_by_month.get(key, 0.0) for key in keys]
    mean = statistics.fmean(values)
    if mean <= 0:
        return 0.0
    variation = statistics.pstdev(values) / mean
    return round(max(0.0, 1.0 - min(variation, 1.0)), 4)


def _balance_statistics(
    txns: Sequence[_Txn],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
    """Time-weighted average, extremes, monthly slope and days overdrawn."""
    balances = [txn.balance for txn in txns if txn.balance is not None]
    if not balances:
        return None, None, None, None, 0

    end_of_day: Dict[date, float] = {}
    negative_days = set()
    for txn in txns:
        if txn.balance is None:
            continue
        end_of_day[txn.when] = txn.balance
        if txn.balance < 0:
            negative_days.add(txn.when)

    days = sorted(end_of_day)
    if len(days) == 1:
        average = end_of_day[days[0]]
        slope = None
    else:
        weighted = 0.0
        for today, tomorrow in zip(days, days[1:], strict=False):
            weighted += end_of_day[today] * (tomorrow - today).days
        span = (days[-1] - days[0]).days
        average = weighted / span if span else end_of_day[days[-1]]

        xs = [(day - days[0]).days for day in days]
        ys = [end_of_day[day] for day in days]
        x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
            if denominator
            else None
        )

    return average, min(balances), max(balances), slope, len(negative_days)


def empty_features() -> FinancialFeatures:
    return FinancialFeatures(
        transaction_count=0,
        history_days=0,
        months_covered=0.0,
        total_income=0.0,
        avg_monthly_income=0.0,
        income_stability=None,
        income_sources=0,
        total_expenses=0.0,
        avg_monthly_expense=0.0,
        essential_expense_ratio=0.0,
        avg_transaction_size=0.0,
        large_transactions=0,
        avg_balance=None,
        min_balance=None,
        max_balance=None,
        balance_trend=None,
        negative_balance_days=0,
        monthly_debt_payments=0.0,
        debt_to_income=0.0,
        savings_rate=0.0,
        expense_to_income=0.0,
        cash_withdrawal_ratio=0.0,
        gambling_transactions=0,
        returned_payments=0,
    )


def extract_features_from_transactions(
    transactions: Sequence[Dict[str, Any]],
) -> FinancialFeatures:
    """
    Compute credit features from transactions.

    Args:
        transactions: Dicts with date, description, withdrawal, deposit, balance.
    """
    txns = _prepare(transactions)
    if not txns:
        return empty_features()

    first, last = txns[0].when, txns[-1].when
    history_days = (last - first).days + 1
    months = max(1.0, history_days / DAYS_PER_MONTH)

    income_by_month: Dict[Tuple[int, int], float] = defaultdict(float)
    income_sources = set()
    total_income = total_expenses = 0.0
    essential = debt = cash_out = 0.0
    gambling = returned = 0

    for txn in txns:
        text = txn.description
        if txn.deposit and not _NOT_INCOME.search(text):
            total_income += txn.deposit
            income_by_month[(txn.when.year, txn.when.month)] += txn.deposit
            if _SALARY.search(text):
                income_sources.add("salary")
            elif _BENEFIT.search(text):
                income_sources.add("benefit")
            elif _CASH_IN.search(text):
                income_sources.add("cash")
            elif _TRANSFER.search(text):
                income_sources.add("transfer")
            else:
                income_sources.add("other")

        if txn.withdrawal:
            total_expenses += txn.withdrawal
            if _DEBT.search(text):
                debt += txn.withdrawal
            elif _ESSENTIAL.search(text):
                essential += txn.withdrawal
            if _CASH_OUT.search(text):
                cash_out += txn.withdrawal
            if _GAMBLING.search(text):
                gambling += 1

        if _RETURNED.search(text):
            returned += 1

    avg_monthly_income = total_income / months
    avg_monthly_expense = total_expenses / months
    monthly_debt = debt / months

    average, lowest, highest, slope, negative_days = _balance_statistics(txns)
    balance_trend: Optional[float] = None
    if slope is not None:
        scale = avg_monthly_income if avg_monthly_income > 0 else max(abs(average or 0.0), 1.0)
        balance_trend = round(slope * DAYS_PER_MONTH / scale, 4)

    amounts = [txn.deposit or txn.withdrawal for txn in txns]
    large_threshold = avg_monthly_income * 0.5
    large = sum(1 for amount in amounts if large_threshold > 0 and amount > large_threshold)

    return FinancialFeatures(
        transaction_count=len(txns),
        history_days=history_days,
        months_covered=round(months, 2),
        total_income=round(total_income, 2),
        avg_monthly_income=round(avg_monthly_income, 2),
        income_stability=_income_stability(income_by_month, first, last),
        income_sources=len(income_sources),
        total_expenses=round(total_expenses, 2),
        avg_monthly_expense=round(avg_monthly_expense, 2),
        essential_expense_ratio=round(essential / total_expenses, 4) if total_expenses else 0.0,
        avg_transaction_size=round(statistics.fmean(amounts), 2),
        large_transactions=large,
        avg_balance=round(average, 2) if average is not None else None,
        min_balance=lowest,
        max_balance=highest,
        balance_trend=balance_trend,
        negative_balance_days=negative_days,
        monthly_debt_payments=round(monthly_debt, 2),
        debt_to_income=round(debt / total_income, 4) if total_income else 0.0,
        savings_rate=(
            round((total_income - total_expenses) / total_income, 4) if total_income else 0.0
        ),
        expense_to_income=round(total_expenses / total_income, 4) if total_income else 0.0,
        cash_withdrawal_ratio=round(cash_out / total_expenses, 4) if total_expenses else 0.0,
        gambling_transactions=gambling,
        returned_payments=returned,
    )


def features_to_dict(features: FinancialFeatures) -> Dict[str, Any]:
    """Convert FinancialFeatures to a plain dictionary."""
    return asdict(features)
