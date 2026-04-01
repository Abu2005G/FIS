"""
Feature extraction from financial transactions for credit risk assessment.
"""
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Transaction:
    date: str
    description: str
    withdrawal: Optional[float]
    deposit: Optional[float]
    balance: Optional[float]


@dataclass
class FinancialFeatures:
    """Extracted features for ML model"""
    # Income features
    total_income_3m: float
    avg_monthly_income: float
    income_stability: float  # coefficient of variation
    income_sources: int

    # Expense features
    total_expenses_3m: float
    avg_monthly_expense: float
    essential_expense_ratio: float

    # Balance features
    avg_balance: float
    min_balance: float
    max_balance: float
    balance_trend: float  # slope of balance over time
    negative_balance_count: int

    # Transaction patterns
    transaction_count: int
    avg_transaction_size: float
    large_transactions: int  # > 50% of avg income
    frequent_small_deposits: int  # possible structuring

    # Ratios
    debt_to_income: float
    savings_rate: float
    expense_to_income: float

    # Derived score components
    creditworthiness_score: float


def parse_amount(amount_str: str) -> Optional[float]:
    """Parse amount string to float, handling commas."""
    if amount_str is None or amount_str == "":
        return None
    try:
        return float(str(amount_str).replace(",", ""))
    except (ValueError, TypeError):
        return None


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_str, "%d/%m/%y")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None


def extract_features_from_transactions(transactions: List[Dict[str, Any]]) -> FinancialFeatures:
    """
    Extract ML features from a list of transactions.

    Args:
        transactions: List of transaction dicts with date, description, withdrawal, deposit, balance

    Returns:
        FinancialFeatures object with computed features
    """
    if not transactions:
        return FinancialFeatures(
            total_income_3m=0, avg_monthly_income=0, income_stability=0, income_sources=0,
            total_expenses_3m=0, avg_monthly_expense=0, essential_expense_ratio=0,
            avg_balance=0, min_balance=0, max_balance=0, balance_trend=0, negative_balance_count=0,
            transaction_count=0, avg_transaction_size=0, large_transactions=0, frequent_small_deposits=0,
            debt_to_income=0, savings_rate=0, expense_to_income=0,
            creditworthiness_score=0
        )

    # Parse transactions
    parsed_txns = []
    for t in transactions:
        txn = Transaction(
            date=t.get("date", ""),
            description=t.get("description", ""),
            withdrawal=parse_amount(t.get("withdrawal")),
            deposit=parse_amount(t.get("deposit")),
            balance=parse_amount(t.get("balance"))
        )
        parsed_txns.append(txn)

    # Sort by date
    parsed_txns.sort(key=lambda x: x.date)

    # Group by month
    monthly_data = defaultdict(lambda: {"income": [], "expenses": []})
    for txn in parsed_txns:
        month_key = txn.date[:5] if len(txn.date) >= 5 else txn.date  # MM/YY
        if txn.deposit:
            monthly_data[month_key]["income"].append(txn.deposit)
        if txn.withdrawal:
            monthly_data[month_key]["expenses"].append(txn.withdrawal)

    # Calculate monthly income
    monthly_incomes = [sum(data["income"]) for data in monthly_data.values()]
    monthly_expenses = [sum(data["expenses"]) for data in monthly_data.values()]

    total_income = sum(monthly_incomes) if monthly_incomes else 0
    total_expenses = sum(monthly_expenses) if monthly_expenses else 0
    avg_monthly_income = total_income / len(monthly_incomes) if monthly_incomes else 0
    avg_monthly_expense = total_expenses / len(monthly_expenses) if monthly_expenses else 0

    # Income stability (coefficient of variation: std/mean)
    if monthly_incomes and len(monthly_incomes) > 1 and avg_monthly_income > 0:
        mean_income = avg_monthly_income
        variance = sum((x - mean_income) ** 2 for x in monthly_incomes) / len(monthly_incomes)
        std_income = variance ** 0.5
        income_stability = 1 - min(std_income / mean_income, 1)  # Higher is more stable
    else:
        income_stability = 0.5 if monthly_incomes else 0

    # Income sources (unique deposit descriptions)
    income_descriptions = set()
    for txn in parsed_txns:
        if txn.deposit and txn.deposit > 100:  # Filter small deposits
            desc = txn.description.lower()
            # Categorize income sources
            if any(keyword in desc for keyword in ["salary", "wage", "payroll"]):
                income_descriptions.add("salary")
            elif any(keyword in desc for keyword in ["transfer", "deposit"]):
                income_descriptions.add("transfer")
            else:
                income_descriptions.add("other")

    # Balance statistics
    balances = [txn.balance for txn in parsed_txns if txn.balance is not None]
    avg_balance = sum(balances) / len(balances) if balances else 0
    min_balance = min(balances) if balances else 0
    max_balance = max(balances) if balances else 0
    negative_balance_count = sum(1 for b in balances if b < 0)

    # Balance trend (simple linear regression slope approximation)
    if len(balances) >= 2:
        n = len(balances)
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = avg_balance
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, balances))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        balance_trend = numerator / denominator if denominator != 0 else 0
    else:
        balance_trend = 0

    # Transaction patterns
    transaction_count = len(parsed_txns)
    all_amounts = [txn.deposit or txn.withdrawal or 0 for txn in parsed_txns]
    avg_transaction_size = sum(all_amounts) / len(all_amounts) if all_amounts else 0

    # Large transactions (> 50% of avg monthly income)
    threshold = avg_monthly_income * 0.5 if avg_monthly_income > 0 else 10000
    large_transactions = sum(
        1 for txn in parsed_txns
        if (txn.deposit and txn.deposit > threshold) or
           (txn.withdrawal and txn.withdrawal > threshold)
    )

    # Frequent small deposits (possible structuring - deposits just under reporting threshold)
    frequent_small_deposits = sum(
        1 for txn in parsed_txns
        if txn.deposit and 5000 < txn.deposit < 10000
    )

    # Expense categorization
    essential_keywords = ["rent", "mortgage", "utilities", "electric", "water", "groceries", "food", "transport", "medical", "insurance"]
    essential_expenses = sum(
        txn.withdrawal for txn in parsed_txns
        if txn.withdrawal and any(kw in txn.description.lower() for kw in essential_keywords)
    )
    essential_expense_ratio = essential_expenses / total_expenses if total_expenses > 0 else 0

    # Ratios
    expense_to_income = total_expenses / total_income if total_income > 0 else 1.0
    savings_rate = (total_income - total_expenses) / total_income if total_income > 0 else 0

    # Estimate debt-to-income from expense patterns
    # Assume recurring large withdrawals are debt payments
    debt_keywords = ["loan", "credit card", "emi", "repayment", "debt"]
    debt_payments = sum(
        txn.withdrawal for txn in parsed_txns
        if txn.withdrawal and any(kw in txn.description.lower() for kw in debt_keywords)
    )
    debt_to_income = debt_payments / total_income if total_income > 0 else 0

    # Calculate creditworthiness score (0-100)
    score = 50  # Base score
    score += income_stability * 20  # Up to 20 points for stable income
    score -= negative_balance_count * 5  # Penalty for overdrafts
    score += min(savings_rate * 30, 15)  # Up to 15 points for savings
    score -= min(expense_to_income * 10, 10)  # Penalty for high expense ratio
    score -= min(debt_to_income * 20, 15)  # Penalty for high debt
    score += min(avg_balance / (avg_monthly_income + 1) * 10, 10)  # Points for maintaining balance
    score = max(0, min(100, score))  # Clamp between 0-100

    return FinancialFeatures(
        total_income_3m=total_income,
        avg_monthly_income=avg_monthly_income,
        income_stability=income_stability,
        income_sources=len(income_descriptions),
        total_expenses_3m=total_expenses,
        avg_monthly_expense=avg_monthly_expense,
        essential_expense_ratio=essential_expense_ratio,
        avg_balance=avg_balance,
        min_balance=min_balance,
        max_balance=max_balance,
        balance_trend=balance_trend,
        negative_balance_count=negative_balance_count,
        transaction_count=transaction_count,
        avg_transaction_size=avg_transaction_size,
        large_transactions=large_transactions,
        frequent_small_deposits=frequent_small_deposits,
        debt_to_income=debt_to_income,
        savings_rate=savings_rate,
        expense_to_income=expense_to_income,
        creditworthiness_score=round(score, 2)
    )


def features_to_dict(features: FinancialFeatures) -> Dict[str, Any]:
    """Convert FinancialFeatures to dictionary."""
    return {
        "total_income_3m": features.total_income_3m,
        "avg_monthly_income": features.avg_monthly_income,
        "income_stability": features.income_stability,
        "income_sources": features.income_sources,
        "total_expenses_3m": features.total_expenses_3m,
        "avg_monthly_expense": features.avg_monthly_expense,
        "essential_expense_ratio": features.essential_expense_ratio,
        "avg_balance": features.avg_balance,
        "min_balance": features.min_balance,
        "max_balance": features.max_balance,
        "balance_trend": features.balance_trend,
        "negative_balance_count": features.negative_balance_count,
        "transaction_count": features.transaction_count,
        "avg_transaction_size": features.avg_transaction_size,
        "large_transactions": features.large_transactions,
        "frequent_small_deposits": features.frequent_small_deposits,
        "debt_to_income": features.debt_to_income,
        "savings_rate": features.savings_rate,
        "expense_to_income": features.expense_to_income,
        "creditworthiness_score": features.creditworthiness_score
    }
