"""
Validate the credit scorecard against the Berka (PKDD'99) bank dataset.

The dataset holds real, anonymised accounts from a Czech bank: about one
million transactions and 682 loans with their repayment outcome. For each loan
this script takes the account's transactions from before the loan was granted,
scores them exactly as the API does, and compares the result with what
actually happened.

Outcome labels (loan.asc "status"):
    A = finished, repaid          C = running, payments on time   -> good
    B = finished, not repaid      D = running, client in debt     -> bad

Get the data (18 MB, public, archived copy of the original challenge file):
    https://web.archive.org/web/20070214120527id_/http://lisp.vse.cz/pkdd99/DATA/data_berka.zip
Unzip it into data/berka/ and run from the repository root:
    python scripts/validate_berka.py --data-dir data/berka
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.ml.features import extract_features_from_transactions  # noqa: E402
from src.app.ml.model import CreditRiskModel  # noqa: E402

# Czech transaction codes -> the English narration a statement would carry
K_SYMBOL = {
    "UVER": "loan payment",
    "SIPO": "household payment utilities",
    "POJISTNE": "insurance payment",
    "SLUZBY": "statement fee",
    "SANKC. UROK": "sanction interest penalty overdraft",
    "UROK": "interest credit",
    "DUCHOD": "pension",
}
OPERATION = {
    "VKLAD": "cash deposit",
    "VYBER": "cash withdrawal",
    "VYBER KARTOU": "card withdrawal",
    "PREVOD Z UCTU": "transfer from another bank",
    "PREVOD NA UCET": "transfer to another bank",
}


def berka_date(value: str) -> date:
    return date(1900 + int(value[:2]), int(value[2:4]), int(value[4:6]))


def load_loans(data_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(data_dir, "loan.asc"), encoding="latin-1") as handle:
        return [
            {
                "loan_id": row["loan_id"],
                "account_id": row["account_id"],
                "date": berka_date(row["date"]),
                "amount": float(row["amount"]),
                "duration": int(row["duration"]),
                "bad": row["status"] in ("B", "D"),
                "status": row["status"],
            }
            for row in csv.DictReader(handle, delimiter=";")
        ]


def load_transactions(data_dir: str, accounts: set) -> Dict[str, List[Dict[str, Any]]]:
    by_account: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(os.path.join(data_dir, "trans.asc"), encoding="latin-1") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row["account_id"] not in accounts:
                continue
            credit = row["type"] == "PRIJEM"
            amount = float(row["amount"])
            narration = " ".join(
                part
                for part in (
                    K_SYMBOL.get(row["k_symbol"].strip(), ""),
                    OPERATION.get(row["operation"].strip(), ""),
                )
                if part
            )
            by_account[row["account_id"]].append(
                {
                    "trans_id": int(row["trans_id"]),
                    "when": berka_date(row["date"]),
                    "description": narration,
                    "withdrawal": None if credit else amount,
                    "deposit": amount if credit else None,
                    "balance": float(row["balance"]),
                }
            )
    for txns in by_account.values():
        txns.sort(key=lambda txn: (txn["when"], txn["trans_id"]))
    return by_account


def auc(good_scores: List[float], bad_scores: List[float]) -> float:
    """Probability that a random good loan outscores a random bad one."""
    ranked = sorted(
        [(score, 0) for score in good_scores] + [(score, 1) for score in bad_scores]
    )
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for item in ranked[index:end] if item[1] == 0)
        index = end
    n_good, n_bad = len(good_scores), len(bad_scores)
    return (rank_sum - n_good * (n_good + 1) / 2.0) / (n_good * n_bad)


def ks_statistic(good_scores: List[float], bad_scores: List[float]) -> float:
    """Largest gap between the cumulative score distributions."""
    best = 0.0
    for cut in sorted(set(good_scores + bad_scores)):
        good_share = sum(1 for score in good_scores if score <= cut) / len(good_scores)
        bad_share = sum(1 for score in bad_scores if score <= cut) / len(bad_scores)
        best = max(best, abs(bad_share - good_share))
    return best


def evaluate(data_dir: str, window_days: int) -> Dict[str, Any]:
    loans = load_loans(data_dir)
    transactions = load_transactions(data_dir, {loan["account_id"] for loan in loans})
    model = CreditRiskModel()

    rows: List[Tuple[Dict[str, Any], Any]] = []
    for loan in loans:
        start = loan["date"] - timedelta(days=window_days)
        history = [
            {
                "date": txn["when"].isoformat(),
                "description": txn["description"],
                "withdrawal": txn["withdrawal"],
                "deposit": txn["deposit"],
                "balance": txn["balance"],
            }
            for txn in transactions[loan["account_id"]]
            if start <= txn["when"] < loan["date"]  # strictly before the loan: no leakage
        ]
        features = extract_features_from_transactions(history)
        rows.append((loan, model.assess(features, loan["amount"], loan["duration"])))

    good = [result.credit_score for loan, result in rows if not loan["bad"]]
    bad = [result.credit_score for loan, result in rows if loan["bad"]]

    by_recommendation: Dict[str, Dict[str, int]] = {}
    for loan, result in rows:
        bucket = by_recommendation.setdefault(result.recommendation, {"loans": 0, "bad": 0})
        bucket["loans"] += 1
        bucket["bad"] += int(loan["bad"])

    bands = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 101)]
    by_band = []
    for low, high in bands:
        members = [loan for loan, result in rows if low <= result.credit_score < high]
        by_band.append(
            {
                "score_band": f"{low}-{min(high, 100)}",
                "loans": len(members),
                "bad": sum(1 for loan in members if loan["bad"]),
            }
        )

    return {
        "window_days": window_days,
        "loans": len(rows),
        "bad_loans": len(bad),
        "base_bad_rate": round(len(bad) / len(rows), 4),
        "auc": round(auc(good, bad), 4),
        "ks": round(ks_statistic(good, bad), 4),
        "mean_score_good": round(sum(good) / len(good), 2),
        "mean_score_bad": round(sum(bad) / len(bad), 2),
        "by_recommendation": by_recommendation,
        "by_score_band": by_band,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\nHistory window: {report['window_days']} days before the loan")
    print(
        f"Loans: {report['loans']}  bad: {report['bad_loans']}  "
        f"base bad rate: {report['base_bad_rate']:.1%}"
    )
    print(f"AUC: {report['auc']:.3f}   KS: {report['ks']:.3f}")
    print(
        f"Mean score - good loans: {report['mean_score_good']}, "
        f"bad loans: {report['mean_score_bad']}"
    )
    print("\nBy recommendation:")
    for name in ("approve", "review", "reject"):
        bucket = report["by_recommendation"].get(name, {"loans": 0, "bad": 0})
        rate = bucket["bad"] / bucket["loans"] if bucket["loans"] else 0.0
        print(f"  {name:8s} loans={bucket['loans']:4d}  bad={bucket['bad']:3d}  bad rate={rate:.1%}")
    print("\nBy score band:")
    for band in report["by_score_band"]:
        rate = band["bad"] / band["loans"] if band["loans"] else 0.0
        print(
            f"  {band['score_band']:7s} loans={band['loans']:4d}  "
            f"bad={band['bad']:3d}  bad rate={rate:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data/berka")
    parser.add_argument("--window-days", type=int, nargs="+", default=[90, 180, 365])
    parser.add_argument("--json", help="Write the reports to this file")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "loan.asc")):
        raise SystemExit(f"loan.asc not found in {args.data_dir}. See this file's docstring.")

    reports = [evaluate(args.data_dir, window) for window in args.window_days]
    for report in reports:
        print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)


if __name__ == "__main__":
    main()
