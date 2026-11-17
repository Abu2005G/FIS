"""
Bank statement parsing.

Turns a PDF, CSV or JSON bank statement into a list of normalised transactions:

    {"date": "2026-01-03", "description": "...", "withdrawal": 130.0,
     "deposit": None, "balance": 4144.10, "reconciled": True}

Design notes
------------
* The direction of a transaction (money in or money out) is decided by the
  running balance wherever possible, because statement layouts that print a
  single amount column give no other reliable signal.
* Every statement gets a reconciliation score: the share of transactions whose
  amount explains the change in the running balance. A low score means the
  statement was misread (or altered) and the caller should not trust it.
* Account numbers, names and addresses are deliberately not extracted.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MAX_PAGES = 300
MAX_TRANSACTIONS = 20_000
MAX_CONTINUATION_LINES = 8
MAX_DESCRIPTION_LENGTH = 200
BALANCE_TOLERANCE = 0.011

SUPPORTED_SUFFIXES = (".pdf", ".csv", ".json")


class StatementParseError(ValueError):
    """The statement could not be read into transactions."""


@dataclass
class ParsedStatement:
    """Result of parsing one statement file."""

    transactions: List[Dict[str, Any]]
    quality: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}

_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))

_DATE_AT_START = re.compile(
    r"^(?P<date>"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"
    rf"|\d{{1,2}}[ -]?(?:{_MONTH_NAMES})\.?,?[ -]?\d{{2,4}}"
    r")(?=\s|$)",
    re.IGNORECASE,
)

_ISO_DATE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$")
_NAMED_DATE = re.compile(
    rf"^(\d{{1,2}})[ -]?({_MONTH_NAMES})\.?,?[ -]?(\d{{2,4}})$", re.IGNORECASE
)


def _expand_year(year: int) -> int:
    """Turn a two-digit year into four digits (pivot: next year)."""
    if year >= 100:
        return year
    pivot = (date.today().year + 1) % 100
    return 2000 + year if year <= pivot else 1900 + year


def parse_date(value: Any, day_first: bool = True) -> Optional[date]:
    """Parse the date formats commonly found on bank statements."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    try:
        match = _ISO_DATE.match(text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return date(year, month, day)

        match = _NUMERIC_DATE.match(text)
        if match:
            first, second, year = (int(part) for part in match.groups())
            day, month = (first, second) if day_first else (second, first)
            if month > 12 >= day:
                day, month = month, day
            return date(_expand_year(year), month, day)

        match = _NAMED_DATE.match(text)
        if match:
            day, month_name, year = match.groups()
            return date(_expand_year(int(year)), _MONTHS[month_name.lower()], int(day))
    except ValueError:
        return None
    return None


# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------

_AMOUNT = re.compile(
    r"^(?P<open>[-(])?"
    r"(?:[₹$€£]|rs\.?|inr)?"
    r"(?P<number>\d{1,3}(?:,\d{2,3})+\.\d{1,2}|\d+\.\d{1,2})"
    r"\)?(?P<marker>cr|dr)?\.?$",
    re.IGNORECASE,
)

_PLAIN_NUMBER = re.compile(r"^[-+(]?(?:[₹$€£]|rs\.?|inr)?\s*[\d,]*\.?\d+\)?$", re.IGNORECASE)


@dataclass
class _Amount:
    value: float
    negative: bool = False
    marker: Optional[str] = None  # "cr" or "dr" printed next to the number

    @property
    def signed(self) -> float:
        return -self.value if self.negative or self.marker == "dr" else self.value


def _parse_amount_token(token: str) -> Optional[_Amount]:
    match = _AMOUNT.match(token)
    if not match:
        return None
    value = float(match.group("number").replace(",", ""))
    marker = match.group("marker")
    return _Amount(
        value=value,
        negative=match.group("open") is not None,
        marker=marker.lower() if marker else None,
    )


def to_number(value: Any) -> Optional[float]:
    """Convert a CSV/JSON cell to a float. Returns None for blanks."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text or text in {"-", "--"}:
        return None
    negative = text.startswith(("-", "(")) or text.lower().endswith("dr")
    cleaned = re.sub(r"(?i)(cr|dr)\.?$", "", text).strip()
    if not _PLAIN_NUMBER.match(cleaned):
        return None
    cleaned = re.sub(r"(?i)[₹$€£(),+\-\s]|rs\.?|inr", "", cleaned)
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return -number if negative else number


def _split_trailing_amounts(text: str) -> Tuple[str, List[_Amount]]:
    """Split 'narration 1,000.00 743.80' into ('narration', [1000.0, 743.8])."""
    tokens = text.split()
    amounts: List[_Amount] = []
    pending_marker: Optional[str] = None

    while tokens:
        token = tokens[-1]
        lowered = token.lower().rstrip(".")
        if lowered in ("cr", "dr") and pending_marker is None:
            pending_marker = lowered
            tokens.pop()
            continue
        amount = _parse_amount_token(token)
        if amount is None:
            break
        if pending_marker and amount.marker is None:
            amount.marker = pending_marker
        pending_marker = None
        amounts.append(amount)
        tokens.pop()

    if pending_marker:
        tokens.append(pending_marker)
    amounts.reverse()
    return " ".join(tokens), amounts[-3:]


# --------------------------------------------------------------------------
# Text-layout statements (PDF)
# --------------------------------------------------------------------------

_COLUMN_WORDS = (
    "date",
    "narration",
    "description",
    "particulars",
    "details",
    "withdrawal",
    "deposit",
    "debit",
    "credit",
    "balance",
    "chq",
    "ref",
)

_STOP_LINE = re.compile(
    r"statement\s*summary|opening\s*balance|closing\s*balance|^\s*totals?\b"
    r"|end\s*of\s*statement|page\s*no|generated\s*(on|by)|brought\s*forward"
    r"|carried\s*forward",
    re.IGNORECASE,
)

_OPENING_BALANCE = re.compile(r"opening\s*balance", re.IGNORECASE)

_CREDIT_HINT = re.compile(
    r"neft\s*cr|imps\s*cr|salary|payroll|refund|reversal|cashback|interest|int\.?\s*pd"
    r"|deposit|credit(?!\s*card)|received",
    re.IGNORECASE,
)

_REFERENCE_NOISE = re.compile(r"\b\d{9,}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")


@dataclass
class _Row:
    when: date
    parts: List[str]
    amounts: List[_Amount]
    continuation_lines: int = 0
    closed: bool = False


def _is_column_header(line: str) -> bool:
    lowered = line.lower()
    return sum(1 for word in _COLUMN_WORDS if word in lowered) >= 3


def _find_page_furniture(pages: Sequence[Sequence[str]]) -> set:
    """Lines repeated on most pages are headers and footers, not transactions."""
    if len(pages) < 3:
        return set()
    counts: Dict[str, int] = {}
    for lines in pages:
        for key in {re.sub(r"\d", "#", line) for line in lines}:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(3, math.ceil(len(pages) * 0.6))
    return {key for key, count in counts.items() if count >= threshold}


def _find_opening_balance(pages: Sequence[Sequence[str]]) -> Optional[float]:
    """Best-effort read of a printed opening balance."""
    for lines in pages:
        for index, line in enumerate(lines):
            if not _OPENING_BALANCE.search(line):
                continue
            for candidate in (line, lines[index + 1] if index + 1 < len(lines) else ""):
                after = candidate.split("alance", 1)[-1] if "alance" in candidate else candidate
                for token in after.split():
                    amount = _parse_amount_token(token)
                    if amount is not None:
                        return amount.signed
    return None


def _collect_rows(pages: Sequence[Sequence[str]], day_first: bool) -> List[_Row]:
    furniture = _find_page_furniture(pages)
    rows: List[_Row] = []
    current: Optional[_Row] = None

    for lines in pages:
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            date_match = _DATE_AT_START.match(line)
            when = parse_date(date_match.group("date"), day_first) if date_match else None

            if when is not None and date_match is not None:
                narration, amounts = _split_trailing_amounts(line[date_match.end():].strip())
                current = _Row(when=when, parts=[narration], amounts=amounts)
                rows.append(current)
                if len(rows) > MAX_TRANSACTIONS:
                    raise StatementParseError(
                        f"Statement has more than {MAX_TRANSACTIONS} transactions"
                    )
                continue

            if current is None or current.closed:
                continue
            if re.sub(r"\d", "#", line) in furniture or _is_column_header(line):
                continue
            if _STOP_LINE.search(line):
                current.closed = True
                continue

            narration, amounts = _split_trailing_amounts(line)
            if amounts and not narration and not current.amounts:
                current.amounts = amounts
                continue
            current.continuation_lines += 1
            if current.continuation_lines <= MAX_CONTINUATION_LINES:
                current.parts.append(line)
            else:
                current.closed = True

    return rows


def _clean_description(parts: Iterable[str]) -> str:
    text = " ".join(part for part in parts if part)
    text = _REFERENCE_NOISE.sub(" ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text[:MAX_DESCRIPTION_LENGTH]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= BALANCE_TOLERANCE


def _resolve_rows(rows: List[_Row], opening_balance: Optional[float]) -> List[Dict[str, Any]]:
    """Decide money-in vs money-out for each row using the running balance."""
    usable = [row for row in rows if row.amounts]
    if not usable:
        return []
    if usable[0].when > usable[-1].when:
        usable.reverse()  # newest-first statement

    transactions: List[Dict[str, Any]] = []
    previous: Optional[float] = None
    first = True

    for row in usable:
        description = _clean_description(row.parts)
        amounts = row.amounts
        direction: Optional[str] = None
        balance: Optional[float] = None
        reconciled: Optional[bool] = None

        if len(amounts) == 1:
            amount = amounts[0].value
            marker = amounts[0].marker
        else:
            balance = amounts[-1].signed
            amount = amounts[-2].value
            marker = amounts[-2].marker
            if len(amounts) == 3:
                withdrawal_col, deposit_col = amounts[0].value, amounts[1].value
                if withdrawal_col > 0 and deposit_col == 0:
                    amount, direction = withdrawal_col, "debit"
                elif deposit_col > 0 and withdrawal_col == 0:
                    amount, direction = deposit_col, "credit"

            if first and previous is None and opening_balance is not None:
                if _close(opening_balance + amount, balance) or _close(
                    opening_balance - amount, balance
                ):
                    previous = opening_balance

            if previous is not None:
                if _close(previous + amount, balance):
                    direction, reconciled = "credit", True
                elif _close(previous - amount, balance):
                    direction, reconciled = "debit", True
                else:
                    reconciled = False

        if direction is None:
            if marker == "cr":
                direction = "credit"
            elif marker == "dr":
                direction = "debit"
            else:
                direction = "credit" if _CREDIT_HINT.search(description) else "debit"

        first = False
        if balance is not None:
            previous = balance

        if amount <= 0:
            continue
        transactions.append(
            {
                "date": row.when.isoformat(),
                "description": description,
                "withdrawal": amount if direction == "debit" else None,
                "deposit": amount if direction == "credit" else None,
                "balance": balance,
                "reconciled": reconciled,
            }
        )

    return transactions


def parse_statement_text(pages: Sequence[str], day_first: bool = True) -> List[Dict[str, Any]]:
    """Parse statement text that has already been extracted, one string per page."""
    page_lines = [page.splitlines() for page in pages]
    rows = _collect_rows(page_lines, day_first)
    return _resolve_rows(rows, _find_opening_balance(page_lines))


def _read_pdf_pages(path: Path) -> List[str]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise StatementParseError("PDF support is not installed (pdfplumber)") from exc

    try:
        with pdfplumber.open(str(path)) as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise StatementParseError(f"PDF has more than {MAX_PAGES} pages")
            return [page.extract_text() or "" for page in pdf.pages]
    except StatementParseError:
        raise
    except Exception as exc:
        raise StatementParseError("The file is not a readable PDF") from exc


# --------------------------------------------------------------------------
# Tabular statements (CSV / JSON)
# --------------------------------------------------------------------------

_FIELD_ALIASES = {
    "date": ("date", "txndate", "transactiondate", "valuedate", "postingdate", "valuedt"),
    "description": (
        "description",
        "narration",
        "particulars",
        "details",
        "remarks",
        "memo",
        "merchant",
    ),
    "withdrawal": (
        "withdrawal",
        "withdrawalamt",
        "withdrawalamount",
        "withdrawals",
        "debit",
        "debitamount",
        "dr",
        "paidout",
    ),
    "deposit": (
        "deposit",
        "depositamt",
        "depositamount",
        "deposits",
        "credit",
        "creditamount",
        "cr",
        "paidin",
    ),
    "amount": ("amount", "transactionamount", "txnamount"),
    "type": ("type", "transactiontype", "drcr", "crdr", "txntype"),
    "balance": ("balance", "closingbalance", "runningbalance", "availablebalance"),
}


def _normalise_key(key: Any) -> str:
    return re.sub(r"[^a-z]", "", str(key).lower())


def _pick(record: Dict[str, Any], field_name: str) -> Any:
    for alias in _FIELD_ALIASES[field_name]:
        if alias in record and record[alias] not in (None, ""):
            return record[alias]
    return None


def _normalise_record(raw: Dict[str, Any], day_first: bool) -> Optional[Dict[str, Any]]:
    record = {_normalise_key(key): value for key, value in raw.items()}
    when = parse_date(_pick(record, "date"), day_first)
    if when is None:
        return None

    withdrawal = to_number(_pick(record, "withdrawal"))
    deposit = to_number(_pick(record, "deposit"))
    amount = to_number(_pick(record, "amount"))
    kind = str(_pick(record, "type") or "").strip().lower()

    if not withdrawal and not deposit and amount:
        is_credit = kind.startswith(("cr", "dep", "in")) or (
            not kind.startswith(("dr", "deb", "with", "out")) and amount > 0 and kind == ""
        )
        if kind.startswith(("dr", "deb", "with", "out")) or amount < 0:
            is_credit = False
        if is_credit:
            deposit = abs(amount)
        else:
            withdrawal = abs(amount)

    withdrawal = abs(withdrawal) if withdrawal else None
    deposit = abs(deposit) if deposit else None
    if not withdrawal and not deposit:
        return None

    description = _clean_description([str(_pick(record, "description") or "")])
    return {
        "date": when.isoformat(),
        "description": description,
        "withdrawal": withdrawal,
        "deposit": deposit,
        "balance": to_number(_pick(record, "balance")),
        "reconciled": None,
    }


def normalise_records(
    records: Iterable[Dict[str, Any]], day_first: bool = True
) -> List[Dict[str, Any]]:
    """Normalise dict-shaped transactions (from CSV, JSON or another system)."""
    transactions: List[Dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        record = _normalise_record(raw, day_first)
        if record is not None:
            transactions.append(record)
        if len(transactions) > MAX_TRANSACTIONS:
            raise StatementParseError(f"Statement has more than {MAX_TRANSACTIONS} transactions")
    if len(transactions) >= 2 and transactions[0]["date"] > transactions[-1]["date"]:
        transactions.reverse()
    return transactions


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return list(csv.DictReader(io.StringIO(text), dialect=dialect))


def _read_json(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StatementParseError("The file is not valid JSON") from exc
    if isinstance(data, dict):
        data = data.get("transactions", [])
    if not isinstance(data, list):
        raise StatementParseError("JSON must be a list of transactions")
    return data


# --------------------------------------------------------------------------
# Quality and public API
# --------------------------------------------------------------------------


def assess_quality(transactions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check that amounts explain the running balance, and mark each row.

    ``reconciliation_rate`` is the lower of two shares: the share of rows that
    reconcile, and the share of money moved that reconciles. The second one
    matters because inflating a single salary credit changes few rows but a
    large part of the value.
    """
    checked = reconciled = 0
    checked_value = reconciled_value = 0.0
    previous: Optional[float] = None

    for txn in transactions:
        balance = txn.get("balance")
        if balance is None:
            continue
        if previous is not None:
            deposit = txn.get("deposit") or 0.0
            withdrawal = txn.get("withdrawal") or 0.0
            ok = _close(previous + deposit - withdrawal, balance)
            txn["reconciled"] = ok
            checked += 1
            checked_value += deposit + withdrawal
            if ok:
                reconciled += 1
                reconciled_value += deposit + withdrawal
        previous = balance

    row_rate = reconciled / checked if checked else None
    value_rate = reconciled_value / checked_value if checked_value else row_rate
    return {
        "transactions": len(transactions),
        "balance_checks": checked,
        "balance_checks_passed": reconciled,
        "row_reconciliation_rate": round(row_rate, 4) if row_rate is not None else None,
        "value_reconciliation_rate": round(value_rate, 4) if value_rate is not None else None,
        "reconciliation_rate": (
            round(min(row_rate, value_rate), 4)
            if row_rate is not None and value_rate is not None
            else None
        ),
    }


def parse_statement(
    file_path: str, suffix: Optional[str] = None, day_first: bool = True
) -> ParsedStatement:
    """
    Parse a statement file.

    Args:
        file_path: Path to the file on disk.
        suffix: File type (".pdf", ".csv", ".json"). Defaults to the path's suffix.
        day_first: Read 03/04/2026 as 3 April (True) or 4 March (False).

    Raises:
        StatementParseError: The file is unreadable or holds no transactions.
    """
    path = Path(file_path)
    if not path.is_file():
        raise StatementParseError("Statement file not found")

    kind = (suffix or path.suffix).lower()
    if kind == ".pdf":
        pages = _read_pdf_pages(path)
        if not any(page.strip() for page in pages):
            raise StatementParseError(
                "No text found in the PDF. Scanned or image-only statements are not supported"
            )
        transactions = parse_statement_text(pages, day_first)
    elif kind == ".csv":
        transactions = normalise_records(_read_csv(path), day_first)
    elif kind == ".json":
        transactions = normalise_records(_read_json(path), day_first)
    else:
        raise StatementParseError(
            f"Unsupported file type '{kind}'. Use one of: {', '.join(SUPPORTED_SUFFIXES)}"
        )

    if not transactions:
        raise StatementParseError("No transactions could be read from the statement")

    quality = assess_quality(transactions)
    metadata = {
        "period_start": min(txn["date"] for txn in transactions),
        "period_end": max(txn["date"] for txn in transactions),
        "format": kind.lstrip("."),
    }
    logger.info(
        "Parsed statement: %s transactions, reconciliation=%s",
        quality["transactions"],
        quality["reconciliation_rate"],
    )
    return ParsedStatement(transactions=transactions, quality=quality, metadata=metadata)


def parse_transaction_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a statement file and return only its transactions."""
    return parse_statement(file_path).transactions


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.app.ml.statement_parser <statement.pdf|csv|json>")
        raise SystemExit(2)
    parsed = parse_statement(sys.argv[1])
    print(json.dumps({"quality": parsed.quality, "metadata": parsed.metadata}, indent=2))
