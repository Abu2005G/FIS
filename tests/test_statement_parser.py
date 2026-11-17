from datetime import date

import pytest

from src.app.ml.statement_parser import (
    StatementParseError,
    assess_quality,
    normalise_records,
    parse_date,
    parse_statement,
    parse_statement_text,
    to_number,
)
from tests.conftest import build_pdf, rows_to_csv, statement_rows

SINGLE_AMOUNT_PAGE = """Date Narration Chq./Ref.No. ValueDt WithdrawalAmt. DepositAmt. ClosingBalance
01/01/26 UPI-MUTUAL FUNDS 0000108885274149 01/01/26 1,000.00 743.80
continuation of the narration
01/01/26 TPT-FAMILY TRANSFER 0000000126560415 01/01/26 5,000.00 5,743.80
02/01/26 UPI-GROCERY STORE 0000116509752956 02/01/26 80.00 5,663.80
03/01/26 NEFTCR-EMPLOYER PRIVATE LIMITED IN22600319205675 03/01/26 8,207.30 13,871.10
STATEMENT SUMMARY
OpeningBalance DrCount CrCount Debits Credits ClosingBal
1,743.80 2 2 1,080.00 13,207.30 13,871.10
"""


def test_direction_comes_from_the_running_balance():
    txns = parse_statement_text([SINGLE_AMOUNT_PAGE])

    assert [(t["withdrawal"], t["deposit"]) for t in txns] == [
        (1000.0, None),
        (None, 5000.0),
        (80.0, None),
        (None, 8207.3),
    ]
    assert all(t["reconciled"] for t in txns)
    assert txns[0]["date"] == "2026-01-01"


def test_summary_block_and_reference_numbers_stay_out_of_descriptions():
    txns = parse_statement_text([SINGLE_AMOUNT_PAGE])

    assert txns[0]["description"] == "UPI-MUTUAL FUNDS continuation of the narration"
    assert "OpeningBalance" not in txns[-1]["description"]
    assert "0000108885274149" not in txns[0]["description"]


def test_three_column_layout_with_explicit_zero_columns():
    page = """Date Description Debit Credit Balance
2026-02-01 SALARY 0.00 50,000.00 60,000.00
2026-02-03 RENT 15,000.00 0.00 45,000.00
"""
    txns = parse_statement_text([page])

    assert txns[0]["deposit"] == 50000.0 and txns[0]["withdrawal"] is None
    assert txns[1]["withdrawal"] == 15000.0 and txns[1]["deposit"] is None


def test_newest_first_statements_are_reordered():
    page = """05 Jan 2026 CARD PURCHASE 100.00 900.00
03 Jan 2026 SALARY 500.00 1,000.00
01 Jan 2026 ATM 200.00 500.00
"""
    txns = parse_statement_text([page])

    assert [t["date"] for t in txns] == ["2026-01-01", "2026-01-03", "2026-01-05"]
    assert txns[1]["deposit"] == 500.0
    assert txns[2]["withdrawal"] == 100.0


def test_overdrawn_balances_and_cr_dr_markers():
    page = """01/03/26 OPENING ENTRY 100.00 Cr 100.00
02/03/26 CARD PAYMENT 250.00 150.00 Dr
03/03/26 FEE 10.00 (160.00)
"""
    txns = parse_statement_text([page])

    assert txns[0]["deposit"] == 100.0
    assert txns[1]["balance"] == -150.0 and txns[1]["withdrawal"] == 250.0
    assert txns[2]["balance"] == -160.0 and txns[2]["withdrawal"] == 10.0


def test_repeated_page_headers_are_not_merged_into_transactions():
    header = "ACME BANK\nCustomer: PRIVATE NAME\nDate Narration Withdrawal Deposit Balance\n"
    pages = [
        header + "01/01/26 FIRST 10.00 990.00\ncarried over",
        header + "next line of first\n02/01/26 SECOND 20.00 970.00\n",
        header + "03/01/26 THIRD 30.00 940.00\n",
    ]
    txns = parse_statement_text(pages)

    assert len(txns) == 3
    assert all("PRIVATE NAME" not in t["description"] for t in txns)
    assert all("ACME BANK" not in t["description"] for t in txns)
    assert "next line of first" in txns[0]["description"]


def test_quality_flags_amounts_that_do_not_explain_the_balance():
    rows = statement_rows(months=1)
    rows[3]["withdrawal"] += 500.0  # tampered amount

    quality = assess_quality(rows)

    assert quality["balance_checks"] == len(rows) - 1
    assert quality["balance_checks_passed"] == len(rows) - 2
    assert rows[3]["reconciled"] is False


def test_csv_statement(tmp_path):
    path = tmp_path / "statement.csv"
    path.write_bytes(rows_to_csv(statement_rows(months=2)))

    parsed = parse_statement(str(path))

    assert len(parsed.transactions) == 14
    assert parsed.quality["reconciliation_rate"] == 1.0
    assert parsed.metadata["period_start"] == "2026-01-01"


def test_csv_with_signed_amount_column(tmp_path):
    path = tmp_path / "signed.csv"
    path.write_text("Txn Date;Narration;Amount;Balance\n01-02-2026;PAY;1000.50;1000.50\n"
                    "02-02-2026;SHOP;-200.50;800.00\n")

    txns = parse_statement(str(path)).transactions

    assert txns[0]["deposit"] == 1000.5
    assert txns[1]["withdrawal"] == 200.5


def test_json_statement(tmp_path):
    path = tmp_path / "statement.json"
    path.write_text('{"transactions": [{"date": "03/01/26", "description": "PAY", '
                    '"deposit": "1,200.00", "balance": 1200}]}')

    txns = parse_statement(str(path)).transactions

    assert txns == [
        {
            "date": "2026-01-03",
            "description": "PAY",
            "withdrawal": None,
            "deposit": 1200.0,
            "balance": 1200.0,
            "reconciled": None,
        }
    ]


def test_pdf_statement_end_to_end(tmp_path):
    path = tmp_path / "statement.pdf"
    path.write_bytes(build_pdf(SINGLE_AMOUNT_PAGE.splitlines()))

    parsed = parse_statement(str(path))

    assert len(parsed.transactions) == 4
    assert parsed.quality["reconciliation_rate"] == 1.0
    assert sum(t["deposit"] or 0 for t in parsed.transactions) == pytest.approx(13207.30)


@pytest.mark.parametrize(
    "name, content, message",
    [
        ("empty.csv", b"Date,Description,Withdrawal,Deposit,Balance\n", "No transactions"),
        ("broken.pdf", b"%PDF-1.4 not really a pdf", "not a readable PDF"),
        ("bad.json", b"{not json", "not valid JSON"),
        ("notes.txt", b"hello", "Unsupported file type"),
    ],
)
def test_unreadable_files_raise_a_clear_error(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_bytes(content)

    with pytest.raises(StatementParseError, match=message):
        parse_statement(str(path))


def test_pdf_without_text_is_rejected(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_pdf([]))

    with pytest.raises(StatementParseError, match="Scanned"):
        parse_statement(str(path))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("03/01/26", date(2026, 1, 3)),
        ("2026-01-03", date(2026, 1, 3)),
        ("3-Jan-2026", date(2026, 1, 3)),
        ("03 January 2026", date(2026, 1, 3)),
        ("13/25/2026", None),
        ("31/02/2026", None),
        ("", None),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_parse_date_month_first():
    assert parse_date("03/04/2026", day_first=False) == date(2026, 3, 4)


@pytest.mark.parametrize(
    "value, expected",
    [("1,00,000.50", 100000.5), ("(250.00)", -250.0), ("75.00 Dr", -75.0), ("", None),
     ("abc", None), (None, None), (12, 12.0), (float("nan"), None), ("₹ 1,200", 1200.0)],
)
def test_to_number(value, expected):
    assert to_number(value) == expected


def test_normalise_records_skips_rows_without_a_date_or_amount():
    rows = [{"date": "bad", "amount": 5}, {"date": "2026-01-01"}, "junk",
            {"date": "2026-01-01", "debit": 5}]

    assert len(normalise_records(rows)) == 1
