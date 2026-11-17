"""Shared test setup. Environment is set before the app is imported."""

from __future__ import annotations

import os
import tempfile

_TEST_DIR = tempfile.mkdtemp(prefix="fis-tests-")
TEST_API_KEY = "unit-test-key-0123456789-abcdefghij"

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "API_KEY": TEST_API_KEY,
        "DATABASE_URL": f"sqlite:///{_TEST_DIR}/test.db",
        "LLM_PROVIDER": "none",
        "LOG_LEVEL": "WARNING",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.app.database import Base, engine  # noqa: E402
from src.app.main import app  # noqa: E402
from src.app.routes import get_pipeline  # noqa: E402


@pytest.fixture()
def client():
    """API client on a clean database, with the LLM switched off."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_pipeline] = lambda: None
    with TestClient(app) as test_client:
        test_client.headers.update({"X-API-Key": TEST_API_KEY})
        yield test_client
    app.dependency_overrides.clear()


def statement_rows(months: int = 4, salary: float = 60000.0, rent: float = 15000.0):
    """A healthy salaried account: income in, bills out, balance growing."""
    rows = []
    balance = 40000.0
    def add(month, day, description, withdrawal=None, deposit=None):
        nonlocal balance
        balance += (deposit or 0) - (withdrawal or 0)
        rows.append(
            {
                "date": f"2026-{month:02d}-{day:02d}",
                "description": description,
                "withdrawal": withdrawal,
                "deposit": deposit,
                "balance": round(balance, 2),
            }
        )

    for month in range(1, months + 1):
        add(month, 1, "SALARY ACME LTD", deposit=salary)
        add(month, 3, "RENT PAYMENT", withdrawal=rent)
        add(month, 6, "ELECTRICITY BILL", withdrawal=2200.0)
        add(month, 10, "GROCERY SUPERMARKET", withdrawal=5200.0)
        add(month, 15, "RESTAURANT", withdrawal=1800.0)
        add(month, 20, "FUEL STATION", withdrawal=2500.0)
        add(month, 25, "ONLINE SHOPPING", withdrawal=4000.0)
    return rows


def rows_to_csv(rows) -> bytes:
    lines = ["Date,Description,Withdrawal,Deposit,Balance"]
    for row in rows:
        lines.append(
            f"{row['date']},{row['description']},"
            f"{row['withdrawal'] or ''},{row['deposit'] or ''},{row['balance']}"
        )
    return "\n".join(lines).encode()


SINGLE_PDF_LINES = [
    "Date Narration Chq./Ref.No. ValueDt WithdrawalAmt. DepositAmt. ClosingBalance",
    "01/01/26 UPI-MUTUAL FUNDS 0000108885274149 01/01/26 1,000.00 743.80",
    "01/01/26 TPT-FAMILY TRANSFER 0000000126560415 01/01/26 5,000.00 5,743.80",
    "02/01/26 UPI-GROCERY STORE 0000116509752956 02/01/26 80.00 5,663.80",
    "03/01/26 NEFTCR-EMPLOYER PRIVATE LIMITED IN22600319205675 03/01/26 8,207.30 13,871.10",
]


def build_pdf(lines) -> bytes:
    """Write a minimal one-page text PDF (enough for pdfplumber to read)."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = ["BT", "/F1 9 Tf", "40 800 Td", "12 TL"]
    for line in lines:
        content.append(f"({escape(line)}) Tj T*")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)
