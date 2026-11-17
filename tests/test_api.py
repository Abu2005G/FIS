"""HTTP API: auth, validation, upload handling and the decision flow."""

import os

import pytest

from src.app.config import ConfigurationError, load_settings
from src.app.main import app
from src.app.prompts.llm_client import LLMError, MockLLMClient
from src.app.prompts.pipeline import FinancialIntelligencePipeline
from src.app.routes import get_pipeline
from tests.conftest import SINGLE_PDF_LINES, build_pdf, rows_to_csv, statement_rows
from tests.test_scoring import distressed_rows

APPLICATION = {
    "applicant_name": "Test Applicant",
    "applicant_email": "Applicant@Example.com",
    "loan_amount": 100000,
    "loan_term_months": 36,
    "loan_purpose": "Home renovation",
}


def apply(client, **overrides):
    response = client.post("/api/v1/loans/apply", json={**APPLICATION, **overrides})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def upload(client, application_id, content, filename="statement.csv"):
    return client.post(
        f"/api/v1/loans/{application_id}/upload-statement",
        files={"file": (filename, content)},
    )


# --------------------------------------------------------------------- auth
def test_health_needs_no_key(client):
    response = client.get("/api/v1/health", headers={"X-API-Key": ""})

    assert response.status_code == 200
    assert response.json()["database"] == "ok"
    assert "X-Request-ID" in response.headers


@pytest.mark.parametrize("key", [None, "", "wrong-key", "my-secret-key-123"])
def test_loan_routes_reject_missing_or_wrong_keys(client, key):
    headers = {"X-API-Key": key} if key is not None else {}
    client.headers.pop("X-API-Key")

    for method, path in [("get", "/api/v1/loans"), ("post", "/api/v1/loans/apply"),
                         ("post", "/api/v1/loans/1/decide")]:
        assert getattr(client, method)(path, headers=headers).status_code == 401


# --------------------------------------------------------------- applications
def test_create_and_fetch_application(client):
    application_id = apply(client)

    body = client.get(f"/api/v1/loans/{application_id}").json()
    assert body["status"] == "pending"
    assert body["applicant_email"] == "applicant@example.com"
    assert body["credit_score"] is None


@pytest.mark.parametrize(
    "overrides",
    [{"loan_amount": 0}, {"loan_amount": -5}, {"loan_amount": "NaN"},
     {"applicant_email": "not-an-email"}, {"applicant_name": "  "},
     {"loan_term_months": 0}, {"loan_term_months": 1000}],
)
def test_invalid_applications_are_rejected(client, overrides):
    response = client.post("/api/v1/loans/apply", json={**APPLICATION, **overrides})

    assert response.status_code == 422


def test_list_is_paginated_and_filterable(client):
    ids = [apply(client) for _ in range(3)]

    assert [a["id"] for a in client.get("/api/v1/loans?limit=2").json()] == ids[::-1][:2]
    assert [a["id"] for a in client.get("/api/v1/loans?limit=2&offset=2").json()] == ids[:1]
    assert client.get("/api/v1/loans?status=approved").json() == []
    assert client.get("/api/v1/loans?status=bogus").status_code == 422
    assert client.get("/api/v1/loans?limit=100000").status_code == 422


def test_unknown_application_is_404_everywhere(client):
    assert client.get("/api/v1/loans/999").status_code == 404
    assert client.get("/api/v1/loans/999/statements").status_code == 404
    assert client.get("/api/v1/loans/999/decisions").status_code == 404
    assert client.post("/api/v1/loans/999/decide").status_code == 404
    assert upload(client, 999, b"x").status_code == 404


# -------------------------------------------------------------------- uploads
def test_upload_csv_statement(client):
    application_id = apply(client)

    response = upload(client, application_id, rows_to_csv(statement_rows(months=4)))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["transactions_found"] == 28
    assert body["quality"]["reconciliation_rate"] == 1.0
    assert body["period_start"] == "2026-01-01"
    assert body["warnings"] == []
    statements = client.get(f"/api/v1/loans/{application_id}/statements").json()
    assert statements[0]["original_filename"] == "statement.csv"


def test_upload_pdf_statement(client):
    application_id = apply(client)

    response = upload(client, application_id, build_pdf(SINGLE_PDF_LINES), "statement.pdf")

    assert response.status_code == 201, response.text
    assert response.json()["transactions_found"] == 4


def test_uploaded_file_is_not_left_on_disk(client, monkeypatch, tmp_path):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    application_id = apply(client)

    assert upload(client, application_id, rows_to_csv(statement_rows())).status_code == 201
    assert upload(client, application_id, b"%PDF-1.4 broken", "x.pdf").status_code == 422
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "filename, content, expected",
    [
        ("statement.exe", b"MZ", 400),
        ("statement.pdf", b"<html>not a pdf</html>", 400),
        ("statement.csv", b"", 400),
        ("statement.csv", b"Date,Description\n", 422),
        ("statement.json", b"{broken", 422),
    ],
)
def test_bad_uploads_get_a_clear_client_error(client, filename, content, expected):
    response = upload(client, apply(client), content, filename)

    assert response.status_code == expected
    assert "Traceback" not in response.text


def test_oversized_upload_is_refused(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    from src.app.config import get_settings

    get_settings.cache_clear()
    try:
        response = upload(client, apply(client), b"a" * 5000)
    finally:
        monkeypatch.delenv("MAX_UPLOAD_BYTES")
        get_settings.cache_clear()

    assert response.status_code == 413


def test_same_file_cannot_be_uploaded_twice(client):
    application_id = apply(client)
    content = rows_to_csv(statement_rows())

    assert upload(client, application_id, content).status_code == 201
    assert upload(client, application_id, content).status_code == 409


def test_tampered_statement_produces_a_warning(client):
    rows = statement_rows(months=4)
    rows[7]["deposit"] += 25000.0  # inflate one salary credit, leave the balances alone

    response = upload(client, apply(client), rows_to_csv(rows))

    assert response.status_code == 201
    quality = response.json()["quality"]
    assert quality["balance_checks_passed"] == quality["balance_checks"] - 1
    assert quality["reconciliation_rate"] < 0.9  # one row, but a big share of the money
    assert "manual review" in response.json()["warnings"][0]


# ------------------------------------------------------------------ decisions
def test_decide_requires_a_statement(client):
    assert client.post(f"/api/v1/loans/{apply(client)}/decide").status_code == 409


def test_strong_applicant_is_approved_and_audited(client):
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(statement_rows(months=4)))

    response = client.post(f"/api/v1/loans/{application_id}/decide")

    assert response.status_code == 200, response.text
    decision = response.json()
    assert decision["outcome"] == "approved" and decision["approved"] is True
    assert decision["llm"]["status"] == "disabled"
    assert decision["credit_score"] >= 65

    application = client.get(f"/api/v1/loans/{application_id}").json()
    assert application["status"] == "approved"
    assert application["credit_score"] == decision["credit_score"]

    stored = client.get(
        f"/api/v1/loans/{application_id}/decisions/{decision['decision_id']}"
    ).json()
    assert stored["scorecard"] == decision["scorecard"]
    assert stored["features"] == decision["features"]


def test_distressed_applicant_is_rejected(client):
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(distressed_rows()))

    decision = client.post(f"/api/v1/loans/{application_id}/decide").json()

    assert decision["outcome"] == "rejected"
    assert decision["red_flags"]
    assert client.get("/api/v1/loans?status=rejected").json()[0]["id"] == application_id


def test_tampered_statement_is_never_auto_approved(client):
    rows = statement_rows(months=4)
    rows[14]["deposit"] *= 3  # a single doctored salary credit
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(rows))

    decision = client.post(f"/api/v1/loans/{application_id}/decide").json()

    assert decision["outcome"] == "manual_review"


def test_every_decision_run_is_kept(client):
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(statement_rows(months=4)))

    first = client.post(f"/api/v1/loans/{application_id}/decide").json()
    second = client.post(f"/api/v1/loans/{application_id}/decide").json()

    history = client.get(f"/api/v1/loans/{application_id}/decisions").json()
    assert [d["id"] for d in history] == [first["decision_id"], second["decision_id"]]
    assert client.get(f"/api/v1/loans/{application_id}/decisions/9999").status_code == 404


def test_decision_with_llm_pipeline(client):
    app.dependency_overrides[get_pipeline] = lambda: FinancialIntelligencePipeline(
        MockLLMClient()
    )
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(statement_rows(months=4)))

    decision = client.post(f"/api/v1/loans/{application_id}/decide").json()

    assert decision["outcome"] == "approved"
    assert decision["llm"]["status"] == "completed"
    assert [stage["status"] for stage in decision["llm"]["stages"]] == ["ok"] * 5


def test_llm_outage_results_in_manual_review_not_rejection(client):
    class DownClient(MockLLMClient):
        def complete_json(self, stage, system_prompt, prompt):
            raise LLMError("Cannot reach Ollama at http://localhost:11434")

    app.dependency_overrides[get_pipeline] = lambda: FinancialIntelligencePipeline(DownClient())
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(statement_rows(months=4)))

    decision = client.post(f"/api/v1/loans/{application_id}/decide").json()

    assert decision["outcome"] == "manual_review"
    assert decision["llm"]["status"] == "failed"
    assert client.get(f"/api/v1/loans/{application_id}").json()["status"] == "manual_review"


def test_unexpected_errors_do_not_leak_details(client):
    def explode():
        raise RuntimeError("secret internal detail")

    app.dependency_overrides[get_pipeline] = explode
    application_id = apply(client)
    upload(client, application_id, rows_to_csv(statement_rows()))

    response = client.post(f"/api/v1/loans/{application_id}/decide")

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


# -------------------------------------------------------------- configuration
@pytest.mark.parametrize(
    "env, message",
    [
        ({"API_KEY": ""}, "API_KEY is not set"),
        ({"API_KEY": "my-secret-key-123"}, "random secret"),
        ({"API_KEY": "short"}, "random secret"),
        ({"LLM_PROVIDER": "gpt"}, "LLM_PROVIDER must be one of"),
        ({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}, "requires OPENAI_API_KEY"),
        ({"LLM_PROVIDER": "mock", "ENVIRONMENT": "production"}, "not allowed in production"),
        ({"LLM_TIMEOUT_SECONDS": "soon"}, "must be a number"),
        ({"MAX_UPLOAD_BYTES": "0"}, "at least 1"),
    ],
)
def test_bad_configuration_stops_startup(monkeypatch, env, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        load_settings()


def test_valid_configuration_loads(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")

    settings = load_settings()

    assert settings.llm_model == "llama3.2"
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.api_key == os.environ["API_KEY"]
