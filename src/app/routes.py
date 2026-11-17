"""HTTP API routes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.app.auth import verify_api_key
from src.app.config import Settings, get_settings
from src.app.database import get_db
from src.app.decision import ENGINE_VERSION, make_decision, merge_transactions
from src.app.ml.model import MODEL_VERSION, RiskPolicy
from src.app.ml.statement_parser import SUPPORTED_SUFFIXES, StatementParseError, parse_statement
from src.app.models import FinancialStatement, LoanApplication, LoanDecision, LoanStatus
from src.app.prompts.llm_client import build_llm_client
from src.app.prompts.pipeline import FinancialIntelligencePipeline
from src.app.schemas import (
    DecisionSummary,
    LoanApplicationCreate,
    LoanApplicationResponse,
    LoanDecisionResponse,
    StatementResponse,
    StatementUploadResponse,
)

logger = logging.getLogger(__name__)

public_router = APIRouter(prefix="/api/v1", tags=["service"])
router = APIRouter(prefix="/api/v1", tags=["loans"], dependencies=[Depends(verify_api_key)])

STATEMENT_TYPES = ("bank_statement",)
UPLOAD_CHUNK_BYTES = 1024 * 1024


@lru_cache(maxsize=1)
def _shared_pipeline() -> Optional[FinancialIntelligencePipeline]:
    settings = get_settings()
    client = build_llm_client(settings)
    if client is None:
        return None
    return FinancialIntelligencePipeline(
        client,
        batch_size=settings.llm_batch_size,
        max_transactions=settings.llm_max_transactions,
    )


def get_pipeline() -> Optional[FinancialIntelligencePipeline]:
    """The LLM pipeline, or None when LLM_PROVIDER=none. Stateless, so shared."""
    return _shared_pipeline()


def _get_application(db: Session, application_id: int) -> LoanApplication:
    application = db.get(LoanApplication, application_id)
    if application is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Loan application not found")
    return application


# ------------------------------------------------------------------ service
@public_router.get("/health")
def health_check(
    db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """Liveness and readiness. Returns 503 if the database is unreachable."""
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Health check: database unreachable")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from None
    return {
        "status": "healthy",
        "database": "ok",
        "scorecard_version": MODEL_VERSION,
        "engine_version": ENGINE_VERSION,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model if settings.llm_enabled else None,
    }


# -------------------------------------------------------------------- loans
@router.post(
    "/loans/apply", response_model=LoanApplicationResponse, status_code=status.HTTP_201_CREATED
)
def create_loan_application(
    application: LoanApplicationCreate, db: Session = Depends(get_db)
) -> LoanApplication:
    """Create a loan application."""
    record = LoanApplication(
        applicant_name=application.applicant_name,
        applicant_email=application.applicant_email,
        loan_amount=application.loan_amount,
        loan_term_months=application.loan_term_months,
        loan_purpose=application.loan_purpose,
        status=LoanStatus.PENDING.value,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("Created loan application id=%s", record.id)
    return record


@router.get("/loans", response_model=List[LoanApplicationResponse])
def list_loan_applications(
    status_filter: Optional[LoanStatus] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> List[LoanApplication]:
    """List applications, newest first."""
    query = select(LoanApplication)
    if status_filter is not None:
        query = query.where(LoanApplication.status == status_filter.value)
    query = query.order_by(LoanApplication.id.desc()).limit(limit).offset(offset)
    return list(db.scalars(query))


@router.get("/loans/{application_id}", response_model=LoanApplicationResponse)
def get_loan_application(application_id: int, db: Session = Depends(get_db)) -> LoanApplication:
    """Get one application."""
    return _get_application(db, application_id)


# --------------------------------------------------------------- statements
def _save_upload(upload: UploadFile, suffix: str, max_bytes: int) -> tuple:
    """Stream the upload to a private temp file. Returns (path, sha256)."""
    digest = hashlib.sha256()
    size = 0
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="fis-upload-")
    try:
        with handle:
            while True:
                chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        413,
                        f"File is larger than the {max_bytes // (1024 * 1024)} MB limit",
                    )
                digest.update(chunk)
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty")
        if suffix == ".pdf":
            with open(handle.name, "rb") as saved:
                if not saved.read(5).startswith(b"%PDF-"):
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST, "The file is not a PDF document"
                    )
    except BaseException:
        os.unlink(handle.name)
        raise
    return handle.name, digest.hexdigest()


@router.post(
    "/loans/{application_id}/upload-statement",
    response_model=StatementUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_financial_statement(
    application_id: int,
    statement_type: str = Form(default="bank_statement"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Upload a bank statement (PDF, CSV or JSON) for an application.

    The file is parsed and then deleted; only the extracted transactions are
    stored.
    """
    _get_application(db, application_id)

    if statement_type not in STATEMENT_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"statement_type must be one of: {', '.join(STATEMENT_TYPES)}",
        )
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type. Upload one of: {', '.join(SUPPORTED_SUFFIXES)}",
        )

    tmp_path, sha256 = _save_upload(file, suffix, settings.max_upload_bytes)
    try:
        duplicate = db.scalar(
            select(FinancialStatement.id).where(
                FinancialStatement.application_id == application_id,
                FinancialStatement.file_sha256 == sha256,
            )
        )
        if duplicate is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"This file was already uploaded as statement {duplicate}",
            )
        try:
            parsed = parse_statement(tmp_path, suffix, day_first=settings.statement_day_first)
        except StatementParseError as exc:
            raise HTTPException(422, str(exc)) from exc
    finally:
        os.unlink(tmp_path)  # statements hold personal data; never keep the file

    transactions = parsed.transactions
    statement = FinancialStatement(
        application_id=application_id,
        statement_type=statement_type,
        original_filename=filename[:255],
        file_sha256=sha256,
        extracted_data=json.dumps({"transactions": transactions}),
        parse_quality=json.dumps(parsed.quality),
        period_start=parsed.metadata.get("period_start"),
        period_end=parsed.metadata.get("period_end"),
        total_deposits=round(sum(txn.get("deposit") or 0 for txn in transactions), 2),
        total_withdrawals=round(sum(txn.get("withdrawal") or 0 for txn in transactions), 2),
        transaction_count=len(transactions),
    )
    db.add(statement)
    db.commit()
    db.refresh(statement)

    warnings: List[str] = []
    rate = parsed.quality.get("reconciliation_rate")
    if rate is None:
        warnings.append("The statement has no running balance, so it could not be cross-checked")
    elif rate < RiskPolicy().min_reconciliation_rate:
        warnings.append(
            f"Only {rate:.0%} of the statement agrees with its running balance; "
            "this application will need a manual review"
        )
    logger.info(
        "Stored statement id=%s application=%s transactions=%s reconciliation=%s",
        statement.id,
        application_id,
        len(transactions),
        rate,
    )
    return {
        "statement_id": statement.id,
        "application_id": application_id,
        "transactions_found": len(transactions),
        "period_start": statement.period_start,
        "period_end": statement.period_end,
        "total_deposits": statement.total_deposits,
        "total_withdrawals": statement.total_withdrawals,
        "quality": parsed.quality,
        "warnings": warnings,
    }


@router.get("/loans/{application_id}/statements", response_model=List[StatementResponse])
def get_application_statements(
    application_id: int, db: Session = Depends(get_db)
) -> List[FinancialStatement]:
    """List the statements uploaded for an application."""
    return list(_get_application(db, application_id).statements)


# ---------------------------------------------------------------- decisions
@router.post("/loans/{application_id}/decide", response_model=LoanDecisionResponse)
def make_loan_decision(
    application_id: int,
    db: Session = Depends(get_db),
    pipeline: Optional[FinancialIntelligencePipeline] = Depends(get_pipeline),
) -> Dict[str, Any]:
    """
    Decide an application from its uploaded statements.

    Each call stores a new, immutable decision record. With an LLM configured
    this request can take minutes; call it from a worker, not a browser.
    """
    application = _get_application(db, application_id)
    statements = list(application.statements)
    if not statements:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "No statements uploaded. Upload a bank statement first."
        )

    transaction_sets: List[List[Dict[str, Any]]] = []
    rates: List[float] = []
    for statement in statements:
        try:
            data = json.loads(statement.extracted_data)
        except json.JSONDecodeError:
            logger.error("Statement id=%s holds unreadable data", statement.id)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "Stored statement data is unreadable"
            ) from None
        transaction_sets.append(data.get("transactions", []) if isinstance(data, dict) else data)
        quality = json.loads(statement.parse_quality) if statement.parse_quality else {}
        if quality.get("reconciliation_rate") is not None:
            rates.append(quality["reconciliation_rate"])

    transactions = merge_transactions(transaction_sets)
    if not transactions:
        raise HTTPException(
            422, "The uploaded statements hold no transactions"
        )

    decision = make_decision(
        transactions=transactions,
        loan_amount=application.loan_amount,
        loan_term_months=application.loan_term_months,
        loan_purpose=application.loan_purpose or "",
        reconciliation_rate=min(rates) if rates else None,
        pipeline=pipeline,
    )

    llm = decision["llm"]
    record = LoanDecision(
        application_id=application.id,
        outcome=decision["outcome"],
        credit_score=decision["credit_score"],
        risk_level=decision["risk_level"],
        model_version=decision["scorecard"]["model_version"],
        llm_provider=llm.get("provider", "none"),
        llm_model=llm.get("model"),
        detail=json.dumps(decision, default=str),
    )
    application.status = decision["outcome"]
    application.credit_score = decision["credit_score"]
    application.risk_level = decision["risk_level"]
    application.decision_reason = json.dumps(decision["reasoning"])
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Decision id=%s application=%s outcome=%s score=%s llm=%s",
        record.id,
        application.id,
        record.outcome,
        record.credit_score,
        llm.get("status"),
    )
    return {
        **decision,
        "decision_id": record.id,
        "application_id": application.id,
        "created_at": record.created_at,
    }


@router.get("/loans/{application_id}/decisions", response_model=List[DecisionSummary])
def list_decisions(application_id: int, db: Session = Depends(get_db)) -> List[LoanDecision]:
    """Decision history for an application, oldest first."""
    return list(_get_application(db, application_id).decisions)


@router.get("/loans/{application_id}/decisions/{decision_id}")
def get_decision(
    application_id: int, decision_id: int, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """The full stored audit record for one decision."""
    record = db.get(LoanDecision, decision_id)
    if record is None or record.application_id != application_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Decision not found")
    return {
        "decision_id": record.id,
        "application_id": record.application_id,
        "created_at": record.created_at,
        **json.loads(record.detail),
    }
