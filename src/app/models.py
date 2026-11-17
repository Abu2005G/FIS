"""Database tables."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LoanStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    applicant_name: Mapped[str] = mapped_column(String(200), nullable=False)
    applicant_email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    loan_amount: Mapped[float] = mapped_column(Float, nullable=False)
    loan_term_months: Mapped[int] = mapped_column(Integer, nullable=False, default=36)
    loan_purpose: Mapped[Optional[str]] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=LoanStatus.PENDING.value, index=True
    )
    credit_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-100, higher is safer
    risk_level: Mapped[Optional[str]] = mapped_column(String(10))
    decision_reason: Mapped[Optional[str]] = mapped_column(Text)  # JSON list of reasons
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    statements: Mapped[List["FinancialStatement"]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )
    decisions: Mapped[List["LoanDecision"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="LoanDecision.id",
    )


class FinancialStatement(Base):
    __tablename__ = "financial_statements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("loan_applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    statement_type: Mapped[str] = mapped_column(String(30), nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    extracted_data: Mapped[str] = mapped_column(Text, nullable=False)  # JSON transactions
    parse_quality: Mapped[Optional[str]] = mapped_column(Text)  # JSON reconciliation stats
    period_start: Mapped[Optional[str]] = mapped_column(String(10))
    period_end: Mapped[Optional[str]] = mapped_column(String(10))
    total_deposits: Mapped[float] = mapped_column(Float, default=0.0)
    total_withdrawals: Mapped[float] = mapped_column(Float, default=0.0)
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped["LoanApplication"] = relationship(back_populates="statements")


class LoanDecision(Base):
    """Immutable record of one decision run: the audit trail."""

    __tablename__ = "loan_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("loan_applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    credit_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(20), nullable=False)
    llm_model: Mapped[Optional[str]] = mapped_column(String(80))
    detail: Mapped[str] = mapped_column(Text, nullable=False)  # full decision as JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped["LoanApplication"] = relationship(back_populates="decisions")
