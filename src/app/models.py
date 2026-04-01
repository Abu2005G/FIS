try:
    from sqlalchemy import (
        Column,
        Integer,
        String,
        DateTime,
        Float,
        Boolean,
        Text,
        ForeignKey,
        Enum,
    )
    from sqlalchemy.orm import relationship
except ModuleNotFoundError:
    raise ImportError(
        "SQLAlchemy is not installed. Please run 'pip install sqlalchemy'."
    )

from datetime import datetime
from src.app.database import Base
import enum


class LoanStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, index=True)
    applicant_name = Column(String, nullable=False)
    applicant_email = Column(String, nullable=False)
    loan_amount = Column(Float, nullable=False)
    loan_purpose = Column(String)
    status = Column(String, default=LoanStatus.PENDING.value)
    credit_score = Column(Float)
    risk_level = Column(String)  # low, medium, high
    decision_reason = Column(Text)
    monthly_income = Column(Float)
    total_debt = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    statements = relationship(
        "FinancialStatement", back_populates="application", cascade="all, delete-orphan"
    )


class FinancialStatement(Base):
    __tablename__ = "financial_statements"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("loan_applications.id"))
    statement_type = Column(String)  # bank_statement, payslip, etc.
    file_path = Column(String)
    extracted_data = Column(Text)  # JSON string of extracted features
    total_deposits = Column(Float, default=0)
    total_withdrawals = Column(Float, default=0)
    average_balance = Column(Float, default=0)
    transaction_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    application = relationship("LoanApplication", back_populates="statements")
