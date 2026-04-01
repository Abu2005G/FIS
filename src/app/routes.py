from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional, cast
from datetime import datetime
import json
import tempfile
import os

from src.app.auth import verify_api_key
from src.app.database import get_db
from src.app.models import Item, LoanApplication, FinancialStatement, LoanStatus
from src.app.ml.statement_parser import parse_transaction_file
from src.app.prompts.pipeline import get_pipeline, FinancialIntelligencePipeline
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")

# ============ Pydantic Models ============

class ItemCreate(BaseModel):
    name: str
    description: str = ""

class LoanApplicationCreate(BaseModel):
    applicant_name: str
    applicant_email: str
    loan_amount: float
    loan_purpose: Optional[str] = None

class LoanApplicationResponse(BaseModel):
    id: int
    applicant_name: str
    applicant_email: str
    loan_amount: float
    loan_purpose: Optional[str]
    status: str
    credit_score: Optional[float]
    risk_level: Optional[str]
    decision_reason: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class LoanDecisionResponse(BaseModel):
    application_id: int
    approved: bool
    probability: float
    credit_score: float
    risk_level: str
    reasoning: List[str]
    loan_amount: float

# ============ Existing Routes ============

@router.get("/health")
def health_check():
    return {"status": "healthy", "ml_model": "ready"}

@router.get("/protected")
def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "You have access! 🎉"}

@router.post("/items", dependencies=[Depends(verify_api_key)])
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@router.get("/items", dependencies=[Depends(verify_api_key)])
def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()

# ============ Loan Application Routes ============

@router.post("/loans/apply", response_model=LoanApplicationResponse, dependencies=[Depends(verify_api_key)])
def create_loan_application(
    application: LoanApplicationCreate,
    db: Session = Depends(get_db)
):
    """Create a new loan application."""
    db_application = LoanApplication(
        applicant_name=application.applicant_name,
        applicant_email=application.applicant_email,
        loan_amount=application.loan_amount,
        loan_purpose=application.loan_purpose,
        status=LoanStatus.PENDING.value
    )
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    return db_application


@router.post("/loans/{application_id}/upload-statement", dependencies=[Depends(verify_api_key)])
def upload_financial_statement(
    application_id: int,
    statement_type: str = Form(..., description="Type: bank_statement, payslip, etc."),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a financial statement (PDF) for a loan application.
    Parses the PDF and stores raw transaction data for the LLM pipeline.
    """
    # Check if application exists
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Loan application not found")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = file.file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Parse the PDF to extract transactions
        transactions = parse_transaction_file(tmp_path)

        # Store raw transactions for LLM pipeline processing
        statement = FinancialStatement(
            application_id=application_id,
            statement_type=statement_type,
            file_path=tmp_path,
            extracted_data=json.dumps({"transactions": transactions}),  # Store raw transactions
            total_deposits=sum(t.get('deposit', 0) or 0 for t in transactions),
            total_withdrawals=sum(t.get('withdrawal', 0) or 0 for t in transactions),
            average_balance=0,  # Calculated during pipeline
            transaction_count=len(transactions)
        )
        db.add(statement)
        db.commit()
        db.refresh(statement)

        return {
            "message": "Statement uploaded successfully. Ready for LLM analysis.",
            "statement_id": statement.id,
            "transactions_found": len(transactions),
            "next_step": f"POST /api/v1/loans/{application_id}/decide to get credit decision"
        }

    except Exception as e:
        os.unlink(tmp_path)  # Clean up temp file
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")


@router.post("/loans/{application_id}/decide", dependencies=[Depends(verify_api_key)])
def make_loan_decision(
    application_id: int,
    db: Session = Depends(get_db)
):
    """
    Make a loan approval decision using the 5-stage LLM Prompt Pipeline.
    Requires financial statements to be uploaded first.

    Pipeline: Structuring → Classification → Behaviour → Risk → Intelligence
    """
    # Get application
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Loan application not found")

    # Get financial statements
    statements = db.query(FinancialStatement).filter(
        FinancialStatement.application_id == application_id
    ).all()

    if not statements:
        raise HTTPException(
            status_code=400,
            detail="No financial statements found. Please upload statements first."
        )

    # Extract raw transactions from statements
    all_transactions = []
    for stmt in statements:
        raw_data = cast(Optional[str], stmt.extracted_data)
        if raw_data:
            data = json.loads(raw_data)
            if isinstance(data, list):
                all_transactions.extend(data)
            elif isinstance(data, dict) and "transactions" in data:
                all_transactions.extend(data["transactions"])

    if not all_transactions:
        raise HTTPException(
            status_code=400,
            detail="No extractable transactions found in statements."
        )

    # Format transactions for pipeline (convert to raw strings)
    raw_transactions = []
    for txn in all_transactions:
        if isinstance(txn, dict):
            # Format as readable string
            date = txn.get('date', 'N/A')
            desc = txn.get('description', txn.get('merchant', 'Unknown'))
            withdrawal = txn.get('withdrawal', txn.get('debit', ''))
            deposit = txn.get('deposit', txn.get('credit', ''))
            amount = deposit if deposit else withdrawal
            txn_type = 'credit' if deposit else 'debit'
            raw_transactions.append(f"{date} {desc} {amount} {txn_type}")
        elif isinstance(txn, str):
            raw_transactions.append(txn)

    # Run the 5-stage LLM Pipeline
    pipeline = get_pipeline()
    result = pipeline.process_application(
        application_id=cast(int, application.id),
        applicant_name=cast(str, application.applicant_name),
        loan_amount=cast(float, application.loan_amount),
        loan_purpose=cast(str, application.loan_purpose or "Personal"),
        raw_transactions=raw_transactions[:30]  # Process first 30 transactions
    )

    decision = result['decision']
    credit_decision = decision.get('credit_decision', {})
    risk_profile = decision.get('risk_profile', {})

    # Update application with decision
    approved = credit_decision.get('approved', False)
    setattr(application, 'status', LoanStatus.APPROVED.value if approved else LoanStatus.REJECTED.value)
    setattr(application, 'credit_score', risk_profile.get('score', 0) * 100)  # Convert to 0-100 scale
    setattr(application, 'risk_level', risk_profile.get('level', 'unknown'))
    setattr(application, 'decision_reason', json.dumps({
        "reasoning": credit_decision.get('reasoning', []),
        "audit_summary": decision.get('audit_summary', {}),
        "pipeline_confidence": result['processing_summary']['overall_confidence']
    }))

    db.commit()

    return {
        "application_id": application_id,
        "approved": approved,
        "confidence": credit_decision.get('confidence', 0),
        "risk_score": risk_profile.get('score', 0),
        "risk_level": risk_profile.get('level', 'unknown'),
        "max_approved_amount": credit_decision.get('max_approved_amount', 0),
        "reasoning": credit_decision.get('reasoning', []),
        "audit_trail": result['audit_log'],
        "processing_summary": result['processing_summary']
    }


@router.get("/loans", response_model=List[LoanApplicationResponse], dependencies=[Depends(verify_api_key)])
def list_loan_applications(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all loan applications with optional status filter."""
    query = db.query(LoanApplication)
    if status:
        query = query.filter(LoanApplication.status == status)
    return query.order_by(LoanApplication.created_at.desc()).all()


@router.get("/loans/{application_id}", response_model=LoanApplicationResponse, dependencies=[Depends(verify_api_key)])
def get_loan_application(
    application_id: int,
    db: Session = Depends(get_db)
):
    """Get details of a specific loan application."""
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Loan application not found")
    return application


@router.get("/loans/{application_id}/statements", dependencies=[Depends(verify_api_key)])
def get_application_statements(
    application_id: int,
    db: Session = Depends(get_db)
):
    """Get all financial statements for a loan application."""
    statements = db.query(FinancialStatement).filter(
        FinancialStatement.application_id == application_id
    ).all()

    return [
        {
            "id": s.id,
            "statement_type": s.statement_type,
            "transaction_count": s.transaction_count,
            "total_deposits": s.total_deposits,
            "total_withdrawals": s.total_withdrawals,
            "average_balance": s.average_balance,
            "created_at": s.created_at
        }
        for s in statements
    ]
