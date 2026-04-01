# Financial Intelligence & Risk Signal Engine

A **prompt-based LLM pipeline** for credit risk assessment that uses multi-stage reasoning to analyze financial statements and make lending decisions.

## Architecture

The system uses a **5-stage prompt pipeline**:

```
Raw Bank Statement
       ↓
Prompt 1 — Transaction Structuring
       ↓
Prompt 2 — Transaction Classification
       ↓
Prompt 3 — Financial Behaviour Analysis
       ↓
Prompt 4 — Risk Signal Detection
       ↓
Prompt 5 — Financial Intelligence Summary → Loan Decision
```

## Files Created

### Core Pipeline
- `app/prompts/__init__.py` - Pipeline module
- `app/prompts/templates.py` - 5 LLM prompt templates
- `app/prompts/pipeline.py` - Pipeline executor with audit logging

### Database Models
- `app/models.py` - Updated with LoanApplication, FinancialStatement models

### API Routes
- `app/routes.py` - Endpoints for loan applications, statement upload, and decisions

### Legacy (Optional)
- `app/ml/features.py` - Traditional feature extraction
- `app/ml/model.py` - Rule-based fallback model
- `app/ml/statement_parser.py` - PDF transaction extractor

## Quick Start

### 1. Install Ollama (for LLM inference)

```bash
# macOS
brew install ollama

# Or download from https://ollama.com

# Pull a model
ollama pull llama3.2
ollama serve
```

### 2. Start the API

```bash
cd /Users/abdulvaseemmasood/FIS
source fintech_env/bin/activate
uvicorn app.main:app --reload
```

### 3. Test the Pipeline

```bash
# 1. Create a loan application
curl -X POST http://localhost:8000/api/v1/loans/apply \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_name": "John Doe",
    "applicant_email": "john@example.com",
    "loan_amount": 100000,
    "loan_purpose": "Home Renovation"
  }'

# Response: {"id": 1, ...}

# 2. Upload a bank statement PDF
curl -X POST http://localhost:8000/api/v1/loans/1/upload-statement \
  -H "X-API-Key: test-key" \
  -F "statement_type=bank_statement" \
  -F "file=@statement.pdf"

# 3. Get credit decision via LLM Pipeline
curl -X POST http://localhost:8000/api/v1/loans/1/decide \
  -H "X-API-Key: test-key"
```

## Pipeline Stages Explained

### Stage 1: Transaction Structuring
Converts messy PDF text into structured JSON with date, merchant, amount, and type.

**Example:**
```
Input:  "01/01/24 UPI-INDIAN CLEARING CORP-MUTUALFUNDS 1000"
Output: {"date": "01/01/24", "merchant": "Mutual Funds", "amount": 1000, "type": "debit"}
```

### Stage 2: Transaction Classification
Classifies transactions into semantic categories:
- Income (salary, freelance)
- Expense (bills, shopping)
- Investment (mutual funds, stocks)
- Transfer
- Cash Withdrawal
- Loan Payment

### Stage 3: Financial Behaviour Analysis
Analyzes patterns over time:
- Income stability
- Spending patterns
- Investment activity
- Cash management
- Debt obligations

### Stage 4: Risk Signal Detection
Identifies risk indicators:
- Irregular income
- High cash withdrawals
- Gambling activity
- Loan repayment stress
- Suspicious patterns
- Overdraft history

### Stage 5: Intelligence Summary
Generates final decision:
```json
{
  "credit_decision": {
    "approved": true,
    "confidence": 0.85,
    "max_approved_amount": 100000,
    "reasoning": ["Stable income", "Low debt burden"]
  },
  "risk_profile": {
    "score": 0.25,
    "level": "low"
  },
  "audit_summary": {
    "recommendation": "approve"
  }
}
```

## Compliance Features

The system is designed for **bank-grade** use with:

1. **Audit Trail** - Every pipeline stage is logged with:
   - Input/output data
   - Confidence scores
   - Execution timestamps
   - Error tracking

2. **Explainability** - Every decision includes:
   - Risk signals detected
   - Reasoning steps
   - Confidence metrics

3. **Deterministic** - Same input always produces same output (temperature=0.2)

4. **Configurable** - Switch LLM providers:
   - Ollama (local, private)
   - OpenAI (cloud)
   - Mock (testing)

## Configuration

Set environment variables:

```bash
# For Ollama (default)
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.2

# For OpenAI
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=your-key
```

## Testing

```bash
# Run pipeline test
python test_pipeline.py

# Run FastAPI
uvicorn app.main:app --reload

# View API docs
open http://localhost:8000/docs
```

## Architecture Decision

This system differs from traditional ML approaches:

| Traditional ML | Prompt Pipeline |
|---------------|-------------------|
| Pre-defined features | LLM extracts context |
| Statistical model | Reasoning-based |
| Black box | Explainable decisions |
| Needs training data | Works out of the box |

## Future Improvements

- [ ] LangGraph integration for more complex workflows
- [ ] Real-time transaction monitoring
- [ ] Graph-based financial analysis
- [ ] Fraud detection module
- [ ] Banking API integrations (Plaid, Yodlee)

## License

MIT
