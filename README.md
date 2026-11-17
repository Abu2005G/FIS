# Financial Intelligence & Risk Signal Engine

Credit risk assessment from bank statements. Upload a statement, get an
explainable decision: **approved**, **rejected** or **manual review**.

Two engines work together:

- A **deterministic scorecard** reads the transactions, computes standard
  underwriting ratios and scores the application. It is the decision of record.
- An optional **5-stage LLM pipeline** analyses the same transactions and can
  flag concerns the scorecard cannot see.

## How a decision is made

```
statement (PDF / CSV / JSON)
        │
        ▼
  Statement parser ──► transactions + reconciliation check
        │
        ▼
  Feature extraction ──► income, spending, balances, debt, risk signals
        │
        ▼
  Scorecard ──► approve / review / reject, with a point-by-point breakdown
        │
        ▼ (only if the scorecard did not reject, and an LLM is configured)
  LLM pipeline: structuring → classification → behaviour → risk → summary
        │
        ▼
  Decision policy ──► approved / manual_review / rejected  (stored as an audit record)
```

The LLM can only make an outcome more cautious:

| Scorecard | LLM result                         | Outcome         |
|-----------|------------------------------------|-----------------|
| reject    | not consulted                      | `rejected`      |
| review    | anything                           | `manual_review` |
| approve   | switched off (`LLM_PROVIDER=none`) | `approved`      |
| approve   | clear approval, not high risk      | `approved`      |
| approve   | concerns, reject or review         | `manual_review` |
| approve   | failed or invalid output           | `manual_review` |

A model outage, a hallucination, or text planted in a transaction description
therefore cannot produce an approval. Model output is validated against a
schema at every stage, the applicant's name is never sent to the model, and
amounts always come from the parser, not the model.

## Quick start

```bash
python3 -m venv fintech_env
./fintech_env/bin/pip install -r requirements-dev.txt

cp .env.example .env
# put a random key in .env:
python -c "import secrets; print(secrets.token_urlsafe(32))"

./fintech_env/bin/uvicorn src.app.main:app --reload
```

Run commands from the repository root. Interactive docs are at
`http://localhost:8000/docs` (hidden when `ENVIRONMENT=production`).

To use a local model, install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
ollama serve
```

Without a model, set `LLM_PROVIDER=none` to run on the scorecard alone.

### Example

```bash
export KEY="<the API_KEY from .env>"
B=http://localhost:8000/api/v1

curl -X POST $B/loans/apply -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"applicant_name": "Jane Doe", "applicant_email": "jane@example.com",
       "loan_amount": 100000, "loan_term_months": 36, "loan_purpose": "Home renovation"}'

curl -X POST $B/loans/1/upload-statement -H "X-API-Key: $KEY" -F "file=@statement.pdf"

curl -X POST $B/loans/1/decide -H "X-API-Key: $KEY"
```

## API

All `/loans` routes need the `X-API-Key` header.

| Method | Path                                      | Purpose                                   |
|--------|-------------------------------------------|-------------------------------------------|
| GET    | `/api/v1/health`                          | Liveness and database check (no key)      |
| POST   | `/api/v1/loans/apply`                     | Create an application                     |
| GET    | `/api/v1/loans?status=&limit=&offset=`    | List applications                         |
| GET    | `/api/v1/loans/{id}`                      | Get one application                       |
| POST   | `/api/v1/loans/{id}/upload-statement`     | Upload a PDF, CSV or JSON statement       |
| GET    | `/api/v1/loans/{id}/statements`           | List uploaded statements                  |
| POST   | `/api/v1/loans/{id}/decide`               | Run a decision and store its audit record |
| GET    | `/api/v1/loans/{id}/decisions`            | Decision history                          |
| GET    | `/api/v1/loans/{id}/decisions/{decision}` | Full audit record for one decision        |

Uploaded files are parsed and deleted straight away; only the extracted
transactions are stored. Each `/decide` call adds a new, immutable decision
record holding the features, score breakdown, LLM stage log and outcome.

## Configuration

Set through environment variables or `.env`. See `.env.example` for the full list.

| Variable              | Default                  | Notes                                           |
|-----------------------|--------------------------|-------------------------------------------------|
| `API_KEY`             | none, required           | Random secret, at least 24 characters           |
| `ENVIRONMENT`         | `development`            | `production` hides docs and forbids the mock LLM |
| `DATABASE_URL`        | `sqlite:///./fis.db`     | Any SQLAlchemy URL                              |
| `LLM_PROVIDER`        | `ollama`                 | `none`, `ollama`, `openai`, `mock`              |
| `LLM_MODEL`           | `llama3.2`/`gpt-4o-mini` | Model name for the chosen provider              |
| `OLLAMA_BASE_URL`     | `http://localhost:11434` |                                                 |
| `OPENAI_API_KEY`      | none                     | Needed for `openai`, plus `pip install openai`  |
| `LLM_TIMEOUT_SECONDS` | `120`                    | Per model call                                  |
| `MAX_UPLOAD_BYTES`    | `10485760`               | Upload size limit                               |
| `STATEMENT_DAY_FIRST` | `true`                   | `03/04/2026` is 3 April when true               |

With `LLM_PROVIDER=openai`, transaction descriptions (which include
counterparty names) are sent to OpenAI. Use `ollama` to keep them on your own
machines.

## The scorecard

`src/app/ml/model.py`. Points out of 100, higher is safer:

| Component         | Max | Measures                                                   |
|-------------------|-----|------------------------------------------------------------|
| Affordability     | 25  | (existing debt payments + new instalment) / monthly income |
| Income stability  | 15  | Month-to-month variation in income                         |
| Savings rate      | 15  | Share of income not spent                                  |
| Liquidity         | 15  | Average balance in months of spending                      |
| Overdrafts        | 15  | Days the account was overdrawn                             |
| Existing debt     | 10  | Loan and card payments as a share of income                |
| Balance trend     | 5   | Whether the balance is rising or falling                   |
| Returned payments | -24 | Bounced payments and penalty charges                       |
| Gambling          | -10 | Gambling transactions                                      |

Approve at 65 or above with no red flags; reject below 40 or when repayments
would exceed 60% of income; otherwise manual review. Applications with under
60 days or 15 transactions of history, or whose statement balances do not add
up, are never approved automatically. Thresholds live in `RiskPolicy`.

The scorecard is hand-built from standard underwriting ratios. It is not a
statistically fitted model and has not been calibrated to any lender's book.

## Validation

### Scorecard against real loan outcomes

`scripts/validate_berka.py` scores the 682 loans in the public
[Berka / PKDD'99 dataset](https://web.archive.org/web/20070214120527/http://lisp.vse.cz/pkdd99/Challenge/chall.htm)
(real, anonymised Czech bank accounts, 76 loans went bad) using only
transactions from before each loan was granted.

| History used | AUC   | Bad rate: approve | review | reject | (all loans) |
|--------------|-------|-------------------|--------|--------|-------------|
| 180 days     | 0.760 | 3.3% (7 of 211)   | 10.2%  | 54.2%  | 11.1%       |
| 365 days     | 0.783 | 3.4% (9 of 261)   | 11.2%  | 55.6%  | 11.1%       |

The previous scoring code, run on the same loans, reached AUC 0.57 (180 days)
and 0.64 (365 days), with a 9.7% bad rate among the loans it approved.

Read these numbers with care: the data is from 1990s Czech accounts, the
sample is small, and no thresholds were tuned on it. It shows the scorecard
ranks risk in the right direction; it does not show how it will perform on
your applicants.

```bash
# data: see the docstring in scripts/validate_berka.py
./fintech_env/bin/python scripts/validate_berka.py --data-dir data/berka
```

### Statement parser

On a real 14-page bank statement (148 transactions) the parser reproduces the
bank's own printed summary exactly: debit and credit counts, both totals and
the closing balance, with every row reconciling against the running balance.

### Tests

```bash
./fintech_env/bin/python -m pytest
```

120 tests cover the parser, features, scorecard, LLM clients, pipeline,
decision policy and the HTTP API.

### Not yet validated

- **The LLM stages with a real model.** They are tested against scripted
  replies and a stand-in server that speaks Ollama's API. Run the pipeline with
  the model you intend to use and review its output before relying on it.
- **Other banks' statement layouts.** The parser is layout-agnostic and has
  tests for several formats, but only one real statement has been checked.
- **The Docker image build.**

## Docker

```bash
docker build -t fis .
docker run -p 8000:8000 -e API_KEY="$KEY" -e LLM_PROVIDER=none -v fis-data:/data fis
```

To reach Ollama on the host from the container, set
`OLLAMA_BASE_URL=http://host.docker.internal:11434`.

## Before real lending use

This codebase is engineered to fail safely, but software alone does not make a
lending system. Still needed:

- Calibration and back-testing of the scorecard on your own loan outcomes.
- Fair-lending, explainability and data-protection review for your jurisdiction.
- Per-user authentication and roles (the API has one shared key), TLS and rate
  limiting at a gateway.
- PostgreSQL and schema migrations (tables are created at startup today), and
  encryption at rest for stored transactions.
- A job queue for `/decide`: with an LLM it can run for minutes inside the request.
- OCR for scanned statements, which are rejected today.

## Project layout

```
src/app/
  main.py, config.py, auth.py, database.py, models.py, schemas.py, routes.py
  decision.py              scorecard + LLM decision policy
  ml/statement_parser.py   PDF / CSV / JSON -> transactions
  ml/features.py           transactions -> features
  ml/model.py              scorecard
  prompts/                 templates, LLM clients, output schemas, pipeline
scripts/validate_berka.py  validation against real loan outcomes
tests/
```

## License

MIT
