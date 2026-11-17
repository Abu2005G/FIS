"""
5-stage LLM pipeline: structuring -> classification -> behaviour -> risk -> summary.

The pipeline is advisory. It produces analysis and a recommendation that
``src.app.decision`` weighs against the deterministic scorecard.

Guarantees
----------
* No shared state: every run keeps its data in local variables, so one
  pipeline object is safe to use from many request threads.
* Every model reply is validated against a schema before it is used.
* A failed stage is recorded as failed. It never turns into a silent
  rejection or approval.
* The audit record holds what actually happened (status, timing, call counts,
  errors), not estimated confidence figures.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.app.prompts.llm_client import LLMClient, LLMError
from src.app.prompts.schemas import (
    UNKNOWN_CATEGORY,
    BehaviourAnalysis,
    ClassificationOutput,
    IntelligenceSummary,
    RiskDetection,
    StructuringOutput,
)
from src.app.prompts.templates import (
    BEHAVIOUR_ANALYSIS_SYSTEM_PROMPT,
    CLASSIFICATION_SYSTEM_PROMPT,
    FINANCIAL_BEHAVIOUR_PROMPT,
    INTELLIGENCE_SUMMARY_PROMPT,
    INTELLIGENCE_SYSTEM_PROMPT,
    RISK_DETECTION_PROMPT,
    RISK_DETECTION_SYSTEM_PROMPT,
    STRUCTURING_SYSTEM_PROMPT,
    TRANSACTION_CLASSIFICATION_PROMPT,
    TRANSACTION_STRUCTURING_PROMPT,
)

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "pipeline-2.0"

OK = "ok"
DEGRADED = "degraded"
FAILED = "failed"
SKIPPED = "skipped"

VALIDATION_ATTEMPTS = 2
BEHAVIOUR_SAMPLE_SIZE = 60
DESCRIPTION_LIMIT = 80

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class StageRecord:
    """What happened in one pipeline stage."""

    stage_name: str
    status: str
    started_at: str
    execution_time_ms: float = 0.0
    llm_calls: int = 0
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Outcome of one pipeline run."""

    status: str  # "completed" or "failed"
    provider: str
    model: str
    transactions_total: int
    transactions_analysed: int
    stages: List[StageRecord] = field(default_factory=list)
    behaviour: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, limit: int = DESCRIPTION_LIMIT) -> str:
    """Neutralise statement text before it goes into a prompt."""
    text = re.sub(r"[\x00-\x1f\x7f<>{}`]", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


class _StageFailure(Exception):
    """A stage could not produce a valid result."""


class FinancialIntelligencePipeline:
    """Runs the five LLM stages for one application."""

    def __init__(self, client: LLMClient, batch_size: int = 25, max_transactions: int = 150):
        self.client = client
        self.batch_size = max(1, batch_size)
        self.max_transactions = max(1, max_transactions)

    # ------------------------------------------------------------- helpers
    def _ask(
        self, stage: str, system_prompt: str, prompt: str, schema: Type[SchemaT], calls: List[int]
    ) -> SchemaT:
        """Call the model and validate its reply, retrying once on bad output."""
        last_error = "no attempt made"
        for _ in range(VALIDATION_ATTEMPTS):
            calls[0] += 1
            try:
                raw = self.client.complete_json(stage, system_prompt, prompt)
            except LLMError as exc:
                raise _StageFailure(str(exc)) from exc
            try:
                return schema.model_validate(raw)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors()[:3]
                )
                last_error = f"Model output failed validation ({problems})"
                logger.warning("Stage %s: %s", stage, last_error)
        raise _StageFailure(last_error)

    def _batches(self, items: Sequence[Dict[str, Any]]) -> List[Sequence[Dict[str, Any]]]:
        return [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    # --------------------------------------------------------------- stages
    def _structure(self, working: List[Dict[str, Any]]) -> StageRecord:
        record = StageRecord("transaction_structuring", OK, _now())
        started = time.perf_counter()
        calls = [0]
        failed_batches = mismatches = structured = 0
        errors: List[str] = []

        attempted = 0
        for batch in self._batches(working):
            attempted += 1
            payload = [
                {
                    "index": txn["index"],
                    "text": f"{txn['date']} {txn['description']}",
                    "amount": txn["amount"],
                    "type": txn["type"],
                }
                for txn in batch
            ]
            prompt = TRANSACTION_STRUCTURING_PROMPT.format(transactions_json=_compact(payload))
            try:
                output = self._ask(
                    record.stage_name, STRUCTURING_SYSTEM_PROMPT, prompt, StructuringOutput, calls
                )
            except _StageFailure as exc:
                failed_batches += 1
                errors.append(str(exc))
                if failed_batches == attempted:
                    break  # nothing has worked yet: do not keep calling a dead backend
                continue

            by_index = {txn["index"]: txn for txn in batch}
            for item in output.transactions:
                target = by_index.get(item.index)
                if target is None or "merchant" in target:
                    continue
                target["merchant"] = _safe_text(item.merchant, 60)
                structured += 1
                # The parser's amount and direction are authoritative. Count
                # disagreements as a measure of how far to trust the model.
                if (
                    abs(item.amount - target["amount"]) > 0.01
                    or item.transaction_type != target["type"]
                ):
                    mismatches += 1

        for txn in working:
            txn.setdefault("merchant", txn["description"][:60] or "Unknown")

        total_batches = len(self._batches(working))
        record.execution_time_ms = round((time.perf_counter() - started) * 1000, 1)
        record.llm_calls = calls[0]
        record.detail = {
            "transactions": len(working),
            "structured_by_model": structured,
            "batches": total_batches,
            "failed_batches": failed_batches,
            "model_disagreements": mismatches,
        }
        if failed_batches == attempted:
            record.status, record.error = FAILED, errors[0]
        elif failed_batches or structured < len(working):
            record.status = DEGRADED
            record.error = errors[0] if errors else "Model skipped some transactions"
        return record

    def _classify(self, working: List[Dict[str, Any]]) -> StageRecord:
        record = StageRecord("transaction_classification", OK, _now())
        started = time.perf_counter()
        calls = [0]
        failed_batches = classified = 0
        errors: List[str] = []

        attempted = 0
        for batch in self._batches(working):
            attempted += 1
            payload = [
                {
                    "index": txn["index"],
                    "merchant": txn["merchant"],
                    "amount": txn["amount"],
                    "type": txn["type"],
                }
                for txn in batch
            ]
            prompt = TRANSACTION_CLASSIFICATION_PROMPT.format(transactions_json=_compact(payload))
            try:
                output = self._ask(
                    record.stage_name,
                    CLASSIFICATION_SYSTEM_PROMPT,
                    prompt,
                    ClassificationOutput,
                    calls,
                )
            except _StageFailure as exc:
                failed_batches += 1
                errors.append(str(exc))
                if failed_batches == attempted:
                    break  # nothing has worked yet: do not keep calling a dead backend
                continue

            by_index = {txn["index"]: txn for txn in batch}
            for item in output.transactions:
                target = by_index.get(item.index)
                if target is None or "category" in target:
                    continue
                target["category"] = item.category
                target["merchant_type"] = _safe_text(item.merchant_type, 40)
                target["confidence"] = item.confidence
                classified += 1

        for txn in working:
            txn.setdefault("category", UNKNOWN_CATEGORY)

        total_batches = len(self._batches(working))
        confidences = [txn["confidence"] for txn in working if "confidence" in txn]
        record.execution_time_ms = round((time.perf_counter() - started) * 1000, 1)
        record.llm_calls = calls[0]
        record.detail = {
            "transactions": len(working),
            "classified_by_model": classified,
            "batches": total_batches,
            "failed_batches": failed_batches,
            "mean_model_confidence": (
                round(sum(confidences) / len(confidences), 3) if confidences else None
            ),
        }
        if failed_batches == attempted:
            record.status, record.error = FAILED, errors[0]
        elif failed_batches or classified < len(working):
            record.status = DEGRADED
            record.error = errors[0] if errors else "Model skipped some transactions"
        return record

    def _single(
        self,
        stage_name: str,
        system_prompt: str,
        prompt: str,
        schema: Type[SchemaT],
    ) -> tuple:
        record = StageRecord(stage_name, OK, _now())
        started = time.perf_counter()
        calls = [0]
        output: Optional[SchemaT] = None
        try:
            output = self._ask(stage_name, system_prompt, prompt, schema, calls)
        except _StageFailure as exc:
            record.status, record.error = FAILED, str(exc)
        record.execution_time_ms = round((time.perf_counter() - started) * 1000, 1)
        record.llm_calls = calls[0]
        return record, output

    # ------------------------------------------------------------------ run
    def run(
        self,
        transactions: Sequence[Dict[str, Any]],
        features: Dict[str, Any],
        scorecard: Dict[str, Any],
        loan_amount: float,
        loan_term_months: int,
        loan_purpose: str,
    ) -> PipelineResult:
        """
        Run all five stages.

        Args:
            transactions: Normalised transactions, oldest first.
            features: Deterministic features computed from the full statement.
            scorecard: Scorecard result (recommendation, score, red flags).
            loan_amount, loan_term_months, loan_purpose: The request.
        """
        recent = list(transactions)[-self.max_transactions :]
        working: List[Dict[str, Any]] = []
        for index, txn in enumerate(recent):
            deposit = txn.get("deposit") or 0.0
            working.append(
                {
                    "index": index,
                    "date": str(txn.get("date", "")),
                    "description": _safe_text(txn.get("description")),
                    "amount": round(float(deposit or txn.get("withdrawal") or 0.0), 2),
                    "type": "credit" if deposit else "debit",
                }
            )

        result = PipelineResult(
            status="failed",
            provider=self.client.provider,
            model=self.client.model,
            transactions_total=len(transactions),
            transactions_analysed=len(working),
        )
        remaining = [
            "transaction_structuring",
            "transaction_classification",
            "behaviour_analysis",
            "risk_detection",
            "intelligence_summary",
        ]

        def finish(record: StageRecord) -> bool:
            result.stages.append(record)
            remaining.remove(record.stage_name)
            if record.error:
                result.errors.append(f"{record.stage_name}: {record.error}")
            if record.status == FAILED:
                for name in remaining:
                    result.stages.append(StageRecord(name, SKIPPED, _now()))
                return False
            return True

        if not working:
            finish(StageRecord(remaining[0], FAILED, _now(), error="No transactions to analyse"))
            return result

        if not finish(self._structure(working)):
            return result
        if not finish(self._classify(working)):
            return result

        totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total": 0.0})
        for txn in working:
            bucket = totals[txn["category"]]
            bucket["count"] += 1
            bucket["total"] = round(bucket["total"] + txn["amount"], 2)

        sample = [
            {
                "date": txn["date"],
                "merchant": txn["merchant"],
                "amount": txn["amount"],
                "type": txn["type"],
                "category": txn["category"],
            }
            for txn in working[-BEHAVIOUR_SAMPLE_SIZE:]
        ]
        features_json = _compact(features)

        record, behaviour = self._single(
            "behaviour_analysis",
            BEHAVIOUR_ANALYSIS_SYSTEM_PROMPT,
            FINANCIAL_BEHAVIOUR_PROMPT.format(
                features_json=features_json,
                category_totals_json=_compact(totals),
                transactions_json=_compact(sample),
            ),
            BehaviourAnalysis,
        )
        if not finish(record) or behaviour is None:
            return result
        result.behaviour = behaviour.model_dump()
        behaviour_json = _compact(result.behaviour)

        record, risk = self._single(
            "risk_detection",
            RISK_DETECTION_SYSTEM_PROMPT,
            RISK_DETECTION_PROMPT.format(
                loan_amount=f"{loan_amount:,.2f}",
                loan_term_months=loan_term_months,
                behaviour_json=behaviour_json,
                features_json=features_json,
            ),
            RiskDetection,
        )
        if not finish(record) or risk is None:
            return result
        result.risk = risk.model_dump()

        record, summary = self._single(
            "intelligence_summary",
            INTELLIGENCE_SYSTEM_PROMPT,
            INTELLIGENCE_SUMMARY_PROMPT.format(
                loan_amount=f"{loan_amount:,.2f}",
                loan_term_months=loan_term_months,
                loan_purpose=_safe_text(loan_purpose, 100) or "Not stated",
                behaviour_json=behaviour_json,
                risk_json=_compact(result.risk),
                scorecard_json=_compact(scorecard),
            ),
            IntelligenceSummary,
        )
        if not finish(record) or summary is None:
            return result
        result.summary = summary.model_dump()
        result.summary["supports_approval"] = summary.supports_approval
        result.status = "completed"
        return result
