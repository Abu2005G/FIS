"""
Financial Intelligence Pipeline Executor
Implements the 5-stage LLM reasoning pipeline from README
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os

from src.app.prompts.templates import (
    TRANSACTION_STRUCTURING_PROMPT,
    TRANSACTION_CLASSIFICATION_PROMPT,
    FINANCIAL_BEHAVIOUR_PROMPT,
    RISK_DETECTION_PROMPT,
    INTELLIGENCE_SUMMARY_PROMPT,
    STRUCTURING_SYSTEM_PROMPT,
    CLASSIFICATION_SYSTEM_PROMPT,
    BEHAVIOUR_ANALYSIS_SYSTEM_PROMPT,
    RISK_DETECTION_SYSTEM_PROMPT,
    INTELLIGENCE_SYSTEM_PROMPT,
)


@dataclass
class PipelineStage:
    """Record of a pipeline stage execution"""

    stage_name: str
    input_data: Any
    output_data: Any
    confidence: float
    execution_time_ms: float
    timestamp: str
    error: Optional[str] = None


@dataclass
class AuditLog:
    """Complete audit trail for a loan decision"""

    application_id: int
    applicant_name: str
    loan_amount: float
    stages: List[PipelineStage] = field(default_factory=list)
    final_decision: Optional[Dict] = None
    overall_confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class LLMClient:
    """Wrapper for LLM API calls - supports multiple backends"""

    def __init__(self, provider: str = "ollama", model: Optional[str] = None):
        self.provider = provider
        self.model = model or self._get_default_model()
        self.api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None

    def _get_default_model(self) -> str:
        if self.provider == "ollama":
            return "llama3.2"  # or mistral, phi4
        elif self.provider == "openai":
            return "gpt-4o-mini"
        return "default"

    def call(
        self, prompt: str, system_prompt: str = "", temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Call LLM and return parsed response"""
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt, temperature)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt, temperature)
        else:
            return self._call_mock(prompt, system_prompt)

    def _call_ollama(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> Dict[str, Any]:
        """Call Ollama local LLM"""
        try:
            import requests  # type: ignore[import-untyped]

            response = requests.post(
                url="http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return self._parse_response(result.get("response", ""))
        except Exception as e:
            return {"error": str(e), "raw_response": ""}

    def _call_openai(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e), "raw_response": ""}

    def _call_mock(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Mock LLM for testing - returns sensible defaults"""
        return {
            "mock": True,
            "message": "Mock LLM response - configure Ollama or OpenAI for real inference",
        }

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text"""
        # Try to find JSON in the response
        try:
            # Look for JSON block
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text.strip()

            return json.loads(json_str)
        except:
            # Return raw text if JSON parsing fails
            return {"raw_response": text, "parsed": False}


class FinancialIntelligencePipeline:
    """
    5-Stage LLM Pipeline for Credit Risk Assessment

    Stage 1: Structuring -> Stage 2: Classification ->
    Stage 3: Behaviour -> Stage 4: Risk -> Stage 5: Intelligence
    """

    def __init__(self, llm_provider: str = "ollama", llm_model: Optional[str] = None):
        self.llm = LLMClient(provider=llm_provider, model=llm_model)
        self.audit_log: Optional[AuditLog] = None

    def process_application(
        self,
        application_id: int,
        applicant_name: str,
        loan_amount: float,
        loan_purpose: str,
        raw_transactions: List[str],
    ) -> Dict[str, Any]:
        """
        Run the complete 5-stage pipeline

        Args:
            application_id: Unique ID for the loan application
            applicant_name: Name of applicant
            loan_amount: Requested loan amount
            loan_purpose: Purpose of loan
            raw_transactions: List of raw transaction strings from PDF

        Returns:
            Final credit decision with complete audit trail
        """
        # Initialize audit log
        self.audit_log = AuditLog(
            application_id=application_id,
            applicant_name=applicant_name,
            loan_amount=loan_amount,
        )
        assert self.audit_log is not None  # Set above; used for full pipeline run
        print(f"Processing Application #{application_id} for {applicant_name}")
        print(f"Loan Amount: ${loan_amount:,.2f}")
        print(f"{'=' * 60}")

        # Stage 1: Transaction Structuring
        structured_transactions = self._stage_1_structure(raw_transactions)

        # Stage 2: Transaction Classification
        classified_transactions = self._stage_2_classify(structured_transactions)

        # Stage 3: Financial Behaviour Analysis
        behaviour_analysis = self._stage_3_analyze_behaviour(classified_transactions)

        # Stage 4: Risk Detection
        risk_assessment = self._stage_4_detect_risk(behaviour_analysis, loan_amount)

        # Stage 5: Intelligence Summary & Decision
        final_decision = self._stage_5_summarize(
            classified_transactions,
            behaviour_analysis,
            risk_assessment,
            loan_amount,
            loan_purpose,
            applicant_name,
        )

        # Calculate overall confidence
        confidences = [s.confidence for s in self.audit_log.stages]
        self.audit_log.overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0
        )
        self.audit_log.final_decision = final_decision

        return {
            "decision": final_decision,
            "audit_log": self._audit_log_to_dict(self.audit_log),
            "processing_summary": {
                "transactions_processed": len(raw_transactions),
                "stages_completed": len(self.audit_log.stages),
                "overall_confidence": self.audit_log.overall_confidence,
                "processing_time_ms": sum(
                    s.execution_time_ms for s in self.audit_log.stages
                ),
            },
        }

    def _stage_1_structure(self, raw_transactions: List[str]) -> List[Dict]:
        """Stage 1: Structure raw transactions"""
        assert self.audit_log is not None
        print("\n📋 Stage 1: Transaction Structuring...")
        start_time = time.time()

        structured = []
        for i, txn_text in enumerate(
            raw_transactions[:20]
        ):  # Process first 20 for speed
            prompt = TRANSACTION_STRUCTURING_PROMPT.format(transaction_text=txn_text)
            result = self.llm.call(prompt, STRUCTURING_SYSTEM_PROMPT)

            if "error" not in result:
                result["raw_text"] = txn_text
                structured.append(result)

        execution_time = (time.time() - start_time) * 1000

        self.audit_log.stages.append(
            PipelineStage(
                stage_name="transaction_structuring",
                input_data=raw_transactions[:10],
                output_data=structured,
                confidence=0.85,  # Estimated
                execution_time_ms=execution_time,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

        print(f"   ✓ Structured {len(structured)} transactions")
        return structured

    def _stage_2_classify(self, structured_transactions: List[Dict]) -> List[Dict]:
        """Stage 2: Classify transactions"""
        assert self.audit_log is not None
        print("\n🏷️  Stage 2: Transaction Classification...")
        start_time = time.time()

        classified = []
        for txn in structured_transactions:
            prompt = TRANSACTION_CLASSIFICATION_PROMPT.format(
                structured_transaction=json.dumps(txn)
            )
            result = self.llm.call(prompt, CLASSIFICATION_SYSTEM_PROMPT)

            # Merge classification with structured data
            txn_with_class = {**txn, **result}
            classified.append(txn_with_class)

        execution_time = (time.time() - start_time) * 1000

        self.audit_log.stages.append(
            PipelineStage(
                stage_name="transaction_classification",
                input_data=structured_transactions,
                output_data=classified,
                confidence=0.80,
                execution_time_ms=execution_time,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

        print(f"   ✓ Classified {len(classified)} transactions")
        return classified

    def _stage_3_analyze_behaviour(self, classified_transactions: List[Dict]) -> Dict:
        """Stage 3: Financial Behaviour Analysis"""
        assert self.audit_log is not None
        print("\n📊 Stage 3: Financial Behaviour Analysis...")
        start_time = time.time()

        # Sample transactions for analysis (first 15)
        txn_sample = classified_transactions[:15]

        prompt = FINANCIAL_BEHAVIOUR_PROMPT.format(
            transaction_list=json.dumps(txn_sample, indent=2)
        )
        result = self.llm.call(
            prompt, BEHAVIOUR_ANALYSIS_SYSTEM_PROMPT, temperature=0.3
        )

        execution_time = (time.time() - start_time) * 1000

        self.audit_log.stages.append(
            PipelineStage(
                stage_name="behaviour_analysis",
                input_data=txn_sample,
                output_data=result,
                confidence=0.75,
                execution_time_ms=execution_time,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

        print(f"   ✓ Analysis complete")
        return result

    def _stage_4_detect_risk(
        self, behaviour_analysis: Dict, loan_amount: float
    ) -> Dict:
        """Stage 4: Risk Signal Detection"""
        assert self.audit_log is not None
        print("\n⚠️  Stage 4: Risk Signal Detection...")
        start_time = time.time()

        prompt = RISK_DETECTION_PROMPT.format(
            behaviour_analysis=json.dumps(behaviour_analysis, indent=2),
            loan_amount=loan_amount,
        )
        result = self.llm.call(prompt, RISK_DETECTION_SYSTEM_PROMPT, temperature=0.2)

        execution_time = (time.time() - start_time) * 1000

        self.audit_log.stages.append(
            PipelineStage(
                stage_name="risk_detection",
                input_data=behaviour_analysis,
                output_data=result,
                confidence=result.get("risk_score", 0.5),
                execution_time_ms=execution_time,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

        risk_level = result.get("risk_level", "unknown")
        print(f"   ✓ Risk Level: {risk_level.upper()}")
        return result

    def _stage_5_summarize(
        self,
        transactions: List[Dict],
        behaviour_analysis: Dict,
        risk_assessment: Dict,
        loan_amount: float,
        loan_purpose: str,
        applicant_name: str,
    ) -> Dict:
        """Stage 5: Final Intelligence Summary"""
        assert self.audit_log is not None
        print("\n🎯 Stage 5: Financial Intelligence Summary...")
        start_time = time.time()

        prompt = INTELLIGENCE_SUMMARY_PROMPT.format(
            transaction_summary=json.dumps(transactions[:10], indent=2),
            behaviour_analysis=json.dumps(behaviour_analysis, indent=2),
            risk_assessment=json.dumps(risk_assessment, indent=2),
            loan_amount=loan_amount,
            loan_purpose=loan_purpose,
            applicant_name=applicant_name,
        )
        result = self.llm.call(prompt, INTELLIGENCE_SYSTEM_PROMPT, temperature=0.2)

        execution_time = (time.time() - start_time) * 1000

        self.audit_log.stages.append(
            PipelineStage(
                stage_name="intelligence_summary",
                input_data={"behaviour": behaviour_analysis, "risk": risk_assessment},
                output_data=result,
                confidence=result.get("credit_decision", {}).get("confidence", 0.5),
                execution_time_ms=execution_time,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

        approved = result.get("credit_decision", {}).get("approved", False)
        print(f"   ✓ Decision: {'APPROVED' if approved else 'REJECTED'}")
        return result

    def _audit_log_to_dict(self, audit_log: AuditLog) -> Dict:
        """Convert audit log to dictionary"""
        return {
            "application_id": audit_log.application_id,
            "applicant_name": audit_log.applicant_name,
            "loan_amount": audit_log.loan_amount,
            "created_at": audit_log.created_at,
            "overall_confidence": audit_log.overall_confidence,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "confidence": s.confidence,
                    "execution_time_ms": s.execution_time_ms,
                    "timestamp": s.timestamp,
                    "error": s.error,
                }
                for s in audit_log.stages
            ],
            "final_decision": audit_log.final_decision,
        }


# Global pipeline instance
_pipeline_instance = None


def get_pipeline(
    llm_provider: str = "ollama", llm_model: Optional[str] = None
) -> FinancialIntelligencePipeline:
    """Get or create pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FinancialIntelligencePipeline(llm_provider, llm_model)
    return _pipeline_instance
