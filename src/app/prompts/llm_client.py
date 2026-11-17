"""
LLM clients.

Each client turns a (system prompt, user prompt) pair into a parsed JSON object
or raises ``LLMError``. Failures are never swallowed: the caller decides what a
failed model call means for the loan decision.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, Optional

from src.app.config import Settings

logger = logging.getLogger(__name__)

# Fixed sampling settings so the same statement gives the same answer wherever
# the backend honours them. LLM output is still not guaranteed to be identical
# between runs, model versions or hardware.
TEMPERATURE = 0.0
SEED = 7


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMTransientError(LLMError):
    """A failure worth retrying (timeout, connection drop, overload)."""


class LLMOutputError(LLMError):
    """The model answered, but not with a usable JSON object."""


def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract one JSON object from model output."""
    if not isinstance(text, str) or not text.strip():
        raise LLMOutputError("Model returned an empty response")
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMOutputError("Model response was not a JSON object")


class LLMClient:
    """Base client: retry policy and JSON parsing shared by all backends."""

    provider = "base"

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep

    def _request(self, system_prompt: str, prompt: str) -> str:
        raise NotImplementedError

    def complete_json(self, stage: str, system_prompt: str, prompt: str) -> Dict[str, Any]:
        """Call the model and return its reply as a dict. Raises LLMError."""
        last_error: Optional[LLMError] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(min(0.5 * 2 ** (attempt - 1), 8.0))
            try:
                return parse_json_object(self._request(system_prompt, prompt))
            except (LLMTransientError, LLMOutputError) as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed (stage=%s attempt=%d/%d): %s",
                    stage,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
        raise last_error or LLMError("LLM call failed")


class OllamaClient(LLMClient):
    """Local models served by Ollama."""

    provider = "ollama"

    def __init__(self, model: str, base_url: str, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")

    def _request(self, system_prompt: str, prompt: str) -> str:
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": TEMPERATURE, "seed": SEED},
                },
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise LLMTransientError(f"Ollama timed out after {self.timeout_seconds:.0f}s") from exc
        except requests.ConnectionError as exc:
            raise LLMTransientError(f"Cannot reach Ollama at {self.base_url}") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {type(exc).__name__}") from exc

        if response.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' is not available. Run: ollama pull {self.model}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(f"Ollama returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMError(f"Ollama returned HTTP {response.status_code}")

        try:
            return response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama returned an unexpected response shape") from exc


class OpenAIClient(LLMClient):
    """OpenAI chat models. Statement text leaves your infrastructure."""

    provider = "openai"

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "LLM_PROVIDER=openai needs the openai package: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key, timeout=self.timeout_seconds, max_retries=0)

    def _request(self, system_prompt: str, prompt: str) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                seed=SEED,
                response_format={"type": "json_object"},
            )
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as exc:
            raise LLMTransientError(f"OpenAI call failed: {type(exc).__name__}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI call failed: {type(exc).__name__}") from exc

        return response.choices[0].message.content or ""


class MockLLMClient(LLMClient):
    """
    Offline stand-in for development and tests. Not a model.

    It returns schema-valid replies that mirror the scorecard, so the pipeline
    and API can be exercised without a model server. Blocked in production.
    """

    provider = "mock"

    def __init__(self, model: str = "mock", **kwargs: Any):
        super().__init__(model, **kwargs)

    @staticmethod
    def _tagged_json(prompt: str, tag: str) -> Any:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", prompt, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _request(self, system_prompt: str, prompt: str) -> str:
        raise NotImplementedError  # complete_json is overridden

    def complete_json(self, stage: str, system_prompt: str, prompt: str) -> Dict[str, Any]:
        items = self._tagged_json(prompt, "transactions") or []
        if stage == "transaction_structuring":
            return {
                "transactions": [
                    {
                        "index": item["index"],
                        "merchant": str(item.get("text", ""))[:60] or "Unknown",
                        "amount": item.get("amount", 0),
                        "transaction_type": item.get("type", "debit"),
                    }
                    for item in items
                ]
            }
        if stage == "transaction_classification":
            return {
                "transactions": [
                    {
                        "index": item["index"],
                        "category": "Income" if item.get("type") == "credit" else "Expense",
                        "merchant_type": "mock",
                        "confidence": 0.5,
                    }
                    for item in items
                ]
            }
        if stage == "behaviour_analysis":
            return {
                "income_analysis": {"stability": "unknown", "assessment": "mock analysis"},
                "spending_analysis": {"pattern": "unknown", "assessment": "mock analysis"},
            }

        scorecard = self._tagged_json(prompt, "scorecard") or {}
        recommendation = scorecard.get("recommendation", "review")
        level = scorecard.get("risk_level", "medium")
        if stage == "risk_detection":
            return {
                "risk_signals": [],
                "risk_level": "medium",
                "risk_score": 0.5,
                "explanation": "Mock provider: no model was consulted.",
            }
        return {
            "risk_profile": {"score": 0.5, "level": level},
            "credit_decision": {
                "approved": recommendation == "approve",
                "confidence": 0.5,
                "reasoning": ["Mock provider: mirrors the scorecard; no model was consulted."],
            },
            "audit_summary": {"recommendation": recommendation},
        }


def build_llm_client(settings: Settings) -> Optional[LLMClient]:
    """Create the configured client, or None when the LLM is switched off."""
    common = {
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.llm_model, settings.ollama_base_url, **common)
    if settings.llm_provider == "openai":
        return OpenAIClient(settings.llm_model, settings.openai_api_key or "", **common)
    if settings.llm_provider == "mock":
        return MockLLMClient()
    raise LLMError(f"Unknown LLM provider: {settings.llm_provider}")
