"""Cost tracking — records token usage and estimated costs per LLM call."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageRecord:
    """A single LLM usage record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_id: str | None = None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts.

    Returns 0.0 if the model is not in the registry.
    """
    from core.model_registry import get_registry

    costs = get_registry().get_cost(model)
    if costs is None:
        return 0.0

    input_cost, output_cost = costs
    return (prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000


class CostTracker:
    """Accumulates usage records across a session or workflow run."""

    def __init__(self):
        self._records: list[UsageRecord] = []

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        step_id: str | None = None,
    ) -> UsageRecord:
        """Record a usage event and return the record."""
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        rec = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            step_id=step_id,
        )
        self._records.append(rec)
        return rec

    def record_from_response(self, response, *, model: str = "", step_id: str | None = None) -> UsageRecord | None:
        """Extract usage from a litellm response object and record it.

        Returns None if the response has no usage information.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        model_name = model or getattr(response, "model", "") or ""
        return self.record(model_name, prompt_tokens, completion_tokens, step_id=step_id)

    @property
    def records(self) -> list[UsageRecord]:
        return list(self._records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def summary(self) -> dict:
        """Return a summary dict of accumulated usage."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_calls": len(self._records),
            "models_used": sorted(set(r.model for r in self._records)),
        }

    def reset(self):
        """Clear all recorded usage."""
        self._records.clear()
