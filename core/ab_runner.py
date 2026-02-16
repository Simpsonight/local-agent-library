"""A/B model testing — run the same prompt through multiple models and compare."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from core.cost_tracker import CostTracker, estimate_cost
from core.engine import Agent
from core.schema import OutputSchema, validate_output

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ABResult:
    """Result from a single model in an A/B test."""

    model: str
    output: str
    parsed: dict | None = None
    schema_valid: bool = True
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ABComparison:
    """Complete comparison of multiple model outputs."""

    prompt: str
    template_name: str
    results: list[ABResult]
    schema: OutputSchema | None = None


class ABRunner:
    """Runs the same prompt through multiple models for comparison."""

    def __init__(self, agent: Agent, template_name: str):
        self.agent = agent
        self.template_name = template_name
        self.schema = agent.get_schema(template_name)

    def run(self, prompt: str, models: list[str]) -> ABComparison:
        """Execute the prompt against each model and collect results.

        Only models present in the model registry are allowed.
        Temporarily overrides the agent's model config for each run.
        """
        from core.model_registry import get_registry

        registry = get_registry()
        results: list[ABResult] = []

        for model in models:
            # Validate model against registry before executing
            if registry.get(model) is None:
                logger.warning("A/B test: model '%s' not in registry, skipping.", model)
                results.append(ABResult(
                    model=model,
                    output=f"ERROR: Model '{model}' not found in registry.",
                    schema_valid=False,
                    errors=[f"Model '{model}' not in registry — only registered models are allowed."],
                ))
                continue
            logger.info("A/B testing model: %s", model)
            result = self._run_single(prompt, model)
            results.append(result)

        return ABComparison(
            prompt=prompt,
            template_name=self.template_name,
            results=results,
            schema=self.schema,
        )

    def _run_single(self, prompt: str, model: str) -> ABResult:
        """Run a single model and return the result."""
        # Temporarily override model
        original_model = self.agent.config["model"]
        self.agent.config["model"] = model

        start_time = time.monotonic()
        full_text = ""
        finish_reason = "stop"

        try:
            for delta, reason in self.agent.run_stream(
                prompt, template_name=self.template_name
            ):
                if delta:
                    full_text += delta
                if reason is not None:
                    finish_reason = reason
        except Exception as exc:
            logger.error("A/B test failed for model %s: %s", model, exc)
            self.agent.config["model"] = original_model
            return ABResult(
                model=model,
                output=f"ERROR: {exc}",
                schema_valid=False,
                errors=[str(exc)],
                duration_seconds=time.monotonic() - start_time,
            )
        finally:
            self.agent.config["model"] = original_model

        duration = time.monotonic() - start_time

        # Validate against schema
        parsed = None
        errors: list[str] = []
        schema_valid = True
        if self.schema is not None:
            parsed, errors = validate_output(full_text, self.schema)
            schema_valid = len(errors) == 0

        # Estimate cost (rough — actual tokens not available from streaming)
        est_prompt_tokens = len(prompt) // 4  # rough estimate
        est_completion_tokens = len(full_text) // 4
        cost = estimate_cost(model, est_prompt_tokens, est_completion_tokens)

        return ABResult(
            model=model,
            output=full_text,
            parsed=parsed,
            schema_valid=schema_valid,
            errors=errors,
            duration_seconds=round(duration, 2),
            prompt_tokens=est_prompt_tokens,
            completion_tokens=est_completion_tokens,
            cost_usd=round(cost, 6),
        )


def format_comparison(comparison: ABComparison) -> str:
    """Format an A/B comparison as a human-readable summary."""
    lines = [f"A/B Comparison: {len(comparison.results)} models\n"]

    for i, result in enumerate(comparison.results, 1):
        status = "VALID" if result.schema_valid else "INVALID"
        lines.append(f"--- Model {i}: {result.model} ---")
        lines.append(f"  Status: {status}")
        lines.append(f"  Duration: {result.duration_seconds}s")
        lines.append(f"  Output length: {len(result.output)} chars")
        lines.append(f"  Est. cost: ${result.cost_usd:.4f}")
        if result.errors:
            lines.append(f"  Errors: {', '.join(result.errors[:3])}")
        lines.append("")

    return "\n".join(lines)
