"""Output pipeline — validation and retry logic for structured LLM outputs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from core.engine import Agent
from core.schema import OutputSchema, validate_output

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputResult:
    """Result of running a prompt through the output pipeline."""

    raw_text: str
    parsed: dict | None = None  # None when no schema
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class OutputPipeline:
    """Runs an LLM call with optional schema validation and retries."""

    def run(
        self,
        prompt: str,
        agent: Agent,
        template_name: str,
        *,
        max_retries: int = 2,
        on_stream: Callable[[str], None] | None = None,
    ) -> OutputResult:
        """Execute prompt and validate output against schema if present.

        On schema violation, feeds errors back to the LLM and retries
        up to *max_retries* times.
        """
        schema = agent.get_schema(template_name)
        attempts = 0
        last_text = ""
        last_errors: list[str] = []
        last_finish = "stop"
        current_prompt = prompt
        total_prompt_tokens = 0
        total_completion_tokens = 0
        model_name = agent.config["model"]

        while attempts <= max_retries:
            attempts += 1

            # Run LLM
            full_text = ""
            finish_reason = "stop"
            for delta, reason in agent.run_stream(current_prompt, template_name=template_name):
                if delta:
                    full_text += delta
                    if on_stream:
                        on_stream(delta)
                if reason is not None:
                    finish_reason = reason

            # Accumulate usage from this attempt
            usage = getattr(agent, "last_usage", None)
            if usage and (usage.get("prompt_tokens", 0) > 0 or usage.get("completion_tokens", 0) > 0):
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                model_name = usage.get("model", model_name)
            else:
                # Fallback: estimate tokens (~4 chars per token)
                total_prompt_tokens += len(current_prompt) // 4
                total_completion_tokens += len(full_text) // 4

            last_text = full_text
            last_finish = finish_reason

            # No schema → return raw
            if schema is None:
                return OutputResult(
                    raw_text=full_text,
                    parsed=None,
                    errors=[],
                    attempts=attempts,
                    finish_reason=finish_reason,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    model=model_name,
                )

            # Validate against schema
            parsed, errors = validate_output(full_text, schema)
            if not errors:
                return OutputResult(
                    raw_text=full_text,
                    parsed=parsed,
                    errors=[],
                    attempts=attempts,
                    finish_reason=finish_reason,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    model=model_name,
                )

            last_errors = errors
            logger.warning(
                "Schema validation failed (attempt %d/%d): %s",
                attempts, max_retries + 1, errors,
            )

            # Build retry prompt with error feedback and full schema
            if attempts <= max_retries:
                error_feedback = "\n".join(f"- {e}" for e in errors)
                schema_json = json.dumps(schema.schema, indent=2, ensure_ascii=False)
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response had validation errors:\n"
                    f"{error_feedback}\n\n"
                    f"The response MUST match this JSON schema (use these exact field names):\n"
                    f"```json\n{schema_json}\n```\n\n"
                    f"Respond with valid JSON only."
                )

        # All retries exhausted
        return OutputResult(
            raw_text=last_text,
            parsed=None,
            errors=last_errors,
            attempts=attempts,
            finish_reason=last_finish,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            model=model_name,
        )
