"""Workflow runtime context — mutable state for a single workflow execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointAction(Enum):
    APPROVE = "approve"
    EDIT = "edit"
    ABORT = "abort"


@dataclass
class StepResult:
    """Result of executing a single workflow step."""

    step_id: str
    output: str
    finish_reason: str
    status: StepStatus = StepStatus.COMPLETED
    error: str | None = None


# Pattern matching {{ input.name }} or {{ steps.id.output }}
_VAR_PATTERN = re.compile(r"\{\{\s*(input|steps)\.([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}")


@dataclass
class WorkflowContext:
    """Mutable runtime state for a workflow execution (one per run, not singleton)."""

    inputs: dict[str, str] = field(default_factory=dict)
    results: dict[str, StepResult] = field(default_factory=dict)

    def set_result(self, step_id: str, result: StepResult) -> None:
        """Store the result of a completed step."""
        self.results[step_id] = result

    def get_output(self, step_id: str) -> str | None:
        """Get the output text of a completed step."""
        result = self.results.get(step_id)
        if result is None:
            return None
        return result.output

    def resolve_variable(self, expression: str) -> str:
        """Resolve a single {{ input.x }} or {{ steps.id.field }} expression.

        Returns the resolved value, or the original expression if unresolvable.
        """
        def _replace(match: re.Match) -> str:
            namespace = match.group(1)
            ref = match.group(2)

            if namespace == "input":
                value = self.inputs.get(ref)
                if value is not None:
                    return value
                return match.group(0)  # Leave unresolved

            if namespace == "steps":
                # ref is like "step_id.output" or "step_id.error"
                parts = ref.split(".", 1)
                if len(parts) != 2:
                    return match.group(0)
                step_id, attr = parts
                result = self.results.get(step_id)
                if result is None:
                    return match.group(0)
                value = getattr(result, attr, None)
                if value is not None:
                    return str(value)
                return match.group(0)

            return match.group(0)

        return _VAR_PATTERN.sub(_replace, expression)

    def resolve_step_variables(self, variable_map: dict[str, str]) -> dict[str, str]:
        """Resolve all variables for a step, returning name -> resolved value."""
        return {
            name: self.resolve_variable(expr)
            for name, expr in variable_map.items()
        }
