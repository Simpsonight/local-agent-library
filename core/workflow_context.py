"""Workflow runtime context — mutable state for a single workflow execution."""

from __future__ import annotations

import json
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
    parsed_output: dict | None = None


# Pattern matching {{ input.name }} or {{ steps.id.output }}
_VAR_PATTERN = re.compile(r"\{\{\s*(input|steps)\.([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\s*\}\}")


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

        Supports deep field access into parsed_output:
        ``{{ steps.analyze.output.keywords[0] }}`` traverses the parsed dict.
        Returns the resolved value, or the original expression if unresolvable.
        """
        def _resolve_deep(obj, path_parts: list[str]):
            """Traverse a nested dict/list by dotted path parts with bracket notation."""
            current = obj
            for part in path_parts:
                if current is None:
                    return None
                # Handle bracket notation: e.g. "keywords[0]"
                if "[" in part:
                    key = part[:part.index("[")]
                    bracket_str = part[part.index("["):]
                    if key:
                        if isinstance(current, dict):
                            current = current.get(key)
                        else:
                            return None
                    # Process bracket indices
                    import re as _re
                    indices = _re.findall(r"\[(\w+)\]", bracket_str)
                    for idx in indices:
                        if current is None:
                            return None
                        if isinstance(current, list):
                            try:
                                current = current[int(idx)]
                            except (ValueError, IndexError):
                                return None
                        elif isinstance(current, dict):
                            current = current.get(idx)
                        else:
                            return None
                else:
                    if isinstance(current, dict):
                        current = current.get(part)
                    else:
                        return None
            return current

        def _replace(match: re.Match) -> str:
            namespace = match.group(1)
            ref = match.group(2)

            if namespace == "input":
                value = self.inputs.get(ref)
                if value is not None:
                    return value
                return match.group(0)  # Leave unresolved

            if namespace == "steps":
                parts = ref.split(".", 1)
                if len(parts) != 2:
                    return match.group(0)
                step_id, attr_path = parts
                result = self.results.get(step_id)
                if result is None:
                    return match.group(0)

                # Deep field access: steps.X.output.field.subfield
                attr_parts = attr_path.split(".", 1)
                attr = attr_parts[0]

                if len(attr_parts) > 1 and attr == "output" and result.parsed_output is not None:
                    # Traverse into parsed_output
                    deep_path = attr_parts[1].split(".")
                    resolved = _resolve_deep(result.parsed_output, deep_path)
                    if resolved is not None:
                        return json.dumps(resolved) if isinstance(resolved, (dict, list)) else str(resolved)
                    return match.group(0)

                # Fallback to simple attribute access
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
