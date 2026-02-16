"""Workflow YAML schema — pure data models for declarative workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowInput:
    """A user-facing input that the workflow requires before execution."""

    name: str
    description: str = ""
    command: bool = False  # Allow /file, /url, /ctx commands


@dataclass(frozen=True)
class StepDefinition:
    """A single step in a sequential workflow."""

    id: str
    agent: str  # Agent folder name
    template: str  # Template filename (.j2)
    variables: dict[str, str] = field(default_factory=dict)  # name -> expression
    checkpoint: bool = False  # Pause for human review after this step
    condition: str | None = None  # Jinja2 expression; step skipped if evaluates to false


@dataclass(frozen=True)
class ParallelGroup:
    """A group of steps that execute concurrently (Phase 2)."""

    steps: list[StepDefinition] = field(default_factory=list)
    on_error: str = "fail_fast"  # "fail_fast" or "continue"


# A workflow step item is either a sequential step or a parallel group.
StepItem = StepDefinition | ParallelGroup


@dataclass(frozen=True)
class WorkflowDefinition:
    """Complete workflow definition loaded from YAML."""

    name: str
    description: str
    steps: list[StepItem]
    inputs: list[WorkflowInput] = field(default_factory=list)
    source_path: Path | None = None


def _parse_step(raw: dict) -> StepDefinition:
    """Parse a single step dict into a StepDefinition."""
    required = {"id", "agent", "template"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Step missing required fields: {missing}")

    return StepDefinition(
        id=raw["id"],
        agent=raw["agent"],
        template=raw["template"],
        variables=dict(raw.get("variables", {})),
        checkpoint=bool(raw.get("checkpoint", False)),
        condition=raw.get("condition"),
    )


def _parse_step_item(raw: dict) -> StepItem:
    """Parse either a step or a parallel group from a YAML dict."""
    if "parallel" in raw:
        parallel_steps = [_parse_step(s) for s in raw["parallel"]]
        on_error = raw.get("on_error", "fail_fast")
        if on_error not in ("fail_fast", "continue"):
            raise ValueError(f"Invalid on_error value: {on_error!r} (expected 'fail_fast' or 'continue')")
        return ParallelGroup(steps=parallel_steps, on_error=on_error)
    return _parse_step(raw)


def load_workflow(path: Path) -> WorkflowDefinition:
    """Load and parse a workflow YAML file into a WorkflowDefinition.

    Raises ValueError for malformed YAML or missing required fields.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Workflow file must be a YAML mapping: {path}")

    # Required top-level fields
    if "name" not in data:
        raise ValueError(f"Workflow missing 'name': {path}")
    if "steps" not in data or not data["steps"]:
        raise ValueError(f"Workflow missing 'steps': {path}")

    # Parse inputs
    inputs = []
    for inp in data.get("inputs", []):
        if "name" not in inp:
            raise ValueError(f"Workflow input missing 'name': {path}")
        inputs.append(WorkflowInput(
            name=inp["name"],
            description=inp.get("description", ""),
            command=bool(inp.get("command", False)),
        ))

    # Parse steps
    steps = [_parse_step_item(s) for s in data["steps"]]

    return WorkflowDefinition(
        name=data["name"],
        description=data.get("description", ""),
        steps=steps,
        inputs=inputs,
        source_path=path,
    )


def _collect_all_steps(definition: WorkflowDefinition) -> list[StepDefinition]:
    """Flatten all StepDefinitions from a workflow (including parallel groups)."""
    result = []
    for item in definition.steps:
        if isinstance(item, ParallelGroup):
            result.extend(item.steps)
        else:
            result.append(item)
    return result


def validate_workflow(
    definition: WorkflowDefinition,
    available_agents: dict[str, list[str]],
) -> list[str]:
    """Validate a workflow definition against available agents.

    *available_agents* maps agent name -> list of template filenames.
    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    all_steps = _collect_all_steps(definition)

    # Check for duplicate step IDs
    seen_ids: set[str] = set()
    for step in all_steps:
        if step.id in seen_ids:
            errors.append(f"Duplicate step ID: {step.id!r}")
        seen_ids.add(step.id)

    # Check agents and templates exist
    for step in all_steps:
        if step.agent not in available_agents:
            errors.append(f"Step {step.id!r}: unknown agent {step.agent!r}")
        elif step.template not in available_agents[step.agent]:
            errors.append(
                f"Step {step.id!r}: template {step.template!r} not found in agent {step.agent!r}"
            )

    # Check variable references
    input_names = {inp.name for inp in definition.inputs}
    for step in all_steps:
        for var_name, expr in step.variables.items():
            if "input." in expr:
                # Extract referenced input name
                ref = _extract_ref(expr, "input.")
                if ref and ref not in input_names:
                    errors.append(
                        f"Step {step.id!r}, variable {var_name!r}: "
                        f"references undefined input {ref!r}"
                    )
            if "steps." in expr:
                ref = _extract_ref(expr, "steps.")
                if ref:
                    # ref should be a step ID
                    step_id = ref.split(".")[0]
                    if step_id not in seen_ids:
                        errors.append(
                            f"Step {step.id!r}, variable {var_name!r}: "
                            f"references undefined step {step_id!r}"
                        )

    return errors


def validate_template_schema_consistency(
    definition: WorkflowDefinition,
    agent_schemas: dict[str, dict[str, dict]],
) -> list[str]:
    """Check that step variable references to parsed fields exist in the target schema.

    *agent_schemas* maps agent_name -> {template_name: schema_dict}.
    """
    errors: list[str] = []
    all_steps = _collect_all_steps(definition)
    step_map: dict[str, StepDefinition] = {s.id: s for s in all_steps}

    for step in all_steps:
        for var_name, expr in step.variables.items():
            if "steps." not in expr:
                continue
            ref = _extract_ref(expr, "steps.")
            if not ref:
                continue
            parts = ref.split(".")
            if len(parts) < 3:
                continue  # e.g. steps.X.output — no deep field
            ref_step_id = parts[0]
            if parts[1] != "output":
                continue
            field_path = parts[2:]
            # Look up the referenced step's schema
            ref_step = step_map.get(ref_step_id)
            if ref_step is None:
                continue
            agent_name = ref_step.agent
            tpl_name = ref_step.template
            schemas = agent_schemas.get(agent_name, {})
            schema = schemas.get(tpl_name)
            if schema is None:
                continue
            # Check top-level field exists in schema properties
            props = schema.get("properties", {})
            top_field = field_path[0].split("[")[0]  # Strip bracket notation
            if top_field not in props:
                errors.append(
                    f"Step {step.id!r}, variable {var_name!r}: "
                    f"references field {top_field!r} not in schema of "
                    f"{agent_name}/{tpl_name}"
                )

    return errors


def _extract_ref(expression: str, prefix: str) -> str | None:
    """Extract a dotted reference from a {{ prefix.name }} expression."""
    # Find the prefix in the expression
    idx = expression.find(prefix)
    if idx < 0:
        return None
    start = idx + len(prefix)
    # Read until non-identifier character
    end = start
    while end < len(expression) and (expression[end].isalnum() or expression[end] in "_."):
        end += 1
    ref = expression[start:end].rstrip(".")
    return ref if ref else None
