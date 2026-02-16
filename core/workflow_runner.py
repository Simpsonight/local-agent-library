"""Workflow execution engine — runs steps sequentially with callback hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.engine import Agent
from core.workflow_context import (
    CheckpointAction,
    StepResult,
    StepStatus,
    WorkflowContext,
)
from core.workflow_schema import (
    ParallelGroup,
    StepDefinition,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)

# Callback type aliases
StepCallback = Callable[[StepDefinition], None]
StepCompleteCallback = Callable[[StepDefinition, StepResult], None]
CheckpointCallback = Callable[[StepDefinition, StepResult], CheckpointAction]
StreamDeltaCallback = Callable[[StepDefinition, str], None]
EditCallback = Callable[[StepDefinition, StepResult], str]


class WorkflowRunner:
    """Executes a workflow definition step by step.

    Uses callbacks for UI interaction, keeping the runner testable
    without a terminal.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        agents: dict[str, Agent],
        *,
        on_step_start: StepCallback | None = None,
        on_step_complete: StepCompleteCallback | None = None,
        on_checkpoint: CheckpointCallback | None = None,
        on_stream_delta: StreamDeltaCallback | None = None,
        on_edit: EditCallback | None = None,
    ):
        self.workflow = workflow
        self.agents = agents
        self.context = WorkflowContext()

        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_checkpoint = on_checkpoint
        self._on_stream_delta = on_stream_delta
        self._on_edit = on_edit

    def set_inputs(self, inputs: dict[str, str]) -> None:
        """Set user-provided input values."""
        self.context.inputs = dict(inputs)

    def run(self) -> WorkflowContext:
        """Execute all workflow steps sequentially.

        Returns the WorkflowContext with all results (including partial
        results if aborted or failed).
        """
        for item in self.workflow.steps:
            if isinstance(item, ParallelGroup):
                # Phase 1: run parallel steps sequentially as fallback
                aborted = self._run_parallel_group_sequential(item)
                if aborted:
                    self._skip_remaining_from(item)
                    break
            else:
                result = self._execute_step(item)
                if result is None:
                    # Step was aborted — mark remaining as skipped
                    self._skip_remaining_from(item)
                    break
                if result.status == StepStatus.FAILED:
                    logger.error("Step %r failed: %s", item.id, result.error)
                    break

        return self.context

    def _execute_step(self, step: StepDefinition) -> StepResult | None:
        """Execute a single step. Returns None if aborted at checkpoint."""
        # Notify step start
        if self._on_step_start:
            self._on_step_start(step)

        # Resolve variables from context
        resolved_vars = self.context.resolve_step_variables(step.variables)

        # Get agent
        agent = self.agents.get(step.agent)
        if agent is None:
            result = StepResult(
                step_id=step.id,
                output="",
                finish_reason="error",
                status=StepStatus.FAILED,
                error=f"Agent {step.agent!r} not found",
            )
            self.context.set_result(step.id, result)
            if self._on_step_complete:
                self._on_step_complete(step, result)
            return result

        # Render template
        rendered = agent.render_template(step.template, resolved_vars)

        # Run LLM with streaming
        full_text = ""
        finish_reason = "stop"
        try:
            for delta, reason in agent.run_stream(rendered):
                if delta and self._on_stream_delta:
                    self._on_stream_delta(step, delta)
                if delta:
                    full_text += delta
                if reason is not None:
                    finish_reason = reason
        except Exception as exc:
            logger.error("Step %r LLM error: %s", step.id, exc)
            result = StepResult(
                step_id=step.id,
                output=full_text,
                finish_reason="error",
                status=StepStatus.FAILED,
                error=str(exc),
            )
            self.context.set_result(step.id, result)
            if self._on_step_complete:
                self._on_step_complete(step, result)
            return result

        result = StepResult(
            step_id=step.id,
            output=full_text,
            finish_reason=finish_reason,
            status=StepStatus.COMPLETED,
        )
        self.context.set_result(step.id, result)

        # Notify step complete
        if self._on_step_complete:
            self._on_step_complete(step, result)

        # Handle checkpoint
        if step.checkpoint and self._on_checkpoint:
            action = self._on_checkpoint(step, result)
            if action == CheckpointAction.ABORT:
                return None
            if action == CheckpointAction.EDIT and self._on_edit:
                edited_text = self._on_edit(step, result)
                result = StepResult(
                    step_id=step.id,
                    output=edited_text,
                    finish_reason=finish_reason,
                    status=StepStatus.COMPLETED,
                )
                self.context.set_result(step.id, result)

        return result

    def _run_parallel_group_sequential(self, group: ParallelGroup) -> bool:
        """Run a parallel group's steps sequentially (Phase 1 fallback).

        Returns True if aborted.
        """
        for step in group.steps:
            result = self._execute_step(step)
            if result is None:
                return True  # Aborted
            if result.status == StepStatus.FAILED:
                if group.on_error == "fail_fast":
                    # Mark remaining steps as skipped
                    remaining = group.steps[group.steps.index(step) + 1:]
                    for s in remaining:
                        self.context.set_result(s.id, StepResult(
                            step_id=s.id, output="", finish_reason="skipped",
                            status=StepStatus.SKIPPED,
                        ))
                    return False  # Not aborted, but group failed
                # on_error == "continue": keep going
        return False

    def _skip_remaining_from(self, current_item: Any) -> None:
        """Mark all steps after current_item as skipped."""
        found = False
        for item in self.workflow.steps:
            if item is current_item:
                found = True
                continue
            if found:
                if isinstance(item, ParallelGroup):
                    for step in item.steps:
                        if step.id not in self.context.results:
                            self.context.set_result(step.id, StepResult(
                                step_id=step.id, output="", finish_reason="skipped",
                                status=StepStatus.SKIPPED,
                            ))
                elif isinstance(item, StepDefinition):
                    if item.id not in self.context.results:
                        self.context.set_result(item.id, StepResult(
                            step_id=item.id, output="", finish_reason="skipped",
                            status=StepStatus.SKIPPED,
                        ))
