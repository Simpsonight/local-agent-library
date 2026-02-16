"""Tests for workflow runtime context and variable resolution."""

import pytest

from core.workflow_context import (
    StepStatus,
    StepResult,
    WorkflowContext,
    CheckpointAction,
)


class TestStepResult:
    def test_defaults(self):
        r = StepResult(step_id="s1", output="hello", finish_reason="stop")
        assert r.status == StepStatus.COMPLETED
        assert r.error is None

    def test_failed_step(self):
        r = StepResult(
            step_id="s1", output="", finish_reason="error",
            status=StepStatus.FAILED, error="Connection timeout",
        )
        assert r.status == StepStatus.FAILED
        assert r.error == "Connection timeout"


class TestWorkflowContext:
    def test_set_and_get_result(self):
        ctx = WorkflowContext(inputs={"x": "hello"})
        result = StepResult(step_id="s1", output="world", finish_reason="stop")
        ctx.set_result("s1", result)
        assert ctx.get_output("s1") == "world"

    def test_get_output_missing(self):
        ctx = WorkflowContext()
        assert ctx.get_output("nonexistent") is None


class TestResolveVariable:
    def test_resolve_input(self):
        ctx = WorkflowContext(inputs={"keywords": "python, ai"})
        assert ctx.resolve_variable("{{ input.keywords }}") == "python, ai"

    def test_resolve_input_with_spaces(self):
        ctx = WorkflowContext(inputs={"keywords": "python"})
        assert ctx.resolve_variable("{{  input.keywords  }}") == "python"

    def test_resolve_step_output(self):
        ctx = WorkflowContext()
        ctx.set_result("analyze", StepResult(
            step_id="analyze", output="Analysis result", finish_reason="stop",
        ))
        assert ctx.resolve_variable("{{ steps.analyze.output }}") == "Analysis result"

    def test_resolve_step_error(self):
        ctx = WorkflowContext()
        ctx.set_result("s1", StepResult(
            step_id="s1", output="", finish_reason="error",
            status=StepStatus.FAILED, error="timeout",
        ))
        assert ctx.resolve_variable("{{ steps.s1.error }}") == "timeout"

    def test_unresolvable_input_left_unchanged(self):
        ctx = WorkflowContext(inputs={})
        expr = "{{ input.missing }}"
        assert ctx.resolve_variable(expr) == expr

    def test_unresolvable_step_left_unchanged(self):
        ctx = WorkflowContext()
        expr = "{{ steps.nonexistent.output }}"
        assert ctx.resolve_variable(expr) == expr

    def test_mixed_text_and_variables(self):
        ctx = WorkflowContext(inputs={"name": "World"})
        result = ctx.resolve_variable("Hello {{ input.name }}!")
        assert result == "Hello World!"

    def test_multiple_variables(self):
        ctx = WorkflowContext(inputs={"a": "X", "b": "Y"})
        result = ctx.resolve_variable("{{ input.a }} and {{ input.b }}")
        assert result == "X and Y"

    def test_step_without_dot_attribute_unchanged(self):
        ctx = WorkflowContext()
        ctx.set_result("s1", StepResult(step_id="s1", output="ok", finish_reason="stop"))
        # "steps.s1" without .output is not a valid ref
        expr = "{{ steps.s1 }}"
        assert ctx.resolve_variable(expr) == expr


class TestResolveStepVariables:
    def test_resolve_all_variables(self):
        ctx = WorkflowContext(inputs={"kw": "seo"})
        ctx.set_result("s1", StepResult(
            step_id="s1", output="plan text", finish_reason="stop",
        ))
        var_map = {
            "keywords": "{{ input.kw }}",
            "plan": "{{ steps.s1.output }}",
            "literal": "just plain text",
        }
        resolved = ctx.resolve_step_variables(var_map)
        assert resolved == {
            "keywords": "seo",
            "plan": "plan text",
            "literal": "just plain text",
        }

    def test_partial_resolution(self):
        ctx = WorkflowContext(inputs={"kw": "seo"})
        var_map = {
            "keywords": "{{ input.kw }}",
            "missing": "{{ steps.unknown.output }}",
        }
        resolved = ctx.resolve_step_variables(var_map)
        assert resolved["keywords"] == "seo"
        assert resolved["missing"] == "{{ steps.unknown.output }}"


class TestCheckpointAction:
    def test_enum_values(self):
        assert CheckpointAction.APPROVE.value == "approve"
        assert CheckpointAction.EDIT.value == "edit"
        assert CheckpointAction.ABORT.value == "abort"
