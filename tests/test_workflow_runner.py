"""Tests for the workflow execution engine."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine import Agent
from core.workflow_context import CheckpointAction, StepResult, StepStatus
from core.workflow_runner import WorkflowRunner
from core.workflow_schema import (
    ParallelGroup,
    StepDefinition,
    WorkflowDefinition,
    WorkflowInput,
)


# ── Helpers ────────────────────────────────────────────────


def _make_mock_stream(text="mock output", finish_reason="stop"):
    """Create a mock completion_fn that streams text in one chunk."""
    def _mock(**kwargs):
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=text),
                finish_reason=None,
            )]
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None),
                finish_reason=finish_reason,
            )]
        )
    return _mock


def _make_failing_stream(error_msg="LLM error"):
    """Create a mock completion_fn that raises an exception."""
    def _mock(**kwargs):
        raise RuntimeError(error_msg)
        yield  # Make it a generator  # noqa: E305
    return _mock


def _create_agent(tmp_path, name="test_agent", mock_fn=None):
    """Create a minimal agent directory and return an Agent."""
    agent_dir = tmp_path / name
    agent_dir.mkdir(exist_ok=True)
    (agent_dir / "system.txt").write_text("You are a test assistant.")
    (agent_dir / "test.j2").write_text("Process: {{ content }}")
    return Agent(agent_dir, completion_fn=mock_fn or _make_mock_stream())


def _simple_workflow(steps, inputs=None):
    """Create a minimal WorkflowDefinition."""
    return WorkflowDefinition(
        name="Test Workflow",
        description="",
        steps=steps,
        inputs=inputs or [],
    )


# ── Basic Execution ───────────────────────────────────────


class TestWorkflowRunner:
    def test_single_step(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("result A"))
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "hello"}),
        ])

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        ctx = runner.run()

        assert ctx.get_output("s1") == "result A"
        assert ctx.results["s1"].status == StepStatus.COMPLETED

    def test_two_step_chaining(self, tmp_path):
        """Second step receives first step's output via variable reference."""
        call_count = 0
        def mock_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            text = f"output_{call_count}"
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=text), finish_reason=None,
            )])
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None), finish_reason="stop",
            )])

        agent = _create_agent(tmp_path, mock_fn=mock_fn)
        workflow = _simple_workflow(
            steps=[
                StepDefinition(id="s1", agent="test_agent", template="test.j2",
                               variables={"content": "start"}),
                StepDefinition(id="s2", agent="test_agent", template="test.j2",
                               variables={"content": "{{ steps.s1.output }}"}),
            ],
        )

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        ctx = runner.run()

        assert ctx.get_output("s1") == "output_1"
        assert ctx.get_output("s2") == "output_2"

    def test_input_resolution(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("done"))
        workflow = _simple_workflow(
            steps=[
                StepDefinition(id="s1", agent="test_agent", template="test.j2",
                               variables={"content": "{{ input.text }}"}),
            ],
            inputs=[WorkflowInput(name="text")],
        )

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        runner.set_inputs({"text": "my input"})
        ctx = runner.run()

        assert ctx.get_output("s1") == "done"
        assert ctx.inputs["text"] == "my input"


# ── Callbacks ─────────────────────────────────────────────


class TestCallbacks:
    def test_step_start_callback(self, tmp_path):
        agent = _create_agent(tmp_path)
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}),
        ])

        started = []
        runner = WorkflowRunner(
            workflow, {"test_agent": agent},
            on_step_start=lambda step: started.append(step.id),
        )
        runner.run()
        assert started == ["s1"]

    def test_step_complete_callback(self, tmp_path):
        agent = _create_agent(tmp_path)
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}),
        ])

        completed = []
        runner = WorkflowRunner(
            workflow, {"test_agent": agent},
            on_step_complete=lambda step, result: completed.append((step.id, result.status)),
        )
        runner.run()
        assert completed == [("s1", StepStatus.COMPLETED)]

    def test_stream_delta_callback(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("hello"))
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}),
        ])

        deltas = []
        runner = WorkflowRunner(
            workflow, {"test_agent": agent},
            on_stream_delta=lambda step, delta: deltas.append(delta),
        )
        runner.run()
        assert "hello" in deltas


# ── Checkpoints ───────────────────────────────────────────


class TestCheckpoints:
    def test_checkpoint_approve(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("result"))
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}, checkpoint=True),
            StepDefinition(id="s2", agent="test_agent", template="test.j2",
                           variables={"content": "y"}),
        ])

        runner = WorkflowRunner(
            workflow, {"test_agent": agent},
            on_checkpoint=lambda step, result: CheckpointAction.APPROVE,
        )
        ctx = runner.run()
        # Both steps should run
        assert "s1" in ctx.results
        assert "s2" in ctx.results

    def test_checkpoint_abort(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("result"))
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}, checkpoint=True),
            StepDefinition(id="s2", agent="test_agent", template="test.j2",
                           variables={"content": "y"}),
        ])

        runner = WorkflowRunner(
            workflow, {"test_agent": agent},
            on_checkpoint=lambda step, result: CheckpointAction.ABORT,
        )
        ctx = runner.run()
        # s1 completed, s2 skipped
        assert ctx.results["s1"].status == StepStatus.COMPLETED
        assert ctx.results["s2"].status == StepStatus.SKIPPED

    def test_checkpoint_edit(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("original"))
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}, checkpoint=True),
            StepDefinition(id="s2", agent="test_agent", template="test.j2",
                           variables={"content": "{{ steps.s1.output }}"}),
        ])

        runner = WorkflowRunner(
            workflow, {"test_agent": agent},
            on_checkpoint=lambda step, result: CheckpointAction.EDIT,
            on_edit=lambda step, result: "edited text",
        )
        ctx = runner.run()
        # s1 output should be replaced with edited text
        assert ctx.get_output("s1") == "edited text"


# ── Error Handling ────────────────────────────────────────


class TestErrorHandling:
    def test_missing_agent(self, tmp_path):
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="nonexistent", template="test.j2",
                           variables={"content": "x"}),
        ])

        runner = WorkflowRunner(workflow, {})
        ctx = runner.run()
        assert ctx.results["s1"].status == StepStatus.FAILED
        assert "not found" in ctx.results["s1"].error

    def test_llm_error(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_failing_stream("API timeout"))
        workflow = _simple_workflow([
            StepDefinition(id="s1", agent="test_agent", template="test.j2",
                           variables={"content": "x"}),
            StepDefinition(id="s2", agent="test_agent", template="test.j2",
                           variables={"content": "y"}),
        ])

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        ctx = runner.run()
        assert ctx.results["s1"].status == StepStatus.FAILED
        assert "API timeout" in ctx.results["s1"].error
        # s2 should not have run
        assert "s2" not in ctx.results


# ── Parallel Group (Sequential Fallback) ──────────────────


class TestParallelGroupSequential:
    def test_parallel_group_runs_all_steps(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_mock_stream("ok"))
        workflow = _simple_workflow([
            ParallelGroup(steps=[
                StepDefinition(id="p1", agent="test_agent", template="test.j2",
                               variables={"content": "a"}),
                StepDefinition(id="p2", agent="test_agent", template="test.j2",
                               variables={"content": "b"}),
            ]),
        ])

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        ctx = runner.run()
        assert ctx.results["p1"].status == StepStatus.COMPLETED
        assert ctx.results["p2"].status == StepStatus.COMPLETED

    def test_parallel_group_fail_fast(self, tmp_path):
        agent = _create_agent(tmp_path, mock_fn=_make_failing_stream("error"))
        workflow = _simple_workflow([
            ParallelGroup(
                steps=[
                    StepDefinition(id="p1", agent="test_agent", template="test.j2",
                                   variables={"content": "a"}),
                    StepDefinition(id="p2", agent="test_agent", template="test.j2",
                                   variables={"content": "b"}),
                ],
                on_error="fail_fast",
            ),
        ])

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        ctx = runner.run()
        assert ctx.results["p1"].status == StepStatus.FAILED
        assert ctx.results["p2"].status == StepStatus.SKIPPED

    def test_parallel_group_continue_on_error(self, tmp_path):
        """First step fails but second step still runs with on_error=continue."""
        call_count = 0
        def alternating_mock(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first fails")
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="second ok"), finish_reason=None,
            )])
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None), finish_reason="stop",
            )])

        agent = _create_agent(tmp_path, mock_fn=alternating_mock)
        workflow = _simple_workflow([
            ParallelGroup(
                steps=[
                    StepDefinition(id="p1", agent="test_agent", template="test.j2",
                                   variables={"content": "a"}),
                    StepDefinition(id="p2", agent="test_agent", template="test.j2",
                                   variables={"content": "b"}),
                ],
                on_error="continue",
            ),
        ])

        runner = WorkflowRunner(workflow, {"test_agent": agent})
        ctx = runner.run()
        assert ctx.results["p1"].status == StepStatus.FAILED
        assert ctx.results["p2"].status == StepStatus.COMPLETED
