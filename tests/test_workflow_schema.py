"""Tests for workflow YAML schema parsing and validation."""

import pytest

from core.workflow_schema import (
    WorkflowInput,
    StepDefinition,
    ParallelGroup,
    WorkflowDefinition,
    load_workflow,
    validate_workflow,
)


# ── load_workflow ──────────────────────────────────────────


class TestLoadWorkflow:
    def test_minimal_workflow(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "description: A test workflow\n"
            "steps:\n"
            "  - id: step1\n"
            "    agent: my_agent\n"
            "    template: tpl.j2\n"
        )
        defn = load_workflow(wf)
        assert defn.name == "Test"
        assert defn.description == "A test workflow"
        assert len(defn.steps) == 1
        assert isinstance(defn.steps[0], StepDefinition)
        assert defn.steps[0].id == "step1"
        assert defn.steps[0].agent == "my_agent"
        assert defn.steps[0].template == "tpl.j2"
        assert defn.steps[0].checkpoint is False
        assert defn.source_path == wf

    def test_workflow_with_inputs(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "steps:\n"
            "  - id: s1\n"
            "    agent: a\n"
            "    template: t.j2\n"
            "    variables:\n"
            "      x: '{{ input.keywords }}'\n"
            "inputs:\n"
            "  - name: keywords\n"
            "    description: Target keywords\n"
            "    command: true\n"
        )
        defn = load_workflow(wf)
        assert len(defn.inputs) == 1
        assert defn.inputs[0].name == "keywords"
        assert defn.inputs[0].description == "Target keywords"
        assert defn.inputs[0].command is True

    def test_workflow_with_checkpoint(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "steps:\n"
            "  - id: s1\n"
            "    agent: a\n"
            "    template: t.j2\n"
            "    checkpoint: true\n"
        )
        defn = load_workflow(wf)
        assert defn.steps[0].checkpoint is True

    def test_workflow_with_variables(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "steps:\n"
            "  - id: s1\n"
            "    agent: a\n"
            "    template: t.j2\n"
            "    variables:\n"
            "      kw: '{{ input.keywords }}'\n"
            "      prev: '{{ steps.s0.output }}'\n"
        )
        defn = load_workflow(wf)
        assert defn.steps[0].variables == {
            "kw": "{{ input.keywords }}",
            "prev": "{{ steps.s0.output }}",
        }

    def test_parallel_group(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "steps:\n"
            "  - parallel:\n"
            "      - id: a\n"
            "        agent: x\n"
            "        template: t.j2\n"
            "      - id: b\n"
            "        agent: y\n"
            "        template: t.j2\n"
            "    on_error: continue\n"
        )
        defn = load_workflow(wf)
        assert len(defn.steps) == 1
        group = defn.steps[0]
        assert isinstance(group, ParallelGroup)
        assert len(group.steps) == 2
        assert group.steps[0].id == "a"
        assert group.steps[1].id == "b"
        assert group.on_error == "continue"

    def test_missing_name_raises(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text("steps:\n  - id: s1\n    agent: a\n    template: t.j2\n")
        with pytest.raises(ValueError, match="missing 'name'"):
            load_workflow(wf)

    def test_missing_steps_raises(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text("name: Test\n")
        with pytest.raises(ValueError, match="missing 'steps'"):
            load_workflow(wf)

    def test_empty_steps_raises(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text("name: Test\nsteps: []\n")
        with pytest.raises(ValueError, match="missing 'steps'"):
            load_workflow(wf)

    def test_step_missing_required_fields(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text("name: Test\nsteps:\n  - id: s1\n")
        with pytest.raises(ValueError, match="missing required fields"):
            load_workflow(wf)

    def test_input_missing_name_raises(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "steps:\n"
            "  - id: s1\n"
            "    agent: a\n"
            "    template: t.j2\n"
            "inputs:\n"
            "  - description: no name field\n"
        )
        with pytest.raises(ValueError, match="input missing 'name'"):
            load_workflow(wf)

    def test_invalid_on_error_raises(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Test\n"
            "steps:\n"
            "  - parallel:\n"
            "      - id: a\n"
            "        agent: x\n"
            "        template: t.j2\n"
            "    on_error: invalid\n"
        )
        with pytest.raises(ValueError, match="Invalid on_error"):
            load_workflow(wf)

    def test_not_a_mapping_raises(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text("- just a list\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_workflow(wf)

    def test_multi_step_workflow(self, tmp_path):
        wf = tmp_path / "test.yaml"
        wf.write_text(
            "name: Pipeline\n"
            "description: Two-step pipeline\n"
            "steps:\n"
            "  - id: analyze\n"
            "    agent: seo_expert\n"
            "    template: analyze.j2\n"
            "    variables:\n"
            "      keywords: '{{ input.keywords }}'\n"
            "  - id: optimize\n"
            "    agent: seo_expert\n"
            "    template: optimize.j2\n"
            "    variables:\n"
            "      analysis: '{{ steps.analyze.output }}'\n"
            "    checkpoint: true\n"
            "inputs:\n"
            "  - name: keywords\n"
        )
        defn = load_workflow(wf)
        assert len(defn.steps) == 2
        assert defn.steps[1].checkpoint is True
        assert "steps.analyze.output" in defn.steps[1].variables["analysis"]


# ── validate_workflow ──────────────────────────────────────


class TestValidateWorkflow:
    def _make_defn(self, steps, inputs=None):
        return WorkflowDefinition(
            name="Test",
            description="",
            steps=steps,
            inputs=inputs or [],
        )

    def test_valid_workflow(self):
        agents = {"my_agent": ["tpl.j2"]}
        defn = self._make_defn(
            steps=[StepDefinition(id="s1", agent="my_agent", template="tpl.j2")],
        )
        assert validate_workflow(defn, agents) == []

    def test_duplicate_step_ids(self):
        agents = {"a": ["t.j2"]}
        defn = self._make_defn(steps=[
            StepDefinition(id="s1", agent="a", template="t.j2"),
            StepDefinition(id="s1", agent="a", template="t.j2"),
        ])
        errors = validate_workflow(defn, agents)
        assert any("Duplicate step ID" in e for e in errors)

    def test_unknown_agent(self):
        agents = {"a": ["t.j2"]}
        defn = self._make_defn(
            steps=[StepDefinition(id="s1", agent="nonexistent", template="t.j2")],
        )
        errors = validate_workflow(defn, agents)
        assert any("unknown agent" in e for e in errors)

    def test_unknown_template(self):
        agents = {"a": ["other.j2"]}
        defn = self._make_defn(
            steps=[StepDefinition(id="s1", agent="a", template="missing.j2")],
        )
        errors = validate_workflow(defn, agents)
        assert any("template" in e and "not found" in e for e in errors)

    def test_undefined_input_reference(self):
        agents = {"a": ["t.j2"]}
        defn = self._make_defn(
            steps=[StepDefinition(
                id="s1", agent="a", template="t.j2",
                variables={"x": "{{ input.missing_input }}"},
            )],
            inputs=[WorkflowInput(name="other")],
        )
        errors = validate_workflow(defn, agents)
        assert any("undefined input" in e for e in errors)

    def test_undefined_step_reference(self):
        agents = {"a": ["t.j2"]}
        defn = self._make_defn(
            steps=[StepDefinition(
                id="s1", agent="a", template="t.j2",
                variables={"x": "{{ steps.nonexistent.output }}"},
            )],
        )
        errors = validate_workflow(defn, agents)
        assert any("undefined step" in e for e in errors)

    def test_valid_step_reference(self):
        agents = {"a": ["t.j2"]}
        defn = self._make_defn(steps=[
            StepDefinition(id="s1", agent="a", template="t.j2"),
            StepDefinition(
                id="s2", agent="a", template="t.j2",
                variables={"x": "{{ steps.s1.output }}"},
            ),
        ])
        errors = validate_workflow(defn, agents)
        assert errors == []

    def test_parallel_group_validation(self):
        agents = {"a": ["t.j2"], "b": ["t.j2"]}
        defn = self._make_defn(steps=[
            ParallelGroup(steps=[
                StepDefinition(id="p1", agent="a", template="t.j2"),
                StepDefinition(id="p2", agent="unknown", template="t.j2"),
            ]),
        ])
        errors = validate_workflow(defn, agents)
        assert any("unknown agent" in e for e in errors)

    def test_valid_input_reference(self):
        agents = {"a": ["t.j2"]}
        defn = self._make_defn(
            steps=[StepDefinition(
                id="s1", agent="a", template="t.j2",
                variables={"x": "{{ input.keywords }}"},
            )],
            inputs=[WorkflowInput(name="keywords")],
        )
        errors = validate_workflow(defn, agents)
        assert errors == []
