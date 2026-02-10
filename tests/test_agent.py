"""Tests for the Agent class."""

import pytest
from pathlib import Path

from core.engine import Agent
from tests.conftest import make_mock_completion


class TestAgentLoading:
    def test_loads_system_prompt(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        assert agent.system_prompt == "You are a test assistant."

    def test_discovers_templates(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        assert agent.templates == ["greet.j2"]

    def test_name_from_folder(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        assert agent.name == "test_agent"

    def test_missing_system_txt_raises(self, tmp_path):
        empty = tmp_path / "empty_agent"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="Missing system.txt"):
            Agent(empty)


class TestAgentConfig:
    def test_defaults_when_no_config(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        assert agent.config["model"] == "gpt-4o"
        assert agent.config["temperature"] == 0.7
        assert agent.config["max_tokens"] == 16384

    def test_config_overrides(self, tmp_agent_dir_with_config):
        agent = Agent(tmp_agent_dir_with_config)
        assert agent.config["model"] == "gpt-3.5-turbo"
        assert agent.config["temperature"] == 0.5
        assert agent.config["max_tokens"] == 1024

    def test_invalid_model_type_raises(self, tmp_agent_dir):
        (tmp_agent_dir / "config.yaml").write_text("model: 123\n")
        with pytest.raises(ValueError, match="model must be a string"):
            Agent(tmp_agent_dir)

    def test_invalid_temperature_type_raises(self, tmp_agent_dir):
        (tmp_agent_dir / "config.yaml").write_text("temperature: hot\n")
        with pytest.raises(ValueError, match="temperature must be a number"):
            Agent(tmp_agent_dir)

    def test_negative_max_tokens_raises(self, tmp_agent_dir):
        (tmp_agent_dir / "config.yaml").write_text("max_tokens: -1\n")
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            Agent(tmp_agent_dir)

    def test_non_int_max_tokens_raises(self, tmp_agent_dir):
        (tmp_agent_dir / "config.yaml").write_text("max_tokens: 1.5\n")
        with pytest.raises(ValueError, match="max_tokens must be an integer"):
            Agent(tmp_agent_dir)


class TestAgentTemplates:
    def test_scan_variables(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        vars_ = agent.scan_variables("greet.j2")
        assert vars_ == {"name"}

    def test_render_template(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        result = agent.render_template("greet.j2", {"name": "World"})
        assert result == "Hello, World!"

    def test_render_missing_var_fills_sentinel(self, tmp_agent_dir):
        agent = Agent(tmp_agent_dir)
        result = agent.render_template("greet.j2", {})
        assert "[MISSING]" in result


class TestAgentRun:
    def test_run_with_mock_completion(self, tmp_agent_dir):
        mock_fn = make_mock_completion(text="test output", finish_reason="stop")
        agent = Agent(tmp_agent_dir, completion_fn=mock_fn)
        text, reason = agent.run("hello")
        assert text == "test output"
        assert reason == "stop"

    def test_run_passes_config_to_completion(self, tmp_agent_dir):
        captured = {}

        def capture_fn(**kwargs):
            captured.update(kwargs)
            return make_mock_completion()(**kwargs)

        agent = Agent(tmp_agent_dir, completion_fn=capture_fn)
        agent.run("test prompt")

        assert captured["model"] == "gpt-4o"
        assert captured["temperature"] == 0.7
        assert captured["max_tokens"] == 16384
        assert len(captured["messages"]) == 2
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][1]["content"] == "test prompt"
