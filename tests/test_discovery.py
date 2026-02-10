"""Tests for core.discovery — robust agent scanning."""

import pytest

from core.discovery import discover_agents


class TestDiscoverAgents:
    def test_discovers_valid_agents(self, tmp_path):
        agent = tmp_path / "my_agent"
        agent.mkdir()
        (agent / "system.txt").write_text("system prompt")
        (agent / "demo.j2").write_text("{{ x }}")

        agents = discover_agents(tmp_path)
        assert len(agents) == 1
        assert agents[0].name == "my_agent"

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden_agent"
        hidden.mkdir()
        (hidden / "system.txt").write_text("prompt")

        visible = tmp_path / "visible_agent"
        visible.mkdir()
        (visible / "system.txt").write_text("prompt")

        agents = discover_agents(tmp_path)
        assert len(agents) == 1
        assert agents[0].name == "visible_agent"

    def test_skips_broken_agent_with_warning(self, tmp_path, caplog):
        broken = tmp_path / "broken_agent"
        broken.mkdir()
        # No system.txt → Agent() will raise FileNotFoundError

        valid = tmp_path / "valid_agent"
        valid.mkdir()
        (valid / "system.txt").write_text("prompt")

        agents = discover_agents(tmp_path)
        assert len(agents) == 1
        assert agents[0].name == "valid_agent"
        assert "Skipping agent 'broken_agent'" in caplog.text

    def test_empty_directory(self, tmp_path):
        agents = discover_agents(tmp_path)
        assert agents == []

    def test_nonexistent_directory(self, tmp_path):
        agents = discover_agents(tmp_path / "does_not_exist")
        assert agents == []
