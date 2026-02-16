"""Tests for core.discovery — robust agent scanning."""

import pytest

from core.discovery import discover_agents


class TestDiscoverAgents:
    def test_discovers_valid_agents(self, tmp_path):
        agent = tmp_path / "my_agent"
        agent.mkdir()
        (agent / "system.txt").write_text("system prompt")
        (agent / "demo.j2").write_text("{{ x }}")
        (agent / "demo.schema.json").write_text('{"type": "object", "properties": {"result": {"type": "string"}}}')

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
        (visible / "demo.j2").write_text("{{ x }}")
        (visible / "demo.schema.json").write_text('{"type": "object", "properties": {}}')

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
        (valid / "demo.j2").write_text("{{ x }}")
        (valid / "demo.schema.json").write_text('{"type": "object", "properties": {}}')

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

    def test_skips_agent_with_missing_schema(self, tmp_path, caplog):
        agent = tmp_path / "no_schema_agent"
        agent.mkdir()
        (agent / "system.txt").write_text("system prompt")
        (agent / "demo.j2").write_text("{{ x }}")
        # No demo.schema.json → should be skipped

        agents = discover_agents(tmp_path)
        assert len(agents) == 0
        assert "Skipping agent 'no_schema_agent'" in caplog.text
        assert "Templates without schema: demo.j2" in caplog.text
        assert "demo.schema.json required" in caplog.text
