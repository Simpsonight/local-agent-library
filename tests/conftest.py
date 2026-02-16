"""Shared fixtures for all test modules."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine import SessionCache


@pytest.fixture()
def fresh_cache():
    """Return a brand-new SessionCache (no global state leak)."""
    return SessionCache()


@pytest.fixture()
def tmp_agent_dir(tmp_path):
    """Minimal agent directory with system.txt and a simple template."""
    agent_dir = tmp_path / "test_agent"
    agent_dir.mkdir()
    (agent_dir / "system.txt").write_text("You are a test assistant.")
    (agent_dir / "greet.j2").write_text("Hello, {{ name }}!")
    return agent_dir


@pytest.fixture()
def tmp_agent_dir_with_config(tmp_agent_dir):
    """Agent directory that also has a config.yaml."""
    (tmp_agent_dir / "config.yaml").write_text(
        "model: gpt-3.5-turbo\ntemperature: 0.5\nmax_tokens: 1024\n"
    )
    return tmp_agent_dir


def make_mock_completion(text="mock response", finish_reason="stop"):
    """Factory that returns a mock completion callable."""
    def _mock(**kwargs):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason=finish_reason,
        )
        return SimpleNamespace(choices=[choice])
    return _mock


def make_mock_stream_completion(chunks, finish_reason="stop"):
    """Factory that returns a mock streaming completion callable.

    *chunks* is a list of strings; each becomes one streamed delta.
    """
    def _mock(**kwargs):
        for text in chunks:
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text),
                    finish_reason=None,
                )],
                usage=None,
            )
        # Final chunk with finish_reason
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None),
                finish_reason=finish_reason,
            )],
            usage=None,
        )
    return _mock
