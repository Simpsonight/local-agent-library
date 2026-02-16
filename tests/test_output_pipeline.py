"""Tests for core.output_pipeline — OutputPipeline validation and retry logic."""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.engine import Agent
from core.output_pipeline import OutputPipeline, OutputResult
from core.schema import OutputSchema


@pytest.fixture
def agent_with_schema(tmp_path):
    """Create an agent with a template and matching schema."""
    (tmp_path / "system.txt").write_text("You are a test assistant.")
    (tmp_path / "analyze.j2").write_text("Analyze: {{ text }}")
    schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["result", "score"],
    }
    (tmp_path / "analyze.schema.json").write_text(json.dumps(schema))

    mock_fn = MagicMock()
    agent = Agent(tmp_path, completion_fn=mock_fn)
    return agent, mock_fn


@pytest.fixture
def agent_without_schema(tmp_path):
    """Create an agent without a schema."""
    (tmp_path / "system.txt").write_text("You are a test assistant.")
    (tmp_path / "greet.j2").write_text("Hello {{ name }}")
    (tmp_path / "greet.schema.json").write_text(json.dumps({
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
    }))

    mock_fn = MagicMock()
    agent = Agent(tmp_path, completion_fn=mock_fn)
    return agent, mock_fn


def _make_stream_response(text):
    """Create a mock streaming response that yields text then finishes."""
    chunks = []
    for char in text:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = char
        chunk.choices[0].finish_reason = None
        chunk.usage = None
        chunks.append(chunk)
    # Final chunk
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta = MagicMock()
    final.choices[0].delta.content = ""
    final.choices[0].finish_reason = "stop"
    final.usage = None
    chunks.append(final)
    return chunks


class TestOutputPipelineNoSchema:
    def test_returns_raw_text(self, agent_without_schema):
        agent, mock_fn = agent_without_schema
        valid_json = json.dumps({"greeting": "Hello!"})
        mock_fn.return_value = iter(_make_stream_response(valid_json))

        pipeline = OutputPipeline()
        result = pipeline.run("test prompt", agent, "greet.j2")

        assert isinstance(result, OutputResult)
        assert result.raw_text == valid_json
        assert result.parsed is not None  # greet.j2 has a schema
        assert result.errors == []
        assert result.attempts == 1


class TestOutputPipelineWithSchema:
    def test_valid_json_passes(self, agent_with_schema):
        agent, mock_fn = agent_with_schema
        valid_json = json.dumps({"result": "good", "score": 95})
        mock_fn.return_value = iter(_make_stream_response(valid_json))

        pipeline = OutputPipeline()
        result = pipeline.run("test prompt", agent, "analyze.j2")

        assert result.parsed is not None
        assert result.parsed["result"] == "good"
        assert result.parsed["score"] == 95
        assert result.errors == []
        assert result.attempts == 1

    def test_retries_on_invalid_json(self, agent_with_schema):
        agent, mock_fn = agent_with_schema
        invalid_json = "not json"
        valid_json = json.dumps({"result": "fixed", "score": 80})
        mock_fn.side_effect = [
            iter(_make_stream_response(invalid_json)),
            iter(_make_stream_response(valid_json)),
        ]

        pipeline = OutputPipeline()
        result = pipeline.run("test prompt", agent, "analyze.j2")

        assert result.parsed is not None
        assert result.parsed["result"] == "fixed"
        assert result.attempts == 2

    def test_retries_on_schema_violation(self, agent_with_schema):
        agent, mock_fn = agent_with_schema
        missing_field = json.dumps({"result": "good"})  # missing score
        valid_json = json.dumps({"result": "good", "score": 90})
        mock_fn.side_effect = [
            iter(_make_stream_response(missing_field)),
            iter(_make_stream_response(valid_json)),
        ]

        pipeline = OutputPipeline()
        result = pipeline.run("test prompt", agent, "analyze.j2")

        assert result.parsed is not None
        assert result.attempts == 2

    def test_max_retries_exhausted(self, agent_with_schema):
        agent, mock_fn = agent_with_schema
        invalid = json.dumps({"result": "bad"})  # always missing score
        mock_fn.return_value = iter(_make_stream_response(invalid))
        mock_fn.side_effect = [
            iter(_make_stream_response(invalid)),
            iter(_make_stream_response(invalid)),
            iter(_make_stream_response(invalid)),
        ]

        pipeline = OutputPipeline()
        result = pipeline.run("test prompt", agent, "analyze.j2", max_retries=2)

        assert result.parsed is None
        assert len(result.errors) > 0
        assert result.attempts == 3

    def test_on_stream_callback(self, agent_with_schema):
        agent, mock_fn = agent_with_schema
        valid_json = json.dumps({"result": "ok", "score": 50})
        mock_fn.return_value = iter(_make_stream_response(valid_json))

        deltas = []
        pipeline = OutputPipeline()
        result = pipeline.run(
            "test prompt", agent, "analyze.j2",
            on_stream=lambda d: deltas.append(d),
        )

        assert result.parsed is not None
        assert "".join(deltas) == valid_json

    def test_zero_retries(self, agent_with_schema):
        agent, mock_fn = agent_with_schema
        invalid = json.dumps({"result": "bad"})
        mock_fn.return_value = iter(_make_stream_response(invalid))

        pipeline = OutputPipeline()
        result = pipeline.run("test prompt", agent, "analyze.j2", max_retries=0)

        assert result.parsed is None
        assert result.attempts == 1
        assert len(result.errors) > 0


class TestOutputResult:
    def test_frozen(self):
        result = OutputResult(raw_text="test", finish_reason="stop")
        with pytest.raises(AttributeError):
            result.raw_text = "changed"

    def test_defaults(self):
        result = OutputResult(raw_text="test", finish_reason="stop")
        assert result.parsed is None
        assert result.errors == []
        assert result.attempts == 1
