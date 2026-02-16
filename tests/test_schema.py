"""Tests for core.schema — OutputSchema loading and validation."""

import json
import pytest
from pathlib import Path

from core.schema import OutputSchema, load_output_schema, validate_output


@pytest.fixture
def schema_dir(tmp_path):
    """Create a temp agent directory with a schema file."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": "number", "minimum": 0, "maximum": 100},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "score"],
    }
    (tmp_path / "analyze.j2").write_text("{{ text }}")
    (tmp_path / "analyze.schema.json").write_text(json.dumps(schema))
    return tmp_path


class TestLoadOutputSchema:
    def test_loads_existing_schema(self, schema_dir):
        result = load_output_schema(schema_dir, "analyze.j2")
        assert result is not None
        assert isinstance(result, OutputSchema)
        assert result.template_name == "analyze.j2"
        assert "title" in result.schema["properties"]
        assert result.source_path == schema_dir / "analyze.schema.json"

    def test_returns_none_when_no_schema(self, tmp_path):
        (tmp_path / "greet.j2").write_text("Hello {{ name }}")
        result = load_output_schema(tmp_path, "greet.j2")
        assert result is None

    def test_raises_on_invalid_json(self, tmp_path):
        (tmp_path / "bad.j2").write_text("{{ x }}")
        (tmp_path / "bad.schema.json").write_text("{not valid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_output_schema(tmp_path, "bad.j2")

    def test_raises_on_invalid_schema(self, tmp_path):
        (tmp_path / "bad.j2").write_text("{{ x }}")
        (tmp_path / "bad.schema.json").write_text(json.dumps({"type": "invalid_type"}))
        with pytest.raises(ValueError, match="Invalid JSON Schema"):
            load_output_schema(tmp_path, "bad.j2")

    def test_output_schema_is_frozen(self, schema_dir):
        result = load_output_schema(schema_dir, "analyze.j2")
        with pytest.raises(AttributeError):
            result.template_name = "other.j2"


class TestValidateOutput:
    @pytest.fixture
    def schema(self, schema_dir):
        return load_output_schema(schema_dir, "analyze.j2")

    def test_valid_output(self, schema):
        text = json.dumps({"title": "Test", "score": 85, "tags": ["a", "b"]})
        parsed, errors = validate_output(text, schema)
        assert parsed is not None
        assert errors == []
        assert parsed["title"] == "Test"
        assert parsed["score"] == 85

    def test_valid_without_optional_field(self, schema):
        text = json.dumps({"title": "Test", "score": 50})
        parsed, errors = validate_output(text, schema)
        assert parsed is not None
        assert errors == []

    def test_invalid_json(self, schema):
        parsed, errors = validate_output("not json at all", schema)
        assert parsed is None
        assert len(errors) == 1
        assert "JSON parse error" in errors[0]

    def test_missing_required_field(self, schema):
        text = json.dumps({"title": "Test"})
        parsed, errors = validate_output(text, schema)
        assert parsed is None
        assert any("score" in e for e in errors)

    def test_wrong_type(self, schema):
        text = json.dumps({"title": "Test", "score": "not_a_number"})
        parsed, errors = validate_output(text, schema)
        assert parsed is None
        assert any("score" in e for e in errors)

    def test_non_object_json(self, schema):
        parsed, errors = validate_output("[1, 2, 3]", schema)
        assert parsed is None
        assert any("object" in e.lower() for e in errors)

    def test_out_of_range(self, schema):
        text = json.dumps({"title": "Test", "score": 150})
        parsed, errors = validate_output(text, schema)
        assert parsed is None
        assert len(errors) > 0

    def test_wrong_array_item_type(self, schema):
        text = json.dumps({"title": "Test", "score": 50, "tags": [1, 2]})
        parsed, errors = validate_output(text, schema)
        assert parsed is None
        assert len(errors) > 0
