"""Tests for core.model_registry — model loading, lookup, and cost queries."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from core.model_registry import ModelEntry, ModelRegistry, get_registry


# ── ModelEntry ───────────────────────────────────────────


class TestModelEntry:
    @pytest.fixture()
    def entry(self):
        return ModelEntry(
            id="gpt-4o",
            display_name="GPT-4o",
            provider="OpenAI",
            api_key_env="OPENAI_API_KEY",
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
        )

    def test_checkbox_label(self, entry):
        label = entry.checkbox_label
        assert "GPT-4o" in label
        assert "OpenAI" in label
        assert "$2.50" in label
        assert "$10.00" in label

    def test_has_api_key_present(self, entry):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            assert entry.has_api_key() is True

    def test_has_api_key_missing(self, entry):
        with patch.dict("os.environ", {}, clear=True):
            assert entry.has_api_key() is False

    def test_frozen(self, entry):
        with pytest.raises(AttributeError):
            entry.id = "other"


# ── ModelRegistry ────────────────────────────────────────


@pytest.fixture()
def sample_models():
    return [
        ModelEntry("gpt-4o", "GPT-4o", "OpenAI", "OPENAI_API_KEY", 2.50, 10.00),
        ModelEntry("claude-sonnet-4", "Claude Sonnet 4", "Anthropic", "ANTHROPIC_API_KEY", 3.00, 15.00),
        ModelEntry("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", "Google", "GEMINI_API_KEY", 0.15, 0.60),
    ]


class TestModelRegistry:
    def test_all_models(self, sample_models):
        reg = ModelRegistry(sample_models)
        assert len(reg.all_models) == 3

    def test_get_exact(self, sample_models):
        reg = ModelRegistry(sample_models)
        entry = reg.get("gpt-4o")
        assert entry is not None
        assert entry.id == "gpt-4o"

    def test_get_prefix_match(self, sample_models):
        reg = ModelRegistry(sample_models)
        entry = reg.get("gpt-4o-2024-11-20")
        assert entry is not None
        assert entry.id == "gpt-4o"

    def test_get_unknown_returns_none(self, sample_models):
        reg = ModelRegistry(sample_models)
        assert reg.get("unknown-model") is None

    def test_get_cost(self, sample_models):
        reg = ModelRegistry(sample_models)
        costs = reg.get_cost("claude-sonnet-4")
        assert costs == (3.00, 15.00)

    def test_get_cost_unknown(self, sample_models):
        reg = ModelRegistry(sample_models)
        assert reg.get_cost("unknown") is None

    def test_available_models_filters(self, sample_models):
        reg = ModelRegistry(sample_models)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            available = reg.available_models()
            assert len(available) == 1
            assert available[0].id == "gpt-4o"

    def test_available_models_all_keys(self, sample_models):
        reg = ModelRegistry(sample_models)
        env = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant",
            "GEMINI_API_KEY": "gkey",
        }
        with patch.dict("os.environ", env, clear=True):
            available = reg.available_models()
            assert len(available) == 3


class TestFromFile:
    def test_loads_yaml(self, tmp_path):
        data = {
            "models": [
                {
                    "id": "test-model",
                    "display_name": "Test Model",
                    "provider": "Test",
                    "api_key_env": "TEST_KEY",
                    "input_cost_per_1m": 1.0,
                    "output_cost_per_1m": 2.0,
                }
            ]
        }
        f = tmp_path / "models.yaml"
        f.write_text(yaml.dump(data))
        reg = ModelRegistry.from_file(f)
        assert len(reg.all_models) == 1
        assert reg.all_models[0].id == "test-model"

    def test_missing_file_returns_empty(self, tmp_path):
        reg = ModelRegistry.from_file(tmp_path / "nonexistent.yaml")
        assert len(reg.all_models) == 0

    def test_malformed_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "models.yaml"
        f.write_text(": invalid: yaml: [")
        reg = ModelRegistry.from_file(f)
        assert len(reg.all_models) == 0


class TestGetRegistry:
    def test_returns_registry(self):
        reg = get_registry()
        assert isinstance(reg, ModelRegistry)

    def test_loads_project_models_yaml(self):
        reg = get_registry()
        # The project models.yaml should have at least the 16 models we defined
        assert len(reg.all_models) >= 15

    def test_singleton_returns_same_instance(self):
        # Reset singleton
        import core.model_registry as mod
        mod._registry = None
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
