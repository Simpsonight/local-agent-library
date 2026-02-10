"""Tests for core.preflight — API key pre-checks."""

from unittest.mock import patch

from core.preflight import check_api_key


class TestCheckApiKey:
    def test_openai_key_present(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            assert check_api_key("gpt-4o") is None

    def test_openai_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            result = check_api_key("gpt-4o")
            assert result is not None
            assert "OPENAI_API_KEY" in result

    def test_anthropic_key_present(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            assert check_api_key("claude-3-opus") is None

    def test_anthropic_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            result = check_api_key("claude-3-opus")
            assert result is not None
            assert "ANTHROPIC_API_KEY" in result

    def test_gemini_key_present(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            assert check_api_key("gemini-pro") is None

    def test_gemini_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            result = check_api_key("gemini-pro")
            assert result is not None
            assert "GEMINI_API_KEY" in result

    def test_o1_model_uses_openai_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            assert check_api_key("o1-preview") is None

    def test_unknown_prefix_passes_through(self):
        with patch.dict("os.environ", {}, clear=True):
            assert check_api_key("custom-model") is None
