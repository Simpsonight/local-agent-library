"""Tests for resolve_command() in main.py — integration with injected cache + mocked I/O."""

from unittest.mock import patch

import pytest

from core.engine import SessionCache
from main import resolve_command


@pytest.fixture()
def cache():
    return SessionCache()


class TestResolveCtx:
    def test_ctx_resolves_last_output(self, cache):
        cache.set("cached value")
        result = resolve_command("/ctx", cache=cache)
        assert result == "cached value"

    def test_ctx_alias_resolves(self, cache):
        cache.set("aliased", alias="foo")
        result = resolve_command("/ctx foo", cache=cache)
        assert result == "aliased"

    def test_ctx_missing_returns_none(self, cache):
        result = resolve_command("/ctx", cache=cache)
        assert result is None


class TestResolveUrl:
    @patch("main.scrape_url", return_value="page content")
    def test_url_calls_scraper(self, mock_scrape, cache):
        result = resolve_command("/url https://example.com", cache=cache)
        assert result == "page content"
        mock_scrape.assert_called_once_with("https://example.com", auth=None)

    @patch("main.scrape_url", side_effect=Exception("network error"))
    def test_url_error_returns_none(self, mock_scrape, cache):
        result = resolve_command("/url https://example.com", cache=cache)
        assert result is None


class TestResolveFile:
    @patch("main.read_file", return_value="file content")
    def test_file_calls_reader(self, mock_read, cache):
        result = resolve_command("/file /tmp/test.txt", cache=cache)
        assert result == "file content"
        mock_read.assert_called_once_with("/tmp/test.txt")

    @patch("main.read_file", side_effect=FileNotFoundError("not found"))
    def test_file_error_returns_none(self, mock_read, cache):
        result = resolve_command("/file /tmp/missing.txt", cache=cache)
        assert result is None


class TestResolveTxt:
    def test_txt_raises_value_error(self, cache):
        """TxtCommand is parsed but resolve_command re-raises ValueError for it."""
        with pytest.raises(ValueError, match="not a command"):
            resolve_command("/txt hello", cache=cache)


class TestResolveNotACommand:
    def test_plain_text_raises(self, cache):
        with pytest.raises(ValueError, match="not a command"):
            resolve_command("just text", cache=cache)
