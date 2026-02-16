"""Tests for core.resolver.resolve_command() — integration with injected cache + mocked I/O."""

from unittest.mock import patch

import pytest

from core.commands import CtxCommand, UrlCommand, FileCommand, TxtCommand, HelpCommand
from core.engine import SessionCache
from core.resolver import resolve_command


@pytest.fixture()
def cache():
    return SessionCache()


class TestResolveCtx:
    def test_ctx_resolves_last_output(self, cache):
        cache.set("cached value")
        result = resolve_command(CtxCommand(), cache=cache)
        assert result == "cached value"

    def test_ctx_alias_resolves(self, cache):
        cache.set("aliased", alias="foo")
        result = resolve_command(CtxCommand(alias="foo"), cache=cache)
        assert result == "aliased"

    def test_ctx_missing_returns_none(self, cache):
        result = resolve_command(CtxCommand(), cache=cache)
        assert result is None


class TestResolveUrl:
    @patch("core.resolver.console")
    @patch("core.resolver.scrape_url", return_value="page content")
    def test_url_calls_scraper(self, mock_scrape, mock_console, cache):
        mock_console.input.return_value = "n"  # No auth
        result = resolve_command(UrlCommand(url="https://example.com"), cache=cache)
        assert result == "page content"
        mock_scrape.assert_called_once_with("https://example.com", auth=None)

    @patch("core.resolver.console")
    @patch("core.resolver.scrape_url", side_effect=Exception("network error"))
    def test_url_error_returns_none(self, mock_scrape, mock_console, cache):
        mock_console.input.return_value = "n"  # No auth
        result = resolve_command(UrlCommand(url="https://example.com"), cache=cache)
        assert result is None

    @patch("core.resolver.getpass", return_value="secret")
    @patch("core.resolver.console")
    @patch("core.resolver.scrape_url", return_value="authed content")
    def test_url_with_interactive_auth(self, mock_scrape, mock_console, mock_getpass, cache):
        mock_console.input.side_effect = ["y", "admin"]  # auth=y, username=admin
        result = resolve_command(UrlCommand(url="https://example.com"), cache=cache)
        assert result == "authed content"
        mock_scrape.assert_called_once_with("https://example.com", auth=("admin", "secret"))


class TestResolveFile:
    @patch("core.resolver.read_file", return_value="file content")
    def test_file_calls_reader(self, mock_read, cache):
        result = resolve_command(FileCommand(path="/tmp/test.txt"), cache=cache)
        assert result == "file content"
        mock_read.assert_called_once_with("/tmp/test.txt")

    @patch("core.resolver.read_file", side_effect=FileNotFoundError("not found"))
    def test_file_error_returns_none(self, mock_read, cache):
        result = resolve_command(FileCommand(path="/tmp/missing.txt"), cache=cache)
        assert result is None


class TestResolveTxtAndHelp:
    def test_txt_raises_type_error(self, cache):
        with pytest.raises(TypeError, match="TxtCommand"):
            resolve_command(TxtCommand(first_line="hello"), cache=cache)

    def test_help_raises_type_error(self, cache):
        with pytest.raises(TypeError, match="HelpCommand"):
            resolve_command(HelpCommand(), cache=cache)
