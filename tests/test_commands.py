"""Tests for core.commands — pure parsing, no I/O."""

import pytest

from core.commands import (
    parse_command,
    CtxCommand,
    UrlCommand,
    FileCommand,
    TxtCommand,
    HelpCommand,
)


class TestCtxCommand:
    def test_bare_ctx(self):
        assert parse_command("/ctx") == CtxCommand(alias="")

    def test_ctx_with_alias(self):
        assert parse_command("/ctx myalias") == CtxCommand(alias="myalias")

    def test_ctx_alias_trimmed(self):
        assert parse_command("/ctx   spaced  ") == CtxCommand(alias="spaced")


class TestUrlCommand:
    def test_simple_url(self):
        cmd = parse_command("/url https://example.com")
        assert cmd == UrlCommand(url="https://example.com", auth=None)

    def test_url_with_auth(self):
        cmd = parse_command("/url user:pass https://example.com")
        assert cmd == UrlCommand(url="https://example.com", auth=("user", "pass"))

    def test_url_missing_argument(self):
        with pytest.raises(ValueError, match="requires a URL"):
            parse_command("/url")

    def test_url_empty_after_slash(self):
        with pytest.raises(ValueError, match="requires a URL"):
            parse_command("/url ")


class TestFileCommand:
    def test_file_path(self):
        assert parse_command("/file /tmp/foo.txt") == FileCommand(path="/tmp/foo.txt")

    def test_file_missing_argument(self):
        with pytest.raises(ValueError, match="requires a path"):
            parse_command("/file")

    def test_file_empty_after_slash(self):
        with pytest.raises(ValueError, match="requires a path"):
            parse_command("/file ")


class TestTxtCommand:
    def test_bare_txt(self):
        assert parse_command("/txt") == TxtCommand(first_line="")

    def test_txt_with_text(self):
        assert parse_command("/txt hello world") == TxtCommand(first_line="hello world")


class TestHelpCommand:
    def test_help(self):
        assert parse_command("/help") == HelpCommand()


class TestNotACommand:
    def test_plain_text_raises(self):
        with pytest.raises(ValueError, match="not a command"):
            parse_command("just some text")

    def test_unknown_slash_raises(self):
        with pytest.raises(ValueError, match="not a command"):
            parse_command("/unknown something")
