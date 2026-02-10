"""Tests for SessionCache."""

import pytest

from core.engine import SessionCache


class TestSessionCacheSetGet:
    def test_set_stores_last_output(self, fresh_cache):
        fresh_cache.set("hello")
        assert fresh_cache.get() == "hello"
        assert fresh_cache.get("last_output") == "hello"

    def test_set_with_alias(self, fresh_cache):
        fresh_cache.set("result", alias="myalias")
        assert fresh_cache.get("myalias") == "result"
        assert fresh_cache.get("last_output") == "result"

    def test_get_missing_returns_none(self, fresh_cache):
        assert fresh_cache.get("nonexistent") is None

    def test_set_overwrites_last_output(self, fresh_cache):
        fresh_cache.set("first")
        fresh_cache.set("second")
        assert fresh_cache.get() == "second"


class TestSessionCacheResolve:
    def test_resolve_ctx_returns_last_output(self, fresh_cache):
        fresh_cache.set("data")
        assert fresh_cache.resolve("ctx") == "data"

    def test_resolve_ctx_alias(self, fresh_cache):
        fresh_cache.set("data", alias="foo")
        assert fresh_cache.resolve("ctx.foo") == "data"

    def test_resolve_ctx_missing_returns_none(self, fresh_cache):
        assert fresh_cache.resolve("ctx") is None

    def test_resolve_unknown_token_returns_none(self, fresh_cache):
        assert fresh_cache.resolve("unknown") is None


class TestSessionCacheVariables:
    def test_set_and_get_variables(self, fresh_cache):
        fresh_cache.set_variables({"a": "1", "b": "2"})
        assert fresh_cache.get_variable("a") == "1"
        assert fresh_cache.get_variable("b") == "2"

    def test_get_variable_missing_returns_none(self, fresh_cache):
        assert fresh_cache.get_variable("nope") is None


class TestSessionCacheContainerProtocol:
    def test_contains(self, fresh_cache):
        fresh_cache.set("val", alias="x")
        assert "x" in fresh_cache
        assert "missing" not in fresh_cache

    def test_keys(self, fresh_cache):
        fresh_cache.set("val", alias="a")
        keys = set(fresh_cache.keys())
        assert "last_output" in keys
        assert "a" in keys

    def test_repr(self, fresh_cache):
        fresh_cache.set("val", alias="r")
        r = repr(fresh_cache)
        assert "SessionCache" in r
        assert "r" in r
