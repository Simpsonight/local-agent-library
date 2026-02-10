"""Pure command parsing — no I/O, no side effects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CtxCommand:
    """Resolve a value from the session cache."""
    alias: str = ""


@dataclass(frozen=True)
class UrlCommand:
    """Scrape a URL."""
    url: str
    auth: tuple[str, str] | None = None


@dataclass(frozen=True)
class FileCommand:
    """Read a local file."""
    path: str


@dataclass(frozen=True)
class TxtCommand:
    """Literal text (bypass command parsing)."""
    first_line: str = ""


@dataclass(frozen=True)
class HelpCommand:
    """Show available commands."""
    pass


Command = CtxCommand | UrlCommand | FileCommand | TxtCommand | HelpCommand


def parse_command(raw: str) -> Command:
    """Parse a slash-command string into a typed Command.

    Raises ValueError if *raw* is not a recognized command.
    """
    if raw == "/ctx" or raw.startswith("/ctx "):
        alias = raw[5:].strip() if raw.startswith("/ctx ") else ""
        return CtxCommand(alias=alias)

    if raw.startswith("/url "):
        rest = raw[5:].strip()
        if not rest:
            raise ValueError("/url requires a URL argument")
        parts = rest.split(None, 1)
        auth = None
        url = rest
        if len(parts) == 2 and ":" in parts[0] and not parts[0].startswith("http"):
            user, password = parts[0].split(":", 1)
            auth = (user, password)
            url = parts[1]
        return UrlCommand(url=url, auth=auth)

    if raw == "/url":
        raise ValueError("/url requires a URL argument")

    if raw.startswith("/file "):
        path = raw[6:].strip()
        if not path:
            raise ValueError("/file requires a path argument")
        return FileCommand(path=path)

    if raw == "/file":
        raise ValueError("/file requires a path argument")

    if raw == "/help":
        return HelpCommand()

    if raw == "/txt":
        return TxtCommand(first_line="")

    if raw.startswith("/txt "):
        return TxtCommand(first_line=raw[5:])

    raise ValueError("not a command")
