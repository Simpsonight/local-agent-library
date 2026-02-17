"""Command resolution and variable collection — bridges parsed commands to I/O."""

from getpass import getpass

from core.cli import console, print_error, print_success, print_warning, prompt_variable, input_zone, prompt_input, print_input_hint
from core.commands import (
    parse_command,
    Command,
    CtxCommand,
    UrlCommand,
    FileCommand,
    TxtCommand,
    HelpCommand,
)
from core.engine import Agent, SessionCache, session_cache
from core.utils import scrape_url, read_file

HELP_TEXT = """\
Available commands:
  /ctx           Inject last LLM output
  /ctx <alias>   Inject a named cached result
  /url <url>     Scrape web page content (prompts for auth if needed)
  /file <path>   Load file content
  /txt <text>    Literal plain text (bypass command parsing)
  /help          Show this help message

Alt+Enter (Esc→Enter) for newline, \\ at line end for continuation, Enter to submit.\
"""


def resolve_command(cmd: Command, cache: SessionCache | None = None) -> str | None:
    """Resolve a data-fetching command (Ctx, Url, File).

    Returns the resolved string on success, None on error.
    Raises TypeError for TxtCommand/HelpCommand (caller dispatches those).
    """
    if isinstance(cmd, (TxtCommand, HelpCommand)):
        raise TypeError(f"resolve_command does not handle {type(cmd).__name__}")

    cache = cache or session_cache

    if isinstance(cmd, CtxCommand):
        token = f"ctx.{cmd.alias}" if cmd.alias else "ctx"
        resolved = cache.resolve(token)
        if resolved is None:
            label = cmd.alias or "last output"
            print_error(f"  Error: '{label}' not found in session cache.")
            return None
        print_success(f"  Loaded from cache ({len(resolved)} chars).")
        return resolved

    if isinstance(cmd, UrlCommand):
        # Prompt for credentials securely (password hidden) if needed
        auth = None
        needs_auth = console.input("[prompt]  Requires authentication? (y/N): [/]").strip().lower()
        if needs_auth == "y":
            username = console.input("[prompt]  Username: [/]").strip()
            password = getpass("  Password: ")
            auth = (username, password)
        print_warning(f"  Loading {cmd.url} ...")
        try:
            content = scrape_url(cmd.url, auth=auth)
            print_success(f"  Loaded ({len(content)} chars).")
            return content
        except Exception as e:
            print_error(f"  Error loading URL: {e}")
            return None

    if isinstance(cmd, FileCommand):
        try:
            content = read_file(cmd.path)
            print_success(f"  File loaded ({len(content)} chars).")
            return content
        except Exception as e:
            print_error(f"  Error reading file: {e}")
            return None

    raise TypeError(f"Unknown command type: {type(cmd).__name__}")


def collect_variables(agent: Agent, template_name: str, cache: SessionCache | None = None) -> dict | None:
    """Prompt the user for each template variable.

    Returns the variables dict, or None if the user aborts.
    """
    cache = cache or session_cache
    variables = {}
    var_names = agent.scan_variables(template_name)
    if not var_names:
        print_warning("  No variables found.")
        return variables

    with input_zone("Variables"):
        print_input_hint()
        console.print("[heading]Enter variables (/ctx, /url, /file = commands | /help for reference):[/]")
        for var in sorted(var_names):
            prev = cache.get_variable(var)

            while True:
                raw = prompt_variable(var, prev_value=prev)

                # Empty input
                if not raw:
                    if prev:
                        variables[var] = prev
                        preview = prev[:60].replace("\n", " ")
                        if len(prev) > 60:
                            preview += "..."
                        print_success(f"  Reused ({len(prev)} chars): {preview}")
                    else:
                        print_warning(f"  Warning: '{var}' empty → [MISSING]")
                        variables[var] = "[MISSING]"
                    break

                # Parse as command
                try:
                    cmd = parse_command(raw)
                except ValueError:
                    cmd = None

                if cmd is not None:
                    # /help — show help, re-prompt
                    if isinstance(cmd, HelpCommand):
                        console.print(HELP_TEXT)
                        continue

                    # /txt — use first_line directly, or prompt for input
                    if isinstance(cmd, TxtCommand):
                        text = cmd.first_line if cmd.first_line else prompt_input(var)
                        print_success(f"  Text captured ({len(text)} chars).")
                        variables[var] = text
                        break

                    # Data-fetching commands
                    resolved = resolve_command(cmd, cache=cache)
                    if resolved is not None:
                        variables[var] = resolved
                        break
                    # resolved is None → command recognized but failed
                    action = console.input("[warning]  Retry (Enter) / skip (s) / abort (q): [/]").strip().lower()
                    if action == "s":
                        variables[var] = "[MISSING]"
                        break
                    if action == "q":
                        return None
                    continue

                # Plain text — raw value is used directly (multiline via Shift+Enter)
                variables[var] = raw
                break

    return variables
