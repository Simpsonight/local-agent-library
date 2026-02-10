#!/usr/bin/env python3
"""Local Agent Library – Interactive CLI for orchestrating AI agents."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

from core.engine import Agent, session_cache, SessionCache
from core.commands import parse_command, CtxCommand, UrlCommand, FileCommand, TxtCommand
from core.utils import scrape_url, read_file

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(__file__).parent / "agents"

# ANSI colors – disabled when stdout is not a terminal or NO_COLOR is set
_NO_COLOR = not sys.stdout.isatty() or "NO_COLOR" in os.environ
CYAN = "" if _NO_COLOR else "\033[96m"
GREEN = "" if _NO_COLOR else "\033[92m"
YELLOW = "" if _NO_COLOR else "\033[93m"
RED = "" if _NO_COLOR else "\033[91m"
BOLD = "" if _NO_COLOR else "\033[1m"
RESET = "" if _NO_COLOR else "\033[0m"


def color(text: str, code: str) -> str:
    if _NO_COLOR:
        return text
    return f"{code}{text}{RESET}"


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    try:
        if sys.platform == "darwin":
            cmd = ["pbcopy"]
        elif sys.platform.startswith("linux"):
            cmd = ["xclip", "-selection", "clipboard"]
        else:
            cmd = ["clip"]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def print_banner():
    print(color("\n╔══════════════════════════════════════╗", CYAN))
    print(color("║     Local Agent Library (LAL)        ║", CYAN))
    print(color("╚══════════════════════════════════════╝", CYAN))


def discover_agents() -> list[Agent]:
    """Find all agent folders under agents/."""
    agents = []
    if not AGENTS_DIR.exists():
        return agents
    for folder in sorted(AGENTS_DIR.iterdir()):
        if folder.is_dir() and (folder / "system.txt").exists():
            agents.append(Agent(folder))
    return agents


def numbered_menu(title: str, items: list[str]) -> int | None:
    """Display a numbered menu and return the selected index, or None on empty input."""
    print(color(f"\n{title}", BOLD))
    for i, item in enumerate(items, 1):
        print(f"  {color(str(i), CYAN)}) {item}")
    while True:
        choice = input(color("\n> ", GREEN)).strip()
        if not choice:
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return idx
        except ValueError:
            pass
        print(color("Ungültige Auswahl, bitte erneut versuchen.", RED))


def read_multiline(first_line: str = "") -> str:
    """Read lines from stdin until a line containing only '---' or EOF (Ctrl+D).

    Returns all collected lines joined with newlines.
    """
    lines: list[str] = []
    if first_line:
        lines.append(first_line)
    print(color("  (Mehrzeilig: Abschluss mit --- in eigener Zeile)", YELLOW))
    try:
        while True:
            line = input()
            if line.strip() == "---":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def resolve_command(raw: str, cache: SessionCache | None = None) -> str | None:
    """Try to resolve a slash-command (/ctx, /url, /file).

    Returns the resolved string on success, None on error,
    or raises ValueError if *raw* is not a recognized command
    (including /txt, which is handled by the caller).
    """
    cmd = parse_command(raw)  # raises ValueError if not a command

    if isinstance(cmd, TxtCommand):
        raise ValueError("not a command")

    cache = cache or session_cache

    if isinstance(cmd, CtxCommand):
        token = f"ctx.{cmd.alias}" if cmd.alias else "ctx"
        resolved = cache.resolve(token)
        if resolved is None:
            label = cmd.alias or "letzte Ausgabe"
            print(color(f"  Fehler: '{label}' nicht im Session-Cache gefunden.", RED))
            return None
        print(color(f"  Aus Cache geladen ({len(resolved)} Zeichen).", GREEN))
        return resolved

    if isinstance(cmd, UrlCommand):
        print(color(f"  Lade {cmd.url} ...", YELLOW))
        try:
            content = scrape_url(cmd.url, auth=cmd.auth)
            print(color(f"  Geladen ({len(content)} Zeichen).", GREEN))
            return content
        except Exception as e:
            print(color(f"  Fehler beim Laden der URL: {e}", RED))
            return None

    if isinstance(cmd, FileCommand):
        try:
            content = read_file(cmd.path)
            print(color(f"  Datei geladen ({len(content)} Zeichen).", GREEN))
            return content
        except Exception as e:
            print(color(f"  Fehler beim Lesen der Datei: {e}", RED))
            return None

    raise ValueError("not a command")


def collect_variables(agent: Agent, template_name: str, cache: SessionCache | None = None) -> dict | None:
    """Prompt the user for each template variable.

    Returns the variables dict, or None if the user aborts.
    """
    cache = cache or session_cache
    variables = {}
    var_names = agent.scan_variables(template_name)
    if not var_names:
        print(color("  Keine Variablen vorhanden.", YELLOW))
        return variables

    print(color("\nVariablen eingeben (/ctx, /url, /file = Befehle | Text = mehrzeilig, --- zum Abschliessen):", BOLD))
    for var in sorted(var_names):
        # Check if a previous value exists for this variable
        prev = cache.get_variable(var)

        while True:
            if prev:
                prompt_suffix = f" [Enter = vorherigen Wert übernehmen ({len(prev)} Z.)]"
                raw = input(color(f"  {var}{prompt_suffix}: ", GREEN)).strip()
            else:
                raw = input(color(f"  {var}: ", GREEN)).strip()

            # Empty input
            if not raw:
                if prev:
                    variables[var] = prev
                    preview = prev[:60].replace("\n", " ")
                    if len(prev) > 60:
                        preview += "..."
                    print(color(f"  Übernommen ({len(prev)} Zeichen): {preview}", GREEN))
                else:
                    print(color(f"  Warnung: '{var}' leer → [MISSING]", YELLOW))
                    variables[var] = "[MISSING]"
                break

            # Try slash-commands (single-line, resolved immediately)
            try:
                resolved = resolve_command(raw, cache=cache)
                if resolved is not None:
                    variables[var] = resolved
                    break
                # resolved is None → command recognized but failed
                action = input(color("  Erneut eingeben (Enter) / überspringen (s) / abbrechen (q): ", YELLOW)).strip().lower()
                if action == "s":
                    variables[var] = "[MISSING]"
                    break
                if action == "q":
                    return None
                continue
            except ValueError:
                pass  # Not a command → treat as text input

            # Plain text → multi-line mode
            # Strip /txt prefix for backwards compatibility
            if raw == "/txt":
                first_line = ""
            elif raw.startswith("/txt "):
                first_line = raw[5:]
            else:
                first_line = raw
            text = read_multiline(first_line)
            print(color(f"  Text übernommen ({len(text)} Zeichen).", GREEN))
            variables[var] = text
            break
    return variables


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("LAL_DEBUG") else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )
    print_banner()

    while True:
        try:
            # Step 1: Agent selection
            agents = discover_agents()
            if not agents:
                print(color("Keine Agents gefunden in agents/. Erstelle einen Agent-Ordner mit system.txt.", RED))
                sys.exit(1)

            agent_names = [a.name for a in agents]
            idx = numbered_menu("Agent auswählen:", agent_names)
            if idx is None:
                continue
            agent = agents[idx]
            print(color(f"\nAgent: {agent.name} ({len(agent.templates)} templates)", BOLD))

            # Step 2: Template selection
            if not agent.templates:
                print(color("  Keine Templates (.j2) für diesen Agent gefunden.", RED))
                continue

            tidx = numbered_menu("Template auswählen:", agent.templates)
            if tidx is None:
                continue
            template_name = agent.templates[tidx]

            # Step 3: Variable input
            variables = collect_variables(agent, template_name)
            if variables is None:
                continue

            # Step 4: Render and execute
            rendered = agent.render_template(template_name, variables)
            print(color("\n── Gerenderter Prompt ──", YELLOW))
            print(rendered)
            print(color("── LLM wird aufgerufen ──", YELLOW))

            result = None
            finish_reason = "stop"
            while True:
                try:
                    result, finish_reason = agent.run(rendered)
                    break
                except AuthenticationError:
                    print(color("\nAuthentifizierung fehlgeschlagen. API-Key in .env prüfen.", RED))
                    logger.debug("AuthenticationError for model %s", agent.config["model"])
                    break
                except BadRequestError as e:
                    print(color(f"\nUngültige Anfrage: {e}", RED))
                    logger.debug("BadRequestError: %s", e, exc_info=True)
                    break
                except RateLimitError:
                    print(color("\nRate-Limit erreicht. Bitte kurz warten.", RED))
                    retry = input(color("Erneut versuchen? (Enter = ja, q = abbrechen): ", YELLOW)).strip().lower()
                    if retry == "q":
                        break
                except (APIConnectionError, Timeout):
                    print(color("\nVerbindung zum API-Server fehlgeschlagen.", RED))
                    retry = input(color("Erneut versuchen? (Enter = ja, q = abbrechen): ", YELLOW)).strip().lower()
                    if retry == "q":
                        break
                except Exception as e:
                    print(color(f"\nLLM-Fehler: {e}", RED))
                    logger.debug("LLM error: %s", e, exc_info=True)
                    retry = input(color("Erneut versuchen? (Enter = ja, q = abbrechen): ", YELLOW)).strip().lower()
                    if retry == "q":
                        break

            if result is None:
                continue

            print(color("\n── Antwort ──", GREEN))
            print(result)

            if finish_reason == "length":
                print(color(f"\n  WARNUNG: Antwort wurde abgeschnitten (max_tokens={agent.config['max_tokens']} erreicht)!", RED))

            # Step 5: Save result + variables for reuse
            session_cache.set(result)
            session_cache.set_variables(variables)

            # Post-result actions: copy / alias / continue
            print(color("\n  [c] In Zwischenablage kopieren  [a] Mit Alias speichern  [Enter] Weiter", YELLOW))
            action = input(color("  > ", GREEN)).strip().lower()
            if action == "c":
                if copy_to_clipboard(result):
                    print(color("  In Zwischenablage kopiert!", GREEN))
                else:
                    print(color("  Fehler: Zwischenablage nicht verfügbar.", RED))
                # Still offer alias
                alias = input(color("  Mit Alias speichern? (Enter = überspringen): ", GREEN)).strip()
                if alias:
                    session_cache.set(result, alias)
                    print(color(f"  Gespeichert als ctx.{alias}", GREEN))
            elif action == "a" or action.startswith("a "):
                alias = action[2:].strip() if action.startswith("a ") else ""
                if not alias:
                    alias = input(color("  Alias: ", GREEN)).strip()
                if alias:
                    session_cache.set(result, alias)
                    print(color(f"  Gespeichert als ctx.{alias}", GREEN))
            # In all cases, result is saved as ctx (last_output) already

            print(color("\n" + "─" * 40, CYAN))

        except KeyboardInterrupt:
            print(color("\n\nAuf Wiedersehen!", CYAN))
            sys.exit(0)
        except EOFError:
            print(color("\n\nAuf Wiedersehen!", CYAN))
            sys.exit(0)


if __name__ == "__main__":
    main()
