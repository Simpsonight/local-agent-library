"""Rich-based terminal UI helpers with questionary interactive menus."""

import sys

import questionary
from questionary import Style as QStyle
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit import prompt as pt_prompt
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "prompt": "green",
    "heading": "bold",
})

console = Console(theme=theme)

# questionary style matching our Rich theme
_Q_STYLE = QStyle([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:green"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:green bold"),
    ("selected", "fg:green"),
    ("instruction", "fg:white dim"),
])

_VERSION = "0.1.0"


def _is_interactive() -> bool:
    """Return True if stdin is a real terminal (not piped)."""
    return sys.stdin.isatty()


def print_banner():
    """Display a styled banner Panel with version info."""
    title = Text()
    title.append("Local ", style="bold cyan")
    title.append("Agent ", style="bold green")
    title.append("Library", style="bold cyan")
    subtitle = Text(f"v{_VERSION}", style="dim white")
    panel = Panel(
        title,
        subtitle=subtitle,
        border_style="cyan",
        padding=(1, 4),
    )
    console.print(panel)


def print_rule():
    """Print a themed horizontal rule separator."""
    console.print(Rule(style="cyan"))


def select_menu(title: str, items: list[str]) -> int | None:
    """Arrow-key select menu. Returns selected index, or None on cancel.

    Falls back to numbered_menu when stdin is not a terminal.
    """
    if not _is_interactive():
        return numbered_menu(title, items)
    try:
        result = questionary.select(
            title,
            choices=items,
            style=_Q_STYLE,
            instruction="(↑↓ navigate, Enter select, Ctrl+C back)",
        ).ask()
        if result is None:
            return None
        return items.index(result)
    except KeyboardInterrupt:
        return None


def numbered_menu(title: str, items: list[str]) -> int | None:
    """Display a numbered menu and return the selected index, or None on empty input."""
    console.print(f"\n[heading]{title}[/]")
    for i, item in enumerate(items, 1):
        console.print(f"  [info]{i}[/]) {item}")
    while True:
        choice = console.input("[prompt]\n> [/]").strip()
        if not choice:
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return idx
        except ValueError:
            pass
        print_error("Invalid selection, please try again.")


def confirm(message: str, default: bool = True) -> bool:
    """Yes/No confirmation. Falls back to text prompt if not interactive."""
    if not _is_interactive():
        result = console.input(f"[warning]{message} (y/n): [/]").strip().lower()
        return result in ("y", "yes", "")
    try:
        result = questionary.confirm(
            message,
            default=default,
            style=_Q_STYLE,
        ).ask()
        return result if result is not None else False
    except KeyboardInterrupt:
        return False


def select_action(title: str, choices: list[str]) -> str | None:
    """Arrow-key action selector. Returns chosen string, or None on cancel."""
    if not _is_interactive():
        console.print(f"[warning]{title}[/]")
        for i, c in enumerate(choices, 1):
            console.print(f"  [info]{i}[/]) {c}")
        raw = console.input("[prompt]  > [/]").strip()
        if not raw:
            return choices[0] if choices else None
        try:
            return choices[int(raw) - 1]
        except (ValueError, IndexError):
            return choices[0] if choices else None
    try:
        result = questionary.select(
            title,
            choices=choices,
            style=_Q_STYLE,
            instruction="(↑↓ navigate, Enter select)",
        ).ask()
        return result
    except KeyboardInterrupt:
        return None


def prompt_text(message: str, default: str = "") -> str | None:
    """Text prompt with questionary. Returns None on cancel."""
    if not _is_interactive():
        return console.input(f"[prompt]{message}: [/]").strip()
    try:
        result = questionary.text(
            message,
            default=default,
            style=_Q_STYLE,
        ).ask()
        return result
    except KeyboardInterrupt:
        return None


# Slash-command completer for variable input
_SLASH_COMPLETER = WordCompleter(
    ["/ctx", "/url", "/file", "/txt", "/help"],
    sentence=True,
)


def prompt_variable(var_name: str, prev_value: str | None = None) -> str:
    """Prompt for a template variable with /command autocomplete.

    Uses prompt_toolkit directly for autocomplete support.
    Falls back to console.input if not interactive.
    """
    suffix = ""
    if prev_value:
        preview_len = min(len(prev_value), 40)
        suffix = f" [Enter = reuse ({len(prev_value)} chars)]"

    if not _is_interactive():
        if prev_value:
            return console.input(f"[prompt]  {var_name}{suffix}: [/]").strip()
        return console.input(f"[prompt]  {var_name}: [/]").strip()

    try:
        result = pt_prompt(
            f"  {var_name}{suffix}: ",
            completer=_SLASH_COMPLETER,
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=False,
        )
        return result.strip() if result else ""
    except (KeyboardInterrupt, EOFError):
        return ""


def read_multiline(first_line: str = "") -> str:
    """Read lines from stdin until '---' or EOF. Returns joined text."""
    lines: list[str] = []
    if first_line:
        lines.append(first_line)
    print_warning("  (Multiline: finish with --- on its own line)")
    try:
        while True:
            line = console.input("")
            if line.strip() == "---":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def print_agent_info(name: str, config: dict, template_count: int):
    """Display agent details in a styled Panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Agent", name)
    table.add_row("Model", config.get("model", "?"))
    table.add_row("Temperature", str(config.get("temperature", "?")))
    table.add_row("Max Tokens", str(config.get("max_tokens", "?")))
    table.add_row("Templates", str(template_count))
    panel = Panel(table, border_style="cyan", title="Agent Info", title_align="left")
    console.print(panel)


def print_variable_summary(variables: dict):
    """Display a summary table of collected variables."""
    if not variables:
        return
    table = Table(title="Variables", border_style="dim", title_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Preview", style="dim")
    for name, value in sorted(variables.items()):
        preview = value[:60].replace("\n", " ")
        if len(value) > 60:
            preview += "..."
        table.add_row(name, preview)
    console.print(table)


def print_rendered_prompt(rendered: str):
    """Display the rendered prompt in a dim Panel."""
    panel = Panel(rendered, title="Rendered Prompt", border_style="dim", title_align="left")
    console.print(panel)


def print_response(text: str):
    """Display LLM response as Markdown in a styled Panel."""
    panel = Panel(
        Markdown(text),
        title="Response",
        border_style="green",
        title_align="left",
    )
    console.print(panel)


def print_error(msg: str):
    console.print(f"[error]{msg}[/]")


def print_success(msg: str):
    console.print(f"[success]{msg}[/]")


def print_warning(msg: str):
    console.print(f"[warning]{msg}[/]")


def print_info(msg: str):
    console.print(f"[info]{msg}[/]")


def print_markdown(text: str):
    """Render LLM output as rich Markdown."""
    console.print(Markdown(text))


def stream_response(token_generator):
    """Consume a streaming token generator with a live display.

    *token_generator* must yield ``(delta_text, finish_reason | None)`` tuples.
    Returns ``(full_text, finish_reason)``.
    """
    full_text = ""
    finish_reason = "stop"
    text_display = Text()
    with Live(text_display, console=console, refresh_per_second=12, transient=True) as live:
        for delta, reason in token_generator:
            if delta:
                full_text += delta
                text_display.append(delta)
                live.update(text_display)
            if reason is not None:
                finish_reason = reason
    return full_text, finish_reason
