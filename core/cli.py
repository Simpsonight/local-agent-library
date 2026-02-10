"""Rich-based terminal UI helpers."""

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
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


def print_banner():
    console.print("\n[info]╔══════════════════════════════════════╗[/]")
    console.print("[info]║     Local Agent Library (LAL)        ║[/]")
    console.print("[info]╚══════════════════════════════════════╝[/]")


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
    with Live(text_display, console=console, refresh_per_second=12) as live:
        for delta, reason in token_generator:
            if delta:
                full_text += delta
                text_display.append(delta)
                live.update(text_display)
            if reason is not None:
                finish_reason = reason
    return full_text, finish_reason
