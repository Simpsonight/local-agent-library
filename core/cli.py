"""Rich-based terminal UI helpers with questionary interactive menus."""

from __future__ import annotations

import sys
from contextlib import contextmanager

import questionary
from questionary import Style as QStyle
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import prompt as pt_prompt
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
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


def _make_input_keybindings() -> KeyBindings:
    """Create key bindings for multiline input.

    Enter submits, Escape→Enter (Alt+Enter) inserts a newline.
    A trailing backslash before Enter is replaced with a newline (continuation).
    """
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        buf = event.app.current_buffer
        text = buf.text
        # Backslash at end of current line → replace with newline (continuation)
        if text.endswith("\\"):
            buf.text = text[:-1]
            buf.insert_text("\n")
        else:
            buf.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline_alt(event):
        event.app.current_buffer.insert_text("\n")

    return kb


_INPUT_KEYBINDINGS = _make_input_keybindings()


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

    Uses prompt_toolkit directly with multiline key bindings.
    Shift+Enter inserts a newline, Enter submits.
    Falls back to console.input if not interactive.
    """
    suffix = ""
    if prev_value:
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
            key_bindings=_INPUT_KEYBINDINGS,
            multiline=True,
            prompt_continuation="  ... ",
        )
        return result.strip() if result else ""
    except (KeyboardInterrupt, EOFError):
        return ""


def _read_lines_until_marker(prompt_str: str = "") -> str:
    """Non-TTY fallback: read lines until ``---`` or EOF."""
    lines: list[str] = []
    if prompt_str:
        console.print(f"[prompt]{prompt_str}[/]")
    print_warning("  (Enter text, finish with --- on its own line)")
    try:
        while True:
            line = console.input("")
            if line.strip() == "---":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def prompt_input(label: str, initial: str = "") -> str:
    """General-purpose text input with multiline key bindings.

    Uses the same Shift+Enter / Alt+Enter bindings as prompt_variable.
    Falls back to _read_lines_until_marker when not interactive.
    """
    if not _is_interactive():
        return _read_lines_until_marker(label)

    try:
        result = pt_prompt(
            f"  {label}: ",
            default=initial,
            key_bindings=_INPUT_KEYBINDINGS,
            multiline=True,
            prompt_continuation="  ... ",
        )
        return result.strip() if result else ""
    except (KeyboardInterrupt, EOFError):
        return initial


@contextmanager
def input_zone(label: str = ""):
    """Context manager that wraps content between two horizontal rules."""
    console.print(Rule(title=label, style="dim cyan") if label else Rule(style="dim cyan"))
    yield
    console.print(Rule(style="dim cyan"))


def print_input_hint():
    """Display a one-time hint about multiline input shortcuts."""
    console.print("[dim]  Alt+Enter (Esc→Enter) für Newline | \\ am Zeilenende für Continuation | Enter zum Absenden[/]")


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


def print_response(text: str, *, parsed: dict | None = None):
    """Display LLM response — structured JSON if parsed, otherwise Markdown."""
    if parsed is not None:
        print_structured_response(parsed)
        return
    panel = Panel(
        Markdown(text),
        title="Response",
        border_style="green",
        title_align="left",
    )
    console.print(panel)


def print_structured_response(data: dict):
    """Display structured JSON output with syntax highlighting."""
    import json
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
    panel = Panel(
        syntax,
        title="Structured Response (JSON)",
        border_style="green",
        title_align="left",
    )
    console.print(panel)


def print_validation_errors(errors: list[str], attempts: int):
    """Display schema validation errors as a warning panel."""
    lines = [f"[warning]Schema validation failed after {attempts} attempt(s):[/]"]
    for err in errors:
        lines.append(f"  [error]• {err}[/]")
    panel = Panel(
        "\n".join(lines),
        title="Validation Errors",
        border_style="yellow",
        title_align="left",
    )
    console.print(panel)


def print_tool_call(name: str, args: dict):
    """Display an MCP tool call as an info panel."""
    import json
    args_str = json.dumps(args, indent=2, ensure_ascii=False) if args else "{}"
    content = f"[info]Tool:[/] {name}\n[info]Args:[/]\n{args_str}"
    panel = Panel(content, title="Tool Call", border_style="cyan", title_align="left")
    console.print(panel)


def print_tool_result(name: str, result: str):
    """Display an MCP tool result (truncated if long)."""
    preview = result[:500]
    if len(result) > 500:
        preview += f"\n... ({len(result)} chars total)"
    panel = Panel(preview, title=f"Tool Result: {name}", border_style="dim", title_align="left")
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


def print_usage_inline(model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float, attempts: int = 1):
    """Display a compact one-line usage summary after an LLM call."""
    total = prompt_tokens + completion_tokens
    parts = [f"[dim]{model}[/]"]
    if total > 0:
        parts.append(f"[dim]{total:,} tokens ({prompt_tokens:,}+{completion_tokens:,})[/]")
    if cost_usd > 0:
        parts.append(f"[dim]~${cost_usd:.4f}[/]")
    if attempts > 1:
        parts.append(f"[warning]{attempts} attempts[/]")
    console.print("  " + "  ·  ".join(parts))


def print_agent_work_summary(model: str, tokens: int, cost: float, duration_sec: float):
    """Compact one-line summary after an LLM call (Claude Code style)."""
    parts = [f"[dim]{model}[/]"]
    if tokens > 0:
        parts.append(f"[dim]{tokens:,} tokens[/]")
    if cost > 0:
        parts.append(f"[dim]~${cost:.4f}[/]")
    if duration_sec > 0:
        parts.append(f"[dim]{duration_sec:.1f}s[/]")
    console.print("  " + "  ·  ".join(parts))


def print_agent_work_details(rendered: str, model: str, prompt_tokens: int, completion_tokens: int, cost: float, attempts: int = 1):
    """Detailed panel showing rendered prompt and full token breakdown."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Model", model)
    table.add_row("Prompt Tokens", f"{prompt_tokens:,}")
    table.add_row("Completion Tokens", f"{completion_tokens:,}")
    table.add_row("Total Tokens", f"{prompt_tokens + completion_tokens:,}")
    if cost > 0:
        table.add_row("Cost", f"~${cost:.4f}")
    if attempts > 1:
        table.add_row("Attempts", str(attempts))
    console.print(Panel(table, title="Token Details", border_style="dim", title_align="left"))
    console.print(Panel(rendered, title="Rendered Prompt", border_style="dim", title_align="left"))


def print_cost_summary(summary: dict):
    """Display a cost/usage summary panel (session totals)."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Prompt Tokens", f"{summary['total_prompt_tokens']:,}")
    table.add_row("Completion Tokens", f"{summary['total_completion_tokens']:,}")
    table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
    cost = summary['total_cost_usd']
    table.add_row("Estimated Cost", f"${cost:.4f}" if cost > 0 else "N/A")
    table.add_row("API Calls", str(summary['num_calls']))
    if summary.get('models_used'):
        table.add_row("Models", ", ".join(summary['models_used']))
    panel = Panel(table, border_style="dim", title="Session Usage Summary", title_align="left")
    console.print(panel)


def stream_response(token_generator, *, model: str = ""):
    """Consume a streaming token generator with a live display.

    *token_generator* must yield ``(delta_text, finish_reason | None)`` tuples.
    Returns ``(full_text, finish_reason)``.

    The streamed text remains visible after completion (``transient=False``).
    A spinner is shown until the first token arrives.
    """
    full_text = ""
    finish_reason = "stop"
    spinner_label = f"{model} generating..." if model else "Generating..."
    started = False
    with Live(
        Text(f"⏳ {spinner_label}", style="dim"),
        console=console,
        refresh_per_second=12,
        transient=False,
    ) as live:
        for delta, reason in token_generator:
            if delta:
                full_text += delta
                if not started:
                    started = True
                live.update(Markdown(full_text))
            if reason is not None:
                finish_reason = reason
    return full_text, finish_reason


# ── Workflow UI helpers ───────────────────────────────────


def print_workflow_info(name: str, description: str, step_count: int, input_count: int):
    """Display workflow details in a styled Panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Workflow", name)
    if description:
        table.add_row("Description", description)
    table.add_row("Steps", str(step_count))
    table.add_row("Inputs", str(input_count))
    panel = Panel(table, border_style="cyan", title="Workflow Info", title_align="left")
    console.print(panel)


def print_step_header(step_id: str, agent_name: str, template: str, step_num: int, total: int):
    """Display a header for a workflow step being executed."""
    console.print(
        f"\n[info]Step {step_num}/{total}:[/] "
        f"[bold]{step_id}[/] → [cyan]{agent_name}[/] / {template}"
    )


def print_step_result(step_id: str, status: str, output_len: int):
    """Display a summary line for a completed step."""
    if status == "completed":
        console.print(f"  [success]✓ {step_id}[/] ({output_len} chars)")
    elif status == "failed":
        console.print(f"  [error]✗ {step_id} FAILED[/]")
    elif status == "skipped":
        console.print(f"  [warning]○ {step_id} skipped[/]")


def prompt_checkpoint(step_id: str) -> str:
    """Show checkpoint menu after a step. Returns 'approve', 'edit', or 'abort'."""
    choices = [
        "Approve & Continue",
        "Edit Output",
        "Abort Workflow",
    ]
    result = select_action(f"Checkpoint [{step_id}]:", choices)
    if result == "Edit Output":
        return "edit"
    if result == "Abort Workflow":
        return "abort"
    return "approve"


def prompt_workflow_input(name: str, description: str, allow_commands: bool) -> str:
    """Prompt for a workflow input value with optional command support."""
    label = name
    if description:
        label = f"{name} ({description})"
    if allow_commands:
        return prompt_variable(label)
    if not _is_interactive():
        return console.input(f"[prompt]  {label}: [/]").strip()
    try:
        result = pt_prompt(f"  {label}: ")
        return result.strip() if result else ""
    except (KeyboardInterrupt, EOFError):
        return ""


def print_workflow_summary(results: dict, total_steps: int):
    """Display a final summary of workflow execution."""
    completed = sum(1 for r in results.values() if r.status.value == "completed")
    failed = sum(1 for r in results.values() if r.status.value == "failed")
    skipped = sum(1 for r in results.values() if r.status.value == "skipped")

    parts = [f"[success]{completed} completed[/]"]
    if failed:
        parts.append(f"[error]{failed} failed[/]")
    if skipped:
        parts.append(f"[warning]{skipped} skipped[/]")

    console.print(f"\n[heading]Workflow complete:[/] {', '.join(parts)}")
