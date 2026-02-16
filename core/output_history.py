"""Output history — persists LLM outputs for comparison across runs."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_HISTORY_DIR = Path(__file__).resolve().parent.parent / ".lal" / "history"


@dataclass(frozen=True)
class OutputRecord:
    """Metadata and content for a single LLM output."""

    agent: str
    template: str
    model: str
    timestamp: str
    raw_output: str
    parsed_output: dict | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    variables_hash: str = ""


def _compute_variables_hash(variables: dict) -> str:
    """Create a stable hash of input variables for grouping runs."""
    serialized = json.dumps(variables, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def save_output(
    agent: str,
    template: str,
    model: str,
    raw_output: str,
    *,
    parsed_output: dict | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    variables: dict | None = None,
    history_dir: Path | None = None,
) -> Path:
    """Save an output to the history directory.

    Returns the path to the saved file.
    """
    base_dir = history_dir or _HISTORY_DIR
    target_dir = base_dir / agent / template.removesuffix(".j2")
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    variables_hash = _compute_variables_hash(variables or {})

    record = OutputRecord(
        agent=agent,
        template=template,
        model=model,
        timestamp=now.isoformat(),
        raw_output=raw_output,
        parsed_output=parsed_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        variables_hash=variables_hash,
    )

    filename = f"{timestamp}.json"
    filepath = target_dir / filename
    filepath.write_text(
        json.dumps(asdict(record), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved output to %s", filepath)
    return filepath


def load_history(
    agent: str,
    template: str,
    *,
    limit: int = 10,
    history_dir: Path | None = None,
) -> list[OutputRecord]:
    """Load recent output records for an agent/template.

    Returns most recent *limit* records, newest first.
    """
    base_dir = history_dir or _HISTORY_DIR
    target_dir = base_dir / agent / template.removesuffix(".j2")
    if not target_dir.exists():
        return []

    files = sorted(target_dir.glob("*.json"), reverse=True)[:limit]
    records = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append(OutputRecord(**data))
        except Exception as exc:
            logger.warning("Failed to load history file %s: %s", f, exc)
    return records


def diff_outputs(old: str, new: str, *, context_lines: int = 3) -> str:
    """Generate a unified diff between two output strings.

    Returns the diff as a string, or empty string if outputs are identical.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile="previous", tofile="current",
        n=context_lines,
    )
    return "".join(diff)
