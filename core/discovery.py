"""Robust agent discovery — scans agent folders under a root directory."""

import logging
from pathlib import Path

from core.engine import Agent

logger = logging.getLogger(__name__)

_DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def discover_agents(agents_dir: Path | None = None) -> list[Agent]:
    """Find all valid agent folders under *agents_dir*.

    - Skips hidden directories (name starts with '.')
    - Logs a warning and continues if an individual agent fails to load
    """
    agents_dir = agents_dir or _DEFAULT_AGENTS_DIR
    agents: list[Agent] = []
    if not agents_dir.exists():
        return agents
    for folder in sorted(agents_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            agents.append(Agent(folder))
        except Exception as exc:
            logger.warning("Skipping agent '%s': %s", folder.name, exc)
    return agents
