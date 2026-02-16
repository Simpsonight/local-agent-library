"""Robust agent and workflow discovery — scans folders under root directories."""

import logging
from pathlib import Path

from core.engine import Agent
from core.workflow_schema import WorkflowDefinition, load_workflow

logger = logging.getLogger(__name__)

_DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
_DEFAULT_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def discover_agents(agents_dir: Path | None = None) -> list[Agent]:
    """Find all valid agent folders under *agents_dir*.

    - Skips hidden directories (name starts with '.')
    - Skips directories starting with '_' (reserved for built-in agents)
    - Logs a warning and continues if an individual agent fails to load
    """
    agents_dir = agents_dir or _DEFAULT_AGENTS_DIR
    agents: list[Agent] = []
    if not agents_dir.exists():
        return agents
    for folder in sorted(agents_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith((".", "_")):
            continue
        try:
            agents.append(Agent(folder))
        except Exception as exc:
            logger.warning("Skipping agent '%s': %s", folder.name, exc)
    return agents


def discover_workflows(workflows_dir: Path | None = None) -> list[WorkflowDefinition]:
    """Find all valid workflow YAML files under *workflows_dir*.

    - Scans for .yaml and .yml files
    - Logs a warning and continues if a workflow fails to load
    """
    workflows_dir = workflows_dir or _DEFAULT_WORKFLOWS_DIR
    workflows: list[WorkflowDefinition] = []
    if not workflows_dir.exists():
        return workflows
    for path in sorted(workflows_dir.iterdir()):
        if path.is_file() and path.suffix in (".yaml", ".yml"):
            try:
                workflows.append(load_workflow(path))
            except Exception as exc:
                logger.warning("Skipping workflow '%s': %s", path.name, exc)
    return workflows
