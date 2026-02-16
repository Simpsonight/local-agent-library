"""Model registry — loads model definitions from models.yaml."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_FILE = _ROOT / "models.yaml"


@dataclass(frozen=True)
class ModelEntry:
    """A single model definition from the registry."""

    id: str
    display_name: str
    provider: str
    api_key_env: str
    input_cost_per_1m: float
    output_cost_per_1m: float

    @property
    def checkbox_label(self) -> str:
        """Label for checkbox UI: ``GPT-4o (OpenAI, ~$2.50/$10.00 per 1M in/out)``."""
        return (
            f"{self.display_name} "
            f"({self.provider}, ~${self.input_cost_per_1m:.2f}/${self.output_cost_per_1m:.2f} per 1M in/out)"
        )

    def has_api_key(self) -> bool:
        """Return True if the required API key is set in the environment."""
        return bool(os.environ.get(self.api_key_env))


class ModelRegistry:
    """Loads and queries model definitions from ``models.yaml``."""

    def __init__(self, models: list[ModelEntry] | None = None):
        self._models: list[ModelEntry] = models or []
        self._by_id: dict[str, ModelEntry] = {m.id: m for m in self._models}

    # ── Loading ──────────────────────────────────────────

    @classmethod
    def from_file(cls, path: Path = _MODELS_FILE) -> ModelRegistry:
        """Load registry from a YAML file. Returns empty registry on error."""
        if not path.exists():
            logger.warning("Model registry file not found: %s", path)
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            entries = []
            for raw in data.get("models", []):
                entries.append(ModelEntry(
                    id=raw["id"],
                    display_name=raw.get("display_name", raw["id"]),
                    provider=raw.get("provider", ""),
                    api_key_env=raw.get("api_key_env", ""),
                    input_cost_per_1m=float(raw.get("input_cost_per_1m", 0)),
                    output_cost_per_1m=float(raw.get("output_cost_per_1m", 0)),
                ))
            return cls(entries)
        except Exception as e:
            logger.warning("Failed to load model registry from %s: %s", path, e)
            return cls()

    # ── Queries ──────────────────────────────────────────

    @property
    def all_models(self) -> list[ModelEntry]:
        return list(self._models)

    def available_models(self) -> list[ModelEntry]:
        """Return only models whose API key is present in the environment."""
        return [m for m in self._models if m.has_api_key()]

    def get(self, model_id: str) -> ModelEntry | None:
        """Lookup by exact id, then by prefix match.

        Prefix match handles versioned ids like ``gpt-4o-2024-11-20`` matching
        the ``gpt-4o`` entry.
        """
        entry = self._by_id.get(model_id)
        if entry is not None:
            return entry
        # Prefix match: find the longest matching prefix
        best: ModelEntry | None = None
        best_len = 0
        for m in self._models:
            if model_id.startswith(m.id) and len(m.id) > best_len:
                best = m
                best_len = len(m.id)
        return best

    def get_cost(self, model_id: str) -> tuple[float, float] | None:
        """Return ``(input_cost_per_1m, output_cost_per_1m)`` or None."""
        entry = self.get(model_id)
        if entry is None:
            return None
        return (entry.input_cost_per_1m, entry.output_cost_per_1m)


# ── Module singleton ─────────────────────────────────────

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the module-level singleton, loading lazily on first call."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.from_file()
    return _registry
