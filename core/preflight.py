"""Pre-flight API key check before calling an LLM."""

import os

# Fallback prefix map for models not in the registry.
_FALLBACK_KEY_MAP: dict[str, str] = {
    "gpt-": "OPENAI_API_KEY",
    "o1-": "OPENAI_API_KEY",
    "o3-": "OPENAI_API_KEY",
    "o4-": "OPENAI_API_KEY",
    "claude-": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def check_api_key(model: str) -> str | None:
    """Return an error message if the expected API key is missing, else None.

    Uses the model registry first; falls back to prefix matching for
    unknown models. Unknown model prefixes pass through (no blocking).
    """
    from core.model_registry import get_registry

    entry = get_registry().get(model)
    if entry is not None:
        if not os.environ.get(entry.api_key_env):
            return f"Missing {entry.api_key_env} in environment for model '{model}'. Add it to your .env file."
        return None

    # Fallback for models not in registry
    for prefix, env_var in _FALLBACK_KEY_MAP.items():
        if model.startswith(prefix):
            if not os.environ.get(env_var):
                return f"Missing {env_var} in environment for model '{model}'. Add it to your .env file."
            return None
    # Unknown prefix — don't block
    return None
