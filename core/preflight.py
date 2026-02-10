"""Pre-flight API key check before calling an LLM."""

import os

MODEL_KEY_MAP: dict[str, str] = {
    "gpt-": "OPENAI_API_KEY",
    "o1-": "OPENAI_API_KEY",
    "o3-": "OPENAI_API_KEY",
    "o4-": "OPENAI_API_KEY",
    "claude-": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def check_api_key(model: str) -> str | None:
    """Return an error message if the expected API key is missing, else None.

    Unknown model prefixes pass through (no blocking).
    """
    for prefix, env_var in MODEL_KEY_MAP.items():
        if model.startswith(prefix):
            if not os.environ.get(env_var):
                return f"Missing {env_var} in environment for model '{model}'. Add it to your .env file."
            return None
    # Unknown prefix — don't block
    return None
