"""Agent engine: loads agents from folder structure, renders templates, calls LLMs."""

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, meta, Undefined
import litellm


DEFAULTS = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 16384,
}


class SessionCache:
    """Dict-like singleton that stores LLM results for chaining across runs."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._variables: dict[str, str] = {}

    def set(self, value: str, alias: str | None = None):
        self._store["last_output"] = value
        if alias:
            self._store[alias] = value

    def get(self, key: str = "last_output") -> str | None:
        return self._store.get(key)

    def resolve(self, token: str) -> str | None:
        """Resolve 'ctx' -> last_output, 'ctx.alias' -> alias lookup."""
        if token == "ctx":
            return self.get("last_output")
        if token.startswith("ctx."):
            alias = token[4:]
            return self.get(alias)
        return None

    def set_variables(self, variables: dict[str, str]):
        """Store input variables from the last run for reuse."""
        self._variables.update(variables)

    def get_variable(self, name: str) -> str | None:
        """Retrieve a previously stored input variable by name."""
        return self._variables.get(name)

    def keys(self):
        return self._store.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __repr__(self):
        return f"SessionCache({list(self._store.keys())})"


# Module-level singleton
session_cache = SessionCache()


class Agent:
    """An agent defined by a folder containing system.txt, optional config.yaml, and .j2 templates."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.name

        # Load system prompt
        system_file = self.path / "system.txt"
        if not system_file.exists():
            raise FileNotFoundError(f"Missing system.txt in {self.path}")
        self.system_prompt = system_file.read_text(encoding="utf-8").strip()

        # Load config with defaults
        self.config = dict(DEFAULTS)
        config_file = self.path / "config.yaml"
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                overrides = yaml.safe_load(f) or {}
            self.config.update(overrides)

        # Discover templates
        self.templates = sorted(p.name for p in self.path.glob("*.j2"))

        # Jinja2 environment scoped to agent folder
        self._env = Environment(
            loader=FileSystemLoader(str(self.path)),
            undefined=Undefined,
        )

    def scan_variables(self, template_name: str) -> set[str]:
        """Return all undeclared variables in a template."""
        source = (self.path / template_name).read_text(encoding="utf-8")
        ast = self._env.parse(source)
        return meta.find_undeclared_variables(ast)

    def render_template(self, template_name: str, variables: dict) -> str:
        """Render a template; missing vars become '[MISSING]'."""
        tpl = self._env.get_template(template_name)
        # Fill missing keys with sentinel
        all_vars = self.scan_variables(template_name)
        for var in all_vars:
            if var not in variables:
                variables[var] = "[MISSING]"
        return tpl.render(**variables)

    def run(self, prompt: str) -> tuple[str, str]:
        """Call litellm with system prompt + user prompt.

        Returns (text, finish_reason).  finish_reason is typically
        'stop' (complete) or 'length' (truncated by max_tokens).
        """
        response = litellm.completion(
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        choice = response.choices[0]
        return choice.message.content, choice.finish_reason or "stop"

    def __repr__(self):
        return f"Agent({self.name}, templates={self.templates})"
