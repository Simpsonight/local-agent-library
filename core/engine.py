"""Agent engine: loads agents from folder structure, renders templates, calls LLMs."""

import json
import logging
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, meta
import litellm

from core.schema import OutputSchema, load_output_schema

logger = logging.getLogger(__name__)

DEFAULTS = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 16384,
}

VALID_CONFIG_KEYS = set(DEFAULTS.keys()) | {"output_format", "mcp_servers"}


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

    def __init__(self, path: Path, completion_fn=None):
        self.path = Path(path)
        self.name = self.path.name
        self._completion_fn = completion_fn or litellm.completion

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
            unknown = set(overrides.keys()) - VALID_CONFIG_KEYS
            if unknown:
                logger.warning("Unknown config keys in %s: %s", config_file, unknown)
            if "model" in overrides and not isinstance(overrides["model"], str):
                raise ValueError(f"model must be a string in {config_file}, got {type(overrides['model']).__name__}")
            if "temperature" in overrides:
                if not isinstance(overrides["temperature"], (int, float)):
                    raise ValueError(f"temperature must be a number in {config_file}, got {type(overrides['temperature']).__name__}")
                if not 0.0 <= overrides["temperature"] <= 2.0:
                    logger.warning("temperature %.2f in %s is outside typical range [0.0, 2.0]", overrides["temperature"], config_file)
            if "max_tokens" in overrides:
                if not isinstance(overrides["max_tokens"], int):
                    raise ValueError(f"max_tokens must be an integer in {config_file}, got {type(overrides['max_tokens']).__name__}")
                if overrides["max_tokens"] <= 0:
                    raise ValueError(f"max_tokens must be positive in {config_file}, got {overrides['max_tokens']}")
            self.config.update(overrides)

        # Discover templates
        self.templates = sorted(p.name for p in self.path.glob("*.j2"))

        # Load output schemas for templates
        self._schemas: dict[str, OutputSchema] = {}
        for tpl_name in self.templates:
            schema = load_output_schema(self.path, tpl_name)
            if schema is not None:
                self._schemas[tpl_name] = schema

        # Jinja2 environment scoped to agent folder
        self._env = Environment(
            loader=FileSystemLoader(str(self.path)),
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

    def get_schema(self, template_name: str) -> OutputSchema | None:
        """Return the output schema for a template, or None if not defined."""
        return self._schemas.get(template_name)

    def _augment_prompt_with_schema(self, prompt: str, template_name: str | None) -> str:
        """If a schema exists for the template, append it to the prompt."""
        if not template_name or template_name not in self._schemas:
            return prompt
        schema = self._schemas[template_name]
        schema_json = json.dumps(schema.schema, indent=2, ensure_ascii=False)
        return (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond with a JSON object matching this exact schema. "
            f"Use the exact field names specified:\n"
            f"```json\n{schema_json}\n```"
        )

    def run_stream(self, prompt: str, *, template_name: str | None = None):
        """Call litellm with streaming enabled.

        Yields ``(delta_text, None)`` per chunk and ``("", finish_reason)`` at end.
        If *template_name* is given and has a schema, adds ``response_format``
        to request JSON output from the model and appends the schema to the prompt.

        After completion, ``self.last_usage`` contains token counts (if available).
        """
        actual_prompt = self._augment_prompt_with_schema(prompt, template_name)
        self.last_usage = None

        kwargs = dict(
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": actual_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        # Request JSON output when a schema exists for the template
        if template_name and template_name in self._schemas:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._completion_fn(**kwargs)
        for chunk in response:
            # Capture usage from final chunk (sent after finish_reason)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "model": getattr(chunk, "model", self.config["model"]) or self.config["model"],
                }
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta_obj = getattr(choice, "delta", None)
            delta = getattr(delta_obj, "content", None) or ""
            if choice.finish_reason:
                yield ("", choice.finish_reason)
                return
            if delta:
                yield (delta, None)
        yield ("", "stop")

    def run_with_tools(self, prompt: str, tools: list[dict], call_tool_fn, *, template_name: str | None = None):
        """Multi-turn LLM call with tool use support.

        Yields events:
        - ``("text", delta)`` for streaming text
        - ``("tool_call", {"name": name, "arguments": args})`` when LLM wants a tool
        - ``("tool_result", {"name": name, "result": result})`` after tool execution
        - ``("finish", full_text)`` at the end

        *call_tool_fn* should accept ``(name: str, arguments: dict)`` and return ``str``.
        """
        actual_prompt = self._augment_prompt_with_schema(prompt, template_name)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": actual_prompt},
        ]

        kwargs = dict(
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            tools=tools,
        )

        if template_name and template_name in self._schemas:
            kwargs["response_format"] = {"type": "json_object"}

        max_turns = 10  # Safety limit for tool-calling loops
        full_text = ""

        for _turn in range(max_turns):
            kwargs["messages"] = messages
            response = self._completion_fn(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # Accumulate any text content
            if message.content:
                full_text += message.content
                yield ("text", message.content)

            # Check for tool calls
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                yield ("finish", full_text)
                return

            # Process tool calls
            messages.append(message)  # Add assistant message with tool_calls

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                yield ("tool_call", {"name": fn_name, "arguments": fn_args})

                try:
                    result = call_tool_fn(fn_name, fn_args)
                except Exception as exc:
                    result = f"Error: {exc}"

                yield ("tool_result", {"name": fn_name, "result": result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Safety: max turns reached
        yield ("finish", full_text)

    def run(self, prompt: str) -> tuple[str, str]:
        """Call litellm with system prompt + user prompt.

        Returns (text, finish_reason).  finish_reason is typically
        'stop' (complete) or 'length' (truncated by max_tokens).
        """
        response = self._completion_fn(
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
