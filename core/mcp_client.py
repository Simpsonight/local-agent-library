"""MCP client — manages connections to Model Context Protocol servers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPTool:
    """A tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict
    server_name: str


class MCPManager:
    """Manages connections to multiple MCP servers and routes tool calls."""

    def __init__(self):
        self._sessions: dict[str, tuple] = {}  # name -> (read, write, session)
        self._tools: dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self._server_sessions: dict[str, object] = {}  # server_name -> ClientSession

    async def connect(self, server_configs: list[MCPServerConfig]) -> list[dict]:
        """Connect to all configured MCP servers and return tool definitions.

        Returns tools in the OpenAI function-calling format for litellm.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        all_tools: list[dict] = []

        for config in server_configs:
            try:
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env or None,
                )

                read, write = await asyncio.to_thread(
                    lambda: None  # placeholder
                )
                # Use the context manager protocol
                transport = stdio_client(server_params)
                read_stream, write_stream = await transport.__aenter__()

                session = ClientSession(read_stream, write_stream)
                await session.__aenter__()
                await session.initialize()

                self._sessions[config.name] = (transport, session)
                self._server_sessions[config.name] = session

                # List available tools
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    mcp_tool = MCPTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        server_name=config.name,
                    )
                    self._tools[tool.name] = mcp_tool
                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        },
                    })

                logger.info(
                    "Connected to MCP server '%s': %d tools available",
                    config.name, len(tools_result.tools),
                )

            except Exception as exc:
                logger.error("Failed to connect to MCP server '%s': %s", config.name, exc)
                raise ConnectionError(
                    f"Failed to connect to MCP server '{config.name}': {exc}"
                ) from exc

        return all_tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool by name, routing to the correct MCP server.

        Returns the tool result as a string.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}")

        session = self._server_sessions.get(tool.server_name)
        if session is None:
            raise ConnectionError(f"No active session for server '{tool.server_name}'")

        try:
            result = await session.call_tool(name, arguments)
            # Combine text content from the result
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts)
        except Exception as exc:
            logger.error("Tool call '%s' failed: %s", name, exc)
            raise RuntimeError(f"Tool call '{name}' failed: {exc}") from exc

    async def close(self):
        """Close all MCP server connections."""
        for name, (transport, session) in self._sessions.items():
            try:
                await session.__aexit__(None, None, None)
                await transport.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error closing MCP server '%s': %s", name, exc)
        self._sessions.clear()
        self._server_sessions.clear()
        self._tools.clear()

    @property
    def available_tools(self) -> list[MCPTool]:
        """Return all discovered tools."""
        return list(self._tools.values())

    def get_tools_openai_format(self) -> list[dict]:
        """Return all tools in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools.values()
        ]


def parse_mcp_configs(raw_configs: list[dict]) -> list[MCPServerConfig]:
    """Parse MCP server configurations from config.yaml format.

    Expected format:
    ```yaml
    mcp_servers:
      - name: filesystem
        command: npx
        args: ["-y", "@anthropic/mcp-filesystem"]
    ```
    """
    configs = []
    for raw in raw_configs:
        if "name" not in raw or "command" not in raw:
            raise ValueError(f"MCP server config missing 'name' or 'command': {raw}")
        configs.append(MCPServerConfig(
            name=raw["name"],
            command=raw["command"],
            args=raw.get("args", []),
            env=raw.get("env", {}),
        ))
    return configs
