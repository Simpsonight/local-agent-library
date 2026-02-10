#!/usr/bin/env python3
"""Local Agent Library – Interactive CLI for orchestrating AI agents."""

import logging
import os
import sys

from dotenv import load_dotenv
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

from core.cli import (
    console,
    print_banner,
    numbered_menu,
    print_error,
    print_info,
    print_markdown,
    print_success,
    print_warning,
    stream_response,
)
from core.clipboard import copy_to_clipboard
from core.discovery import discover_agents
from core.engine import session_cache
from core.preflight import check_api_key
from core.resolver import collect_variables

logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("LAL_DEBUG") else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )
    print_banner()

    while True:
        try:
            # Agent selection
            agents = discover_agents()
            if not agents:
                print_error("No agents found in agents/. Create an agent folder with system.txt.")
                sys.exit(1)

            agent_names = [a.name for a in agents]
            idx = numbered_menu("Select agent:", agent_names)
            if idx is None:
                continue
            agent = agents[idx]
            console.print(f"\n[heading]Agent: {agent.name} ({len(agent.templates)} templates)[/]")

            # Pre-flight API key check
            key_err = check_api_key(agent.config["model"])
            if key_err:
                print_error(key_err)
                continue

            # Template selection
            if not agent.templates:
                print_error("  No templates (.j2) found for this agent.")
                continue

            tidx = numbered_menu("Select template:", agent.templates)
            if tidx is None:
                continue
            template_name = agent.templates[tidx]

            # Variable input
            variables = collect_variables(agent, template_name)
            if variables is None:
                continue

            # Render and execute
            rendered = agent.render_template(template_name, variables)
            print_warning("\n── Rendered Prompt ──")
            console.print(rendered)

            # Streaming LLM call
            result = None
            finish_reason = "stop"
            while True:
                try:
                    with console.status("Calling LLM..."):
                        # Initialize streaming generator (actual response starts in stream_response)
                        token_gen = agent.run_stream(rendered)
                    result, finish_reason = stream_response(token_gen)
                    break
                except AuthenticationError:
                    print_error("\nAuthentication failed. Check API key in .env.")
                    logger.debug("AuthenticationError for model %s", agent.config["model"])
                    break
                except BadRequestError as e:
                    print_error(f"\nBad request: {e}")
                    logger.debug("BadRequestError: %s", e, exc_info=True)
                    break
                except RateLimitError:
                    print_error("\nRate limit reached. Please wait a moment.")
                    retry = console.input("[warning]Retry? (Enter = yes, q = abort): [/]").strip().lower()
                    if retry == "q":
                        break
                except (APIConnectionError, Timeout):
                    print_error("\nConnection to API server failed.")
                    retry = console.input("[warning]Retry? (Enter = yes, q = abort): [/]").strip().lower()
                    if retry == "q":
                        break
                except Exception as e:
                    print_error(f"\nLLM error: {e}")
                    logger.debug("LLM error: %s", e, exc_info=True)
                    retry = console.input("[warning]Retry? (Enter = yes, q = abort): [/]").strip().lower()
                    if retry == "q":
                        break

            if result is None:
                continue

            # Display response as Markdown
            print_success("\n── Response ──")
            print_markdown(result)

            if finish_reason == "length":
                print_warning(f"\n  Note: Response was truncated (max_tokens={agent.config['max_tokens']} reached).")

            # Save result + variables for reuse
            session_cache.set(result)
            session_cache.set_variables(variables)

            # Post-result actions
            print_warning("\n  [c] Copy to clipboard  [a] Save with alias  [Enter] Continue")
            action = console.input("[prompt]  > [/]").strip().lower()
            if action == "c":
                if copy_to_clipboard(result):
                    print_success("  Copied to clipboard!")
                else:
                    print_error("  Error: clipboard not available.")
                alias = console.input("[prompt]  Save with alias? (Enter = skip): [/]").strip()
                if alias:
                    session_cache.set(result, alias)
                    print_success(f"  Saved as ctx.{alias}")
            elif action == "a" or action.startswith("a "):
                alias = action[2:].strip() if action.startswith("a ") else ""
                if not alias:
                    alias = console.input("[prompt]  Alias: [/]").strip()
                if alias:
                    session_cache.set(result, alias)
                    print_success(f"  Saved as ctx.{alias}")

            print_info("\n" + "─" * 40)

        except KeyboardInterrupt:
            print_info("\n\nGoodbye!")
            sys.exit(0)
        except EOFError:
            print_info("\n\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
