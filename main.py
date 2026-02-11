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
    confirm,
    console,
    print_agent_info,
    print_banner,
    print_error,
    print_info,
    print_rendered_prompt,
    print_response,
    print_rule,
    print_success,
    print_variable_summary,
    print_warning,
    prompt_text,
    select_action,
    select_menu,
    stream_response,
)
from core.clipboard import copy_to_clipboard
from core.discovery import discover_agents
from core.engine import session_cache
from core.preflight import check_api_key
from core.resolver import collect_variables

logger = logging.getLogger(__name__)

POST_ACTIONS = [
    "Continue",
    "Copy to clipboard",
    "Save with alias",
    "Copy + Save with alias",
]


def _handle_post_actions(result: str):
    """Handle post-result user actions (copy, save, etc.)."""
    action = select_action("What next?", POST_ACTIONS)

    if action == "Copy to clipboard":
        if copy_to_clipboard(result):
            print_success("  Copied to clipboard!")
        else:
            print_error("  Error: clipboard not available.")

    elif action == "Save with alias":
        alias = prompt_text("Alias")
        if alias:
            session_cache.set(result, alias)
            print_success(f"  Saved as ctx.{alias}")

    elif action == "Copy + Save with alias":
        if copy_to_clipboard(result):
            print_success("  Copied to clipboard!")
        else:
            print_error("  Error: clipboard not available.")
        alias = prompt_text("Alias")
        if alias:
            session_cache.set(result, alias)
            print_success(f"  Saved as ctx.{alias}")


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

            agent_labels = [
                f"{a.name} ({len(a.templates)} templates)" for a in agents
            ]
            idx = select_menu("Select agent:", agent_labels)
            if idx is None:
                print_info("\nGoodbye!")
                sys.exit(0)
            agent = agents[idx]

            # Show agent info panel
            print_agent_info(agent.name, agent.config, len(agent.templates))

            # Pre-flight API key check
            key_err = check_api_key(agent.config["model"])
            if key_err:
                print_error(key_err)
                continue

            # Template selection
            if not agent.templates:
                print_error("  No templates (.j2) found for this agent.")
                continue

            tidx = select_menu("Select template:", agent.templates)
            if tidx is None:
                continue
            template_name = agent.templates[tidx]

            # Variable input
            variables = collect_variables(agent, template_name)
            if variables is None:
                continue

            # Variable summary
            print_variable_summary(variables)

            # Render and display prompt
            rendered = agent.render_template(template_name, variables)
            print_rendered_prompt(rendered)

            # Streaming LLM call
            result = None
            finish_reason = "stop"
            while True:
                try:
                    with console.status("Calling LLM..."):
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
                    if not confirm("Retry?"):
                        break
                except (APIConnectionError, Timeout):
                    print_error("\nConnection to API server failed.")
                    if not confirm("Retry?"):
                        break
                except Exception as e:
                    print_error(f"\nLLM error: {e}")
                    logger.debug("LLM error: %s", e, exc_info=True)
                    if not confirm("Retry?"):
                        break

            if result is None:
                continue

            # Display response in styled panel
            print_response(result)

            if finish_reason == "length":
                print_warning(f"\n  Note: Response was truncated (max_tokens={agent.config['max_tokens']} reached).")

            # Save result + variables for reuse
            session_cache.set(result)
            session_cache.set_variables(variables)

            # Post-result actions
            _handle_post_actions(result)

            print_rule()

        except KeyboardInterrupt:
            print_info("\n\nGoodbye!")
            sys.exit(0)
        except EOFError:
            print_info("\n\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
