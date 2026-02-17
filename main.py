#!/usr/bin/env python3
"""Local Agent Library – Interactive CLI for orchestrating AI agents."""

import logging
import os
import sys
import time

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
    input_zone,
    print_agent_info,
    print_agent_work_details,
    print_agent_work_summary,
    print_banner,
    print_cost_summary,
    print_error,
    print_info,
    print_input_hint,
    print_rendered_prompt,
    print_response,
    print_rule,
    print_step_header,
    print_step_result,
    print_structured_response,
    print_success,
    print_usage_inline,
    print_validation_errors,
    print_variable_summary,
    print_warning,
    print_workflow_info,
    print_workflow_summary,
    prompt_checkpoint,
    prompt_input,
    prompt_text,
    prompt_workflow_input,
    select_action,
    select_menu,
    stream_response,
)
from core.clipboard import copy_to_clipboard
from core.commands import parse_command
from core.cost_tracker import CostTracker, estimate_cost
from core.discovery import discover_agents, discover_workflows
from core.engine import session_cache
from core.output_pipeline import OutputPipeline, OutputResult
from core.preflight import check_api_key
from core.resolver import collect_variables, resolve_command
from core.workflow_context import CheckpointAction, StepStatus
from core.workflow_runner import WorkflowRunner
from core.workflow_schema import (
    ParallelGroup,
    StepDefinition,
    WorkflowDefinition,
    validate_workflow,
)

logger = logging.getLogger(__name__)

# Session-wide cost tracker
cost_tracker = CostTracker()

POST_ACTIONS = [
    "Continue",
    "Show run details",
    "Copy to clipboard",
    "Save with alias",
    "Copy + Save with alias",
]

TOP_LEVEL_MODES = [
    "Run Single Agent",
    "Run Workflow",
]


def _handle_post_actions(result: str, *, details: dict | None = None):
    """Handle post-result user actions (copy, save, show details, etc.)."""
    action = select_action("What next?", POST_ACTIONS)

    if action == "Show run details":
        if details:
            print_agent_work_details(
                details["rendered"],
                details["model"],
                details["prompt_tokens"],
                details["completion_tokens"],
                details["cost"],
                details.get("attempts", 1),
            )
        else:
            print_warning("  No run details available.")

    elif action == "Copy to clipboard":
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


def _collect_workflow_inputs(workflow: WorkflowDefinition) -> dict[str, str] | None:
    """Prompt the user for all workflow inputs. Returns None if aborted."""
    if not workflow.inputs:
        return {}

    inputs: dict[str, str] = {}
    with input_zone("Workflow Inputs"):
        print_input_hint()
        for inp in workflow.inputs:
            while True:
                raw = prompt_workflow_input(inp.name, inp.description, inp.command)
                if not raw:
                    print_warning(f"  Warning: '{inp.name}' empty")
                    inputs[inp.name] = ""
                    break

                # If commands are allowed, try to resolve them
                if inp.command:
                    try:
                        cmd = parse_command(raw)
                    except ValueError:
                        cmd = None

                    if cmd is not None:
                        from core.commands import TxtCommand, HelpCommand
                        if isinstance(cmd, HelpCommand):
                            from core.resolver import HELP_TEXT
                            console.print(HELP_TEXT)
                            continue
                        if isinstance(cmd, TxtCommand):
                            text = cmd.first_line if cmd.first_line else prompt_input(inp.name)
                            inputs[inp.name] = text
                            print_success(f"  Text captured ({len(text)} chars).")
                            break
                        resolved = resolve_command(cmd)
                        if resolved is not None:
                            inputs[inp.name] = resolved
                            break
                        action = console.input("[warning]  Retry (Enter) / skip (s) / abort (q): [/]").strip().lower()
                        if action == "s":
                            inputs[inp.name] = ""
                            break
                        if action == "q":
                            return None
                        continue

                    # Plain text — raw value used directly (multiline via Shift+Enter)
                    inputs[inp.name] = raw
                    break
                else:
                    inputs[inp.name] = raw
                    break

    return inputs


def _count_all_steps(workflow: WorkflowDefinition) -> int:
    """Count total steps including those inside parallel groups."""
    total = 0
    for item in workflow.steps:
        if isinstance(item, ParallelGroup):
            total += len(item.steps)
        else:
            total += 1
    return total


def _run_workflow(workflows: list[WorkflowDefinition], agents_list: list):
    """Interactive workflow execution."""
    if not workflows:
        print_error("No workflows found in workflows/. Create a .yaml file there.")
        return

    # Select workflow
    labels = [f"{w.name} — {w.description}" if w.description else w.name for w in workflows]
    idx = select_menu("Select workflow:", labels)
    if idx is None:
        return
    workflow = workflows[idx]

    # Build agent map
    agents_map = {a.name: a for a in agents_list}

    # Validate workflow
    available = {a.name: a.templates for a in agents_list}
    errors = validate_workflow(workflow, available)
    if errors:
        print_error("Workflow validation failed:")
        for err in errors:
            print_error(f"  - {err}")
        return

    # Show info
    total_steps = _count_all_steps(workflow)
    print_workflow_info(workflow.name, workflow.description, total_steps, len(workflow.inputs))

    # Pre-flight API key checks for all agents used
    step_agents = set()
    for item in workflow.steps:
        if isinstance(item, ParallelGroup):
            for s in item.steps:
                step_agents.add(s.agent)
        else:
            step_agents.add(item.agent)
    for agent_name in step_agents:
        agent = agents_map.get(agent_name)
        if agent:
            key_err = check_api_key(agent.config["model"])
            if key_err:
                print_error(key_err)
                return

    # Collect inputs
    inputs = _collect_workflow_inputs(workflow)
    if inputs is None:
        return

    # Track step numbering
    step_counter = [0]

    def on_step_start(step: StepDefinition):
        step_counter[0] += 1
        print_step_header(step.id, step.agent, step.template, step_counter[0], total_steps)

    def on_step_complete(step, result):
        print_step_result(step.id, result.status.value, len(result.output))
        # Track cost per step
        agent = agents_map.get(step.agent)
        if agent and result.status.value == "completed":
            usage = getattr(agent, "last_usage", None)
            if usage and (usage.get("prompt_tokens", 0) > 0 or usage.get("completion_tokens", 0) > 0):
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)
                model = usage.get("model", agent.config["model"])
            else:
                # Fallback: estimate tokens (~4 chars per token)
                pt = len(result.output) // 4  # rough prompt estimate
                ct = len(result.output) // 4
                model = agent.config["model"]
            step_cost = estimate_cost(model, pt, ct)
            cost_tracker.record(model, pt, ct, step_id=step.id)
            print_usage_inline(model, pt, ct, step_cost)

    def on_checkpoint(step, result) -> CheckpointAction:
        # Show the output (with parsed JSON if available)
        if result.parsed_output is not None:
            print_response(result.output, parsed=result.parsed_output)
        else:
            print_response(result.output)
        choice = prompt_checkpoint(step.id)
        if choice == "edit":
            return CheckpointAction.EDIT
        if choice == "abort":
            return CheckpointAction.ABORT
        return CheckpointAction.APPROVE

    def on_edit(step, result) -> str:
        return prompt_input("Edit output", initial=result.output)

    def on_stream_delta(step, delta):
        pass  # Streaming display is handled per-step differently in workflow mode

    # Create and run
    runner = WorkflowRunner(
        workflow, agents_map,
        on_step_start=on_step_start,
        on_step_complete=on_step_complete,
        on_checkpoint=on_checkpoint,
        on_edit=on_edit,
        on_stream_delta=on_stream_delta,
    )
    runner.set_inputs(inputs)

    try:
        ctx = runner.run()
    except Exception as e:
        print_error(f"Workflow error: {e}")
        logger.debug("Workflow error: %s", e, exc_info=True)
        return

    # Show summary
    print_workflow_summary(ctx.results, total_steps)

    # Find the last completed step's result
    last_output = None
    last_result = None
    for item in reversed(workflow.steps):
        if isinstance(item, ParallelGroup):
            for s in reversed(item.steps):
                if s.id in ctx.results and ctx.results[s.id].status == StepStatus.COMPLETED:
                    last_result = ctx.results[s.id]
                    last_output = last_result.output
                    break
        elif item.id in ctx.results and ctx.results[item.id].status == StepStatus.COMPLETED:
            last_result = ctx.results[item.id]
            last_output = last_result.output
        if last_output:
            break

    if last_output:
        session_cache.set(last_output)

        # Display final output like single-agent mode
        if last_result.parsed_output is not None:
            print_response(last_output, parsed=last_result.parsed_output)
        else:
            print_response(last_output)

        _handle_post_actions(last_output)


def _run_single_agent(agents_list: list):
    """Original single-agent interactive mode."""
    if not agents_list:
        print_error("No agents found in agents/. Create an agent folder with system.txt.")
        sys.exit(1)

    agent_labels = [
        f"{a.name} ({len(a.templates)} templates)" for a in agents_list
    ]
    idx = select_menu("Select agent:", agent_labels)
    if idx is None:
        return
    agent = agents_list[idx]

    # Show agent info panel
    print_agent_info(agent.name, agent.config, len(agent.templates))

    # Pre-flight API key check
    key_err = check_api_key(agent.config["model"])
    if key_err:
        print_error(key_err)
        return

    # Template selection
    if not agent.templates:
        print_error("  No templates (.j2) found for this agent.")
        return

    tidx = select_menu("Select template:", agent.templates)
    if tidx is None:
        return
    template_name = agent.templates[tidx]

    # Variable input
    variables = collect_variables(agent, template_name)
    if variables is None:
        return

    # Variable summary
    print_variable_summary(variables)

    # Render prompt (displayed later on-demand via "Show run details")
    rendered = agent.render_template(template_name, variables)

    # Run through output pipeline (handles schema validation + retries)
    pipeline = OutputPipeline()
    output_result = None
    t_start = time.monotonic()
    while True:
        try:
            with console.status("Calling LLM..."):
                pass  # Status spinner for initial connection
            output_result = pipeline.run(
                rendered, agent, template_name,
                on_stream=lambda delta: None,  # Streaming handled by pipeline
            )
            break
        except AuthenticationError:
            print_error("\nAuthentication failed. Check API key in .env.")
            logger.debug("AuthenticationError for model %s", agent.config["model"])
            break
        except BadRequestError as e:
            print_error(f"\nBad request: {e}")
            logger.debug("BadRequestError: %s", e, exc_info=True)
            break
        except RateLimitError as e:
            msg = str(e)
            if "quota" in msg.lower() or "billing" in msg.lower():
                print_error(f"\nQuota exceeded — check your plan and billing details at your provider.")
                print_error(f"  {msg}")
                break
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
    duration = time.monotonic() - t_start

    if output_result is None:
        return

    # Display response — only for structured/parsed outputs or errors
    if output_result.errors:
        print_validation_errors(output_result.errors, output_result.attempts)
        print_response(output_result.raw_text)
    elif output_result.parsed is not None:
        print_response(output_result.raw_text, parsed=output_result.parsed)

    if output_result.finish_reason == "length":
        print_warning(f"\n  Note: Response was truncated (max_tokens={agent.config['max_tokens']} reached).")

    # Track and display cost — compact summary line
    cost = estimate_cost(output_result.model, output_result.prompt_tokens, output_result.completion_tokens)
    cost_tracker.record(
        output_result.model,
        output_result.prompt_tokens,
        output_result.completion_tokens,
    )
    total_tokens = output_result.prompt_tokens + output_result.completion_tokens
    print_agent_work_summary(output_result.model, total_tokens, cost, duration)

    # Save result + variables for reuse
    session_cache.set(output_result.raw_text)
    session_cache.set_variables(variables)

    # Post-result actions with details for on-demand display
    run_details = {
        "rendered": rendered,
        "model": output_result.model,
        "prompt_tokens": output_result.prompt_tokens,
        "completion_tokens": output_result.completion_tokens,
        "cost": cost,
        "attempts": output_result.attempts,
    }
    _handle_post_actions(output_result.raw_text, details=run_details)

    print_rule()


def _show_session_costs():
    """Display session cost summary if any calls were made."""
    if cost_tracker.total_tokens > 0:
        print_cost_summary(cost_tracker.summary())


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("LAL_DEBUG") else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )
    print_banner()

    while True:
        try:
            # Discover agents and workflows
            agents_list = discover_agents()
            workflows = discover_workflows()

            # Top-level mode selection (skip if no workflows available)
            if workflows:
                mode_idx = select_menu("Select mode:", TOP_LEVEL_MODES)
                if mode_idx is None:
                    _show_session_costs()
                    print_info("\nGoodbye!")
                    sys.exit(0)
                mode = TOP_LEVEL_MODES[mode_idx]
            else:
                mode = "Run Single Agent"

            if mode == "Run Workflow":
                _run_workflow(workflows, agents_list)
            else:
                _run_single_agent(agents_list)

        except KeyboardInterrupt:
            _show_session_costs()
            print_info("\n\nGoodbye!")
            sys.exit(0)
        except EOFError:
            _show_session_costs()
            print_info("\n\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
