"""Shared infrastructure for iterative pipelines.

This module provides common utilities and abstractions used by both the
shell pipeline and coding pipeline. Both pipelines follow the same
staged model:

1. INVESTIGATE → Gather context about the project/task
2. PLAN → Create a complete plan for achieving the goal
3. VALIDATE_PLAN → Check if plan will achieve the goal (max 5 iterations)
4. EXECUTE → Run the plan (generate+write for code, commands for shell)
5. VALIDATE → Verify the execution achieved the goal
"""

import asyncio
import json
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI

from axono import config

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_llm(model_type: str = "instruction"):  # pragma: no cover
    """Build an LLM instance using centralised config.

    Args:
        model_type: The type of model to use ("instruction" or "reasoning").
                    Defaults to "instruction".
    """
    return init_chat_model(
        model=config.get_model_name(model_type) or "default",
        model_provider=config.LLM_MODEL_PROVIDER,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )


# ---------------------------------------------------------------------------
# Thinking queue (side-channel for reasoning model thinking tokens)
# ---------------------------------------------------------------------------

_thinking_queue: asyncio.Queue | None = None


def set_thinking_queue(queue: asyncio.Queue | None) -> None:
    """Set the global thinking queue. Pass None to disable."""
    global _thinking_queue
    _thinking_queue = queue


def _emit_thinking(text: str | None) -> None:
    """Push a thinking update to the queue if one is set.

    Args:
        text: Accumulated thinking text, or None to signal end of thinking.
    """
    if _thinking_queue is not None:
        _thinking_queue.put_nowait(text)


def _get_openai_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from config."""
    return AsyncOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )


async def stream_response(
    messages: list,
    model_type: str = "instruction",
) -> str:
    """Stream an LLM response, emitting thinking tokens via the global queue.

    Uses the raw OpenAI client (not LangChain) so that ``reasoning_content``
    from reasoning models is captured.  Falls back gracefully for models that
    don't produce reasoning tokens.

    Args:
        messages: LangChain message objects (SystemMessage / HumanMessage).
        model_type: Model type key ("instruction" or "reasoning").

    Returns:
        The full content text of the response (thinking stripped).
    """
    client = _get_openai_client()
    model_name = config.get_model_name(model_type) or "default"

    # Convert LangChain messages to OpenAI format
    openai_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": str(msg.content)})
        elif isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": str(msg.content)})
        else:
            openai_messages.append({"role": "assistant", "content": str(msg.content)})

    stream = await client.chat.completions.create(
        model=model_name,
        messages=openai_messages,
        stream=True,
    )

    content_parts: list[str] = []
    thinking_parts: list[str] = []
    in_thinking = False

    async for chunk in stream:
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning_content", None)
        c = delta.content

        if rc:
            if not in_thinking:
                in_thinking = True
            thinking_parts.append(rc)
            _emit_thinking("".join(thinking_parts))
        elif c:
            if in_thinking:
                in_thinking = False
                _emit_thinking(None)  # Signal end of thinking
            content_parts.append(c)

    # Ensure thinking is ended even if no content follows
    if in_thinking:
        _emit_thinking(None)

    return "".join(content_parts)


def coerce_response_text(content) -> str:
    """Convert LLM response content to a string."""
    if content is None:
        return ""
    if isinstance(content, list):
        return json.dumps(content)
    return str(content)


def parse_json(raw: str) -> dict | list | None:
    """Parse JSON from a response, tolerating markdown fences and preamble text.

    Tries three strategies in order:
    1. Direct parse (response is pure JSON)
    2. Strip markdown fences and parse
    3. Find JSON within free-text (for chain-of-thought responses)
    """
    text = raw.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences
    stripped = text
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find JSON within free-text
    # First try fenced JSON blocks
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Scan for first { or [ that leads to valid JSON
    for i, ch in enumerate(text):
        if ch in "{[":
            # Try parsing from this position
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                pass
            # Try finding the matching closing bracket
            closing = "}" if ch == "{" else "]"
            for j in range(len(text) - 1, i, -1):
                if text[j] == closing:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        continue

    return None


def truncate(text: str, max_len: int = 500) -> str:
    """Truncate text for display."""
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Pipeline context (legacy - kept for compatibility)
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Result of executing a single action."""

    action: str
    success: bool
    output: str = ""
    error: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Shared context for iterative pipelines."""

    task: str
    cwd: str
    history: list[ActionResult] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ActionResult) -> None:
        """Add an action result to history."""
        self.history.append(result)

    def last_n_results(self, n: int = 5) -> list[ActionResult]:
        """Get the last N results from history."""
        return self.history[-n:] if self.history else []


# ---------------------------------------------------------------------------
# Unified pipeline data structures
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step in a plan with user-friendly description."""

    description: str  # Human-readable description shown to user
    action: dict = field(default_factory=dict)  # Action details (command, patch, etc.)


@dataclass
class PlanValidation:
    """Result of plan validation."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class StepExecution:
    """Result of executing a single step."""

    step: PlanStep
    success: bool
    output: str = ""
    error: str = ""


@dataclass
class ExecutionResult:
    """Result of executing all steps in a plan."""

    success: bool
    step_results: list[StepExecution] = field(default_factory=list)
    summary: str = ""


@dataclass
class FinalValidation:
    """Result of final validation after execution."""

    ok: bool
    issues: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class Investigation:
    """Generic investigation result for any pipeline."""

    cwd: str
    dir_listing: list[str] | str  # list for coding, str for shell display
    files: list[Any] = field(default_factory=list)  # FileContent for coding
    project_type: str | None = None  # for shell
    summary: str = ""
    skipped: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """Unified plan structure for any pipeline."""

    summary: str
    steps: list[PlanStep] = field(default_factory=list)
    files_to_read: list[str] = field(default_factory=list)  # coding uses this
    patches: list[dict] = field(default_factory=list)  # coding planned edits
    raw: str = ""


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    max_plan_iterations: int = 5
    max_action_steps: int = 15  # for iterative mode
    iterative: bool = False  # True = coding style, False = shell style


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

PLAN_VALIDATOR_SYSTEM = """\
You are a plan validation agent. Given:
- The original user request
- The proposed plan with steps

Evaluate if executing this plan will achieve the user's goal.

Consider:
1. Does the plan address all requirements in the request?
2. Are the steps in the correct order?
3. Are there any missing steps?
4. Are there any unnecessary steps?
5. Will the plan actually achieve the desired outcome?

You may reason briefly, but your response MUST contain a JSON object:
{
    "valid": true/false,
    "issues": ["list of problems found"],
    "suggestions": ["list of improvements"],
    "summary": "brief assessment"
}
"""


async def validate_plan(
    task: str,
    plan_summary: str,
    steps: list[PlanStep],
    context: str = "",
) -> PlanValidation:
    """Validate that a plan will achieve the task goal.

    Args:
        task: The original user request.
        plan_summary: Summary of the plan.
        steps: List of planned steps.
        context: Optional additional context (e.g., project type, files).

    Returns:
        PlanValidation with validation result.
    """
    steps_text = "\n".join(f"{i+1}. {step.description}" for i, step in enumerate(steps))

    user_prompt = f"## User Request\n{task}\n\n## Plan Summary\n{plan_summary}\n\n## Steps\n{steps_text}"
    if context:
        user_prompt += f"\n\n## Context\n{context}"

    messages = [
        SystemMessage(content=PLAN_VALIDATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    raw = (await stream_response(messages, "reasoning")).strip()

    data = parse_json(raw)
    if data is None:
        return PlanValidation(
            valid=False,
            issues=["Could not parse plan validation response"],
            summary="Plan validation response was not valid JSON",
        )

    return PlanValidation(
        valid=data.get("valid", True),
        issues=data.get("issues", []),
        suggestions=data.get("suggestions", []),
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Generic LLM step planner
# ---------------------------------------------------------------------------


async def plan_next_action(
    system_prompt: str,
    user_prompt: str,
    llm=None,
) -> dict:
    """Plan the next action using an LLM.

    Args:
        system_prompt: The system prompt describing available actions.
        user_prompt: The user prompt with current context.
        llm: Optional LLM instance. If not provided, creates one.

    Returns:
        Parsed JSON dict from LLM, or {"done": True, "summary": "..."} on parse failure.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    raw = (await stream_response(messages)).strip()

    data = parse_json(raw)
    if data is None:
        return {"done": True, "summary": "Could not parse response"}

    return data


# ---------------------------------------------------------------------------
# Generic LLM stage functions
# ---------------------------------------------------------------------------


async def llm_plan(
    task: str,
    context: str,
    system_prompt: str,
) -> Plan:
    """Create a plan using an LLM.

    Args:
        task: The user's goal/task.
        context: Context string (directory listing, project type, etc.).
        system_prompt: The system prompt for the planner.

    Returns:
        Plan with steps and summary.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Goal: {task}\n\n{context}"),
    ]

    raw = (await stream_response(messages, "reasoning")).strip()

    data = parse_json(raw)
    if data is None:
        # Fallback: treat entire response as summary with no steps
        return Plan(summary=raw, raw=raw)

    steps = []
    for step_data in data.get("steps", []):
        if isinstance(step_data, dict) and "description" in step_data:
            steps.append(
                PlanStep(
                    description=step_data.get("description", ""),
                    action=step_data,
                )
            )

    return Plan(
        summary=data.get("summary", ""),
        steps=steps,
        files_to_read=data.get("files_to_read", []),
        patches=data.get("patches", []),
        raw=raw,
    )


FINAL_VALIDATOR_SYSTEM = """\
You are a task validation agent. Given:
- The original user request
- The plan that was followed
- The execution results

Evaluate if the task was completed successfully.

You may reason briefly, but your response MUST contain a JSON object:
{
    "ok": true/false,
    "issues": ["list of issues if any"],
    "summary": "brief validation summary"
}
"""


async def llm_validate(
    task: str,
    plan_summary: str,
    execution_summary: str,
    system_prompt: str | None = None,
) -> FinalValidation:
    """Validate execution results using an LLM.

    Args:
        task: The original user request.
        plan_summary: Summary of the plan that was executed.
        execution_summary: Summary of what was executed and results.
        system_prompt: Optional custom system prompt. Uses default if not provided.

    Returns:
        FinalValidation with ok status, issues, and summary.
    """
    user_prompt = (
        f"## Original Request\n{task}\n\n"
        f"## Plan\n{plan_summary}\n\n"
        f"## Execution Results\n{execution_summary}"
    )

    messages = [
        SystemMessage(content=system_prompt or FINAL_VALIDATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    raw = (await stream_response(messages, "reasoning")).strip()

    data = parse_json(raw)
    if data is None:
        return FinalValidation(
            ok=False,
            issues=["Could not parse validation response"],
            summary="Validation response was not valid JSON",
        )

    return FinalValidation(
        ok=data.get("ok", True),
        issues=data.get("issues", []),
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Generic pipeline orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    task: str,
    working_dir: str,
    config: PipelineConfig,
    investigate_fn: Callable[[str], Investigation],
    plan_fn: Callable[[str, Investigation, list[str] | None], Awaitable[Plan]],
    execute_fn: Callable[[Plan, str], Awaitable[tuple[ExecutionResult, str]]],
    validate_fn: (
        Callable[[str, Plan, ExecutionResult], Awaitable[FinalValidation]] | None
    ) = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Generic pipeline orchestrator for shell and coding pipelines.

    Stages:
    1. Investigate - gather directory context
    2. Plan - create complete plan with descriptions (with validation loop)
    3. Execute - run planned steps
    4. Validate - verify execution achieved goal

    Args:
        task: The user's goal/task.
        working_dir: The working directory.
        config: Pipeline configuration.
        investigate_fn: Function to investigate the project.
        plan_fn: Function to create a plan. Takes (task, investigation, previous_issues).
        execute_fn: Function to execute the plan. Returns (ExecutionResult, final_cwd).
        validate_fn: Optional function to validate execution.

    Yields:
        Tuples of (event_type, message) where event_type is one of:
        - "status": Progress update
        - "output": Command/action output to show
        - "result": Final result summary
        - "error": Error message
        - "cwd": Updated working directory
    """
    # Stage 1: Investigation
    yield ("status", "Investigating project...")
    investigation = investigate_fn(working_dir)
    if investigation.project_type:
        yield ("status", f"Detected: {investigation.project_type}")

    # Stage 2 & 3: Plan with validation loop
    plan: Plan | None = None
    plan_issues: list[str] = []

    for iteration in range(config.max_plan_iterations):
        # Stage 2: Planning
        try:
            if iteration == 0:
                yield ("status", "Creating plan...")
            else:
                yield ("status", f"Revising plan (attempt {iteration + 1})...")

            plan = await plan_fn(
                task,
                investigation,
                plan_issues if plan_issues else None,
            )

            if not plan.steps:
                yield ("error", "No steps in plan")
                yield ("cwd", working_dir)
                return

            yield ("status", f"Plan: {plan.summary}")

        except Exception as e:
            yield ("error", f"Planning failed: {e}")
            yield ("cwd", working_dir)
            return

        # Stage 3: Validate Plan
        try:
            yield ("status", "Validating plan...")

            # Build context for validation
            context = f"Directory: {investigation.cwd}"
            if investigation.project_type:
                context += f"\nProject type: {investigation.project_type}"

            validation = await validate_plan(
                task=task,
                plan_summary=plan.summary,
                steps=plan.steps,
                context=context,
            )

            if validation.valid:
                yield ("status", "Plan validated ✓")
                break
            else:
                plan_issues = validation.issues
                issues_str = "; ".join(validation.issues[:2])
                yield ("status", f"Plan issues: {issues_str}")
                # Continue to next iteration to re-plan

        except Exception as e:
            # If validation fails, proceed anyway
            yield ("status", f"Plan validation skipped: {e}")
            break
    else:
        # Max iterations reached, proceed with last plan
        yield (
            "status",
            f"Proceeding after {config.max_plan_iterations} plan attempts",
        )

    # Safety check - plan should be set at this point
    if plan is None:  # pragma: no cover
        yield ("error", "No plan generated")
        yield ("cwd", working_dir)
        return

    # Stage 4: Execution
    yield ("status", "Executing plan...")
    cwd = working_dir

    try:
        for step in plan.steps:
            yield ("status", step.description)

        execution, cwd = await execute_fn(plan, working_dir)

        # Report step results
        for sr in execution.step_results:
            if sr.success:
                if sr.output and sr.output != "✓":
                    yield ("output", sr.output)
            else:
                yield ("error", sr.error)

    except Exception as e:
        yield ("error", f"Execution failed: {e}")
        yield ("cwd", cwd)
        return

    # Stage 5: Final Validation
    if validate_fn:
        try:
            yield ("status", "Validating results...")
            final_validation = await validate_fn(task, plan, execution)

            if final_validation.ok:
                yield (
                    "result",
                    final_validation.summary or "Task completed successfully",
                )
            else:
                issues = "; ".join(final_validation.issues[:2])
                yield (
                    "result",
                    (
                        f"{final_validation.summary}. Issues: {issues}"
                        if issues
                        else final_validation.summary
                    ),
                )

        except Exception as e:
            # If validation fails, still report completion
            yield ("result", f"Execution complete (validation error: {e})")
    else:
        yield ("result", "Task completed")

    yield ("cwd", cwd)
