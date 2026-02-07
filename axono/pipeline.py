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

import json
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

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
        model=config.get_model_name(model_type),
        model_provider=config.LLM_MODEL_PROVIDER,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )


def coerce_response_text(content) -> str:
    """Convert LLM response content to a string."""
    if content is None:
        return ""
    if isinstance(content, list):
        return json.dumps(content)
    return str(content)


def parse_json(raw: str) -> dict | None:
    """Parse JSON, tolerating markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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

Respond ONLY with JSON (no markdown fences, no commentary):
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
    llm = get_llm("reasoning")

    steps_text = "\n".join(f"{i+1}. {step.description}" for i, step in enumerate(steps))

    user_prompt = f"## User Request\n{task}\n\n## Plan Summary\n{plan_summary}\n\n## Steps\n{steps_text}"
    if context:
        user_prompt += f"\n\n## Context\n{context}"

    messages = [
        SystemMessage(content=PLAN_VALIDATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        # If we can't parse, assume valid (don't block on parse failures)
        return PlanValidation(valid=True, summary=raw)

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
    if llm is None:
        llm = get_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        return {"done": True, "summary": "Could not parse response"}

    return data
