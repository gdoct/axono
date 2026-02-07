"""Shared infrastructure for iterative pipelines.

This module provides common utilities and abstractions used by both the
shell pipeline and coding pipeline. Both pipelines follow the same
iterative model: Plan one step → Execute → Observe → Repeat.
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
# Pipeline context
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
