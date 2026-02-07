"""Intent analyzer for classifying user messages.

This module provides the first node in the processing pipeline. It classifies
user input into:
- chat: Direct conversation (greetings, questions, explanations)
- task: Actionable requests that need tool execution

For task intents, it also produces a high-level task list that will be
executed in order.
"""

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from axono.pipeline import coerce_response_text, get_llm, parse_json

INTENT_ANALYZER_SYSTEM = """\
You are an intent classifier for an AI coding assistant. Analyze the user's message and classify it.

INTENT TYPES:
- "chat": Greetings, questions about concepts, explanations, chitchat, asking for information
- "task": Actionable requests like create, build, install, fix, implement, edit, run, deploy, etc.

For "task" intents, create a HIGH-LEVEL task list. Each task should be a logical unit of work, not a shell command.

EXAMPLES:

User: "Hello, how are you?"
→ {"type": "chat", "task_list": [], "reasoning": "Greeting"}

User: "What is a Python decorator?"
→ {"type": "chat", "task_list": [], "reasoning": "Asking for explanation"}

User: "Create a new Python project with tests"
→ {"type": "task", "task_list": ["Create project directory structure", "Initialize git repository", "Create pyproject.toml with dependencies", "Create initial test file", "Run tests to verify setup"], "reasoning": "Request to create something"}

User: "Install numpy and pandas"
→ {"type": "task", "task_list": ["Install numpy and pandas packages"], "reasoning": "Installation request"}

User: "Clone https://github.com/foo/bar and build it"
→ {"type": "task", "task_list": ["Clone the repository", "Identify build system", "Install dependencies", "Build the project"], "reasoning": "Multi-step build request"}

User: "Fix the bug in auth.py where login fails"
→ {"type": "task", "task_list": ["Investigate auth.py to understand the login flow", "Identify the bug causing login failures", "Fix the bug", "Verify the fix works"], "reasoning": "Bug fix request"}

Respond ONLY with JSON (no markdown fences, no commentary):
{"type": "chat"|"task", "task_list": [...], "reasoning": "..."}
"""


@dataclass
class Intent:
    """Result of intent analysis."""

    type: Literal["chat", "task"]
    task_list: list[str] = field(default_factory=list)
    reasoning: str = ""


async def analyze_intent(message: str, cwd: str) -> Intent:
    """Analyze user message to determine intent.

    Args:
        message: The user's message text.
        cwd: Current working directory (provides context).

    Returns:
        Intent with type classification and optional task list.
    """
    llm = get_llm("instruction")

    user_prompt = f"Current directory: {cwd}\n\nUser message: {message}"

    messages = [
        SystemMessage(content=INTENT_ANALYZER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None or not isinstance(data, dict):
        # Default to chat if we can't parse or got non-dict (e.g., list)
        return Intent(type="chat", reasoning="Could not parse intent response")

    intent_type = data.get("type", "chat")
    if intent_type not in ("chat", "task"):
        intent_type = "chat"

    task_list = data.get("task_list", [])
    if not isinstance(task_list, list):
        task_list = []
    # Ensure all items are strings
    task_list = [str(t) for t in task_list if t]

    return Intent(
        type=intent_type,
        task_list=task_list,
        reasoning=data.get("reasoning", ""),
    )
