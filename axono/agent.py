import asyncio
import os
import subprocess  # nosec B404 -- intentional: bash tool requires subprocess
import sys
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from axono import config
from axono.coding import run_coding_pipeline
from axono.safety import judge_command
from axono.shell import run_shell_pipeline
from langchain_community.tools import DuckDuckGoSearchRun

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _HAS_MCP = True
except ImportError:  # pragma: no cover
    MultiServerMCPClient = None  # type: ignore[misc, assignment]
    _HAS_MCP = False

SYSTEM_PROMPT = (
    "You are a helpful personal AI assistant running in a terminal. "
    "You have access to the local filesystem and can execute shell commands.\n\n"
    "TOOLS:\n"
    "- `shell`: Use for task-oriented requests like 'install X', 'clone and build Y', "
    "'set up a Python environment'. It plans steps, executes them, and verifies.\n"
    "- `bash`: Use for simple direct commands when you know exactly what to run.\n"
    "- `code`: Use for editing/writing source code files.\n"
    "- `duckduckgo_search`: Use to search the web for current information.\n\n"
    "RESPONSE STYLE:\n"
    "- Be extremely concise. One-line answers when possible.\n"
    "- Do NOT narrate what you're doing or add filler.\n"
    "- After a tool runs, only reply if there's something meaningful to report.\n\n"
    "SAFETY: Tools have safety checks. If blocked, tell the user why and ask "
    "if they want to proceed. Only set `unsafe=true` after explicit confirmation."
)

_CURRENT_DIR = os.path.expanduser("~")


@tool
async def bash(command: str, unsafe: bool = False) -> str:
    """Execute a shell command and return the output.

    Commands are checked for safety before execution. If a command is judged
    dangerous it will be blocked. Set ``unsafe=True`` to bypass the safety
    check — only do this when the user has explicitly confirmed.

    Args:
        command: The shell command to execute.
        unsafe: Skip the safety check. Only use after the user confirms.
    """
    global _CURRENT_DIR
    if not unsafe:
        try:
            verdict = await judge_command(command)
            if verdict.get("dangerous", False):
                reason = verdict.get("reason", "Potentially dangerous command")
                return (
                    f"BLOCKED: Command was not executed because it was judged dangerous.\n"
                    f"Reason: {reason}\n"
                    f"Command: {command}"
                )
        except Exception as exc:
            print(f"Warning: safety check failed: {exc}", file=sys.stderr)
    cmd = command.strip()
    cwd = _CURRENT_DIR

    # Handle pure "cd <path>" command
    if cmd.startswith("cd ") and "&&" not in cmd and ";" not in cmd:
        target = cmd[3:].strip()
        target = os.path.expanduser(target)
        if not os.path.isabs(target):
            target = os.path.abspath(os.path.join(cwd, target))
        if os.path.isdir(target):
            _CURRENT_DIR = target
            return f"__CWD__:{_CURRENT_DIR}"
        else:
            return f"Error: Directory not found: {target}\n__CWD__:{_CURRENT_DIR}"

    # Check if command contains cd (we'll need to track the final directory)
    has_cd = " cd " in f" {cmd} " or cmd.startswith("cd ") or "&&cd " in cmd or "; cd " in cmd

    try:
        # Run the command, appending pwd to capture final directory if cd is involved
        if has_cd:
            exec_cmd = f"{cmd} && pwd"
        else:
            exec_cmd = cmd

        # Use asyncio.to_thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            partial(
                subprocess.run,  # nosec B602 B604 -- intentional: user-facing shell tool, guarded by LLM safety judge
                exec_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=config.COMMAND_TIMEOUT,
                cwd=cwd,
            )
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n[stderr]\n" + result.stderr) if output else result.stderr

        # If we appended pwd, extract and update the final directory
        if has_cd and result.returncode == 0 and output.strip():
            lines = output.strip().split("\n")
            potential_dir = lines[-1].strip()
            if os.path.isdir(potential_dir):
                _CURRENT_DIR = potential_dir
                # Remove the pwd output from display
                output = "\n".join(lines[:-1]) if len(lines) > 1 else ""

        if not output.strip():
            output = f"(command completed with exit code {result.returncode})"
        max_len = 8000
        if len(output) > max_len:
            output = output[:max_len] + f"\n... (truncated, {len(output)} total chars)"
        return f"{output}\n__CWD__:{_CURRENT_DIR}"
    except subprocess.TimeoutExpired:
        return (
            f"Error: Command timed out after {config.COMMAND_TIMEOUT} seconds. "
            "The command may be interactive or long-running."
        )
    except Exception as e:
        return f"Error executing command: {type(e).__name__}: {e}"


@tool
async def code(task: str, working_dir: str) -> str:
    """Generate or edit source code files in a project directory.

    Use this tool when the user asks you to write, implement, edit, fix, or
    refactor code. It scans the project, plans changes, generates code, writes
    files, and validates the result.

    Args:
        task: A description of what the user wants done (e.g. "implement the
              CalculateAnnuity method in AnnuityCalculator.cs").
        working_dir: Absolute path to the project directory to work in.
    """
    output_parts: list[str] = []
    async for event_type, data in run_coding_pipeline(task, working_dir):
        if event_type == "status":
            output_parts.append(f"[status] {data}")
        elif event_type == "result":
            output_parts.append(data)
        elif event_type == "error":
            output_parts.append(f"[error] {data}")
    return "\n".join(output_parts)


@tool
async def shell(task: str, working_dir: str, unsafe: bool = False) -> str:
    """Execute a multi-step shell task with planning and verification.

    Use this tool for complex shell tasks like "clone this repo and build it",
    "install these packages", or "set up a Python environment". The tool will:
    1. Plan the necessary commands
    2. Execute them step by step
    3. Verify the result

    For simple single commands, it will detect this and run directly.

    Args:
        task: What the user wants to do (e.g. "install numpy and pandas").
        working_dir: Absolute path to run commands in.
        unsafe: Skip safety checks. Only use after user confirms.
    """
    global _CURRENT_DIR
    new_cwd = None
    steps_run = 0
    last_status = ""
    last_output = ""
    result_msg = ""
    errors: list[str] = []

    async for event_type, data in run_shell_pipeline(task, working_dir, unsafe):
        if event_type == "status":
            last_status = str(data)
            steps_run += 1
        elif event_type == "output":
            if data:
                last_output = str(data)
        elif event_type == "result":
            if data:
                result_msg = str(data)
        elif event_type == "error":
            errors.append(str(data))
        elif event_type == "cwd":
            new_cwd = str(data)

    # Build compact output
    output_parts: list[str] = []
    if steps_run > 1:
        output_parts.append(f"Ran {steps_run} commands")
    if last_status:
        output_parts.append(last_status)
    if last_output and len(last_output) < 200:
        output_parts.append(last_output)
    if errors:
        output_parts.append(f"Errors: {'; '.join(errors[-2:])}")  # Last 2 errors
    if result_msg:
        output_parts.append(result_msg)

    # Update global cwd if changed
    if new_cwd and new_cwd != working_dir:
        _CURRENT_DIR = new_cwd
        output_parts.append(f"__CWD__:{new_cwd}")

    return "\n".join(output_parts) if output_parts else "(done)"


async def _load_mcp_tools() -> list:
    """Load tools from configured MCP servers.

    Returns an empty list if MCP support is not installed, no servers are
    configured, or a connection fails.
    """
    if not _HAS_MCP or MultiServerMCPClient is None:
        return []

    server_config = config.load_mcp_config()
    if not server_config:
        return []

    try:
        client = MultiServerMCPClient(server_config)
        return await client.get_tools()
    except Exception as exc:
        print(f"Warning: Failed to load MCP tools: {exc}", file=sys.stderr)
        return []


async def build_agent(on_status=None):
    """Build and return the LangChain agent graph.

    Args:
        on_status: Optional callback that receives status messages (str).
    """
    llm = init_chat_model(
        model=config.LLM_MODEL_NAME,
        model_provider=config.LLM_MODEL_PROVIDER,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )

    builtin_tools = [bash, shell, code]
    mcp_tools = await _load_mcp_tools()
    search_tool = DuckDuckGoSearchRun()
    all_tools = builtin_tools + mcp_tools + [search_tool]

    system_prompt = SYSTEM_PROMPT
    if mcp_tools:
        names = [t.name for t in mcp_tools]
        system_prompt += (
            "\n\nYou also have access to the following tools provided by "
            "external MCP servers:\n"
            + "\n".join(f"  - {name}" for name in names)
            + "\nUse these tools when the user's request matches their capabilities."
        )
        if on_status:
            on_status(f"Loaded {len(mcp_tools)} MCP tool(s): {', '.join(names)}")

    graph = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
    )
    return graph


async def run_agent(
    graph, messages
) -> AsyncGenerator[tuple[str, Any], None]:
    """Run the agent and yield UI events after each node completes.

    Yields tuples of (event_type, data):
      - ("assistant", str)   - Full assistant message text
      - ("tool_result", str) - Tool output text
      - ("messages", list)   - Updated full message history
      - ("error", str)       - An error occurred
    """
    inputs = {"messages": list(messages)}
    collected_messages = list(messages)

    try:
        async for update in graph.astream(inputs, stream_mode="updates"):
            for node_name, state_update in update.items():
                new_msgs = state_update.get("messages", [])
                for msg in new_msgs:
                    collected_messages.append(msg)

                if node_name == "model":
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                yield ("assistant", msg.content)
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield (
                                        "tool_call",
                                        f"{tc['name']}({tc['args']})",
                                    )
                elif node_name == "tools":
                    for msg in new_msgs:
                        if isinstance(msg, ToolMessage):
                            yield ("tool_result", msg.content)

        yield ("messages", collected_messages)

    except Exception as e:
        yield ("error", f"{type(e).__name__}: {e}")
