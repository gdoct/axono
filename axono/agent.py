import os
import subprocess  # nosec B404 -- intentional: bash tool requires subprocess
import sys
from collections.abc import AsyncGenerator
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from axono import config
from axono.coding import run_coding_pipeline
from axono.safety import judge_command

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _HAS_MCP = True
except ImportError:  # pragma: no cover
    MultiServerMCPClient = None  # type: ignore[misc, assignment]
    _HAS_MCP = False

SYSTEM_PROMPT = (
    "You are a helpful personal AI assistant running in a terminal. "
    "You have access to the local filesystem and can execute shell commands "
    "using the bash tool. When the user asks you to perform a system task, "
    "use the bash tool to execute the appropriate command. "
    "\n\n"
    "You also have a `code` tool for software engineering tasks. Use the "
    "`code` tool when the user asks you to: write code, implement a method "
    "or feature, edit source files, create new source files, fix bugs in "
    "code, refactor code, or any task that involves reading and modifying "
    "source code files. The `code` tool will scan the project directory, "
    "plan the changes, generate code, write it to disk, and validate the "
    "result. Pass the user's request as the `task` parameter and the "
    "project directory as `working_dir`.\n\n"
    "IMPORTANT RESPONSE STYLE RULES:\n"
    "- Be extremely concise. Give short, to-the-point replies.\n"
    "- Do NOT narrate what you are about to do or what you just did.\n"
    "- Do NOT add filler like 'What would you like to do next?' or "
    "'Let me know if you need anything else.'\n"
    "- After running a tool, only reply if there is something meaningful "
    "to report (e.g. an error, a summary of results). A simple action "
    "like changing directories needs no commentary at all.\n"
    "- Prefer one-line answers when possible.\n\n"
    "The bash tool has a built-in safety check. If a command is blocked as "
    "dangerous, tell the user what was blocked and why, then ask if they "
    "want to proceed. If the user confirms, re-run the same command with "
    "the `unsafe` flag set to true to bypass the safety check. Never set "
    "`unsafe` to true unless the user has explicitly confirmed."
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
    if cmd.startswith("cd "):
        target = cmd[3:].strip()
        if "&&" in target:
            target, remainder = target.split("&&", 1)
            target = target.strip()
            cmd = remainder.strip()
        else:
            cmd = ""
        target = os.path.expanduser(target)
        if not os.path.isabs(target):
            target = os.path.abspath(os.path.join(cwd, target))
        if os.path.isdir(target):
            _CURRENT_DIR = target
            cwd = _CURRENT_DIR
        else:
            return f"Error: Directory not found: {target}\n__CWD__:{_CURRENT_DIR}"

    try:
        if not cmd:
            return f"__CWD__:{_CURRENT_DIR}"
        result = subprocess.run(  # nosec B602 -- intentional: user-facing shell tool, guarded by LLM safety judge
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=config.COMMAND_TIMEOUT,
            cwd=cwd,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n[stderr]\n" + result.stderr) if output else result.stderr
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

    builtin_tools = [bash, code]
    mcp_tools = await _load_mcp_tools()
    all_tools = builtin_tools + mcp_tools

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
