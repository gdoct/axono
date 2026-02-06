"""Iterative shell pipeline: Plan one step -> Execute -> Observe -> Repeat.

This module implements an adaptive pipeline for shell tasks. Instead of
planning all steps upfront, it plans one step at a time based on the
current state (directory contents, previous outputs, etc.).
"""

import asyncio
import os
import subprocess  # nosec B404 -- intentional: shell tool requires subprocess
import sys
from dataclasses import dataclass
from functools import partial
from langchain_core.messages import HumanMessage, SystemMessage

from axono import config
from axono.pipeline import get_llm, parse_json, coerce_response_text, truncate
from axono.safety import judge_command


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of executing a single step."""

    command: str
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    blocked: bool = False
    block_reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_project_type(cwd: str) -> str | None:
    """Detect project type from build files."""
    indicators = {
        "Cargo.toml": "Rust (use: cargo build)",
        "package.json": "Node.js (use: npm install && npm run build)",
        "requirements.txt": "Python (use: pip install -r requirements.txt)",
        "pyproject.toml": "Python (use: pip install . or pip install -e .)",
        "setup.py": "Python (use: pip install . or python setup.py install)",
        "Makefile": "Make (use: make)",
        "CMakeLists.txt": "CMake (use: mkdir build && cd build && cmake .. && make)",
        "go.mod": "Go (use: go build)",
        "pom.xml": "Java/Maven (use: mvn package)",
        "build.gradle": "Java/Gradle (use: gradle build)",
        "build.gradle.kts": "Kotlin/Gradle (use: gradle build)",
        "Gemfile": "Ruby (use: bundle install)",
        "composer.json": "PHP (use: composer install)",
        "mix.exs": "Elixir (use: mix deps.get && mix compile)",
        "stack.yaml": "Haskell (use: stack build)",
        "dune-project": "OCaml (use: dune build)",
        "Package.swift": "Swift (use: swift build)",
        "*.csproj": ".NET (use: dotnet build)",
        "*.sln": ".NET (use: dotnet build)",
    }

    try:
        entries = set(os.listdir(cwd))
        for indicator, project_type in indicators.items():
            if indicator.startswith("*"):
                # Glob pattern
                ext = indicator[1:]
                if any(f.endswith(ext) for f in entries):
                    return project_type
            elif indicator in entries:
                return project_type
    except OSError:
        pass
    return None


def _get_dir_context(cwd: str, max_files: int = 20) -> tuple[str, str | None]:
    """Get a brief listing of the current directory and detect project type."""
    project_type = _detect_project_type(cwd)

    try:
        entries = os.listdir(cwd)
        # Sort: directories first, then files
        dirs = sorted(d for d in entries if os.path.isdir(os.path.join(cwd, d)))
        files = sorted(f for f in entries if os.path.isfile(os.path.join(cwd, f)))
        all_entries = dirs + files

        if len(all_entries) > max_files:
            shown = all_entries[:max_files]
            listing = ", ".join(shown) + f" (+{len(all_entries) - max_files} more)"
        else:
            listing = ", ".join(all_entries) if all_entries else "(empty)"

        return listing, project_type
    except OSError:
        return "(unreadable)", project_type


async def _run_command(cmd: str, cwd: str, unsafe: bool = False) -> StepResult:
    """Execute a single shell command."""
    # Safety check
    if not unsafe:
        try:
            verdict = await judge_command(cmd)
            if verdict.get("dangerous", False):
                return StepResult(
                    command=cmd,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    success=False,
                    blocked=True,
                    block_reason=verdict.get("reason", "Potentially dangerous"),
                )
        except Exception as exc:
            print(f"Warning: safety check failed: {exc}", file=sys.stderr)

    try:
        # Use asyncio.to_thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            partial(
                subprocess.run,  # nosec B602
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=config.COMMAND_TIMEOUT,
                cwd=cwd,
            )
        )
        return StepResult(
            command=cmd,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            success=result.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        return StepResult(
            command=cmd,
            stdout="",
            stderr=f"Timed out after {config.COMMAND_TIMEOUT}s",
            exit_code=-1,
            success=False,
        )
    except Exception as e:
        return StepResult(
            command=cmd,
            stdout="",
            stderr=str(e),
            exit_code=-1,
            success=False,
        )


# ---------------------------------------------------------------------------
# Iterative Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """\
You are a shell command agent. Execute ONE command at a time, observe, then decide next.

Respond with JSON only:
{"done": false, "command": "cmd", "reason": "3-5 words"}
OR
{"done": true, "summary": "brief result"}

CRITICAL RULES:
- ONE command at a time. NEVER use && or ;
- If "Project type" is provided, USE THAT BUILD SYSTEM. Do not guess.
- If no project type shown, investigate first (ls, cat README, etc.)
- Keep reasons SHORT
- Respond ONLY with JSON
"""


async def _plan_next_step(
    task: str,
    cwd: str,
    history: list[StepResult],
) -> dict:
    """Plan the next step based on current state."""
    llm = get_llm()

    # Build context
    dir_contents, project_type = _get_dir_context(cwd)

    history_text = ""
    if history:
        lines = []
        for r in history[-5:]:  # Last 5 commands only
            status = "✓" if r.success else "✗"
            out = truncate(r.stdout, 200) if r.stdout else ""
            err = truncate(r.stderr, 100) if r.stderr and not r.success else ""
            line = f"{status} $ {r.command}"
            if out:
                line += f"\n   {out}"
            if err:
                line += f"\n   [err] {err}"
            lines.append(line)
        history_text = "\n".join(lines)

    user_prompt = f"Goal: {task}\nCWD: {cwd}\nContents: {dir_contents}"
    if project_type:
        user_prompt += f"\nProject type: {project_type}"
    if history_text:
        user_prompt += f"\nHistory:\n{history_text}"

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        return {"done": True, "summary": "Could not parse response"}

    return data


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_shell_pipeline(task: str, working_dir: str, unsafe: bool = False):
    """Run the iterative shell pipeline.

    Yields status tuples:
      ("status", str)   - progress updates
      ("output", str)   - command output to show
      ("result", str)   - final result
      ("error", str)    - if something went wrong
      ("cwd", str)      - updated working directory
    """
    cwd = working_dir
    history: list[StepResult] = []
    max_steps = 10

    for step_num in range(max_steps):
        # Plan next step
        try:
            plan = await _plan_next_step(task, cwd, history)
        except Exception as e:
            yield ("error", f"Planning failed: {e}")
            break

        # Check if done
        if plan.get("done"):
            summary = plan.get("summary", "Done")
            yield ("result", summary)
            break

        command = plan.get("command", "").strip()
        if not command:
            yield ("error", "No command provided")
            break

        reason = plan.get("reason", "")
        if reason:
            yield ("status", f"$ {command}  # {reason}")
        else:
            yield ("status", f"$ {command}")

        # Handle cd specially to track directory changes
        if command.startswith("cd "):
            target = command[3:].strip()
            target = os.path.expanduser(target)
            if not os.path.isabs(target):
                target = os.path.abspath(os.path.join(cwd, target))
            if os.path.isdir(target):
                cwd = target
                history.append(StepResult(
                    command=command,
                    stdout=f"Changed to {cwd}",
                    stderr="",
                    exit_code=0,
                    success=True,
                ))
                yield ("output", f"→ {cwd}")
            else:
                yield ("error", f"Directory not found: {target}")
                history.append(StepResult(
                    command=command,
                    stdout="",
                    stderr=f"Directory not found: {target}",
                    exit_code=1,
                    success=False,
                ))
            continue

        # Execute the command
        result = await _run_command(command, cwd, unsafe)
        history.append(result)

        if result.blocked:
            yield ("error", f"BLOCKED: {result.block_reason}")
            # Don't break - let planner try something else
            continue

        # Show output
        if result.success:
            out = truncate(result.stdout, 300)
            if out:
                yield ("output", out)
            else:
                yield ("output", "✓")
        else:
            err = truncate(result.stderr or f"Exit code {result.exit_code}", 300)
            yield ("error", err)
            # Don't break - let planner adapt

    else:
        # Reached max steps
        yield ("result", f"Stopped after {max_steps} steps")

    yield ("cwd", cwd)
