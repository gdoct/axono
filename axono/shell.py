"""Iterative shell pipeline with staged architecture.

This module implements a shell pipeline that follows the same stages as the
coding pipeline:

1. INVESTIGATE → Gather directory context and detect project type
2. PLAN → Create a complete plan of shell commands with descriptions
3. VALIDATE_PLAN → Check if plan will achieve the goal (max 5 iterations)
4. EXECUTE → Run the planned commands
5. VALIDATE → Verify the execution achieved the goal
"""

import asyncio
import os
import subprocess  # nosec B404 -- intentional: shell tool requires subprocess
import sys
from dataclasses import dataclass, field
from functools import partial

from langchain_core.messages import HumanMessage, SystemMessage

from axono import config
from axono.pipeline import (
    ExecutionResult,
    FinalValidation,
    PlanStep,
    PlanValidation,
    StepExecution,
    coerce_response_text,
    get_llm,
    parse_json,
    truncate,
    validate_plan,
)
from axono.safety import judge_command

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of executing a single step (legacy, kept for compatibility)."""

    command: str
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    blocked: bool = False
    block_reason: str = ""


@dataclass
class ShellInvestigation:
    """Result of the investigation stage."""

    cwd: str
    dir_listing: str
    project_type: str | None = None
    summary: str = ""


@dataclass
class ShellPlan:
    """A complete plan for a shell task."""

    summary: str
    steps: list[PlanStep] = field(default_factory=list)
    raw: str = ""


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
                subprocess.run,  # nosec B602 B604 -- intentional shell execution for coding assistant
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
# Stage 1: Investigation
# ---------------------------------------------------------------------------


def investigate_shell(working_dir: str) -> ShellInvestigation:
    """Gather directory context and detect project type."""
    dir_listing, project_type = _get_dir_context(working_dir)

    summary_parts = [f"Directory: {working_dir}"]
    if project_type:
        summary_parts.append(f"Project type: {project_type}")
    summary_parts.append(f"Contents: {dir_listing}")

    return ShellInvestigation(
        cwd=working_dir,
        dir_listing=dir_listing,
        project_type=project_type,
        summary="; ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# Stage 2: Planning
# ---------------------------------------------------------------------------

SHELL_PLANNER_SYSTEM = """\
You are a shell planning agent. Create a complete plan to achieve the user's goal.

Respond ONLY with JSON (no markdown fences, no commentary):
{
    "summary": "Brief description of what will be done",
    "steps": [
        {
            "description": "Human-readable step description (shown to user)",
            "command": "the actual shell command"
        }
    ]
}

RULES:
- Each step should have ONE command (no && or ;)
- "description" should be short and user-friendly (e.g., "Installing dependencies")
- If "Project type" is provided, USE THAT BUILD SYSTEM. Do not guess.
- If no project type shown, start by investigating (ls, cat README, etc.)
- Think through all steps needed to complete the task
- Include verification steps if appropriate (e.g., check if build succeeded)
"""


async def plan_shell(
    task: str,
    investigation: ShellInvestigation,
    previous_issues: list[str] | None = None,
) -> ShellPlan:
    """Create a complete plan of shell commands with descriptions.

    Args:
        task: The goal to achieve.
        investigation: Directory context and project type.
        previous_issues: Issues from a previous plan validation attempt.
                        If provided, the planner will try a different approach.
    """
    llm = get_llm("reasoning")

    user_prompt = (
        f"Goal: {task}\nCWD: {investigation.cwd}\nContents: {investigation.dir_listing}"
    )
    if investigation.project_type:
        user_prompt += f"\nProject type: {investigation.project_type}"

    if previous_issues:
        user_prompt += "\n\n## Previous Plan Issues (avoid these problems):\n"
        for issue in previous_issues:
            user_prompt += f"- {issue}\n"
        user_prompt += "\nCreate a DIFFERENT plan that addresses these issues."

    messages = [
        SystemMessage(content=SHELL_PLANNER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        # Fallback: treat entire response as summary with no steps
        return ShellPlan(summary=raw, steps=[], raw=raw)

    steps = []
    for step_data in data.get("steps", []):
        if isinstance(step_data, dict) and "description" in step_data:
            steps.append(
                PlanStep(
                    description=step_data.get("description", ""),
                    action={"command": step_data.get("command", "")},
                )
            )

    return ShellPlan(
        summary=data.get("summary", ""),
        steps=steps,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Stage 3: Plan Validation (uses shared validate_plan from pipeline.py)
# ---------------------------------------------------------------------------


async def validate_shell_plan(
    task: str,
    plan: ShellPlan,
    investigation: ShellInvestigation,
) -> PlanValidation:
    """Validate that a shell plan will achieve the task goal."""
    context = f"Directory: {investigation.cwd}"
    if investigation.project_type:
        context += f"\nProject type: {investigation.project_type}"
    context += f"\nContents: {investigation.dir_listing}"

    return await validate_plan(
        task=task,
        plan_summary=plan.summary,
        steps=plan.steps,
        context=context,
    )


# ---------------------------------------------------------------------------
# Stage 4: Execution
# ---------------------------------------------------------------------------


async def execute_shell(
    plan: ShellPlan,
    working_dir: str,
    unsafe: bool = False,
) -> tuple[ExecutionResult, str]:
    """Execute the planned commands.

    Returns:
        Tuple of (ExecutionResult, final_cwd)
    """
    cwd = working_dir
    step_results: list[StepExecution] = []
    all_success = True

    for step in plan.steps:
        command = step.action.get("command", "")
        if not command:
            continue

        # Handle cd specially to track directory changes
        if command.startswith("cd "):
            target = command[3:].strip()
            target = os.path.expanduser(target)
            if not os.path.isabs(target):
                target = os.path.abspath(os.path.join(cwd, target))
            if os.path.isdir(target):
                cwd = target
                step_results.append(
                    StepExecution(
                        step=step,
                        success=True,
                        output=f"Changed to {cwd}",
                    )
                )
            else:
                all_success = False
                step_results.append(
                    StepExecution(
                        step=step,
                        success=False,
                        error=f"Directory not found: {target}",
                    )
                )
            continue

        # Execute the command
        result = await _run_command(command, cwd, unsafe)

        if result.blocked:
            all_success = False
            step_results.append(
                StepExecution(
                    step=step,
                    success=False,
                    error=f"BLOCKED: {result.block_reason}",
                )
            )
        elif result.success:
            step_results.append(
                StepExecution(
                    step=step,
                    success=True,
                    output=truncate(result.stdout, 300) if result.stdout else "✓",
                )
            )
        else:
            all_success = False
            step_results.append(
                StepExecution(
                    step=step,
                    success=False,
                    output=truncate(result.stdout, 200) if result.stdout else "",
                    error=truncate(
                        result.stderr or f"Exit code {result.exit_code}", 200
                    ),
                )
            )

    return (
        ExecutionResult(
            success=all_success,
            step_results=step_results,
            summary=f"Executed {len(step_results)} steps",
        ),
        cwd,
    )


# ---------------------------------------------------------------------------
# Stage 5: Final Validation
# ---------------------------------------------------------------------------

SHELL_VALIDATOR_SYSTEM = """\
You are a shell task validation agent. Given:
- The original user request
- The plan that was followed
- The execution results

Evaluate if the task was completed successfully.

Respond ONLY with JSON (no markdown fences, no commentary):
{
    "ok": true/false,
    "issues": ["list of issues if any"],
    "summary": "brief validation summary"
}
"""


async def validate_shell_execution(
    task: str,
    plan: ShellPlan,
    execution: ExecutionResult,
) -> FinalValidation:
    """Validate that the execution achieved the goal."""
    llm = get_llm("reasoning")

    results_text = "\n".join(
        f"- {sr.step.description}: {'✓' if sr.success else '✗'} "
        + (sr.output if sr.success else sr.error)
        for sr in execution.step_results
    )

    user_prompt = (
        f"## Original Request\n{task}\n\n"
        f"## Plan\n{plan.summary}\n\n"
        f"## Execution Results\n{results_text}"
    )

    messages = [
        SystemMessage(content=SHELL_VALIDATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        # Default to success if we can't parse
        return FinalValidation(ok=True, summary=raw)

    return FinalValidation(
        ok=data.get("ok", True),
        issues=data.get("issues", []),
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

MAX_PLAN_ITERATIONS = 5


async def run_shell_pipeline(task: str, working_dir: str, unsafe: bool = False):
    """Run the staged shell pipeline.

    Stages:
    1. Investigate - gather directory context
    2. Plan - create complete plan with descriptions
    3. Validate Plan - check plan will achieve goal (max 5 iterations)
    4. Execute - run planned commands
    5. Validate - verify execution achieved goal

    Yields status tuples:
      ("status", str)   - progress updates (step descriptions)
      ("output", str)   - command output to show
      ("result", str)   - final result
      ("error", str)    - if something went wrong
      ("cwd", str)      - updated working directory
    """
    # Stage 1: Investigation
    yield ("status", "Investigating project...")
    investigation = investigate_shell(working_dir)
    if investigation.project_type:
        yield ("status", f"Detected: {investigation.project_type}")

    # Stage 2 & 3: Plan with validation loop
    plan: ShellPlan | None = None
    plan_issues: list[str] = []

    for iteration in range(MAX_PLAN_ITERATIONS):
        # Stage 2: Planning
        try:
            if iteration == 0:
                yield ("status", "Creating plan...")
            else:
                yield ("status", f"Revising plan (attempt {iteration + 1})...")

            plan = await plan_shell(
                task,
                investigation,
                previous_issues=plan_issues if plan_issues else None,
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
            validation = await validate_shell_plan(task, plan, investigation)

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
            f"Proceeding after {MAX_PLAN_ITERATIONS} plan attempts",
        )

    # Note: plan cannot be None here because:
    # - If plan_shell raises, we return early (line 536)
    # - If plan_shell returns empty steps, we return early (line 530)
    # - Otherwise, plan is set to a valid ShellPlan
    # This check satisfies the type checker without using assert
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

        execution, cwd = await execute_shell(plan, working_dir, unsafe)

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
    try:
        yield ("status", "Validating results...")
        final_validation = await validate_shell_execution(task, plan, execution)

        if final_validation.ok:
            yield ("result", final_validation.summary or "Task completed successfully")
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

    yield ("cwd", cwd)
