"""Iterative coding pipeline: Plan one action -> Execute -> Observe -> Repeat.

This module implements an adaptive coding pipeline. Instead of following a
fixed sequence of stages, the LLM decides which action to take next based
on the current state (files read, changes made, validation results, etc.).

Available actions:
- investigate: Scan project to find relevant files
- read_files: Read specific files into context
- plan: Create a coding plan for what changes to make
- generate: Generate code based on the plan
- write: Write generated patches to disk
- validate: Validate the changes
- done: Task complete
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from axono import config
from axono.pipeline import (
    PlanStep,
    PlanValidation,
    coerce_response_text,
    get_llm,
    parse_json,
    truncate,
    validate_plan,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FileContent:
    """Represents a file and its content."""

    path: str
    content: str


@dataclass
class FilePatch:
    """Describes a write operation for a single file."""

    path: str
    content: str
    action: str = "update"  # "create" or "update"


@dataclass
class CodingPlan:
    """Output of the planner stage."""

    summary: str
    files_to_read: list[str] = field(default_factory=list)
    patches: list[dict] = field(default_factory=list)  # planned edits
    raw: str = ""


@dataclass
class GeneratedCode:
    """Output of the code generator stage."""

    patches: list[FilePatch] = field(default_factory=list)
    explanation: str = ""


@dataclass
class ValidationResult:
    """Output of the validator stage."""

    ok: bool
    issues: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class InvestigationResult:
    """Output of the investigation stage."""

    files: list[FileContent] = field(default_factory=list)
    summary: str = ""
    skipped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_directory(directory: str, max_depth: int = 3) -> list[str]:
    """Return a list of file paths in *directory* up to *max_depth* levels."""
    paths: list[str] = []
    directory = os.path.abspath(directory)
    for root, dirs, files in os.walk(directory):
        depth = root.replace(directory, "").count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        # Skip hidden dirs and common noise
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", "bin", "obj", ".git"}
        ]
        for fname in files:
            if fname.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(root, fname), directory)
            paths.append(rel)
    return sorted(paths)


def _read_file(path: str) -> str | None:
    """Read a file; return None if it doesn't exist or is binary."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _write_file(path: str, content: str) -> None:
    """Write content to a file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(t) >= 3]


def _extension_weight(path: str) -> float:
    ext = os.path.splitext(path)[1].lower()
    weights = {
        ".py": 2.5,
        ".js": 2.0,
        ".ts": 2.2,
        ".tsx": 2.2,
        ".jsx": 2.0,
        ".java": 2.2,
        ".go": 2.2,
        ".rs": 2.2,
        ".cpp": 2.0,
        ".c": 1.8,
        ".cs": 2.2,
        ".rb": 2.0,
        ".php": 2.0,
        ".kt": 2.1,
        ".swift": 2.0,
        ".md": 1.6,
        ".txt": 1.0,
        ".yaml": 1.2,
        ".yml": 1.2,
        ".json": 1.2,
        ".toml": 1.2,
        ".ini": 1.0,
    }
    return weights.get(ext, 0.8)


def _recency_score(mtime: float) -> float:
    age_days = max((time.time() - mtime) / 86_400, 0.0)
    return 1.0 / (1.0 + age_days)


def _score_path_keywords(path: str, keywords: list[str]) -> float:
    haystack = path.lower()
    return sum(1.0 for kw in keywords if kw in haystack)


def _read_snippet(path: str, max_chars: int = 2000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except (OSError, UnicodeDecodeError):
        return ""


def _score_snippet(snippet: str, keywords: list[str]) -> float:
    if not snippet:
        return 0.0
    text = snippet.lower()
    return sum(1.5 for kw in keywords if kw in text)


def _select_seed_files(dir_listing: list[str]) -> list[str]:
    candidates = [
        "README.md",
        "readme.md",
        "specs.md",
        "SPEC.md",
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    ]
    available = set(dir_listing)
    return [c for c in candidates if c in available]


def _read_seed_files(working_dir: str, dir_listing: list[str]) -> list[FileContent]:
    seeds: list[FileContent] = []
    for rel in _select_seed_files(dir_listing):
        full = os.path.join(working_dir, rel)
        content = _read_file(full)
        if content is not None:
            seeds.append(FileContent(path=rel, content=content))
    return seeds


def _dedupe_file_contents(files: list[FileContent]) -> list[FileContent]:
    seen: set[str] = set()
    result: list[FileContent] = []
    for fc in files:
        if fc.path in seen:
            continue
        seen.add(fc.path)
        result.append(fc)
    return result


def _total_chars(files: list[FileContent]) -> int:
    return sum(len(fc.content) for fc in files)


# ---------------------------------------------------------------------------
# Stage 0 – Investigation
# ---------------------------------------------------------------------------


def investigate(
    task: str,
    working_dir: str,
    dir_listing: list[str],
    seed_files: list[FileContent],
) -> InvestigationResult:
    """Select relevant files using heuristics and lightweight retrieval."""
    if not config.INVESTIGATION_ENABLED:
        return InvestigationResult(
            files=seed_files,
            summary="Investigation disabled. Using seed files only.",
        )

    keywords = _tokenize(task)
    max_files = max(config.MAX_CONTEXT_FILES, 0)
    max_chars = max(config.MAX_CONTEXT_CHARS, 0)

    files: list[FileContent] = list(seed_files)
    skipped: list[str] = []
    existing = {fc.path for fc in files}

    total_chars = _total_chars(files)
    remaining_files = max_files - len(files)
    if remaining_files <= 0 or max_chars <= total_chars:
        return InvestigationResult(
            files=files,
            summary="Seed files filled the configured context budget.",
            skipped=skipped,
        )

    candidates: list[tuple[float, str]] = []
    for rel in dir_listing:
        if rel in existing:
            continue
        full_path = os.path.join(working_dir, rel)
        try:
            stat = os.stat(full_path)
        except OSError:
            continue

        score = _extension_weight(rel)
        score += 2.0 * _score_path_keywords(rel, keywords)
        score += _recency_score(stat.st_mtime)
        candidates.append((score, rel))

    candidates.sort(reverse=True, key=lambda item: item[0])
    probe_limit = max(min(len(candidates), max_files * 5), 15)
    probed: list[tuple[float, str]] = []
    for score, rel in candidates[:probe_limit]:
        snippet = _read_snippet(os.path.join(working_dir, rel), max_chars=2000)
        score += _score_snippet(snippet, keywords)
        probed.append((score, rel))

    probed.sort(reverse=True, key=lambda item: item[0])

    for score, rel in probed:
        if remaining_files <= 0:
            break
        full_path = os.path.join(working_dir, rel)
        content = _read_file(full_path)
        if content is None:
            skipped.append(f"{rel} (unreadable)")
            continue

        remaining_chars = max_chars - total_chars
        if remaining_chars <= 0:
            break
        if len(content) > remaining_chars:
            skipped.append(f"{rel} (over budget)")
            continue

        files.append(FileContent(path=rel, content=content))
        total_chars += len(content)
        remaining_files -= 1

    summary = (
        f"Investigation selected {len(files)} files "
        f"({total_chars} chars) within budget of "
        f"{max_files} files / {max_chars} chars."
    )
    return InvestigationResult(
        files=_dedupe_file_contents(files), summary=summary, skipped=skipped
    )


# ---------------------------------------------------------------------------
# Stage 1 – Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """\
You are a senior software engineer acting as a **planning agent**.
You receive:
  - The user's coding request.
  - A directory listing of the project.
  - The contents of files that appear relevant.

Your job is to produce a JSON plan (and nothing else) with the structure:
{
  "summary": "Short human-readable summary of what needs to happen",
  "files_to_read": ["list", "of", "files", "to", "read", "for", "more", "context"],
  "patches": [
    {
      "path": "relative/path/to/file",
      "action": "create" or "update",
      "description": "What change to make in this file"
    }
  ]
}

Rules:
- Only include files that actually need changes in `patches`.
- `files_to_read` should list any files whose contents are needed by the code
  generator but were NOT already provided to you.
- Respond ONLY with valid JSON. No markdown fences. No commentary.
"""


async def plan(
    task: str,
    working_dir: str,
    dir_listing: list[str],
    initial_files: list[FileContent],
) -> tuple[CodingPlan, list[FileContent]]:
    """Stage 1: Analyse the project and create a coding plan.

    Returns the plan and the file contents that were gathered.
    """
    llm = get_llm("reasoning")
    files_block = "\n\n".join(
        f"### {fc.path}\n```\n{fc.content}\n```" for fc in initial_files
    )

    user_prompt = (
        f"## User request\n{task}\n\n"
        f"## Directory listing\n{chr(10).join(dir_listing)}\n\n"
        f"## File contents\n{files_block}"
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    # Parse the JSON plan, tolerating markdown fences
    data = parse_json(raw)
    if data is None:
        # Fallback: treat entire response as summary
        data = {"summary": raw, "files_to_read": [], "patches": []}

    coding_plan = CodingPlan(
        summary=data.get("summary", ""),
        files_to_read=data.get("files_to_read", []),
        patches=data.get("patches", []),
        raw=raw,
    )

    # Read any extra files the planner requested
    total_chars = _total_chars(initial_files)
    remaining_files = max(config.MAX_CONTEXT_FILES - len(initial_files), 0)
    for rel in coding_plan.files_to_read:
        if remaining_files <= 0:
            break
        full = os.path.join(working_dir, rel)
        content = _read_file(full)
        if content is None:
            continue
        if any(fc.path == rel for fc in initial_files):
            continue
        remaining_chars = config.MAX_CONTEXT_CHARS - total_chars
        if remaining_chars <= 0 or len(content) > remaining_chars:
            continue
        initial_files.append(FileContent(path=rel, content=content))
        total_chars += len(content)
        remaining_files -= 1

    return coding_plan, _dedupe_file_contents(initial_files)


# ---------------------------------------------------------------------------
# Stage 2 – Code Generator
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = """\
You are a senior software engineer acting as a **code generation agent**.
You receive:
  - A coding plan describing what needs to happen.
  - The current contents of relevant source files.

Your job is to produce the final file contents for every file that needs to be
created or modified.

Respond ONLY with a JSON array (no markdown fences, no commentary):
[
  {
    "path": "relative/path/to/file",
    "action": "create" or "update",
    "content": "full file content as a string"
  }
]

Rules:
- For "update" actions, return the COMPLETE updated file content (not a diff).
- Preserve existing code that should not change.
- Follow the style and conventions of the existing codebase.
- Each planned class/module should map to an explicit file path. Do NOT emit
    multiple outputs targeting the same file unless explicitly instructed.
- If you need to create new files, choose clear, unique file paths and include
    them as separate entries in the JSON array.
- Respond ONLY with valid JSON. No markdown fences. No commentary.
"""


async def generate(
    coding_plan: CodingPlan,
    file_contents: list[FileContent],
) -> GeneratedCode:
    """Stage 2: Generate code based on the plan and file context."""
    llm = get_llm("instruction")

    files_block = "\n\n".join(
        f"### {fc.path}\n```\n{fc.content}\n```" for fc in file_contents
    )

    patches_block = json.dumps(coding_plan.patches, indent=2)

    user_prompt = (
        f"## Plan summary\n{coding_plan.summary}\n\n"
        f"## Planned changes\n{patches_block}\n\n"
        f"## Current file contents\n{files_block}"
    )

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    # Parse JSON, tolerating markdown fences
    items = parse_json(raw)
    if items is None:
        return GeneratedCode(explanation=f"Failed to parse generator output:\n{raw}")

    # Ensure items is a list
    if not isinstance(items, list):
        return GeneratedCode(explanation=f"Expected JSON array, got:\n{raw}")

    seen_paths: set[str] = set()
    duplicate_paths: list[str] = []
    for item in items:
        path = item.get("path") if isinstance(item, dict) else None
        if not path:
            continue
        if path in seen_paths:
            duplicate_paths.append(path)
        else:
            seen_paths.add(path)

    if duplicate_paths:
        dupes = ", ".join(sorted(set(duplicate_paths)))
        return GeneratedCode(
            explanation=(
                "Generator produced duplicate file paths: "
                f"{dupes}. Each output file must be unique."
            )
        )

    patches = [
        FilePatch(
            path=item["path"],
            content=item["content"],
            action=item.get("action", "update"),
        )
        for item in items
        if "path" in item and "content" in item
    ]
    return GeneratedCode(patches=patches)


# ---------------------------------------------------------------------------
# Stage 3 – File Writer (apply)
# ---------------------------------------------------------------------------


def apply(generated: GeneratedCode, working_dir: str) -> list[str]:
    """Stage 3: Write generated code to disk.

    Returns a list of human-readable descriptions of what was written.
    """
    results: list[str] = []
    for patch in generated.patches:
        full_path = os.path.join(working_dir, patch.path)
        verb = "Created" if patch.action == "create" else "Updated"
        _write_file(full_path, patch.content)
        results.append(f"{verb}: {patch.path}")
    return results


# ---------------------------------------------------------------------------
# Stage 4 – Validator
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM = """\
You are a senior software engineer acting as a **code review / validation agent**.
You receive:
  - The original user request.
  - The plan that was followed.
  - The final file contents that were written to disk.

Your job is to review the code and check for:
  1. Correctness: Does the code fulfil the user's request?
  2. Completeness: Are there any missing pieces?
  3. Syntax errors or obvious bugs.
  4. Style issues or deviations from the project conventions.

Respond ONLY with a JSON object (no markdown fences, no commentary):
{
  "ok": true/false,
  "issues": ["list of issues if any"],
  "summary": "Brief validation summary"
}
"""


async def validate(
    task: str,
    coding_plan: CodingPlan,
    generated: GeneratedCode,
) -> ValidationResult:
    """Stage 4: Validate the generated code."""
    llm = get_llm("reasoning")

    files_block = "\n\n".join(
        f"### {p.path}\n```\n{p.content}\n```" for p in generated.patches
    )

    user_prompt = (
        f"## Original request\n{task}\n\n"
        f"## Plan\n{coding_plan.summary}\n\n"
        f"## Written files\n{files_block}"
    )

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        return ValidationResult(ok=True, summary=raw)

    return ValidationResult(
        ok=data.get("ok", True),
        issues=data.get("issues", []),
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Iterative Planner
# ---------------------------------------------------------------------------

ITERATIVE_PLANNER_SYSTEM = """\
You are a coding agent that decides what action to take next.

Available actions:
- {"action": "investigate", "reason": "..."} - Scan project to find relevant files
- {"action": "read_files", "files": ["path1", "path2"], "reason": "..."} - Read specific files
- {"action": "plan", "reason": "..."} - Create a detailed coding plan
- {"action": "validate_plan", "reason": "..."} - Validate if the plan will achieve the goal
- {"action": "generate", "reason": "..."} - Generate code based on the plan
- {"action": "write", "reason": "..."} - Write generated patches to disk
- {"action": "validate", "reason": "..."} - Validate the changes
- {"done": true, "summary": "..."} - Task complete

RULES:
- Take ONE action at a time
- After creating a plan, validate it before generating code
- After writing files, ALWAYS validate the result
- If plan validation fails, revise the plan
- If code validation fails, you can re-generate and re-write
- Respond ONLY with JSON
"""


def _build_iterative_prompt(
    task: str,
    cwd: str,
    history: list[dict],
    state: dict[str, Any],
) -> str:
    """Build the user prompt for the iterative planner."""
    parts = [f"Goal: {task}", f"CWD: {cwd}"]

    # Show available files if we have a directory listing
    if "dir_listing" in state:
        listing = state["dir_listing"][:30]  # First 30 files
        parts.append(f"Project files: {', '.join(listing)}")
        if len(state["dir_listing"]) > 30:
            parts.append(f"  (+{len(state['dir_listing']) - 30} more)")

    # Show files in context
    if "files" in state and state["files"]:
        file_paths = [fc.path for fc in state["files"]]
        parts.append(f"Files in context: {', '.join(file_paths)}")

    # Show current plan if we have one
    if "coding_plan" in state:
        plan_summary = state["coding_plan"].summary
        parts.append(f"Current plan: {plan_summary}")

    # Show plan validation result if we have one
    if "plan_validation" in state:
        pv = state["plan_validation"]
        if pv.valid:
            parts.append(f"Plan validation: PASSED - {pv.summary}")
        else:
            parts.append(f"Plan validation: FAILED - {pv.summary}")
            for issue in pv.issues[:3]:
                parts.append(f"  - {issue}")
            if pv.suggestions:
                parts.append("  Suggestions:")
                for sug in pv.suggestions[:2]:
                    parts.append(f"    - {sug}")

    # Show generated patches if we have them
    if "generated" in state and state["generated"].patches:
        patch_paths = [p.path for p in state["generated"].patches]
        parts.append(f"Generated patches: {', '.join(patch_paths)}")

    # Show written files
    if "written" in state and state["written"]:
        parts.append(f"Written: {', '.join(state['written'])}")

    # Show validation result if we have one
    if "validation" in state:
        val = state["validation"]
        if val.ok:
            parts.append(f"Validation: PASSED - {val.summary}")
        else:
            parts.append(f"Validation: FAILED - {val.summary}")
            for issue in val.issues[:3]:
                parts.append(f"  - {issue}")

    # Show history (last 5 actions)
    if history:
        parts.append("\nRecent actions:")
        for h in history[-5:]:
            action = h.get("action", "unknown")
            success = "✓" if h.get("success", False) else "✗"
            reason = h.get("reason", "")
            error = h.get("error", "")
            line = f"  {success} {action}"
            if reason:
                line += f" - {truncate(reason, 50)}"
            if error:
                line += f" [error: {truncate(error, 50)}]"
            parts.append(line)

    return "\n".join(parts)


async def _plan_next_action(
    task: str,
    cwd: str,
    history: list[dict],
    state: dict[str, Any],
) -> dict:
    """Plan the next action using the LLM."""
    llm = get_llm("reasoning")
    user_prompt = _build_iterative_prompt(task, cwd, history, state)

    messages = [
        SystemMessage(content=ITERATIVE_PLANNER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None:
        return {"done": True, "summary": "Could not parse response"}

    return data


# ---------------------------------------------------------------------------
# Orchestrator – iterative pipeline
# ---------------------------------------------------------------------------


async def run_coding_pipeline(task: str, working_dir: str):
    """Run the iterative coding pipeline.

    The LLM decides which action to take next based on the current state.
    Actions: investigate, read_files, plan, generate, write, validate, done.

    Yields status tuples that the caller can use to display progress:
      ("status", str)   - progress updates
      ("result", str)   - final result summary
      ("error", str)    - if something went wrong
    """
    # Initialize state
    dir_listing = _scan_directory(working_dir)
    seed_files = _read_seed_files(working_dir, dir_listing)

    state: dict[str, Any] = {
        "dir_listing": dir_listing,
        "files": list(seed_files),
    }
    history: list[dict] = []
    max_steps = 15

    for step_num in range(max_steps):
        # Plan next action
        try:
            action_plan = await _plan_next_action(task, working_dir, history, state)
        except Exception as e:
            yield ("error", f"Planning failed: {e}")
            return

        # Check if done
        if action_plan.get("done"):
            summary = action_plan.get("summary", "Done")
            yield ("result", _build_final_summary(state, summary))
            return

        action = action_plan.get("action", "").strip()
        reason = action_plan.get("reason", "")

        if not action:
            yield ("error", "No action provided")
            return

        yield ("status", f"[{action}] {reason}" if reason else f"[{action}]")

        # Execute the action
        try:
            if action == "investigate":
                result = _handle_investigate(task, working_dir, state)
                history.append(
                    {"action": "investigate", "success": True, "reason": reason}
                )
                yield ("status", result)

            elif action == "read_files":
                files_to_read = action_plan.get("files", [])
                result = _handle_read_files(working_dir, files_to_read, state)
                history.append(
                    {"action": "read_files", "success": True, "reason": reason}
                )
                yield ("status", result)

            elif action == "plan":
                result = await _handle_plan(task, working_dir, state)
                # Clear any previous plan validation when a new plan is created
                state.pop("plan_validation", None)
                history.append({"action": "plan", "success": True, "reason": reason})
                yield ("status", result)

            elif action == "validate_plan":
                result = await _handle_validate_plan(task, state)
                history.append(
                    {"action": "validate_plan", "success": True, "reason": reason}
                )
                yield ("status", result)

            elif action == "generate":
                result = await _handle_generate(state)
                history.append(
                    {"action": "generate", "success": True, "reason": reason}
                )
                yield ("status", result)

            elif action == "write":
                result = _handle_write(working_dir, state)
                history.append({"action": "write", "success": True, "reason": reason})
                yield ("status", result)

            elif action == "validate":
                result = await _handle_validate(task, state)
                history.append(
                    {"action": "validate", "success": True, "reason": reason}
                )
                yield ("status", result)

            else:
                yield ("error", f"Unknown action: {action}")
                history.append(
                    {"action": action, "success": False, "error": "Unknown action"}
                )

        except Exception as e:
            error_msg = f"{action} failed: {e}"
            yield ("error", error_msg)
            history.append(
                {"action": action, "success": False, "error": str(e), "reason": reason}
            )
            # Continue to let the LLM decide what to do next

    # Reached max steps
    yield ("result", _build_final_summary(state, f"Stopped after {max_steps} steps"))


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _handle_investigate(task: str, working_dir: str, state: dict[str, Any]) -> str:
    """Handle the investigate action."""
    investigation = investigate(
        task,
        working_dir,
        state.get("dir_listing", []),
        state.get("files", []),
    )
    state["files"] = investigation.files
    state["investigation"] = investigation
    return investigation.summary


def _handle_read_files(
    working_dir: str, files_to_read: list[str], state: dict[str, Any]
) -> str:
    """Handle the read_files action."""
    current_files = state.get("files", [])
    existing_paths = {fc.path for fc in current_files}
    added = []

    for rel in files_to_read:
        if rel in existing_paths:
            continue
        full = os.path.join(working_dir, rel)
        content = _read_file(full)
        if content is not None:
            current_files.append(FileContent(path=rel, content=content))
            added.append(rel)
            existing_paths.add(rel)

    state["files"] = current_files
    if added:
        return f"Read {len(added)} files: {', '.join(added)}"
    return "No new files read"


async def _handle_plan(task: str, working_dir: str, state: dict[str, Any]) -> str:
    """Handle the plan action."""
    coding_plan, file_contents = await plan(
        task,
        working_dir,
        state.get("dir_listing", []),
        state.get("files", []),
    )
    state["coding_plan"] = coding_plan
    state["files"] = file_contents
    return f"Plan: {coding_plan.summary}"


async def _handle_validate_plan(task: str, state: dict[str, Any]) -> str:
    """Handle the validate_plan action."""
    coding_plan = state.get("coding_plan")
    if not coding_plan:
        raise ValueError("No plan available. Run 'plan' first.")

    # Convert CodingPlan patches to PlanStep format
    steps = [
        PlanStep(
            description=p.get("description")
            or f"{p.get('action', 'modify')} {p.get('path', '')}",
            action={"path": p.get("path", ""), "action": p.get("action", "update")},
        )
        for p in coding_plan.patches
    ]

    # Build context from files in state
    files_in_context = state.get("files", [])
    context = ""
    if files_in_context:
        context = f"Files in context: {', '.join(fc.path for fc in files_in_context)}"

    validation = await validate_plan(
        task=task,
        plan_summary=coding_plan.summary,
        steps=steps,
        context=context,
    )
    state["plan_validation"] = validation

    if validation.valid:
        return f"Plan validated ✓ - {validation.summary}"
    else:
        issues = (
            "; ".join(validation.issues[:2]) if validation.issues else "Issues found"
        )
        return f"Plan validation failed: {issues}"


async def _handle_generate(state: dict[str, Any]) -> str:
    """Handle the generate action."""
    coding_plan = state.get("coding_plan")
    if not coding_plan:
        raise ValueError("No plan available. Run 'plan' first.")

    generated = await generate(coding_plan, state.get("files", []))
    if not generated.patches:
        error = generated.explanation or "No patches generated"
        raise ValueError(error)

    state["generated"] = generated
    return f"Generated {len(generated.patches)} patches: {', '.join(p.path for p in generated.patches)}"


def _handle_write(working_dir: str, state: dict[str, Any]) -> str:
    """Handle the write action."""
    generated = state.get("generated")
    if not generated or not generated.patches:
        raise ValueError("No patches to write. Run 'generate' first.")

    written = apply(generated, working_dir)
    state["written"] = written
    return f"Wrote {len(written)} files: {', '.join(written)}"


async def _handle_validate(task: str, state: dict[str, Any]) -> str:
    """Handle the validate action."""
    coding_plan = state.get("coding_plan")
    generated = state.get("generated")
    if not coding_plan or not generated:
        raise ValueError("Nothing to validate. Run 'plan' and 'generate' first.")

    validation = await validate(task, coding_plan, generated)
    state["validation"] = validation

    if validation.ok:
        return f"Validation passed: {validation.summary}"
    else:
        issues = "; ".join(validation.issues[:3])
        return f"Validation failed: {validation.summary}. Issues: {issues}"


def _build_final_summary(state: dict[str, Any], summary: str) -> str:
    """Build the final result summary."""
    parts = ["## Coding complete", "", f"**Summary:** {summary}"]

    if "coding_plan" in state:
        parts.append(f"**Plan:** {state['coding_plan'].summary}")

    if "written" in state and state["written"]:
        parts.append("")
        parts.append("**Files changed:**")
        for line in state["written"]:
            parts.append(f"  - {line}")

    if "validation" in state:
        parts.append("")
        val = state["validation"]
        if val.ok:
            parts.append(f"**Validation:** Passed. {val.summary}")
        else:
            parts.append(f"**Validation:** Issues found. {val.summary}")
            for issue in val.issues:
                parts.append(f"  - {issue}")

    return "\n".join(parts)
