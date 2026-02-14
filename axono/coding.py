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

import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from axono import config, prompt
from axono.langdetect import detect_project_type_from_folderpath

try:
    from axono.folderembed import query_similar as _query_similar
except Exception:  # ImportError, missing optional deps, etc.
    _query_similar = None  # type: ignore[assignment]
from axono.pipeline import (
    FinalValidation,
    Investigation,
    Plan,
    PlanStep,
    coerce_response_text,
    get_llm,
    parse_json,
    truncate,
    validate_plan,
)

# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------

# Load prompts from YAML files on module initialization
_planner_prompt = prompt.load_prompt("coding", "planner")
_generator_prompt = prompt.load_prompt("coding", "generator")
_validator_prompt = prompt.load_prompt("coding", "validator")
_iterative_planner_prompt = prompt.load_prompt("coding", "iterative_planner")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FileContent:
    """Represents a file and its content."""

    path: str
    content: str


@dataclass
class Replacement:
    """A single search-and-replace edit within a file."""

    search: str
    replace: str


class ReplacementError(Exception):
    """Raised when a search string is not found in the file."""

    def __init__(self, path: str, search: str):
        self.path = path
        self.search = search
        preview = search[:80] + "..." if len(search) > 80 else search
        super().__init__(f"Search text not found in {path}: {preview}")


@dataclass
class FilePatch:
    """Describes a write operation for a single file."""

    path: str
    content: str = ""
    action: str = "update"  # "create" or "update"
    replacements: list[Replacement] = field(default_factory=list)


@dataclass
class GeneratedCode:
    """Output of the code generator stage."""

    patches: list[FilePatch] = field(default_factory=list)
    explanation: str = ""


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


def _backup_file(path: str) -> str | None:
    """Create a backup of an existing file before overwriting.

    Returns the backup path if a backup was created, None if file doesn't exist.
    """
    if not os.path.isfile(path):
        return None
    import shutil

    backup_path = path + ".axono-backup"
    shutil.copy2(path, backup_path)
    return backup_path


def cleanup_backups(backup_paths: list[str]) -> None:
    """Remove backup files after successful validation."""
    for path in backup_paths:
        try:
            os.remove(path)
        except OSError:
            pass


def restore_backups(backup_paths: list[str]) -> list[str]:
    """Restore files from backups. Returns list of restored paths."""
    import shutil

    restored = []
    for backup_path in backup_paths:
        original = backup_path.removesuffix(".axono-backup")
        if os.path.isfile(backup_path):
            shutil.copy2(backup_path, original)
            os.remove(backup_path)
            restored.append(original)
    return restored


def apply_replacements(
    original: str, replacements: list[Replacement], path: str
) -> str:
    """Apply a sequence of search-and-replace operations to file content.

    Each replacement replaces the first occurrence of its search string.

    Raises:
        ReplacementError: If a search string is not found.
    """
    content = original
    for rep in replacements:
        if rep.search not in content:
            raise ReplacementError(path, rep.search)
        content = content.replace(rep.search, rep.replace, 1)
    return content


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


async def _query_embedding_context(
    task: str,
    working_dir: str,
    existing_files: list[FileContent],
    max_chars: int,
) -> list[FileContent]:
    """Query the embedding database for semantically related code chunks.

    Returns a list of FileContent objects representing relevant code snippets.
    Returns an empty list if embeddings are unavailable.
    """
    if _query_similar is None:
        return []

    try:
        chunks = await _query_similar(task, top_k=20)
    except Exception:
        return []

    if not chunks:
        return []

    existing_paths = {fc.path for fc in existing_files}

    seen: set[tuple[str, int, int]] = set()
    results: list[FileContent] = []
    total_chars = 0

    for chunk in chunks:
        rel_path = os.path.relpath(chunk.file_path, working_dir)

        if rel_path in existing_paths:
            continue

        key = (rel_path, chunk.line_start, chunk.line_end)
        if key in seen:
            continue
        seen.add(key)

        code_len = len(chunk.source_code)
        if total_chars + code_len > max_chars:
            continue

        symbol_info = ""
        if chunk.symbol_name:
            prefix = ""
            if chunk.metadata and chunk.metadata.parent_class:
                prefix = f"{chunk.metadata.parent_class}."
            symbol_info = (
                f" :: {prefix}{chunk.symbol_name}"
                f" ({chunk.symbol_type}, lines {chunk.line_start}-{chunk.line_end})"
            )

        label = f"{rel_path}{symbol_info}"
        results.append(FileContent(path=label, content=chunk.source_code))
        total_chars += code_len

    return results


# ---------------------------------------------------------------------------
# Stage 0 – Investigation
# ---------------------------------------------------------------------------


def investigate(
    task: str,
    working_dir: str,
    dir_listing: list[str],
    seed_files: list[FileContent],
) -> Investigation:
    """Select relevant files using heuristics and lightweight retrieval."""
    if not config.INVESTIGATION_ENABLED:
        return Investigation(
            cwd=working_dir,
            dir_listing=dir_listing,
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
        return Investigation(
            cwd=working_dir,
            dir_listing=dir_listing,
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
    return Investigation(
        cwd=working_dir,
        dir_listing=dir_listing,
        files=_dedupe_file_contents(files),
        summary=summary,
        skipped=skipped,
    )


_LLM_RANK_SYSTEM = """\
You are a file relevance ranking agent. Given a task and a list of file paths,
rank them by relevance to the task. Most relevant files first.

You may reason briefly, but your response MUST contain a JSON array of file
paths, ordered by relevance:
["most_relevant.py", "second.py", ...]

Only include files from the provided list. You may omit clearly irrelevant files."""


async def _llm_rank_files(
    task: str, candidates: list[str], working_dir: str
) -> list[str]:
    """Ask the LLM to rank candidate files by relevance to the task.

    Falls back to the original order on failure.
    """
    llm = get_llm("reasoning")

    files_list = "\n".join(f"- {c}" for c in candidates)
    user_prompt = f"Task: {task}\n\nCandidate files:\n{files_list}"

    messages = [
        SystemMessage(content=_LLM_RANK_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw = coerce_response_text(response.content).strip()
        data = parse_json(raw)
        if isinstance(data, list):
            candidate_set = set(candidates)
            valid = [f for f in data if isinstance(f, str) and f in candidate_set]
            remaining = [c for c in candidates if c not in set(valid)]
            return valid + remaining
    except Exception:
        pass

    return candidates


async def investigate_with_llm(
    task: str,
    working_dir: str,
    dir_listing: list[str],
    seed_files: list[FileContent],
) -> Investigation:
    """Run heuristic investigation, then optionally LLM-rank the results."""
    investigation = investigate(task, working_dir, dir_listing, seed_files)

    if not config.LLM_INVESTIGATION:
        return investigation

    candidate_paths = [fc.path for fc in investigation.files]
    if len(candidate_paths) <= 1:
        return investigation

    ranked = await _llm_rank_files(task, candidate_paths, working_dir)

    path_to_file = {fc.path: fc for fc in investigation.files}
    reordered = [path_to_file[p] for p in ranked if p in path_to_file]

    return Investigation(
        cwd=investigation.cwd,
        dir_listing=investigation.dir_listing,
        files=reordered,
        project_type=investigation.project_type,
        summary=investigation.summary + " (LLM-ranked)",
        skipped=investigation.skipped,
    )


# ---------------------------------------------------------------------------
# Stage 1 – Planner
# ---------------------------------------------------------------------------


async def plan(
    task: str,
    working_dir: str,
    dir_listing: list[str],
    initial_files: list[FileContent],
    embedding_context: list[FileContent] | None = None,
) -> tuple[Plan, list[FileContent]]:
    """Stage 1: Analyse the project and create a coding plan.

    Returns the plan and the file contents that were gathered.
    """
    llm = get_llm("reasoning")
    files_block = "\n\n".join(
        f"### {fc.path}\n```\n{fc.content}\n```" for fc in initial_files
    )

    embedding_block = ""
    if embedding_context:
        embedding_block = (
            "\n\n## Semantically related code\n"
            "The following code snippets were found via semantic search"
            " and may be relevant:\n\n"
            + "\n\n".join(
                f"### {fc.path}\n```\n{fc.content}\n```" for fc in embedding_context
            )
        )

    user_prompt = (
        f"## User request\n{task}\n\n"
        f"## Directory listing\n{chr(10).join(dir_listing)}\n\n"
        f"## File contents\n{files_block}"
        f"{embedding_block}"
    )

    messages = [
        SystemMessage(content=_planner_prompt.system),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    # Parse the JSON plan, tolerating markdown fences
    data = parse_json(raw)
    if data is None or not isinstance(data, dict):
        # Fallback: treat entire response as summary
        data = {"summary": raw, "files_to_read": [], "patches": []}

    # Convert patches to PlanStep format
    steps = [
        PlanStep(
            description=p.get("description")
            or f"{p.get('action', 'modify')} {p.get('path', '')}",
            action={"path": p.get("path", ""), "action": p.get("action", "update")},
        )
        for p in data.get("patches", [])
    ]

    coding_plan = Plan(
        summary=data.get("summary", ""),
        files_to_read=data.get("files_to_read", []),
        patches=data.get("patches", []),
        steps=steps,
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


async def generate(
    coding_plan: Plan,
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
        SystemMessage(content=_generator_prompt.system),
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

    patches = []
    for item in items:
        if not isinstance(item, dict) or "path" not in item:
            continue
        action = item.get("action", "update")
        if action == "create":
            if "content" not in item:
                continue
            patches.append(
                FilePatch(path=item["path"], content=item["content"], action="create")
            )
        else:  # "update"
            raw_replacements = item.get("replacements", [])
            if raw_replacements and isinstance(raw_replacements, list):
                replacements = [
                    Replacement(search=r["search"], replace=r["replace"])
                    for r in raw_replacements
                    if isinstance(r, dict) and "search" in r and "replace" in r
                ]
                if replacements:
                    patches.append(
                        FilePatch(
                            path=item["path"],
                            action="update",
                            replacements=replacements,
                        )
                    )
            elif "content" in item:
                # Fallback: full-file content for update (legacy compatibility)
                patches.append(
                    FilePatch(
                        path=item["path"],
                        content=item["content"],
                        action="update",
                    )
                )
    return GeneratedCode(patches=patches)


# ---------------------------------------------------------------------------
# Stage 3 – File Writer (apply)
# ---------------------------------------------------------------------------


def _validate_patch_path(patch_path: str, working_dir: str) -> str:
    """Validate that a patch path stays within working_dir.

    Returns the resolved absolute path.

    Raises:
        ValueError: If the path escapes the working directory.
    """
    if os.path.isabs(patch_path):
        raise ValueError(f"Absolute path not allowed in patch: {patch_path}")

    full_path = os.path.join(working_dir, patch_path)
    real_full = os.path.realpath(full_path)
    real_wd = os.path.realpath(working_dir)

    if real_full != real_wd and not real_full.startswith(real_wd + os.sep):
        raise ValueError(
            f"Path escapes working directory: {patch_path} "
            f"resolves to {real_full}, outside {real_wd}"
        )
    return full_path


def apply(generated: GeneratedCode, working_dir: str) -> tuple[list[str], list[str]]:
    """Stage 3: Write generated code to disk.

    Returns:
        A tuple of (results, backup_paths) where results is a list of
        human-readable descriptions and backup_paths is a list of backup
        file paths created before overwriting.

    Raises:
        ReplacementError: If a search string is not found during update.
        ValueError: If a patch path escapes the working directory.
    """
    results: list[str] = []
    backup_paths: list[str] = []
    for patch in generated.patches:
        full_path = _validate_patch_path(patch.path, working_dir)
        # Backup existing files before overwriting
        backup = _backup_file(full_path)
        if backup:
            backup_paths.append(backup)
        if patch.action == "create":
            _write_file(full_path, patch.content)
            results.append(f"Created: {patch.path}")
        elif patch.replacements:
            original = _read_file(full_path)
            if original is None:
                raise ReplacementError(patch.path, "(file does not exist for update)")
            new_content = apply_replacements(original, patch.replacements, patch.path)
            _write_file(full_path, new_content)
            results.append(
                f"Updated: {patch.path} ({len(patch.replacements)} replacements)"
            )
        else:
            # Full-file content fallback
            _write_file(full_path, patch.content)
            results.append(f"Updated: {patch.path}")
    return results, backup_paths


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------


def _check_python_syntax(content: str, path: str) -> str | None:
    """Check Python syntax using ast.parse().

    Returns error message or None if syntax is valid.
    """
    try:
        ast.parse(content, filename=path)
        return None
    except SyntaxError as e:
        lineno = f":{e.lineno}" if e.lineno is not None else ""
        return f"{path}{lineno}: {e.msg}"


def _is_command_available(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    import shutil

    return shutil.which(cmd) is not None


def _check_syntax_external(content: str, path: str, command: list[str]) -> str | None:
    """Check syntax by writing to a temp file and running a command.

    Returns error message or None if syntax is valid.
    """
    import subprocess  # nosec B404
    import tempfile

    ext = os.path.splitext(path)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        result = subprocess.run(  # nosec B603
            command + [tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            return f"{path}: {error}" if error else f"{path}: syntax check failed"
        return None
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"{path}: syntax check error: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _find_file_content(path: str, files: list[FileContent]) -> str | None:
    """Find file content by path in a list of FileContent."""
    for fc in files:
        if fc.path == path:
            return fc.content
    return None


def check_syntax(patch: FilePatch, project_type: str, content: str) -> str | None:
    """Check syntax of generated content for a patch.

    Routes to the appropriate checker based on file extension.
    Returns error message or None if valid.
    """
    ext = os.path.splitext(patch.path)[1].lower()

    if ext == ".py":
        return _check_python_syntax(content, patch.path)

    if ext in (".js", ".jsx") and _is_command_available("node"):
        return _check_syntax_external(content, patch.path, ["node", "--check"])

    return None


# ---------------------------------------------------------------------------
# Stage 4 – Validator
# ---------------------------------------------------------------------------


async def validate(
    task: str,
    coding_plan: Plan,
    generated: GeneratedCode,
) -> FinalValidation:
    """Stage 4: Validate the generated code."""
    llm = get_llm("reasoning")

    file_blocks = []
    for p in generated.patches:
        if p.action == "create":
            file_blocks.append(f"### {p.path} (CREATE)\n```\n{p.content}\n```")
        elif p.replacements:
            edits = "\n".join(
                f"  SEARCH: {r.search!r}\n  REPLACE: {r.replace!r}"
                for r in p.replacements
            )
            file_blocks.append(
                f"### {p.path} (UPDATE - {len(p.replacements)} edits)\n{edits}"
            )
        else:
            file_blocks.append(
                f"### {p.path} (UPDATE - full rewrite)\n```\n{p.content}\n```"
            )
    files_block = "\n\n".join(file_blocks)

    user_prompt = (
        f"## Original request\n{task}\n\n"
        f"## Plan\n{coding_plan.summary}\n\n"
        f"## Written files\n{files_block}"
    )

    messages = [
        SystemMessage(content=_validator_prompt.system),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None or not isinstance(data, dict):
        return FinalValidation(
            ok=False,
            issues=[f"Could not parse validation response: {truncate(raw, 200)}"],
            summary="Validation response was not valid JSON",
        )

    return FinalValidation(
        ok=data.get("ok", True),
        issues=data.get("issues", []),
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Iterative Planner
# ---------------------------------------------------------------------------


def _build_iterative_prompt(
    task: str,
    cwd: str,
    history: list[dict],
    state: dict[str, Any],
) -> str:
    """Build the user prompt for the iterative planner."""
    parts = [f"Goal: {task}", f"CWD: {cwd}"]

    # Show detected project type if available
    if "project_type" in state and state["project_type"] != "unknown":
        parts.append(f"Project type: {state['project_type']}")

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
        patch_info = []
        for p in state["generated"].patches:
            if p.replacements:
                patch_info.append(f"{p.path} ({len(p.replacements)} edits)")
            else:
                patch_info.append(p.path)
        parts.append(f"Generated patches: {', '.join(patch_info)}")

    # Show syntax errors if any
    if "syntax_errors" in state and state["syntax_errors"]:
        parts.append("Syntax errors in generated code:")
        for err in state["syntax_errors"]:
            parts.append(f"  - {err}")

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
        SystemMessage(content=_iterative_planner_prompt.system),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = coerce_response_text(response.content).strip()

    data = parse_json(raw)
    if data is None or not isinstance(data, dict):
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
    project_type = detect_project_type_from_folderpath(working_dir)

    state: dict[str, Any] = {
        "dir_listing": dir_listing,
        "files": list(seed_files),
        "project_type": project_type,
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
                result = await _handle_investigate(task, working_dir, state)
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


async def _handle_investigate(
    task: str, working_dir: str, state: dict[str, Any]
) -> str:
    """Handle the investigate action."""
    investigation = await investigate_with_llm(
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
    embedding_budget = max(config.MAX_CONTEXT_CHARS // 5, 5000)
    embedding_context = await _query_embedding_context(
        task,
        working_dir,
        state.get("files", []),
        max_chars=embedding_budget,
    )

    coding_plan, file_contents = await plan(
        task,
        working_dir,
        state.get("dir_listing", []),
        state.get("files", []),
        embedding_context=embedding_context,
    )
    state["coding_plan"] = coding_plan
    state["files"] = file_contents
    return f"Plan: {coding_plan.summary}"


async def _handle_validate_plan(task: str, state: dict[str, Any]) -> str:
    """Handle the validate_plan action."""
    coding_plan = state.get("coding_plan")
    if not coding_plan:
        raise ValueError("No plan available. Run 'plan' first.")

    # Build context from files in state
    files_in_context = state.get("files", [])
    context = ""
    if files_in_context:
        context = f"Files in context: {', '.join(fc.path for fc in files_in_context)}"

    validation = await validate_plan(
        task=task,
        plan_summary=coding_plan.summary,
        steps=coding_plan.steps,
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

    # Run syntax checks on generated patches
    syntax_errors: list[str] = []
    project_type = state.get("project_type", "unknown")
    files = state.get("files", [])

    for patch in generated.patches:
        if patch.action == "create" or not patch.replacements:
            content = patch.content
        else:
            original = _find_file_content(patch.path, files)
            if original is None:
                continue
            try:
                content = apply_replacements(original, patch.replacements, patch.path)
            except ReplacementError:
                continue

        if not content:
            continue
        error = check_syntax(patch, project_type, content)
        if error:
            syntax_errors.append(error)

    if syntax_errors:
        state["syntax_errors"] = syntax_errors
    else:
        state.pop("syntax_errors", None)

    return f"Generated {len(generated.patches)} patches: {', '.join(p.path for p in generated.patches)}"


def _handle_write(working_dir: str, state: dict[str, Any]) -> str:
    """Handle the write action."""
    generated = state.get("generated")
    if not generated or not generated.patches:
        raise ValueError("No patches to write. Run 'generate' first.")

    written, backup_paths = apply(generated, working_dir)
    state["written"] = written
    state["backup_paths"] = backup_paths
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
        # Clean up backups on successful validation
        backup_paths = state.pop("backup_paths", [])
        cleanup_backups(backup_paths)
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
