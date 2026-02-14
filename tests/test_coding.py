"""Unit tests for axono.coding."""

import json
import os
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from axono import coding
from axono.coding import (
    FileContent,
    FilePatch,
    GeneratedCode,
    Replacement,
    ReplacementError,
    _backup_file,
    _build_final_summary,
    _build_iterative_prompt,
    _check_python_syntax,
    _check_syntax_external,
    _dedupe_file_contents,
    _extension_weight,
    _find_file_content,
    _handle_generate,
    _handle_investigate,
    _handle_plan,
    _handle_read_files,
    _handle_validate,
    _handle_validate_plan,
    _handle_write,
    _is_command_available,
    _llm_rank_files,
    _plan_next_action,
    _query_embedding_context,
    _read_file,
    _read_seed_files,
    _read_snippet,
    _recency_score,
    _scan_directory,
    _score_path_keywords,
    _score_snippet,
    _select_seed_files,
    _tokenize,
    _total_chars,
    _validate_patch_path,
    _write_file,
    apply,
    apply_replacements,
    check_syntax,
    cleanup_backups,
    generate,
    investigate,
    investigate_with_llm,
    plan,
    restore_backups,
    run_coding_pipeline,
    validate,
)
from axono.pipeline import (
    FinalValidation,
    Investigation,
    Plan,
    PlanStep,
    PlanValidation,
)

# ---------------------------------------------------------------------------
# FileContent
# ---------------------------------------------------------------------------


class TestFileContent:

    def test_basic(self):
        fc = FileContent(path="test.py", content="code")
        assert fc.path == "test.py"
        assert fc.content == "code"


# ---------------------------------------------------------------------------
# FilePatch
# ---------------------------------------------------------------------------


class TestFilePatch:

    def test_default_action(self):
        patch = FilePatch(path="test.py", content="code")
        assert patch.action == "update"

    def test_create_action(self):
        patch = FilePatch(path="new.py", content="code", action="create")
        assert patch.action == "create"

    def test_default_replacements(self):
        patch = FilePatch(path="test.py", content="code")
        assert patch.replacements == []

    def test_with_replacements(self):
        reps = [Replacement(search="old", replace="new")]
        patch = FilePatch(path="test.py", replacements=reps)
        assert len(patch.replacements) == 1
        assert patch.content == ""


# ---------------------------------------------------------------------------
# Replacement / ReplacementError
# ---------------------------------------------------------------------------


class TestReplacement:

    def test_basic(self):
        r = Replacement(search="hello", replace="world")
        assert r.search == "hello"
        assert r.replace == "world"


class TestReplacementError:

    def test_message(self):
        err = ReplacementError("file.py", "missing text")
        assert "file.py" in str(err)
        assert "missing text" in str(err)

    def test_long_search_truncated(self):
        long_search = "x" * 100
        err = ReplacementError("file.py", long_search)
        assert "..." in str(err)
        assert len(str(err)) < 200


class TestApplyReplacements:

    def test_single_replacement(self):
        result = apply_replacements("hello world", [Replacement("hello", "hi")], "f.py")
        assert result == "hi world"

    def test_multiple_replacements(self):
        result = apply_replacements(
            "aaa bbb ccc",
            [Replacement("aaa", "AAA"), Replacement("ccc", "CCC")],
            "f.py",
        )
        assert result == "AAA bbb CCC"

    def test_first_occurrence_only(self):
        result = apply_replacements("foo bar foo", [Replacement("foo", "baz")], "f.py")
        assert result == "baz bar foo"

    def test_search_not_found_raises(self):
        with pytest.raises(coding.ReplacementError, match="file.py"):
            apply_replacements("hello", [Replacement("missing", "x")], "file.py")

    def test_empty_replacements(self):
        result = apply_replacements("hello", [], "f.py")
        assert result == "hello"

    def test_sequential_application(self):
        """Each replacement sees the result of the previous one."""
        result = apply_replacements(
            "a + b",
            [Replacement("a + b", "c"), Replacement("c", "d")],
            "f.py",
        )
        assert result == "d"


# ---------------------------------------------------------------------------
# GeneratedCode
# ---------------------------------------------------------------------------


class TestGeneratedCode:

    def test_defaults(self):
        gc = GeneratedCode()
        assert gc.patches == []
        assert gc.explanation == ""

    def test_with_patches(self):
        patches = [FilePatch(path="a.py", content="code")]
        gc = GeneratedCode(patches=patches)
        assert len(gc.patches) == 1


# ---------------------------------------------------------------------------
# _scan_directory
# ---------------------------------------------------------------------------


class TestScanDirectory:

    def test_basic_scan(self, tmp_path):
        (tmp_path / "file1.py").write_text("code")
        (tmp_path / "file2.txt").write_text("text")

        result = _scan_directory(str(tmp_path))

        assert "file1.py" in result
        assert "file2.txt" in result

    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.py").write_text("")

        result = _scan_directory(str(tmp_path))

        assert ".hidden" not in result
        assert "visible.py" in result

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "file.py").write_text("")

        visible = tmp_path / "visible"
        visible.mkdir()
        (visible / "file.py").write_text("")

        result = _scan_directory(str(tmp_path))

        assert not any(".hidden" in p for p in result)
        assert any("visible" in p for p in result)

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "file.js").write_text("")
        (tmp_path / "src.js").write_text("")

        result = _scan_directory(str(tmp_path))

        assert not any("node_modules" in p for p in result)
        assert "src.js" in result

    def test_skips_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "cache.pyc").write_text("")
        (tmp_path / "main.py").write_text("")

        result = _scan_directory(str(tmp_path))

        assert not any("__pycache__" in p for p in result)
        assert "main.py" in result

    def test_max_depth(self, tmp_path):
        # Create deep structure
        d1 = tmp_path / "d1"
        d2 = d1 / "d2"
        d3 = d2 / "d3"
        d4 = d3 / "d4"
        d4.mkdir(parents=True)

        (d1 / "f1.py").write_text("")
        (d2 / "f2.py").write_text("")
        (d3 / "f3.py").write_text("")
        (d4 / "f4.py").write_text("")

        result = _scan_directory(str(tmp_path), max_depth=3)

        # Depth 0, 1, 2 are processed (files in tmp_path, d1, d2)
        # Depth 3 (d3) triggers continue, so f3.py not included
        assert any("f1.py" in p for p in result)
        assert any("f2.py" in p for p in result)
        # Depth 3+ should NOT be included
        assert not any("f3.py" in p for p in result)
        assert not any("f4.py" in p for p in result)

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        (tmp_path / "m.py").write_text("")

        result = _scan_directory(str(tmp_path))

        assert result == sorted(result)


# ---------------------------------------------------------------------------
# _read_file
# ---------------------------------------------------------------------------


class TestReadFile:

    def test_reads_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")

        result = _read_file(str(f))

        assert result == "content"

    def test_nonexistent_returns_none(self, tmp_path):
        result = _read_file(str(tmp_path / "nonexistent.txt"))
        assert result is None

    def test_binary_returns_content_with_errors_replaced(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02hello\xff")

        result = _read_file(str(f))

        # Should return something (errors="replace")
        assert result is not None
        assert "hello" in result


# ---------------------------------------------------------------------------
# _write_file
# ---------------------------------------------------------------------------


class TestWriteFile:

    def test_writes_file(self, tmp_path):
        f = tmp_path / "test.txt"
        _write_file(str(f), "content")

        assert f.read_text() == "content"

    def test_creates_dirs(self, tmp_path):
        f = tmp_path / "nested" / "dir" / "test.txt"
        _write_file(str(f), "content")

        assert f.read_text() == "content"

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("old")
        _write_file(str(f), "new")

        assert f.read_text() == "new"


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:

    def test_basic(self):
        result = _tokenize("add user authentication")
        assert "add" in result
        assert "user" in result
        assert "authentication" in result

    def test_filters_short(self):
        # _tokenize keeps tokens with len >= 3
        result = _tokenize("a an the add")
        assert "a" not in result  # len 1 < 3
        assert "an" not in result  # len 2 < 3
        assert "the" in result  # len 3 >= 3 (kept!)
        assert "add" in result  # len 3 >= 3

    def test_lowercase(self):
        result = _tokenize("ADD User AUTHENTICATION")
        assert all(t.islower() for t in result)


# ---------------------------------------------------------------------------
# _extension_weight
# ---------------------------------------------------------------------------


class TestExtensionWeight:

    def test_python(self):
        assert _extension_weight("test.py") == 2.5

    def test_typescript(self):
        assert _extension_weight("test.ts") == 2.2

    def test_unknown(self):
        assert _extension_weight("test.xyz") == 0.8


# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------


class TestRecencyScore:

    def test_recent(self):
        now = time.time()
        score = _recency_score(now)
        assert score > 0.9

    def test_old(self):
        old = time.time() - 30 * 86400  # 30 days ago
        score = _recency_score(old)
        assert score < 0.1


# ---------------------------------------------------------------------------
# _score_path_keywords
# ---------------------------------------------------------------------------


class TestScorePathKeywords:

    def test_matches(self):
        score = _score_path_keywords("src/user/auth.py", ["user", "auth"])
        assert score == 2.0

    def test_no_matches(self):
        score = _score_path_keywords("src/utils.py", ["auth", "login"])
        assert score == 0.0


# ---------------------------------------------------------------------------
# _read_snippet
# ---------------------------------------------------------------------------


class TestReadSnippet:

    def test_reads_snippet(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x" * 5000)

        result = _read_snippet(str(f), max_chars=100)

        assert len(result) == 100

    def test_nonexistent_returns_empty(self, tmp_path):
        result = _read_snippet(str(tmp_path / "nonexistent.txt"))
        assert result == ""


# ---------------------------------------------------------------------------
# _score_snippet
# ---------------------------------------------------------------------------


class TestScoreSnippet:

    def test_matches(self):
        snippet = "def authenticate_user(): pass"
        score = _score_snippet(snippet, ["authenticate", "user"])
        assert score == 3.0  # 1.5 * 2

    def test_empty_snippet(self):
        score = _score_snippet("", ["auth"])
        assert score == 0.0


# ---------------------------------------------------------------------------
# _select_seed_files
# ---------------------------------------------------------------------------


class TestSelectSeedFiles:

    def test_selects_readme(self):
        listing = ["README.md", "src.py"]
        result = _select_seed_files(listing)
        assert "README.md" in result

    def test_selects_package_json(self):
        listing = ["package.json", "src.js"]
        result = _select_seed_files(listing)
        assert "package.json" in result

    def test_none_available(self):
        listing = ["random.txt"]
        result = _select_seed_files(listing)
        assert result == []


# ---------------------------------------------------------------------------
# _read_seed_files
# ---------------------------------------------------------------------------


class TestReadSeedFiles:

    def test_reads_seeds(self, tmp_path):
        (tmp_path / "README.md").write_text("# Hello")
        (tmp_path / "other.py").write_text("code")

        result = _read_seed_files(str(tmp_path), ["README.md", "other.py"])

        assert len(result) == 1
        assert result[0].path == "README.md"

    def test_skips_unreadable(self, tmp_path):
        # Just make sure it doesn't crash
        result = _read_seed_files(str(tmp_path), ["README.md"])
        assert result == []


# ---------------------------------------------------------------------------
# _dedupe_file_contents
# ---------------------------------------------------------------------------


class TestDedupeFileContents:

    def test_removes_dupes(self):
        files = [
            FileContent(path="a.py", content="1"),
            FileContent(path="b.py", content="2"),
            FileContent(path="a.py", content="3"),  # duplicate
        ]
        result = _dedupe_file_contents(files)

        assert len(result) == 2
        assert result[0].content == "1"  # keeps first


# ---------------------------------------------------------------------------
# _total_chars
# ---------------------------------------------------------------------------


class TestTotalChars:

    def test_sums_content(self):
        files = [
            FileContent(path="a.py", content="12345"),
            FileContent(path="b.py", content="123"),
        ]
        result = _total_chars(files)
        assert result == 8


# ---------------------------------------------------------------------------
# investigate
# ---------------------------------------------------------------------------


class TestInvestigate:

    def test_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", False)
        seeds = [FileContent(path="readme.md", content="hello")]

        result = investigate("task", str(tmp_path), [], seeds)

        assert result.files == seeds
        assert "disabled" in result.summary.lower()

    def test_selects_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "auth.py").write_text("def login(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        result = investigate(
            "implement authentication",
            str(tmp_path),
            ["auth.py", "utils.py"],
            [],
        )

        # auth.py should score higher for "authentication"
        assert len(result.files) > 0

    def test_respects_budget(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 1)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "a.py").write_text("x" * 100)
        (tmp_path / "b.py").write_text("y" * 100)

        seeds = [FileContent(path="seed.py", content="seed")]
        result = investigate(
            "task",
            str(tmp_path),
            ["a.py", "b.py"],
            seeds,
        )

        # Budget is 1, seed takes it
        assert len(result.files) == 1

    def test_skips_over_budget(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 50)

        (tmp_path / "big.py").write_text("x" * 100)
        (tmp_path / "small.py").write_text("y" * 10)

        result = investigate(
            "task",
            str(tmp_path),
            ["big.py", "small.py"],
            [],
        )

        # big.py is over budget
        assert any("over budget" in s for s in result.skipped)

    def test_budget_already_full(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 1)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        seeds = [FileContent(path="seed.py", content="seed")]

        result = investigate(
            "task",
            str(tmp_path),
            [],
            seeds,
        )

        assert "budget" in result.summary.lower()


# ---------------------------------------------------------------------------
# _llm_rank_files
# ---------------------------------------------------------------------------


def _fake_llm(content):
    """Return a mock LLM whose ``ainvoke`` resolves to the given content."""
    response = SimpleNamespace(content=content)
    llm = mock.AsyncMock()
    llm.ainvoke.return_value = response
    return llm


class TestLlmRankFiles:

    @pytest.mark.asyncio
    async def test_ranks_files(self):
        resp = json.dumps(["b.py", "a.py", "c.py"])
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _llm_rank_files("task", ["a.py", "b.py", "c.py"], "/tmp")

        assert result == ["b.py", "a.py", "c.py"]

    @pytest.mark.asyncio
    async def test_adds_missing_candidates(self):
        """LLM omits some files; they appear at the end."""
        resp = json.dumps(["b.py"])
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _llm_rank_files("task", ["a.py", "b.py", "c.py"], "/tmp")

        assert result[0] == "b.py"
        assert set(result) == {"a.py", "b.py", "c.py"}

    @pytest.mark.asyncio
    async def test_filters_invalid_entries(self):
        """LLM returns paths not in candidates; they are ignored."""
        resp = json.dumps(["b.py", "nonexistent.py", "a.py"])
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _llm_rank_files("task", ["a.py", "b.py"], "/tmp")

        assert result == ["b.py", "a.py"]

    @pytest.mark.asyncio
    async def test_parse_failure_returns_original(self):
        llm = _fake_llm("not json")

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _llm_rank_files("task", ["a.py", "b.py"], "/tmp")

        assert result == ["a.py", "b.py"]

    @pytest.mark.asyncio
    async def test_llm_exception_returns_original(self):
        llm = mock.AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("LLM down")

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _llm_rank_files("task", ["a.py", "b.py"], "/tmp")

        assert result == ["a.py", "b.py"]

    @pytest.mark.asyncio
    async def test_non_list_response_returns_original(self):
        resp = json.dumps({"not": "a list"})
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _llm_rank_files("task", ["a.py", "b.py"], "/tmp")

        assert result == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# investigate_with_llm
# ---------------------------------------------------------------------------


class TestInvestigateWithLlm:

    @pytest.mark.asyncio
    async def test_disabled_returns_heuristic(self, tmp_path, monkeypatch):
        """When LLM_INVESTIGATION is False, returns heuristic results."""
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.LLM_INVESTIGATION", False)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "a.py").write_text("code")

        result = await investigate_with_llm("task", str(tmp_path), ["a.py"], [])

        assert "LLM-ranked" not in result.summary

    @pytest.mark.asyncio
    async def test_enabled_ranks_files(self, tmp_path, monkeypatch):
        """When LLM_INVESTIGATION is True, files are LLM-ranked."""
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.LLM_INVESTIGATION", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "a.py").write_text("aaaa")
        (tmp_path / "b.py").write_text("bbbb")

        resp = json.dumps(["b.py", "a.py"])
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await investigate_with_llm(
                "task", str(tmp_path), ["a.py", "b.py"], []
            )

        assert "LLM-ranked" in result.summary
        assert result.files[0].path == "b.py"

    @pytest.mark.asyncio
    async def test_single_file_skips_llm(self, tmp_path, monkeypatch):
        """With 0 or 1 files, LLM ranking is skipped."""
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.LLM_INVESTIGATION", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "only.py").write_text("code")

        result = await investigate_with_llm("task", str(tmp_path), ["only.py"], [])

        assert "LLM-ranked" not in result.summary


class TestPlan:

    @pytest.mark.asyncio
    async def test_creates_plan(self, tmp_path):
        plan_json = json.dumps(
            {
                "summary": "Add authentication",
                "files_to_read": [],
                "patches": [
                    {
                        "path": "auth.py",
                        "action": "create",
                        "description": "Create auth module",
                    }
                ],
            }
        )
        llm = _fake_llm(plan_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan(
                "Add authentication",
                str(tmp_path),
                [],
                [],
            )

        assert coding_plan.summary == "Add authentication"
        assert len(coding_plan.patches) == 1

    @pytest.mark.asyncio
    async def test_reads_extra_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "extra.py").write_text("extra code")

        plan_json = json.dumps(
            {
                "summary": "Update code",
                "files_to_read": ["extra.py"],
                "patches": [],
            }
        )
        llm = _fake_llm(plan_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan(
                "Update code",
                str(tmp_path),
                ["extra.py"],
                [],
            )

        assert any(fc.path == "extra.py" for fc in files)

    @pytest.mark.asyncio
    async def test_parse_failure_fallback(self, tmp_path):
        llm = _fake_llm("Not valid JSON at all")

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan("task", str(tmp_path), [], [])

        # Should use the raw response as summary
        assert "Not valid JSON" in coding_plan.summary

    @pytest.mark.asyncio
    async def test_includes_embedding_section(self, tmp_path):
        plan_json = json.dumps({"summary": "s", "files_to_read": [], "patches": []})
        llm = _fake_llm(plan_json)
        embedding = [
            FileContent(
                path="auth.py :: login (function, lines 1-5)",
                content="def login(): pass",
            )
        ]

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan(
                "task", str(tmp_path), [], [], embedding_context=embedding
            )

        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "Semantically related code" in user_msg
        assert "login" in user_msg

    @pytest.mark.asyncio
    async def test_no_embedding_section_when_none(self, tmp_path):
        plan_json = json.dumps({"summary": "s", "files_to_read": [], "patches": []})
        llm = _fake_llm(plan_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan("task", str(tmp_path), [], [])

        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "Semantically related code" not in user_msg

    @pytest.mark.asyncio
    async def test_no_embedding_section_when_empty(self, tmp_path):
        plan_json = json.dumps({"summary": "s", "files_to_read": [], "patches": []})
        llm = _fake_llm(plan_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan(
                "task", str(tmp_path), [], [], embedding_context=[]
            )

        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "Semantically related code" not in user_msg


# ---------------------------------------------------------------------------
# _query_embedding_context
# ---------------------------------------------------------------------------


class TestQueryEmbeddingContext:

    @pytest.mark.asyncio
    async def test_returns_relevant_chunks(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        chunk = Chunk(
            file_path=str(tmp_path / "auth.py"),
            file_md5="abc",
            symbol_name="login",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="def login(): pass",
            embedded_text="",
            metadata=ChunkMetadata(),
        )

        qs = mock.AsyncMock(return_value=[chunk])
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "auth login", str(tmp_path), [], max_chars=10000
            )

        assert len(result) == 1
        assert "auth.py" in result[0].path
        assert "login" in result[0].path
        assert "def login" in result[0].content

    @pytest.mark.asyncio
    async def test_returns_empty_when_unavailable(self):
        with mock.patch("axono.coding._query_similar", None):
            result = await _query_embedding_context("task", "/tmp", [], max_chars=10000)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self, tmp_path):
        qs = mock.AsyncMock(side_effect=RuntimeError("no FAISS"))
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), [], max_chars=10000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chunks(self, tmp_path):
        qs = mock.AsyncMock(return_value=[])
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), [], max_chars=10000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_files_already_in_context(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        chunk = Chunk(
            file_path=str(tmp_path / "existing.py"),
            file_md5="abc",
            symbol_name="func",
            symbol_type="function",
            line_start=1,
            line_end=3,
            source_code="def func(): pass",
            embedded_text="",
            metadata=ChunkMetadata(),
        )
        existing = [FileContent(path="existing.py", content="full file")]

        qs = mock.AsyncMock(return_value=[chunk])
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), existing, max_chars=10000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_respects_char_budget(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        chunks = [
            Chunk(
                file_path=str(tmp_path / f"file{i}.py"),
                file_md5="",
                symbol_name=f"f{i}",
                symbol_type="function",
                line_start=1,
                line_end=5,
                source_code="x" * 100,
                embedded_text="",
                metadata=ChunkMetadata(),
            )
            for i in range(5)
        ]

        qs = mock.AsyncMock(return_value=chunks)
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), [], max_chars=250
            )
        # Only 2 chunks fit: 100 + 100 = 200 <= 250, third would be 300 > 250
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_deduplicates_chunks(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        chunk = Chunk(
            file_path=str(tmp_path / "dup.py"),
            file_md5="",
            symbol_name="f",
            symbol_type="function",
            line_start=1,
            line_end=5,
            source_code="code",
            embedded_text="",
            metadata=ChunkMetadata(),
        )

        qs = mock.AsyncMock(return_value=[chunk, chunk])
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), [], max_chars=10000
            )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_chunk_without_symbol_name(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        chunk = Chunk(
            file_path=str(tmp_path / "module.py"),
            file_md5="",
            symbol_name=None,
            symbol_type="file",
            line_start=1,
            line_end=10,
            source_code="import os",
            embedded_text="",
            metadata=ChunkMetadata(),
        )

        qs = mock.AsyncMock(return_value=[chunk])
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), [], max_chars=10000
            )
        assert len(result) == 1
        # No symbol info in label, just the path
        assert "module.py" in result[0].path
        assert "::" not in result[0].path

    @pytest.mark.asyncio
    async def test_chunk_with_parent_class(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        chunk = Chunk(
            file_path=str(tmp_path / "models.py"),
            file_md5="",
            symbol_name="save",
            symbol_type="method",
            line_start=10,
            line_end=20,
            source_code="def save(self): pass",
            embedded_text="",
            metadata=ChunkMetadata(parent_class="User"),
        )

        qs = mock.AsyncMock(return_value=[chunk])
        with mock.patch("axono.coding._query_similar", qs):
            result = await _query_embedding_context(
                "task", str(tmp_path), [], max_chars=10000
            )
        assert len(result) == 1
        assert "User.save" in result[0].path

    def test_import_fallback_sets_none(self):
        """When folderembed is not importable, _query_similar is set to None."""
        import importlib
        import sys

        saved = sys.modules.get("axono.folderembed")
        sys.modules["axono.folderembed"] = None  # type: ignore[assignment]
        try:
            importlib.reload(coding)
            assert coding._query_similar is None
        finally:
            sys.modules["axono.folderembed"] = saved
            importlib.reload(coding)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:

    @pytest.mark.asyncio
    async def test_generates_patches(self):
        gen_json = json.dumps(
            [{"path": "new.py", "action": "create", "content": "# new file"}]
        )
        llm = _fake_llm(gen_json)

        coding_plan = Plan(
            summary="Create file",
            patches=[{"path": "new.py", "action": "create", "description": "New file"}],
        )

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert len(result.patches) == 1
        assert result.patches[0].path == "new.py"

    @pytest.mark.asyncio
    async def test_parse_failure(self):
        llm = _fake_llm("Not JSON")

        coding_plan = Plan(summary="s", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert result.patches == []
        assert "Failed to parse" in result.explanation

    @pytest.mark.asyncio
    async def test_not_a_list(self):
        llm = _fake_llm('{"not": "a list"}')

        coding_plan = Plan(summary="s", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert result.patches == []
        assert "Expected JSON array" in result.explanation

    @pytest.mark.asyncio
    async def test_duplicate_paths(self):
        gen_json = json.dumps(
            [
                {"path": "same.py", "action": "create", "content": "1"},
                {"path": "same.py", "action": "update", "content": "2"},
            ]
        )
        llm = _fake_llm(gen_json)

        coding_plan = Plan(summary="s", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert result.patches == []
        assert "duplicate" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_generates_update_with_replacements(self):
        gen_json = json.dumps(
            [
                {
                    "path": "mod.py",
                    "action": "update",
                    "replacements": [
                        {"search": "old_func", "replace": "new_func"},
                    ],
                }
            ]
        )
        llm = _fake_llm(gen_json)
        coding_plan = Plan(summary="Rename", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert len(result.patches) == 1
        assert result.patches[0].action == "update"
        assert len(result.patches[0].replacements) == 1
        assert result.patches[0].replacements[0].search == "old_func"

    @pytest.mark.asyncio
    async def test_generates_update_fallback_with_content(self):
        gen_json = json.dumps(
            [{"path": "mod.py", "action": "update", "content": "full content"}]
        )
        llm = _fake_llm(gen_json)
        coding_plan = Plan(summary="Rewrite", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert len(result.patches) == 1
        assert result.patches[0].content == "full content"
        assert result.patches[0].replacements == []

    @pytest.mark.asyncio
    async def test_skips_create_without_content(self):
        gen_json = json.dumps([{"path": "a.py", "action": "create"}])
        llm = _fake_llm(gen_json)
        coding_plan = Plan(summary="s", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert result.patches == []

    @pytest.mark.asyncio
    async def test_skips_malformed_replacements(self):
        gen_json = json.dumps(
            [
                {
                    "path": "mod.py",
                    "action": "update",
                    "replacements": [{"bad": "format"}],
                }
            ]
        )
        llm = _fake_llm(gen_json)
        coding_plan = Plan(summary="s", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        assert result.patches == []


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:

    def test_writes_create_patches(self, tmp_path):
        generated = GeneratedCode(
            patches=[
                FilePatch(path="new.py", content="code", action="create"),
            ]
        )

        result, backups = apply(generated, str(tmp_path))

        assert len(result) == 1
        assert (tmp_path / "new.py").read_text() == "code"
        assert "Created" in result[0]
        assert backups == []  # No backup for new file

    def test_writes_update_with_content_fallback(self, tmp_path):
        generated = GeneratedCode(
            patches=[
                FilePatch(path="existing.py", content="updated"),
            ]
        )

        result, backups = apply(generated, str(tmp_path))

        assert len(result) == 1
        assert (tmp_path / "existing.py").read_text() == "updated"
        assert "Updated" in result[0]

    def test_apply_with_replacements(self, tmp_path):
        (tmp_path / "mod.py").write_text("def hello():\n    return 'hello'\n")

        generated = GeneratedCode(
            patches=[
                FilePatch(
                    path="mod.py",
                    action="update",
                    replacements=[Replacement(search="'hello'", replace="'world'")],
                ),
            ]
        )

        result, backups = apply(generated, str(tmp_path))

        assert len(result) == 1
        assert "1 replacements" in result[0]
        assert (tmp_path / "mod.py").read_text() == "def hello():\n    return 'world'\n"
        assert len(backups) == 1
        assert backups[0].endswith(".axono-backup")

    def test_backup_created_for_existing_file(self, tmp_path):
        (tmp_path / "old.py").write_text("original")

        generated = GeneratedCode(
            patches=[FilePatch(path="old.py", content="new content")]
        )

        result, backups = apply(generated, str(tmp_path))

        assert len(backups) == 1
        assert os.path.isfile(backups[0])
        # Backup contains original content
        with open(backups[0]) as f:
            assert f.read() == "original"
        # File has new content
        assert (tmp_path / "old.py").read_text() == "new content"

    def test_apply_replacement_file_not_found(self, tmp_path):
        generated = GeneratedCode(
            patches=[
                FilePatch(
                    path="missing.py",
                    action="update",
                    replacements=[Replacement(search="x", replace="y")],
                ),
            ]
        )

        with pytest.raises(coding.ReplacementError, match="does not exist"):
            apply(generated, str(tmp_path))

    def test_apply_replacement_search_not_found(self, tmp_path):
        (tmp_path / "mod.py").write_text("original content")

        generated = GeneratedCode(
            patches=[
                FilePatch(
                    path="mod.py",
                    action="update",
                    replacements=[Replacement(search="missing text", replace="y")],
                ),
            ]
        )

        with pytest.raises(coding.ReplacementError, match="mod.py"):
            apply(generated, str(tmp_path))


# ---------------------------------------------------------------------------
# _validate_patch_path
# ---------------------------------------------------------------------------


class TestValidatePatchPath:

    def test_rejects_absolute_path(self, tmp_path):
        with pytest.raises(ValueError, match="Absolute path"):
            _validate_patch_path("/etc/evil.py", str(tmp_path))

    def test_rejects_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="escapes working directory"):
            _validate_patch_path("../escape.py", str(tmp_path))

    def test_rejects_deep_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="escapes working directory"):
            _validate_patch_path("sub/../../escape.py", str(tmp_path))

    def test_allows_nested_path(self, tmp_path):
        result = _validate_patch_path("sub/dir/file.py", str(tmp_path))
        assert result == os.path.join(str(tmp_path), "sub/dir/file.py")

    def test_allows_simple_path(self, tmp_path):
        result = _validate_patch_path("file.py", str(tmp_path))
        assert result == os.path.join(str(tmp_path), "file.py")


class TestApplyPathValidation:

    def test_rejects_absolute_path_in_patch(self, tmp_path):
        generated = GeneratedCode(
            patches=[FilePatch(path="/etc/evil.py", content="bad", action="create")]
        )
        with pytest.raises(ValueError, match="Absolute path"):
            apply(generated, str(tmp_path))

    def test_rejects_traversal_in_patch(self, tmp_path):
        generated = GeneratedCode(
            patches=[FilePatch(path="../escape.py", content="bad", action="create")]
        )
        with pytest.raises(ValueError, match="escapes working directory"):
            apply(generated, str(tmp_path))

    def test_allows_nested_patch(self, tmp_path):
        generated = GeneratedCode(
            patches=[FilePatch(path="sub/dir/file.py", content="ok", action="create")]
        )
        result, _ = apply(generated, str(tmp_path))
        assert len(result) == 1
        assert (tmp_path / "sub" / "dir" / "file.py").exists()


# ---------------------------------------------------------------------------
# Backup helpers
# ---------------------------------------------------------------------------


class TestBackupFile:

    def test_creates_backup(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("original")
        result = _backup_file(str(f))
        assert result is not None
        assert result.endswith(".axono-backup")
        assert os.path.isfile(result)
        with open(result) as fh:
            assert fh.read() == "original"

    def test_nonexistent_returns_none(self, tmp_path):
        result = _backup_file(str(tmp_path / "missing.py"))
        assert result is None


class TestCleanupBackups:

    def test_removes_backup_files(self, tmp_path):
        f = tmp_path / "test.py.axono-backup"
        f.write_text("backup")
        cleanup_backups([str(f)])
        assert not f.exists()

    def test_handles_missing_gracefully(self, tmp_path):
        cleanup_backups([str(tmp_path / "missing.axono-backup")])


class TestRestoreBackups:

    def test_restores_files(self, tmp_path):
        original = tmp_path / "test.py"
        backup = tmp_path / "test.py.axono-backup"
        original.write_text("modified")
        backup.write_text("original")

        restored = restore_backups([str(backup)])

        assert len(restored) == 1
        assert original.read_text() == "original"
        assert not backup.exists()

    def test_handles_missing_backup(self, tmp_path):
        restored = restore_backups([str(tmp_path / "missing.axono-backup")])
        assert restored == []


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------


class TestCheckPythonSyntax:

    def test_valid_python(self):
        result = _check_python_syntax("def foo():\n    return 1\n", "mod.py")
        assert result is None

    def test_invalid_python(self):
        result = _check_python_syntax("def foo(\n", "bad.py")
        assert result is not None
        assert "bad.py" in result

    def test_error_includes_line_number(self):
        result = _check_python_syntax("x = 1\ndef foo(\n", "err.py")
        assert result is not None
        assert ":2:" in result  # Error on line 2


class TestIsCommandAvailable:

    def test_existing_command(self):
        # python3 should be available in test environments
        assert _is_command_available("python3") is True

    def test_nonexistent_command(self):
        assert _is_command_available("nonexistent_command_xyz_123") is False


class TestCheckSyntaxExternal:

    def test_success(self):
        import subprocess

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            result = _check_syntax_external("var x = 1;", "app.js", ["node", "--check"])

        assert result is None

    def test_failure_with_stderr(self):
        import subprocess

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="SyntaxError: unexpected token"
            )
            result = _check_syntax_external("var x =", "app.js", ["node", "--check"])

        assert result is not None
        assert "app.js" in result
        assert "SyntaxError" in result

    def test_failure_no_stderr(self):
        import subprocess

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr=""
            )
            result = _check_syntax_external("bad", "app.js", ["node", "--check"])

        assert result is not None
        assert "syntax check failed" in result

    def test_timeout(self):
        import subprocess

        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("node", 10),
        ):
            result = _check_syntax_external("code", "app.js", ["node", "--check"])

        assert result is not None
        assert "syntax check error" in result

    def test_oserror(self):
        with mock.patch("subprocess.run", side_effect=OSError("not found")):
            result = _check_syntax_external("code", "app.js", ["node", "--check"])

        assert result is not None
        assert "syntax check error" in result

    def test_unlink_oserror_ignored(self):
        import subprocess

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            with mock.patch("os.unlink", side_effect=OSError("busy")):
                result = _check_syntax_external("code", "app.js", ["node", "--check"])

        assert result is None


class TestFindFileContent:

    def test_found(self):
        files = [
            FileContent(path="a.py", content="aaa"),
            FileContent(path="b.py", content="bbb"),
        ]
        assert _find_file_content("b.py", files) == "bbb"

    def test_not_found(self):
        files = [FileContent(path="a.py", content="aaa")]
        assert _find_file_content("missing.py", files) is None

    def test_empty_list(self):
        assert _find_file_content("a.py", []) is None


class TestCheckSyntax:

    def test_python_valid(self):
        patch = FilePatch(path="mod.py", content="x = 1")
        result = check_syntax(patch, "python", "x = 1")
        assert result is None

    def test_python_invalid(self):
        patch = FilePatch(path="mod.py", content="def (")
        result = check_syntax(patch, "python", "def (")
        assert result is not None
        assert "mod.py" in result

    def test_js_with_node_available(self):
        patch = FilePatch(path="app.js", content="var x = 1;")
        with mock.patch("axono.coding._is_command_available", return_value=True):
            with mock.patch(
                "axono.coding._check_syntax_external", return_value=None
            ) as mock_ext:
                result = check_syntax(patch, "node", "var x = 1;")
        mock_ext.assert_called_once_with("var x = 1;", "app.js", ["node", "--check"])
        assert result is None

    def test_js_without_node(self):
        patch = FilePatch(path="app.js", content="var x = 1;")
        with mock.patch("axono.coding._is_command_available", return_value=False):
            result = check_syntax(patch, "node", "var x = 1;")
        assert result is None

    def test_jsx_routes_to_node(self):
        patch = FilePatch(path="App.jsx", content="code")
        with mock.patch("axono.coding._is_command_available", return_value=True):
            with mock.patch(
                "axono.coding._check_syntax_external", return_value=None
            ) as mock_ext:
                check_syntax(patch, "node", "code")
        mock_ext.assert_called_once()

    def test_unsupported_extension(self):
        patch = FilePatch(path="data.csv", content="a,b,c")
        result = check_syntax(patch, "unknown", "a,b,c")
        assert result is None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:

    @pytest.mark.asyncio
    async def test_validation_passes(self):
        resp = json.dumps({"ok": True, "issues": [], "summary": "Looks good"})
        llm = _fake_llm(resp)

        coding_plan = Plan(summary="Add auth")
        generated = GeneratedCode(patches=[FilePatch(path="auth.py", content="code")])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate("Add auth", coding_plan, generated)

        assert result.ok is True
        assert result.summary == "Looks good"

    @pytest.mark.asyncio
    async def test_validation_fails(self):
        resp = json.dumps(
            {"ok": False, "issues": ["Missing tests"], "summary": "Incomplete"}
        )
        llm = _fake_llm(resp)

        coding_plan = Plan(summary="Add auth")
        generated = GeneratedCode(patches=[FilePatch(path="auth.py", content="code")])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate("Add auth", coding_plan, generated)

        assert result.ok is False
        assert "Missing tests" in result.issues

    @pytest.mark.asyncio
    async def test_validation_with_replacements(self):
        resp = json.dumps({"ok": True, "issues": [], "summary": "Edits look good"})
        llm = _fake_llm(resp)

        coding_plan = Plan(summary="Refactor")
        generated = GeneratedCode(
            patches=[
                FilePatch(
                    path="mod.py",
                    action="update",
                    replacements=[
                        Replacement(search="old_name", replace="new_name"),
                        Replacement(search="def foo", replace="def bar"),
                    ],
                )
            ]
        )

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate("Refactor", coding_plan, generated)

        assert result.ok is True
        # Verify the prompt includes replacement info
        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "UPDATE - 2 edits" in user_msg
        assert "SEARCH:" in user_msg
        assert "REPLACE:" in user_msg

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_not_ok(self):
        llm = _fake_llm("Everything looks fine!")

        coding_plan = Plan(summary="s")
        generated = GeneratedCode(patches=[FilePatch(path="a.py", content="c")])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate("task", coding_plan, generated)

        assert result.ok is False
        assert len(result.issues) >= 1
        assert "not valid JSON" in result.summary


# ---------------------------------------------------------------------------
# _build_iterative_prompt
# ---------------------------------------------------------------------------


class TestBuildIterativePrompt:

    def test_includes_goal_and_cwd(self):
        result = _build_iterative_prompt("Add feature", "/tmp/project", [], {})
        assert "Add feature" in result
        assert "/tmp/project" in result

    def test_includes_dir_listing(self):
        state = {"dir_listing": ["file1.py", "file2.py"]}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "file1.py" in result

    def test_truncates_long_listing(self):
        state = {"dir_listing": [f"file{i}.py" for i in range(50)]}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "+20 more" in result

    def test_includes_files_in_context(self):
        state = {"files": [FileContent(path="a.py", content="code")]}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "a.py" in result

    def test_includes_plan(self):
        state = {"coding_plan": Plan(summary="Do the thing")}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "Do the thing" in result

    def test_includes_plan_validation_passed(self):
        state = {"plan_validation": PlanValidation(valid=True, summary="OK")}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "PASSED" in result

    def test_includes_plan_validation_failed(self):
        state = {
            "plan_validation": PlanValidation(
                valid=False,
                issues=["Issue 1", "Issue 2"],
                suggestions=["Fix it"],
                summary="Bad plan",
            )
        }
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "FAILED" in result
        assert "Issue 1" in result

    def test_includes_generated_patches(self):
        state = {
            "generated": GeneratedCode(
                patches=[FilePatch(path="new.py", content="code")]
            )
        }
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "new.py" in result

    def test_includes_generated_patches_with_replacements(self):
        state = {
            "generated": GeneratedCode(
                patches=[
                    FilePatch(
                        path="mod.py",
                        action="update",
                        replacements=[Replacement(search="a", replace="b")],
                    )
                ]
            )
        }
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "mod.py (1 edits)" in result

    def test_includes_syntax_errors(self):
        state = {
            "syntax_errors": ["bad.py:1: invalid syntax", "err.py:5: unexpected EOF"]
        }
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "Syntax errors" in result
        assert "bad.py" in result
        assert "err.py" in result

    def test_includes_written_files(self):
        state = {"written": ["Created: new.py"]}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "Created: new.py" in result

    def test_includes_validation(self):
        state = {"validation": FinalValidation(ok=True, summary="Good")}
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "Good" in result

    def test_includes_history(self):
        history = [
            {"action": "investigate", "success": True, "reason": "Find files"},
            {"action": "plan", "success": False, "error": "Failed"},
        ]
        result = _build_iterative_prompt("task", "/tmp", history, {})
        assert "investigate" in result
        assert "plan" in result


# ---------------------------------------------------------------------------
# _plan_next_action
# ---------------------------------------------------------------------------


class TestPlanNextAction:

    @pytest.mark.asyncio
    async def test_returns_action(self):
        resp = json.dumps({"action": "investigate", "reason": "Need more context"})
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _plan_next_action("task", "/tmp", [], {})

        assert result["action"] == "investigate"

    @pytest.mark.asyncio
    async def test_returns_done(self):
        resp = json.dumps({"done": True, "summary": "All done"})
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _plan_next_action("task", "/tmp", [], {})

        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_parse_failure_returns_done(self):
        llm = _fake_llm("Not JSON")

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _plan_next_action("task", "/tmp", [], {})

        assert result["done"] is True


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


class TestHandleInvestigate:

    @pytest.mark.asyncio
    async def test_updates_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.LLM_INVESTIGATION", False)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "test.py").write_text("code")

        state = {"dir_listing": ["test.py"], "files": []}

        result = await _handle_investigate("task", str(tmp_path), state)

        assert "investigation" in state
        assert len(state["files"]) > 0


class TestHandleReadFiles:

    def test_reads_files(self, tmp_path):
        (tmp_path / "file.py").write_text("content")

        state = {"files": []}

        result = _handle_read_files(str(tmp_path), ["file.py"], state)

        assert len(state["files"]) == 1
        assert "Read 1 files" in result

    def test_skips_existing(self, tmp_path):
        (tmp_path / "file.py").write_text("content")

        state = {"files": [FileContent(path="file.py", content="old")]}

        result = _handle_read_files(str(tmp_path), ["file.py"], state)

        assert len(state["files"]) == 1
        assert "No new files" in result


class TestHandlePlan:

    @pytest.mark.asyncio
    async def test_creates_plan(self, tmp_path):
        plan_json = json.dumps({"summary": "Plan", "files_to_read": [], "patches": []})
        llm = _fake_llm(plan_json)

        state = {"dir_listing": [], "files": []}

        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding._query_similar", return_value=[]):
                result = await _handle_plan("task", str(tmp_path), state)

        assert "coding_plan" in state
        assert "Plan" in result

    @pytest.mark.asyncio
    async def test_integrates_embedding_context(self, tmp_path):
        from axono.folderembed import Chunk, ChunkMetadata

        plan_json = json.dumps(
            {"summary": "Plan with embeddings", "files_to_read": [], "patches": []}
        )
        llm = _fake_llm(plan_json)

        chunk = Chunk(
            file_path=str(tmp_path / "related.py"),
            file_md5="",
            symbol_name="helper",
            symbol_type="function",
            line_start=1,
            line_end=3,
            source_code="def helper(): pass",
            embedded_text="",
            metadata=ChunkMetadata(),
        )

        state = {"dir_listing": [], "files": []}

        qs = mock.AsyncMock(return_value=[chunk])
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding._query_similar", qs):
                result = await _handle_plan("task", str(tmp_path), state)

        assert "coding_plan" in state
        # Verify the LLM prompt included embedding context
        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "Semantically related code" in user_msg
        assert "helper" in user_msg

    @pytest.mark.asyncio
    async def test_handles_embedding_failure(self, tmp_path):
        plan_json = json.dumps({"summary": "Plan", "files_to_read": [], "patches": []})
        llm = _fake_llm(plan_json)

        state = {"dir_listing": [], "files": []}

        qs = mock.AsyncMock(side_effect=RuntimeError("FAISS broken"))
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding._query_similar", qs):
                result = await _handle_plan("task", str(tmp_path), state)

        # Should still succeed without embedding context
        assert "coding_plan" in state


class TestHandleValidatePlan:

    @pytest.mark.asyncio
    async def test_validates_plan(self, tmp_path):
        validation_json = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "OK",
            }
        )
        state = {
            "coding_plan": Plan(
                summary="s",
                patches=[{"path": "a.py", "action": "create"}],
                steps=[PlanStep(description="Create a.py")],
            ),
            "files": [],
        }

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=validation_json,
        ):
            result = await _handle_validate_plan("task", state)

        assert "plan_validation" in state
        assert "validated" in result.lower()

    @pytest.mark.asyncio
    async def test_no_plan_raises(self):
        state = {}

        with pytest.raises(ValueError, match="No plan available"):
            await _handle_validate_plan("task", state)


class TestHandleGenerate:

    @pytest.mark.asyncio
    async def test_generates_code(self):
        gen_json = json.dumps(
            [{"path": "new.py", "action": "create", "content": "code"}]
        )
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [],
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _handle_generate(state)

        assert "generated" in state
        assert "new.py" in result

    @pytest.mark.asyncio
    async def test_no_plan_raises(self):
        state = {}

        with pytest.raises(ValueError, match="No plan available"):
            await _handle_generate(state)

    @pytest.mark.asyncio
    async def test_no_patches_raises(self):
        llm = _fake_llm("[]")

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [],
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            with pytest.raises(ValueError, match="No patches generated"):
                await _handle_generate(state)

    @pytest.mark.asyncio
    async def test_syntax_errors_stored(self):
        gen_json = json.dumps(
            [{"path": "bad.py", "action": "create", "content": "def foo(\n"}]
        )
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [],
            "project_type": "python",
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _handle_generate(state)

        assert "syntax_errors" in state
        assert len(state["syntax_errors"]) == 1
        assert "bad.py" in state["syntax_errors"][0]

    @pytest.mark.asyncio
    async def test_no_syntax_errors_clears_state(self):
        gen_json = json.dumps(
            [{"path": "good.py", "action": "create", "content": "x = 1\n"}]
        )
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [],
            "project_type": "python",
            "syntax_errors": ["previous error"],
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            await _handle_generate(state)

        assert "syntax_errors" not in state

    @pytest.mark.asyncio
    async def test_syntax_check_with_replacements(self):
        gen_json = json.dumps(
            [
                {
                    "path": "mod.py",
                    "action": "update",
                    "replacements": [{"search": "good", "replace": "def (\n"}],
                }
            ]
        )
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [FileContent(path="mod.py", content="good")],
            "project_type": "python",
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            await _handle_generate(state)

        assert "syntax_errors" in state
        assert len(state["syntax_errors"]) == 1

    @pytest.mark.asyncio
    async def test_syntax_check_skips_missing_original(self):
        """When original file not in context, skip syntax check for replacements."""
        gen_json = json.dumps(
            [
                {
                    "path": "unknown.py",
                    "action": "update",
                    "replacements": [{"search": "x", "replace": "def (\n"}],
                }
            ]
        )
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [],
            "project_type": "python",
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            await _handle_generate(state)

        # No syntax errors because we couldn't resolve the content
        assert "syntax_errors" not in state

    @pytest.mark.asyncio
    async def test_syntax_check_skips_replacement_error(self):
        """When replacements fail to apply, skip syntax check."""
        gen_json = json.dumps(
            [
                {
                    "path": "mod.py",
                    "action": "update",
                    "replacements": [{"search": "missing", "replace": "new"}],
                }
            ]
        )
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [FileContent(path="mod.py", content="original")],
            "project_type": "python",
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            await _handle_generate(state)

        # No syntax errors because replacement failed
        assert "syntax_errors" not in state

    @pytest.mark.asyncio
    async def test_syntax_check_skips_empty_content(self):
        """Patches with no content skip syntax check."""
        gen_json = json.dumps([{"path": "empty.py", "action": "create", "content": ""}])
        llm = _fake_llm(gen_json)

        state = {
            "coding_plan": Plan(summary="s", patches=[]),
            "files": [],
            "project_type": "python",
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            await _handle_generate(state)

        assert "syntax_errors" not in state


class TestHandleWrite:

    def test_writes_files(self, tmp_path):
        state = {
            "generated": GeneratedCode(
                patches=[FilePatch(path="new.py", content="code")]
            )
        }

        result = _handle_write(str(tmp_path), state)

        assert "written" in state
        assert "new.py" in result
        assert (tmp_path / "new.py").exists()

    def test_no_generated_raises(self, tmp_path):
        state = {}

        with pytest.raises(ValueError, match="No patches to write"):
            _handle_write(str(tmp_path), state)


class TestHandleValidate:

    @pytest.mark.asyncio
    async def test_validates(self):
        resp = json.dumps({"ok": True, "issues": [], "summary": "Good"})
        llm = _fake_llm(resp)

        state = {
            "coding_plan": Plan(summary="s"),
            "generated": GeneratedCode(patches=[FilePatch(path="a.py", content="c")]),
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _handle_validate("task", state)

        assert "validation" in state
        assert "passed" in result.lower()

    @pytest.mark.asyncio
    async def test_no_plan_raises(self):
        state = {}

        with pytest.raises(ValueError, match="Nothing to validate"):
            await _handle_validate("task", state)


# ---------------------------------------------------------------------------
# _build_final_summary
# ---------------------------------------------------------------------------


class TestBuildFinalSummary:

    def test_includes_summary(self):
        result = _build_final_summary({}, "Done!")
        assert "Done!" in result

    def test_includes_plan(self):
        state = {"coding_plan": Plan(summary="Add feature")}
        result = _build_final_summary(state, "Done")
        assert "Add feature" in result

    def test_includes_written(self):
        state = {"written": ["Created: new.py", "Updated: old.py"]}
        result = _build_final_summary(state, "Done")
        assert "Created: new.py" in result
        assert "Updated: old.py" in result

    def test_includes_validation_passed(self):
        state = {"validation": FinalValidation(ok=True, summary="Good")}
        result = _build_final_summary(state, "Done")
        assert "Passed" in result
        assert "Good" in result

    def test_includes_validation_failed(self):
        state = {
            "validation": FinalValidation(ok=False, issues=["Issue 1"], summary="Bad")
        }
        result = _build_final_summary(state, "Done")
        assert "Issues found" in result
        assert "Issue 1" in result


# ---------------------------------------------------------------------------
# run_coding_pipeline
# ---------------------------------------------------------------------------


class TestRunCodingPipeline:

    @pytest.mark.asyncio
    async def test_done_immediately(self, tmp_path):
        resp = json.dumps({"done": True, "summary": "Nothing to do"})
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "result" in types

    @pytest.mark.asyncio
    async def test_investigate_action(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        (tmp_path / "test.py").write_text("code")

        call_count = 0

        def make_resp():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"action": "investigate", "reason": "Find files"})
            return json.dumps({"done": True, "summary": "Done"})

        async def fake_ainvoke(messages):
            return SimpleNamespace(content=make_resp())

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        assert "result" in types

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path):
        call_count = 0

        def make_resp():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"action": "unknown_action", "reason": "test"})
            return json.dumps({"done": True, "summary": "Done"})

        async def fake_ainvoke(messages):
            return SimpleNamespace(content=make_resp())

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        error_events = [e for e in events if e[0] == "error"]
        assert any("Unknown action" in e[1] for e in error_events)

    @pytest.mark.asyncio
    async def test_planning_exception(self, tmp_path):
        with mock.patch("axono.coding.get_llm", side_effect=RuntimeError("LLM down")):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "error" in types

    @pytest.mark.asyncio
    async def test_no_action_provided(self, tmp_path):
        resp = json.dumps({"reason": "no action"})  # Missing action
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        error_events = [e for e in events if e[0] == "error"]
        assert any("No action" in e[1] for e in error_events)

    @pytest.mark.asyncio
    async def test_max_steps(self, tmp_path):
        """Pipeline stops after max steps."""
        # Always return investigate action
        resp = json.dumps({"action": "investigate", "reason": "keep going"})
        llm = _fake_llm(resp)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "Stopped after" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_action_exception_continues(self, tmp_path):
        """Action exception allows LLM to decide next step."""
        call_count = 0

        def make_resp():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"action": "generate", "reason": "Generate"})
            return json.dumps({"done": True, "summary": "Done"})

        async def fake_ainvoke(messages):
            return SimpleNamespace(content=make_resp())

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        # Should have error but also continue to result
        types = [e[0] for e in events]
        assert "error" in types
        assert "result" in types

    @pytest.mark.asyncio
    async def test_full_flow(self, tmp_path, monkeypatch):
        """Test complete flow: plan -> validate_plan -> generate -> write -> validate -> done."""
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100000)

        call_count = 0

        plan_json = json.dumps(
            {
                "summary": "Create file",
                "files_to_read": [],
                "patches": [
                    {"path": "new.py", "action": "create", "description": "New"}
                ],
            }
        )
        plan_valid = json.dumps(
            {"valid": True, "issues": [], "suggestions": [], "summary": "OK"}
        )
        gen_json = json.dumps(
            [{"path": "new.py", "action": "create", "content": "# new file"}]
        )
        validate_json = json.dumps({"ok": True, "issues": [], "summary": "Good"})

        def make_resp(system_content):
            nonlocal call_count
            call_count += 1

            # Route based on call count and content
            if call_count == 1:
                return json.dumps({"action": "plan", "reason": "Create plan"})
            elif call_count == 2:
                return plan_json
            elif call_count == 3:
                return json.dumps({"action": "validate_plan", "reason": "Validate"})
            elif call_count == 4:
                return plan_valid
            elif call_count == 5:
                return json.dumps({"action": "generate", "reason": "Generate code"})
            elif call_count == 6:
                return gen_json
            elif call_count == 7:
                return json.dumps({"action": "write", "reason": "Write files"})
            elif call_count == 8:
                return json.dumps({"action": "validate", "reason": "Validate code"})
            elif call_count == 9:
                return validate_json
            else:
                return json.dumps({"done": True, "summary": "Complete"})

        async def fake_stream(messages, model_type="instruction"):
            return make_resp(messages[0].content)

        async def fake_ainvoke(messages):
            return SimpleNamespace(content=make_resp(messages[0].content))

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.pipeline.stream_response", side_effect=fake_stream):
                with mock.patch("axono.coding._query_similar", return_value=[]):
                    events = []
                    async for ev in run_coding_pipeline("Create new.py", str(tmp_path)):
                        events.append(ev)

        # Should have written the file
        assert (tmp_path / "new.py").exists()

        # Should have result
        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1


# ---------------------------------------------------------------------------
# Additional coverage tests for edge cases
# ---------------------------------------------------------------------------


class TestInvestigateEdgeCases:
    """Cover remaining edge cases in investigate()."""

    def test_os_stat_oserror(self, tmp_path):
        """Files that exist in listing but fail os.stat are skipped (line 262-263)."""
        (tmp_path / "good.py").write_text("code")
        listing = ["good.py", "ghost.py"]  # ghost doesn't exist

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate("task", str(tmp_path), listing, [])

        # good.py should be included, ghost.py should be silently skipped
        paths = [fc.path for fc in result.files]
        assert "good.py" in paths
        assert "ghost.py" not in paths

    def test_file_already_in_existing_set(self, tmp_path):
        """Files already in existing set are skipped (line 258)."""
        (tmp_path / "seed.py").write_text("seed content")
        (tmp_path / "other.py").write_text("other content")
        seeds = [FileContent("seed.py", "seed content")]
        listing = ["seed.py", "other.py"]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate("task", str(tmp_path), listing, seeds)

        # Should not have duplicates
        paths = [fc.path for fc in result.files]
        assert paths.count("seed.py") == 1

    def test_remaining_chars_zero_breaks(self, tmp_path):
        """When remaining_chars hits 0, the probed loop breaks (line 291)."""
        # Create a file that exactly fills the budget, then another
        (tmp_path / "exact.py").write_text("x" * 100)
        (tmp_path / "extra.py").write_text("y" * 50)
        listing = ["exact.py", "extra.py"]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100  # exact.py fills it exactly
            result = investigate("task", str(tmp_path), listing, [])

        paths = [fc.path for fc in result.files]
        # Only exact.py should be added (fills budget exactly)
        # extra.py should not be added (no room)
        assert len(result.files) <= 1


class TestRunCodingPipelineReadFiles:
    """Test the read_files action in run_coding_pipeline (lines 791-796)."""

    @pytest.mark.asyncio
    async def test_read_files_action_in_pipeline(self, tmp_path):
        """Pipeline handles read_files action and adds files to context."""
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "extra.py").write_text("print('extra')")

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(
                    content='{"action": "read_files", "files": ["extra.py"], "reason": "need context"}'
                )
            return SimpleNamespace(
                content='{"done": true, "summary": "Read files complete"}'
            )

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        # Should have status about reading files
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any("Read" in msg or "extra.py" in msg for msg in status_msgs)


# ---------------------------------------------------------------------------
# Additional coverage: investigate remaining_files break in probed loop (line 292)
# ---------------------------------------------------------------------------


class TestInvestigateRemainingFilesBreak:
    """Cover the remaining_files break inside the probed loop (line 292)."""

    def test_remaining_files_exhausted_in_probed_loop(self, tmp_path, monkeypatch):
        """When remaining_files drops to 0 mid-loop, break is hit."""
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 2)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100_000)

        (tmp_path / "a.py").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")
        (tmp_path / "c.py").write_text("ccc")

        # 1 seed → remaining_files = 1
        # probed loop picks one more → remaining_files = 0 → break
        seeds = [FileContent(path="seed.py", content="seed")]
        result = investigate(
            "task",
            str(tmp_path),
            ["a.py", "b.py", "c.py"],
            seeds,
        )
        # Should have at most 2 files (seed + 1 from probed)
        assert len(result.files) <= 2


class TestInvestigateUnreadableInProbed:
    """Cover the unreadable-file skip inside the probed loop (lines 296-297)."""

    def test_unreadable_file_skipped_in_probed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("axono.config.INVESTIGATION_ENABLED", True)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100_000)

        (tmp_path / "good.py").write_text("code")
        (tmp_path / "bad.py").write_text("code")

        original_read = coding._read_file

        def patched_read(path):
            if "bad.py" in path:
                return None
            return original_read(path)

        with mock.patch("axono.coding._read_file", side_effect=patched_read):
            result = investigate(
                "task",
                str(tmp_path),
                ["good.py", "bad.py"],
                [],
            )

        assert any("bad.py" in s and "unreadable" in s for s in result.skipped)


# ---------------------------------------------------------------------------
# Additional coverage: plan() edge cases (lines 388, 392, 394, 397)
# ---------------------------------------------------------------------------


class TestPlanExtraFilesEdgeCases:

    @pytest.mark.asyncio
    async def test_remaining_files_zero_breaks(self, tmp_path, monkeypatch):
        """When remaining_files is 0, the files_to_read loop breaks (line 388)."""
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 1)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100_000)

        (tmp_path / "extra.py").write_text("extra")

        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["extra.py"],
                "patches": [],
            }
        )
        llm = _fake_llm(plan_json)

        # 1 initial file fills MAX_CONTEXT_FILES=1 → remaining_files=0 → break
        initial = [FileContent(path="seed.py", content="seed")]
        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan("task", str(tmp_path), [], initial)

        # extra.py should NOT have been added
        assert not any(fc.path == "extra.py" for fc in files)

    @pytest.mark.asyncio
    async def test_unreadable_extra_file_skipped(self, tmp_path, monkeypatch):
        """Extra file that can't be read is skipped (line 392)."""
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100_000)

        # Don't create the file → _read_file returns None
        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["nonexistent.py"],
                "patches": [],
            }
        )
        llm = _fake_llm(plan_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan("task", str(tmp_path), [], [])

        assert not any(fc.path == "nonexistent.py" for fc in files)

    @pytest.mark.asyncio
    async def test_duplicate_extra_file_skipped(self, tmp_path, monkeypatch):
        """Extra file already in initial_files is skipped (line 394)."""
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 100_000)

        (tmp_path / "already.py").write_text("already here")

        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["already.py"],
                "patches": [],
            }
        )
        llm = _fake_llm(plan_json)

        initial = [FileContent(path="already.py", content="already here")]
        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan("task", str(tmp_path), [], initial)

        # Should still have exactly 1 copy
        assert sum(1 for fc in files if fc.path == "already.py") == 1

    @pytest.mark.asyncio
    async def test_extra_file_over_char_budget(self, tmp_path, monkeypatch):
        """Extra file exceeding char budget is skipped (lines 396-397)."""
        monkeypatch.setattr("axono.config.MAX_CONTEXT_FILES", 10)
        monkeypatch.setattr("axono.config.MAX_CONTEXT_CHARS", 10)

        (tmp_path / "big.py").write_text("x" * 100)

        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["big.py"],
                "patches": [],
            }
        )
        llm = _fake_llm(plan_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan("task", str(tmp_path), [], [])

        assert not any(fc.path == "big.py" for fc in files)


# ---------------------------------------------------------------------------
# Additional coverage: generate() with items missing path (line 452)
# ---------------------------------------------------------------------------


class TestGenerateItemsWithoutPath:

    @pytest.mark.asyncio
    async def test_items_without_path_skipped(self):
        """Items without a 'path' key or with empty path are skipped (line 450-452)."""
        gen_json = json.dumps(
            [
                {"no_path": True, "content": "ignored"},
                {"path": "", "content": "empty path"},
                "not a dict",
                {"path": "real.py", "action": "create", "content": "# real"},
            ]
        )
        llm = _fake_llm(gen_json)

        coding_plan = Plan(summary="s", patches=[])

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(coding_plan, [])

        # Items without path or with empty path are skipped in duplicate check,
        # but the final filter (line 467-475) checks "path" in item and "content" in item,
        # so the empty-path item also produces a patch. The key coverage is line 452 (continue).
        assert any(p.path == "real.py" for p in result.patches)


# ---------------------------------------------------------------------------
# Additional coverage: _build_iterative_prompt validation failed (lines 607-609)
# ---------------------------------------------------------------------------


class TestBuildIterativePromptValidationFailed:

    def test_includes_failed_validation(self):
        state = {
            "validation": FinalValidation(
                ok=False, issues=["Bug found", "Missing test"], summary="Problems"
            )
        }
        result = _build_iterative_prompt("task", "/tmp", [], state)
        assert "FAILED" in result
        assert "Problems" in result
        assert "Bug found" in result
        assert "Missing test" in result


# ---------------------------------------------------------------------------
# Additional coverage: _handle_validate_plan with files + failed (lines 839, 852-855)
# ---------------------------------------------------------------------------


class TestHandleValidatePlanEdgeCases:

    @pytest.mark.asyncio
    async def test_with_files_in_context(self):
        """files_in_context builds context string (line 839)."""
        validation_json = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "OK",
            }
        )

        state = {
            "coding_plan": Plan(
                summary="s",
                patches=[{"path": "a.py", "action": "create"}],
                steps=[PlanStep(description="Create a.py")],
            ),
            "files": [FileContent(path="a.py", content="code")],
        }

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=validation_json,
        ):
            result = await _handle_validate_plan("task", state)

        assert "validated" in result.lower()

    @pytest.mark.asyncio
    async def test_failed_validation(self):
        """Failed validation returns issues (lines 852-855)."""
        validation_json = json.dumps(
            {
                "valid": False,
                "issues": ["Missing error handling", "No tests"],
                "suggestions": [],
                "summary": "Needs work",
            }
        )

        state = {
            "coding_plan": Plan(
                summary="s",
                patches=[{"path": "a.py", "action": "create"}],
                steps=[PlanStep(description="Create a.py")],
            ),
            "files": [],
        }

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=validation_json,
        ):
            result = await _handle_validate_plan("task", state)

        assert "failed" in result.lower()
        assert "Missing error handling" in result

    @pytest.mark.asyncio
    async def test_failed_validation_no_issues(self):
        """Failed validation with empty issues list (line 853 else branch)."""
        validation_json = json.dumps(
            {
                "valid": False,
                "issues": [],
                "suggestions": [],
                "summary": "Bad",
            }
        )

        state = {
            "coding_plan": Plan(
                summary="s",
                patches=[{"path": "a.py", "action": "create"}],
                steps=[PlanStep(description="Create a.py")],
            ),
            "files": [],
        }

        with mock.patch(
            "axono.pipeline.stream_response",
            new_callable=mock.AsyncMock,
            return_value=validation_json,
        ):
            result = await _handle_validate_plan("task", state)

        assert "failed" in result.lower()
        assert "Issues found" in result


# ---------------------------------------------------------------------------
# Additional coverage: _handle_validate failed path (lines 897-898)
# ---------------------------------------------------------------------------


class TestHandleValidateFailed:

    @pytest.mark.asyncio
    async def test_validation_fails(self):
        """Failed validation returns issues (lines 897-898)."""
        resp = json.dumps(
            {
                "ok": False,
                "issues": ["Bug", "Typo"],
                "summary": "Bad code",
            }
        )
        llm = _fake_llm(resp)

        state = {
            "coding_plan": Plan(summary="s"),
            "generated": GeneratedCode(patches=[FilePatch(path="a.py", content="c")]),
        }

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await _handle_validate("task", state)

        assert "failed" in result.lower()
        assert "Bug" in result
        assert "Typo" in result
