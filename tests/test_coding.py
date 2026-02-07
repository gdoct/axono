"""Unit tests for axono.coding."""

import json
import os
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from axono.coding import (
    CodingPlan,
    FileContent,
    FilePatch,
    GeneratedCode,
    _dedupe_file_contents,
    _extension_weight,
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
    _write_file,
    apply,
    generate,
    investigate,
    plan,
    run_coding_pipeline,
    validate,
)
from axono.pipeline import coerce_response_text

# ---------------------------------------------------------------------------
# _scan_directory
# ---------------------------------------------------------------------------


class TestScanDirectory:

    def test_lists_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        result = _scan_directory(str(tmp_path))
        assert sorted(result) == ["a.py", "b.txt"]

    def test_respects_max_depth(self, tmp_path):
        deep = tmp_path / "l1" / "l2" / "l3" / "l4"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("")
        (tmp_path / "top.py").write_text("")
        result = _scan_directory(str(tmp_path), max_depth=2)
        assert "top.py" in result
        # l1/l2/l3/l4/deep.py is depth 4 → excluded at max_depth=2
        assert not any("deep.py" in r for r in result)

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("")
        (tmp_path / "visible.py").write_text("")
        result = _scan_directory(str(tmp_path))
        assert "visible.py" in result
        assert not any(".hidden" in r for r in result)

    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / ".dotfile").write_text("")
        (tmp_path / "normal.py").write_text("")
        result = _scan_directory(str(tmp_path))
        assert "normal.py" in result
        assert ".dotfile" not in result

    def test_skips_noise_dirs(self, tmp_path):
        for name in ["node_modules", "__pycache__", ".git"]:
            d = tmp_path / name
            d.mkdir()
            (d / "file.py").write_text("")
        (tmp_path / "src.py").write_text("")
        result = _scan_directory(str(tmp_path))
        assert result == ["src.py"]

    def test_returns_sorted(self, tmp_path):
        for name in ["c.py", "a.py", "b.py"]:
            (tmp_path / name).write_text("")
        result = _scan_directory(str(tmp_path))
        assert result == ["a.py", "b.py", "c.py"]

    def test_empty_directory(self, tmp_path):
        assert _scan_directory(str(tmp_path)) == []

    def test_nested_files(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("")
        result = _scan_directory(str(tmp_path))
        assert os.path.join("src", "main.py") in result


# ---------------------------------------------------------------------------
# _read_file / _write_file
# ---------------------------------------------------------------------------


class TestReadWriteFile:

    def test_read_existing(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        assert _read_file(str(f)) == "hello world"

    def test_read_missing(self, tmp_path):
        assert _read_file(str(tmp_path / "nope.txt")) is None

    def test_write_creates_parents(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        _write_file(str(target), "content")
        assert target.read_text() == "content"

    def test_write_overwrites(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("old")
        _write_file(str(f), "new")
        assert f.read_text() == "new"


# ---------------------------------------------------------------------------
# _read_snippet
# ---------------------------------------------------------------------------


class TestReadSnippet:

    def test_reads_up_to_max_chars(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("A" * 5000)
        snippet = _read_snippet(str(f), max_chars=100)
        assert len(snippet) == 100

    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_snippet(str(tmp_path / "nope.txt")) == ""


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:

    def test_splits_on_non_alnum(self):
        assert _tokenize("hello-world_foo") == ["hello", "world", "foo"]

    def test_filters_short_tokens(self):
        assert _tokenize("a ab abc abcd") == ["abc", "abcd"]

    def test_lowercases(self):
        assert _tokenize("FooBar") == ["foobar"]

    def test_empty(self):
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _extension_weight
# ---------------------------------------------------------------------------


class TestExtensionWeight:

    def test_python_file(self):
        assert _extension_weight("main.py") == 2.5

    def test_js_file(self):
        assert _extension_weight("app.js") == 2.0

    def test_unknown_extension(self):
        assert _extension_weight("data.xyz") == 0.8

    def test_no_extension(self):
        assert _extension_weight("Makefile") == 0.8


# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------


class TestRecencyScore:

    def test_just_now(self):
        score = _recency_score(time.time())
        assert score > 0.9

    def test_old_file(self):
        old_mtime = time.time() - 365 * 86_400
        score = _recency_score(old_mtime)
        assert score < 0.01

    def test_always_positive(self):
        assert _recency_score(0) > 0


# ---------------------------------------------------------------------------
# _score_path_keywords / _score_snippet
# ---------------------------------------------------------------------------


class TestScoring:

    def test_path_keyword_match(self):
        assert _score_path_keywords("src/auth/login.py", ["auth", "login"]) == 2.0

    def test_path_keyword_no_match(self):
        assert _score_path_keywords("readme.md", ["auth"]) == 0.0

    def test_snippet_match(self):
        assert _score_snippet("def authenticate(user):", ["authenticate"]) == 1.5

    def test_snippet_no_match(self):
        assert _score_snippet("hello world", ["zebra"]) == 0.0

    def test_snippet_empty(self):
        assert _score_snippet("", ["anything"]) == 0.0


# ---------------------------------------------------------------------------
# _select_seed_files / _read_seed_files
# ---------------------------------------------------------------------------


class TestSeedFiles:

    def test_select_known_files(self):
        listing = ["README.md", "src/main.py", "pyproject.toml"]
        seeds = _select_seed_files(listing)
        assert "README.md" in seeds
        assert "pyproject.toml" in seeds
        assert "src/main.py" not in seeds

    def test_select_preserves_priority_order(self):
        listing = ["pyproject.toml", "README.md", "package.json"]
        seeds = _select_seed_files(listing)
        assert seeds.index("README.md") < seeds.index("pyproject.toml")

    def test_select_none_match(self):
        assert _select_seed_files(["foo.py", "bar.rs"]) == []

    def test_read_seed_files(self, tmp_path):
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "other.py").write_text("code")
        listing = ["README.md", "other.py"]
        seeds = _read_seed_files(str(tmp_path), listing)
        assert len(seeds) == 1
        assert seeds[0].path == "README.md"
        assert seeds[0].content == "# Project"


# ---------------------------------------------------------------------------
# _dedupe_file_contents / _total_chars / coerce_response_text (from pipeline)
# ---------------------------------------------------------------------------


class TestMiscHelpers:

    def test_dedupe(self):
        files = [
            FileContent("a.py", "aaa"),
            FileContent("b.py", "bbb"),
            FileContent("a.py", "aaa-dup"),
        ]
        result = _dedupe_file_contents(files)
        assert len(result) == 2
        # First occurrence wins
        assert result[0].content == "aaa"

    def test_total_chars(self):
        files = [FileContent("a", "12345"), FileContent("b", "67890")]
        assert _total_chars(files) == 10

    def test_coerce_none(self):
        assert coerce_response_text(None) == ""

    def test_coerce_list(self):
        result = coerce_response_text([{"a": 1}])
        assert result == json.dumps([{"a": 1}])

    def test_coerce_string(self):
        assert coerce_response_text("hello") == "hello"

    def test_coerce_int(self):
        assert coerce_response_text(42) == "42"


# ---------------------------------------------------------------------------
# investigate (stage 0)
# ---------------------------------------------------------------------------


class TestInvestigate:

    def test_disabled_returns_seeds_only(self):
        seeds = [FileContent("README.md", "# Hi")]
        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = False
            result = investigate("task", "/tmp", ["README.md", "a.py"], seeds)
        assert result.files == seeds
        assert "disabled" in result.summary.lower()

    def test_seed_files_fill_budget(self):
        seeds = [FileContent(f"f{i}.py", "x" * 100) for i in range(10)]
        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 5
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate("task", "/tmp", [], seeds)
        assert "budget" in result.summary.lower()

    def test_selects_relevant_files(self, tmp_path):
        (tmp_path / "auth.py").write_text("def login(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        listing = ["auth.py", "utils.py"]
        seeds = []

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate(
                "fix authentication login", str(tmp_path), listing, seeds
            )

        paths = [fc.path for fc in result.files]
        assert "auth.py" in paths

    def test_skips_unreadable_files(self, tmp_path):
        (tmp_path / "good.py").write_text("code")
        listing = ["good.py", "ghost.py"]
        seeds = []

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate("task", str(tmp_path), listing, seeds)

        paths = [fc.path for fc in result.files]
        assert "good.py" in paths
        assert any("ghost.py" in s for s in result.skipped) or "ghost.py" not in paths

    def test_respects_char_budget(self, tmp_path):
        (tmp_path / "big.py").write_text("x" * 5000)
        (tmp_path / "small.py").write_text("y" * 10)
        listing = ["big.py", "small.py"]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100  # tiny budget
            result = investigate("task", str(tmp_path), listing, [])

        total = _total_chars(result.files)
        assert total <= 100

    def test_dedupes_output(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        seeds = [FileContent("a.py", "code")]
        listing = ["a.py"]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate("task", str(tmp_path), listing, seeds)

        paths = [fc.path for fc in result.files]
        assert paths.count("a.py") == 1

    def test_file_limit_breaks_probed_loop(self, tmp_path):
        """When remaining_files hits 0 during probed iteration, the loop breaks."""
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"content {i}")
        listing = [f"f{i}.py" for i in range(5)]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 1  # only room for 1 file
            cfg.MAX_CONTEXT_CHARS = 100_000
            result = investigate("task", str(tmp_path), listing, [])

        assert len(result.files) <= 1

    def test_unreadable_file_in_probed_loop(self, tmp_path):
        """A file that exists for os.stat but can't be read is skipped."""
        (tmp_path / "ok.py").write_text("good")
        bad = tmp_path / "bad.py"
        bad.write_text("secret")

        listing = ["ok.py", "bad.py"]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100_000
            # Make _read_file return None for bad.py only
            orig_read = _read_file

            def selective_read(path):
                if path.endswith("bad.py"):
                    return None
                return orig_read(path)

            with mock.patch("axono.coding._read_file", side_effect=selective_read):
                result = investigate("task", str(tmp_path), listing, [])

        paths = [fc.path for fc in result.files]
        assert "ok.py" in paths
        assert "bad.py" not in paths
        assert any("bad.py" in s and "unreadable" in s for s in result.skipped)

    def test_char_budget_exhausted_breaks_probed_loop(self, tmp_path):
        """When remaining_chars hits 0 after adding a file, the next iteration breaks."""
        (tmp_path / "a.py").write_text("x" * 100)  # exactly fills budget
        (tmp_path / "b.py").write_text("y" * 10)
        listing = ["a.py", "b.py"]

        with mock.patch("axono.coding.config") as cfg:
            cfg.INVESTIGATION_ENABLED = True
            cfg.MAX_CONTEXT_FILES = 10
            cfg.MAX_CONTEXT_CHARS = 100  # a.py fills it exactly, b.py triggers break
            result = investigate("task", str(tmp_path), listing, [])

        paths = [fc.path for fc in result.files]
        # a.py should be added (exactly 100 chars = budget), b.py should not
        assert len(result.files) <= 1


# ---------------------------------------------------------------------------
# plan (stage 1)
# ---------------------------------------------------------------------------


def _fake_llm(content):
    """Return a mock LLM whose ``ainvoke`` resolves to the given content."""
    response = SimpleNamespace(content=content)
    llm = mock.AsyncMock()
    llm.ainvoke.return_value = response
    return llm


class TestPlan:

    @pytest.mark.asyncio
    async def test_parses_valid_json_plan(self):
        plan_json = json.dumps(
            {
                "summary": "Add tests",
                "files_to_read": [],
                "patches": [
                    {"path": "test.py", "action": "create", "description": "new test"}
                ],
            }
        )
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, files = await plan(
                "add tests", "/tmp", ["test.py"], [FileContent("readme.md", "hi")]
            )
        assert coding_plan.summary == "Add tests"
        assert len(coding_plan.patches) == 1

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        plan_json = (
            '```json\n{"summary": "ok", "files_to_read": [], "patches": []}\n```'
        )
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, _ = await plan("task", "/tmp", [], [])
        assert coding_plan.summary == "ok"

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        llm = _fake_llm("This is not JSON at all")
        with mock.patch("axono.coding.get_llm", return_value=llm):
            coding_plan, _ = await plan("task", "/tmp", [], [])
        assert "This is not JSON at all" in coding_plan.summary
        assert coding_plan.patches == []

    @pytest.mark.asyncio
    async def test_reads_extra_files(self, tmp_path):
        (tmp_path / "extra.py").write_text("extra content")
        plan_json = json.dumps(
            {
                "summary": "Need extra",
                "files_to_read": ["extra.py"],
                "patches": [
                    {"path": "main.py", "action": "update", "description": "fix"}
                ],
            }
        )
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 100_000
                _, files = await plan("task", str(tmp_path), ["extra.py"], [])
        paths = [fc.path for fc in files]
        assert "extra.py" in paths

    @pytest.mark.asyncio
    async def test_respects_file_budget_for_extra_reads(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["a.py", "b.py"],
                "patches": [],
            }
        )
        initial = [FileContent(f"f{i}.py", "x") for i in range(8)]
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 8  # already full
                cfg.MAX_CONTEXT_CHARS = 100_000
                _, files = await plan("task", str(tmp_path), [], initial)
        # Extra files shouldn't be added since budget is full
        paths = [fc.path for fc in files]
        assert "a.py" not in paths

    @pytest.mark.asyncio
    async def test_skips_unreadable_extra_file(self, tmp_path):
        """Extra files that can't be read are silently skipped (line 411)."""
        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["ghost.py"],
                "patches": [],
            }
        )
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 100_000
                _, files = await plan("task", str(tmp_path), [], [])
        paths = [fc.path for fc in files]
        assert "ghost.py" not in paths

    @pytest.mark.asyncio
    async def test_skips_duplicate_extra_file(self, tmp_path):
        """Extra files already in initial_files are skipped (line 413)."""
        (tmp_path / "dup.py").write_text("content")
        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["dup.py"],
                "patches": [],
            }
        )
        initial = [FileContent("dup.py", "content")]
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 100_000
                _, files = await plan("task", str(tmp_path), [], initial)
        # Should not be duplicated
        assert [fc.path for fc in files].count("dup.py") == 1

    @pytest.mark.asyncio
    async def test_skips_over_budget_extra_file(self, tmp_path):
        """Extra files that exceed remaining char budget are skipped (line 416)."""
        (tmp_path / "big.py").write_text("x" * 5000)
        plan_json = json.dumps(
            {
                "summary": "s",
                "files_to_read": ["big.py"],
                "patches": [],
            }
        )
        initial = [FileContent("seed.py", "y" * 9990)]
        llm = _fake_llm(plan_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 10_000  # 9990 used, only 10 left
                _, files = await plan("task", str(tmp_path), [], initial)
        paths = [fc.path for fc in files]
        assert "big.py" not in paths


# ---------------------------------------------------------------------------
# generate (stage 2)
# ---------------------------------------------------------------------------


class TestGenerate:

    @pytest.mark.asyncio
    async def test_parses_valid_output(self):
        gen_json = json.dumps(
            [
                {"path": "foo.py", "action": "create", "content": "print('hi')"},
            ]
        )
        llm = _fake_llm(gen_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(
                CodingPlan(summary="s", patches=[]),
                [FileContent("a.py", "old")],
            )
        assert len(result.patches) == 1
        assert result.patches[0].path == "foo.py"
        assert result.patches[0].content == "print('hi')"

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        gen_json = (
            '```json\n[{"path": "a.py", "action": "update", "content": "new"}]\n```'
        )
        llm = _fake_llm(gen_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(CodingPlan(summary="s", patches=[]), [])
        assert len(result.patches) == 1

    @pytest.mark.asyncio
    async def test_invalid_json_returns_explanation(self):
        llm = _fake_llm("Not JSON")
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(CodingPlan(summary="s", patches=[]), [])
        assert result.patches == []
        assert "Failed to parse" in result.explanation

    @pytest.mark.asyncio
    async def test_duplicate_paths_rejected(self):
        gen_json = json.dumps(
            [
                {"path": "a.py", "action": "update", "content": "v1"},
                {"path": "a.py", "action": "update", "content": "v2"},
            ]
        )
        llm = _fake_llm(gen_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(CodingPlan(summary="s", patches=[]), [])
        assert result.patches == []
        assert "duplicate" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_skips_items_without_path_or_content(self):
        gen_json = json.dumps(
            [
                {"path": "a.py", "content": "good"},
                {"path": "b.py"},  # missing content
                {"content": "orphan"},  # missing path
            ]
        )
        llm = _fake_llm(gen_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(CodingPlan(summary="s", patches=[]), [])
        assert len(result.patches) == 1
        assert result.patches[0].path == "a.py"

    @pytest.mark.asyncio
    async def test_default_action_is_update(self):
        gen_json = json.dumps([{"path": "a.py", "content": "code"}])
        llm = _fake_llm(gen_json)
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(CodingPlan(summary="s", patches=[]), [])
        assert result.patches[0].action == "update"


# ---------------------------------------------------------------------------
# apply (stage 3)
# ---------------------------------------------------------------------------


class TestApply:

    def test_creates_files(self, tmp_path):
        gen = GeneratedCode(
            patches=[
                FilePatch("new.py", "print('hello')", action="create"),
            ]
        )
        results = apply(gen, str(tmp_path))
        assert "Created: new.py" in results
        assert (tmp_path / "new.py").read_text() == "print('hello')"

    def test_updates_files(self, tmp_path):
        (tmp_path / "old.py").write_text("old content")
        gen = GeneratedCode(
            patches=[
                FilePatch("old.py", "new content", action="update"),
            ]
        )
        results = apply(gen, str(tmp_path))
        assert "Updated: old.py" in results
        assert (tmp_path / "old.py").read_text() == "new content"

    def test_creates_nested_dirs(self, tmp_path):
        gen = GeneratedCode(
            patches=[
                FilePatch("a/b/c.py", "deep", action="create"),
            ]
        )
        apply(gen, str(tmp_path))
        assert (tmp_path / "a" / "b" / "c.py").read_text() == "deep"

    def test_multiple_patches(self, tmp_path):
        gen = GeneratedCode(
            patches=[
                FilePatch("a.py", "aaa", action="create"),
                FilePatch("b.py", "bbb", action="create"),
            ]
        )
        results = apply(gen, str(tmp_path))
        assert len(results) == 2

    def test_empty_patches(self, tmp_path):
        gen = GeneratedCode(patches=[])
        results = apply(gen, str(tmp_path))
        assert results == []


# ---------------------------------------------------------------------------
# validate (stage 4)
# ---------------------------------------------------------------------------


class TestValidate:

    @pytest.mark.asyncio
    async def test_valid_code(self):
        llm = _fake_llm(json.dumps({"ok": True, "issues": [], "summary": "LGTM"}))
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate(
                "task",
                CodingPlan(summary="s", patches=[]),
                GeneratedCode(patches=[FilePatch("a.py", "code")]),
            )
        assert result.ok is True
        assert result.summary == "LGTM"
        assert result.issues == []

    @pytest.mark.asyncio
    async def test_issues_found(self):
        llm = _fake_llm(
            json.dumps(
                {
                    "ok": False,
                    "issues": ["missing import", "typo"],
                    "summary": "Needs fixes",
                }
            )
        )
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate(
                "task",
                CodingPlan(summary="s", patches=[]),
                GeneratedCode(patches=[]),
            )
        assert result.ok is False
        assert len(result.issues) == 2

    @pytest.mark.asyncio
    async def test_invalid_json_defaults_to_ok(self):
        llm = _fake_llm("Everything looks great!")
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate(
                "task",
                CodingPlan(summary="s", patches=[]),
                GeneratedCode(patches=[]),
            )
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_strips_fences(self):
        llm = _fake_llm('```json\n{"ok": true, "issues": [], "summary": "ok"}\n```')
        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await validate(
                "task",
                CodingPlan(summary="s", patches=[]),
                GeneratedCode(patches=[]),
            )
        assert result.ok is True


# ---------------------------------------------------------------------------
# run_coding_pipeline (iterative orchestrator)
# ---------------------------------------------------------------------------


class TestRunCodingPipeline:
    """Tests for the iterative coding pipeline orchestrator.

    The iterative pipeline works by having an LLM decide what action to take
    next. The sequence typically goes:
    1. investigate -> find relevant files
    2. plan -> create a coding plan
    3. generate -> generate code patches
    4. write -> write patches to disk
    5. validate -> validate the changes
    6. done -> complete the task
    """

    @pytest.mark.asyncio
    async def test_full_success(self, tmp_path):
        """Test a complete successful pipeline run.

        The iterative planner calls the LLM to decide actions, and then the
        individual stage functions (plan, generate, validate) also call the LLM.
        The calls are interleaved:
        1. planner -> investigate
        2. planner -> plan
        3. plan() stage LLM call
        4. planner -> generate
        5. generate() stage LLM call
        ... and so on
        """
        (tmp_path / "README.md").write_text("# Project")

        # Responses for the individual stages
        plan_json = json.dumps(
            {
                "summary": "Add file",
                "files_to_read": [],
                "patches": [
                    {"path": "new.py", "action": "create", "description": "new"}
                ],
            }
        )
        gen_json = json.dumps(
            [
                {"path": "new.py", "action": "create", "content": "print('hi')"},
            ]
        )
        validate_json = json.dumps({"ok": True, "issues": [], "summary": "Good"})

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            # Detect what kind of call this is by looking at system message
            system_msg = messages[0].content if messages else ""

            if "coding agent that decides what action" in system_msg:
                # This is the iterative planner
                if call_count == 1:
                    return SimpleNamespace(
                        content='{"action": "investigate", "reason": "find files"}'
                    )
                elif call_count == 2:
                    return SimpleNamespace(
                        content='{"action": "plan", "reason": "create plan"}'
                    )
                elif call_count == 4:
                    return SimpleNamespace(
                        content='{"action": "generate", "reason": "generate code"}'
                    )
                elif call_count == 6:
                    return SimpleNamespace(
                        content='{"action": "write", "reason": "write files"}'
                    )
                elif call_count == 7:
                    return SimpleNamespace(
                        content='{"action": "validate", "reason": "check code"}'
                    )
                else:
                    return SimpleNamespace(
                        content='{"done": true, "summary": "Task complete"}'
                    )
            elif "planning agent" in system_msg:
                return SimpleNamespace(content=plan_json)
            elif "code generation agent" in system_msg:
                return SimpleNamespace(content=gen_json)
            elif "validation agent" in system_msg:
                return SimpleNamespace(content=validate_json)
            else:  # pragma: no cover
                # Default fallback
                return SimpleNamespace(
                    content='{"done": true, "summary": "Unknown call"}'
                )

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.INVESTIGATION_ENABLED = True
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 100_000
                events = []
                async for ev in run_coding_pipeline("add file", str(tmp_path)):
                    events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        assert "result" in types
        # The file should have been written
        assert (tmp_path / "new.py").read_text() == "print('hi')"

    @pytest.mark.asyncio
    async def test_immediate_done(self, tmp_path):
        """Test that the pipeline handles immediate done response."""
        done_response = json.dumps({"done": True, "summary": "Nothing to do"})

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = SimpleNamespace(content=done_response)

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "result" in types
        assert "error" not in types

    @pytest.mark.asyncio
    async def test_investigation_failure(self, tmp_path):
        """Test that investigation failures are handled and reported."""
        action_response = json.dumps({"action": "investigate", "reason": "find files"})

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = SimpleNamespace(content=action_response)

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.coding._scan_directory", return_value=[]):
                with mock.patch("axono.coding._read_seed_files", return_value=[]):
                    with mock.patch(
                        "axono.coding.investigate", side_effect=RuntimeError("boom")
                    ):
                        events = []
                        async for ev in run_coding_pipeline("task", str(tmp_path)):
                            events.append(ev)

        assert any(
            e[0] == "error" and "investigate failed" in e[1].lower() for e in events
        )

    @pytest.mark.asyncio
    async def test_plan_failure(self, tmp_path):
        """Test that plan failures are handled and reported."""
        action_response = json.dumps({"action": "plan", "reason": "create plan"})
        done_response = json.dumps({"done": True, "summary": "Gave up"})

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(content=action_response)
            return SimpleNamespace(content=done_response)

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.coding.plan", side_effect=RuntimeError("LLM down")):
                events = []
                async for ev in run_coding_pipeline("task", str(tmp_path)):
                    events.append(ev)

        assert any(e[0] == "error" and "plan failed" in e[1].lower() for e in events)

    @pytest.mark.asyncio
    async def test_generate_without_plan_fails(self, tmp_path):
        """Test that generate action without a plan fails gracefully."""
        action_responses = [
            json.dumps({"action": "generate", "reason": "no plan yet"}),
            json.dumps({"done": True, "summary": "Gave up"}),
        ]

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(content=action_responses[min(call_count - 1, 1)])

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        assert any(e[0] == "error" and "No plan" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_write_without_generate_fails(self, tmp_path):
        """Test that write action without generated patches fails."""
        action_responses = [
            json.dumps({"action": "write", "reason": "no patches yet"}),
            json.dumps({"done": True, "summary": "Gave up"}),
        ]

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(content=action_responses[min(call_count - 1, 1)])

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        assert any(e[0] == "error" and "No patches" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_max_steps_limit(self, tmp_path):
        """Test that the pipeline stops after max steps."""
        # Always return investigate action, never done
        action_response = json.dumps({"action": "investigate", "reason": "loop"})

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = SimpleNamespace(content=action_response)

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.INVESTIGATION_ENABLED = False
                events = []
                async for ev in run_coding_pipeline("task", str(tmp_path)):
                    events.append(ev)

        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "15 steps" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_validation_issues_in_result(self, tmp_path):
        """Test that validation issues appear in the final result."""
        (tmp_path / "README.md").write_text("# Project")

        plan_json = json.dumps(
            {
                "summary": "Fix bug",
                "files_to_read": [],
                "patches": [{"path": "a.py", "action": "update", "description": "fix"}],
            }
        )
        gen_json = json.dumps([{"path": "a.py", "action": "update", "content": "code"}])
        validate_json = json.dumps(
            {
                "ok": False,
                "issues": ["missing import", "typo"],
                "summary": "bad code",
            }
        )

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            system_msg = messages[0].content if messages else ""

            if "coding agent that decides what action" in system_msg:
                # Iterative planner decisions
                if call_count == 1:
                    return SimpleNamespace(
                        content='{"action": "plan", "reason": "plan"}'
                    )
                elif call_count == 3:
                    return SimpleNamespace(
                        content='{"action": "generate", "reason": "generate"}'
                    )
                elif call_count == 5:
                    return SimpleNamespace(
                        content='{"action": "write", "reason": "write"}'
                    )
                elif call_count == 6:
                    return SimpleNamespace(
                        content='{"action": "validate", "reason": "validate"}'
                    )
                else:
                    return SimpleNamespace(
                        content='{"done": true, "summary": "Done with issues"}'
                    )
            elif "planning agent" in system_msg:
                return SimpleNamespace(content=plan_json)
            elif "code generation agent" in system_msg:
                return SimpleNamespace(content=gen_json)
            elif "validation agent" in system_msg:
                return SimpleNamespace(content=validate_json)
            else:  # pragma: no cover
                return SimpleNamespace(content='{"done": true, "summary": "Unknown"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 100_000
                events = []
                async for ev in run_coding_pipeline("fix bug", str(tmp_path)):
                    events.append(ev)

        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 1
        assert "Issues found" in result_events[0][1]
        assert "bad code" in result_events[0][1]

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path):
        """Unknown actions yield error and continue."""
        action_responses = [
            json.dumps({"action": "unknown_action", "reason": "test"}),
            json.dumps({"done": True, "summary": "Gave up"}),
        ]

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(content=action_responses[min(call_count - 1, 1)])

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        assert any(e[0] == "error" and "Unknown action" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_read_files_action(self, tmp_path):
        """Test the read_files action."""
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "extra.py").write_text("print('extra')")

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(
                    content='{"action": "read_files", "files": ["extra.py"], "reason": "need more context"}'
                )
            return SimpleNamespace(content='{"done": true, "summary": "Read file"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        # Should have read the file
        status_events = [e for e in events if e[0] == "status"]
        assert any("Read" in e[1] or "extra.py" in e[1] for e in status_events)

    @pytest.mark.asyncio
    async def test_validate_without_plan_fails(self, tmp_path):
        """Validate action without plan/generated raises error."""
        action_responses = [
            json.dumps({"action": "validate", "reason": "check nothing"}),
            json.dumps({"done": True, "summary": "Gave up"}),
        ]

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(content=action_responses[min(call_count - 1, 1)])

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        assert any(e[0] == "error" and "Nothing to validate" in e[1] for e in events)

    @pytest.mark.asyncio
    async def test_generate_fails_with_no_patches(self, tmp_path):
        """Generate returns error when no patches generated."""
        (tmp_path / "README.md").write_text("# Project")

        plan_json = json.dumps(
            {
                "summary": "Do nothing",
                "files_to_read": [],
                "patches": [],
            }
        )
        # Generator returns empty array - no patches
        gen_json = json.dumps([])

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            system_msg = messages[0].content if messages else ""

            if "coding agent that decides what action" in system_msg:
                if call_count == 1:
                    return SimpleNamespace(
                        content='{"action": "plan", "reason": "plan"}'
                    )
                elif call_count == 3:
                    return SimpleNamespace(
                        content='{"action": "generate", "reason": "generate"}'
                    )
                else:
                    return SimpleNamespace(content='{"done": true, "summary": "Done"}')
            elif "planning agent" in system_msg:
                return SimpleNamespace(content=plan_json)
            elif "code generation agent" in system_msg:
                return SimpleNamespace(content=gen_json)
            else:  # pragma: no cover
                return SimpleNamespace(content='{"done": true, "summary": "Unknown"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.coding.config") as cfg:
                cfg.MAX_CONTEXT_FILES = 10
                cfg.MAX_CONTEXT_CHARS = 100_000
                events = []
                async for ev in run_coding_pipeline("task", str(tmp_path)):
                    events.append(ev)

        # Should have an error about no patches
        assert any(
            e[0] == "error" and ("No patches" in e[1] or "generate" in e[1].lower())
            for e in events
        )


# ---------------------------------------------------------------------------
# _build_iterative_prompt
# ---------------------------------------------------------------------------


class TestBuildIterativePrompt:

    def test_includes_dir_listing_overflow(self):
        """Prompt shows +N more when files exceed 30."""
        from axono.coding import _build_iterative_prompt

        state = {
            "dir_listing": [f"file{i}.py" for i in range(50)],
        }
        prompt = _build_iterative_prompt("task", "/tmp", [], state)

        assert "Project files:" in prompt
        assert "+20 more" in prompt

    def test_includes_history_with_errors(self):
        """Prompt shows history with errors."""
        from axono.coding import _build_iterative_prompt

        history = [
            {
                "action": "plan",
                "success": False,
                "error": "LLM failed",
                "reason": "create plan",
            },
        ]
        state = {}
        prompt = _build_iterative_prompt("task", "/tmp", history, state)

        assert "Recent actions:" in prompt
        assert "plan" in prompt
        assert "LLM failed" in prompt or "error" in prompt

    def test_includes_validation_failed(self):
        """Prompt shows failed validation with issues."""
        from axono.coding import ValidationResult, _build_iterative_prompt

        state = {
            "validation": ValidationResult(
                ok=False, issues=["missing import", "typo"], summary="bad code"
            ),
        }
        prompt = _build_iterative_prompt("task", "/tmp", [], state)

        assert "Validation: FAILED" in prompt
        assert "bad code" in prompt
        assert "missing import" in prompt


# ---------------------------------------------------------------------------
# generate — edge cases
# ---------------------------------------------------------------------------


class TestGenerateEdgeCases:

    @pytest.mark.asyncio
    async def test_json_object_instead_of_array(self):
        """Generate returns error when JSON is object instead of array."""
        from axono.coding import CodingPlan, generate

        gen_json = json.dumps({"path": "a.py", "content": "code"})  # Object, not array
        llm = _fake_llm(gen_json)

        with mock.patch("axono.coding.get_llm", return_value=llm):
            result = await generate(CodingPlan(summary="s", patches=[]), [])

        assert result.patches == []
        assert "Expected JSON array" in result.explanation


# ---------------------------------------------------------------------------
# _handle_read_files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _plan_next_action
# ---------------------------------------------------------------------------


class TestPlanNextActionCoding:

    @pytest.mark.asyncio
    async def test_returns_done_on_parse_failure(self):
        """_plan_next_action returns done when JSON can't be parsed."""
        from axono.coding import _plan_next_action

        response = SimpleNamespace(content="not valid json at all")
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            result = await _plan_next_action("task", "/tmp", [], {})

        assert result["done"] is True
        assert "parse" in result["summary"].lower()


# ---------------------------------------------------------------------------
# run_coding_pipeline — additional edge cases
# ---------------------------------------------------------------------------


class TestRunCodingPipelineEdgeCases:

    @pytest.mark.asyncio
    async def test_planning_exception_yields_error_and_returns(self, tmp_path):
        """When _plan_next_action raises, pipeline yields error and returns."""
        with mock.patch(
            "axono.coding._plan_next_action",
            side_effect=RuntimeError("LLM connection lost"),
        ):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        assert any(e[0] == "error" and "Planning failed" in e[1] for e in events)
        # Should have returned after the error
        result_events = [e for e in events if e[0] == "result"]
        assert len(result_events) == 0  # No result because we returned early

    @pytest.mark.asyncio
    async def test_empty_action_yields_error(self, tmp_path):
        """When action is empty string, pipeline yields error."""
        response = SimpleNamespace(content='{"action": "  ", "reason": "oops"}')
        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke.return_value = response

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            events = []
            async for ev in run_coding_pipeline("task", str(tmp_path)):
                events.append(ev)

        assert any(e[0] == "error" and "No action" in e[1] for e in events)


class TestHandleReadFiles:

    def test_skips_existing_files(self, tmp_path):
        """Files already in state are skipped."""
        from axono.coding import FileContent, _handle_read_files

        (tmp_path / "a.py").write_text("a content")

        state = {
            "files": [FileContent("a.py", "original content")],
        }

        result = _handle_read_files(str(tmp_path), ["a.py"], state)

        # Should not duplicate
        assert len(state["files"]) == 1
        assert "No new files" in result

    def test_reads_new_files(self, tmp_path):
        """New files are added to state."""
        from axono.coding import FileContent, _handle_read_files

        (tmp_path / "new.py").write_text("new content")

        state = {"files": []}

        result = _handle_read_files(str(tmp_path), ["new.py"], state)

        assert len(state["files"]) == 1
        assert state["files"][0].path == "new.py"
        assert "Read 1 files" in result

    def test_skips_unreadable_files(self, tmp_path):
        """Unreadable files are skipped."""
        from axono.coding import _handle_read_files

        state = {"files": []}

        result = _handle_read_files(str(tmp_path), ["ghost.py"], state)

        assert len(state["files"]) == 0
        assert "No new files" in result


# ---------------------------------------------------------------------------
# _build_iterative_prompt with plan_validation
# ---------------------------------------------------------------------------


class TestBuildIterativePromptPlanValidation:

    def test_includes_valid_plan_validation(self):
        """Prompt shows passed plan validation."""
        from axono.coding import _build_iterative_prompt
        from axono.pipeline import PlanValidation

        state = {
            "plan_validation": PlanValidation(valid=True, summary="Plan looks good"),
        }
        prompt = _build_iterative_prompt("task", "/tmp", [], state)

        assert "Plan validation: PASSED" in prompt
        assert "Plan looks good" in prompt

    def test_includes_failed_plan_validation_with_issues(self):
        """Prompt shows failed plan validation with issues."""
        from axono.coding import _build_iterative_prompt
        from axono.pipeline import PlanValidation

        state = {
            "plan_validation": PlanValidation(
                valid=False,
                issues=["Missing test step", "Wrong order"],
                suggestions=["Add tests first", "Reorder steps"],
                summary="Plan incomplete",
            ),
        }
        prompt = _build_iterative_prompt("task", "/tmp", [], state)

        assert "Plan validation: FAILED" in prompt
        assert "Plan incomplete" in prompt
        assert "Missing test step" in prompt
        assert "Wrong order" in prompt
        assert "Suggestions:" in prompt
        assert "Add tests first" in prompt


# ---------------------------------------------------------------------------
# _handle_validate_plan
# ---------------------------------------------------------------------------


class TestHandleValidatePlan:

    @pytest.mark.asyncio
    async def test_validate_plan_no_plan(self):
        """Raises error when no plan available."""
        from axono.coding import _handle_validate_plan

        state = {}

        with pytest.raises(ValueError, match="No plan available"):
            await _handle_validate_plan("task", state)

    @pytest.mark.asyncio
    async def test_validate_plan_success(self):
        """Returns success message when plan is valid."""
        from axono.coding import CodingPlan, _handle_validate_plan

        valid_resp = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "Good plan",
            }
        )
        llm = _fake_llm(valid_resp)

        state = {
            "coding_plan": CodingPlan(
                summary="Add feature",
                patches=[
                    {"path": "a.py", "action": "create", "description": "Create file"}
                ],
            ),
        }

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await _handle_validate_plan("Add feature", state)

        assert "Plan validated ✓" in result
        assert "plan_validation" in state
        assert state["plan_validation"].valid is True

    @pytest.mark.asyncio
    async def test_validate_plan_failure(self):
        """Returns failure message when plan is invalid."""
        from axono.coding import CodingPlan, _handle_validate_plan

        invalid_resp = json.dumps(
            {
                "valid": False,
                "issues": ["Missing tests", "Wrong approach"],
                "suggestions": ["Add test file"],
                "summary": "Incomplete",
            }
        )
        llm = _fake_llm(invalid_resp)

        state = {
            "coding_plan": CodingPlan(
                summary="Add feature",
                patches=[
                    {"path": "a.py", "action": "create", "description": "Create file"}
                ],
            ),
        }

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await _handle_validate_plan("Add feature", state)

        assert "Plan validation failed" in result
        assert "Missing tests" in result
        assert state["plan_validation"].valid is False

    @pytest.mark.asyncio
    async def test_validate_plan_with_files_context(self):
        """Includes files in context when available."""
        from axono.coding import (
            CodingPlan,
            FileContent,
            _handle_validate_plan,
        )

        valid_resp = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "OK",
            }
        )
        llm = _fake_llm(valid_resp)

        state = {
            "coding_plan": CodingPlan(
                summary="Add feature",
                patches=[{"path": "a.py", "action": "create", "description": "Create"}],
            ),
            "files": [
                FileContent(path="existing.py", content="code"),
            ],
        }

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            await _handle_validate_plan("task", state)

        # Verify context was passed (check ainvoke was called)
        call_args = llm.ainvoke.call_args
        user_msg = call_args[0][0][1].content
        assert "existing.py" in user_msg

    @pytest.mark.asyncio
    async def test_validate_plan_uses_description_fallback(self):
        """Uses fallback description when none provided."""
        from axono.coding import CodingPlan, _handle_validate_plan

        valid_resp = json.dumps(
            {
                "valid": True,
                "issues": [],
                "suggestions": [],
                "summary": "OK",
            }
        )
        llm = _fake_llm(valid_resp)

        state = {
            "coding_plan": CodingPlan(
                summary="Add feature",
                patches=[
                    {"path": "a.py", "action": "create", "description": ""},  # Empty
                ],
            ),
        }

        with mock.patch("axono.pipeline.get_llm", return_value=llm):
            result = await _handle_validate_plan("task", state)

        assert "Plan validated" in result


# ---------------------------------------------------------------------------
# run_coding_pipeline with validate_plan action
# ---------------------------------------------------------------------------


class TestRunCodingPipelineValidatePlan:

    @pytest.mark.asyncio
    async def test_validate_plan_action(self, tmp_path):
        """Pipeline handles validate_plan action."""
        (tmp_path / "README.md").write_text("# Project")

        call_count = 0

        async def fake_ainvoke(messages):
            nonlocal call_count
            call_count += 1

            # First call: plan
            if call_count == 1:
                return SimpleNamespace(
                    content='{"action": "plan", "reason": "create plan"}'
                )
            # Second call: planner response
            if call_count == 2:
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "summary": "Add file",
                            "files_to_read": [],
                            "patches": [
                                {
                                    "path": "new.py",
                                    "action": "create",
                                    "description": "Create",
                                }
                            ],
                        }
                    )
                )
            # Third call: validate_plan action
            if call_count == 3:
                return SimpleNamespace(
                    content='{"action": "validate_plan", "reason": "check plan"}'
                )
            # Fourth call: plan validation response
            if call_count == 4:
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "valid": True,
                            "issues": [],
                            "suggestions": [],
                            "summary": "Good",
                        }
                    )
                )
            # Fifth call: done
            return SimpleNamespace(content='{"done": true, "summary": "Validated"}')

        fake_llm = mock.AsyncMock()
        fake_llm.ainvoke = fake_ainvoke

        with mock.patch("axono.coding.get_llm", return_value=fake_llm):
            with mock.patch("axono.pipeline.get_llm", return_value=fake_llm):
                events = []
                async for ev in run_coding_pipeline("task", str(tmp_path)):
                    events.append(ev)

        types = [e[0] for e in events]
        assert "status" in types
        assert "result" in types

        # Check that validate_plan status was emitted
        status_msgs = [e[1] for e in events if e[0] == "status"]
        assert any(
            "validate_plan" in msg.lower() or "validated" in msg.lower()
            for msg in status_msgs
        )
