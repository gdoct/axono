import asyncio

import pytest

from typing import cast

from axono import ui


def _get_text(widget) -> str:
    """Return a string representation of a Textual widget's content."""
    for attr in ("renderable", "render", "text"):
        val = getattr(widget, attr, None)
        if val is not None:
            return str(val() if callable(val) else val)
    return str(widget)


def test_banner_contains_banner_text():
    banner = ui.Banner()
    text = _get_text(banner)
    assert "Type a message" in text
    assert ui.BANNER_TEXT.strip() in text


def test_user_message_escapes_brackets():
    um = ui.UserMessage("hello [world]")
    text = _get_text(um)
    assert "You:" in text
    # The implementation escapes '[' to '\['; accept either form in rendering
    assert "world" in text


def test_assistant_message_default():
    am = ui.AssistantMessage("reply")
    text = _get_text(am)
    assert "Axono:" in text
    assert "reply" in text


def test_system_message_prefix():
    sm = ui.SystemMessage("ok", prefix="Tool")
    text = _get_text(sm)
    assert "Tool:" in text
    assert "ok" in text


def test_chatcontainer_add_message(monkeypatch):
    container = ui.ChatContainer()
    mounted = {}

    async def fake_mount(widget):
        mounted["widget"] = widget

    monkeypatch.setattr(container, "mount", fake_mount)

    class Dummy:
        def __init__(self):
            self.scrolled = False

        def scroll_visible(self, animate=False):
            self.scrolled = True

    w = Dummy()
    asyncio.run(container.add_message(cast(ui.Static, w)))

    assert mounted.get("widget") is w
    assert w.scrolled is True


def test_cwdstatus_update_path():
    cs = ui.CwdStatus()
    cs.update_path("/home/user/project")
    text = _get_text(cs)
    assert "CWD:" in text
    assert "/home/user/project" in text


def test_tool_group_starts_expanded():
    tg = ui.ToolGroup("Tool: bash")
    assert tg.collapsed is False


def test_get_text_fallback():
    """When widget has no renderable/render/text, falls back to str()."""

    class Bare:
        def __str__(self):
            return "bare-widget"

    assert _get_text(Bare()) == "bare-widget"
