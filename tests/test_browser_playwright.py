"""Tests against the real `PlaywrightBrowserAgent` — skipped if no usable
browser can actually be launched (playwright package and/or a Chromium
binary missing). Unlike `tools.computer.fedora.is_available()` (a cheap
`shutil.which` check), browser availability can only really be known by
attempting a launch, so this skips per-test via a fixture rather than a
module-level `skipif` — see `docs/DEVICES.md`-style reasoning applied
here in `docs/BROWSER.md`.

`tests/test_browser_tools.py` covers the tool layer via
`FakeBrowserAgent`, so that coverage never depends on a real browser.
"""
from __future__ import annotations

import pytest

from core.errors import BrowserActionFailed, BrowserUnavailable
from tools.browser.playwright_backend import PlaywrightBrowserAgent


@pytest.fixture
def agent():
    instance = PlaywrightBrowserAgent(timeout_ms=5000)
    try:
        instance._ensure_page()
    except BrowserUnavailable as exc:
        pytest.skip(f"no usable browser available: {exc}")
    yield instance
    instance.close()


@pytest.fixture
def page_path(tmp_path):
    html = """<html><head><title>Test Page</title></head><body>
<h1>Hello Kanna</h1>
<a href="#" id="link" onclick="document.getElementById('result').innerText='clicked'">click me</a>
<input id="box"/>
<div id="result">not clicked</div>
</body></html>"""
    path = tmp_path / "page.html"
    path.write_text(html)
    return path


def test_navigate_returns_observation_of_the_real_page(agent, page_path):
    obs = agent.navigate(f"file://{page_path}")
    assert obs.title == "Test Page"
    assert "Hello Kanna" in obs.text_excerpt
    assert obs.url.startswith("file://")


def test_get_text_reads_the_real_page(agent, page_path):
    agent.navigate(f"file://{page_path}")
    assert "Hello Kanna" in agent.get_text()


def test_screenshot_returns_valid_png(agent, page_path):
    agent.navigate(f"file://{page_path}")
    data = agent.screenshot()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 100


def test_click_actually_changes_the_page_not_just_returns_ok(agent, page_path):
    """The strongest real test: click an element wired to mutate the DOM,
    then confirm the mutation is visible in the *next* observation — this
    is what "never assume a click succeeded" means in practice."""
    agent.navigate(f"file://{page_path}")
    before = agent.get_text()
    assert "not clicked" in before

    obs = agent.click("#link")

    assert "not clicked" not in obs.text_excerpt
    assert "clicked" in obs.text_excerpt


def test_fill_sets_the_real_input_value(agent, page_path):
    agent.navigate(f"file://{page_path}")
    agent.fill("#box", "hello from the test suite")
    # Read it back through a fresh get_text-independent check: query the
    # input's value via evaluate, since inner_text doesn't include input values.
    value = agent._page.input_value("#box")
    assert value == "hello from the test suite"


def test_click_on_missing_selector_raises_browser_action_failed(agent, page_path):
    agent.navigate(f"file://{page_path}")
    with pytest.raises(BrowserActionFailed, match="does-not-exist"):
        agent.click("#does-not-exist")


def test_fill_on_missing_selector_raises_browser_action_failed(agent, page_path):
    agent.navigate(f"file://{page_path}")
    with pytest.raises(BrowserActionFailed):
        agent.fill("#does-not-exist", "text")


def test_go_back_returns_to_previous_page(agent, page_path, tmp_path):
    second_html = "<html><head><title>Second</title></head><body>Second page</body></html>"
    second_path = tmp_path / "second.html"
    second_path.write_text(second_html)

    agent.navigate(f"file://{page_path}")
    agent.navigate(f"file://{second_path}")
    obs = agent.go_back()

    assert obs.title == "Test Page"


def test_current_url_reflects_navigation(agent, page_path):
    agent.navigate(f"file://{page_path}")
    assert agent.current_url() == f"file://{page_path}"


def test_missing_package_raises_browser_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "playwright.sync_api" or name.startswith("playwright"):
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    fresh_agent = PlaywrightBrowserAgent()
    with pytest.raises(BrowserUnavailable, match="not installed"):
        fresh_agent._ensure_page()
