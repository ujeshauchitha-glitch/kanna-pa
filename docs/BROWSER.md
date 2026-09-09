# Browser automation

## Status: one real backend — `PlaywrightBrowserAgent` (Chromium)

`tools/browser/base.py` declares the device-independent surface:

```python
@dataclass
class PageObservation:
    url: str
    title: str
    text_excerpt: str  # truncated visible text — enough to see what actually changed

class BrowserAgent(Protocol):
    def navigate(self, url: str) -> PageObservation: ...
    def get_text(self) -> str: ...
    def screenshot(self) -> bytes: ...
    def click(self, selector: str) -> PageObservation: ...
    def fill(self, selector: str, text: str) -> PageObservation: ...
    def go_back(self) -> PageObservation: ...
    def current_url(self) -> str: ...
    def close(self) -> None: ...
```

Every state-changing method (`navigate`, `click`, `fill`, `go_back`) returns a `PageObservation` of
what the page actually looks like *after* the action — never a bare success flag the caller has to
trust blindly. This is the same "never assume an action succeeded" discipline `docs/DEVICES.md`
applies to computer control: a plan step's `expected` postcondition, or an LLM planner reading the
tool result, has real evidence (the resulting URL, title, and visible text) to check against.

Two implementations exist:

- **`tools/browser/playwright_backend.py::PlaywrightBrowserAgent`** — a real backend, built on
  [Playwright](https://playwright.dev/python/)'s sync API driving headless Chromium. Lazily imports
  `playwright` (extra `[browser]`) so nothing outside this module needs it installed, matching
  `core/llm/anthropic_provider.py` and `vision/_common.py`'s discipline for optional heavy deps.
  Raises `BrowserUnavailable` if the package isn't installed or no browser binary can be launched, and
  `BrowserActionFailed` for a specific navigate/click/fill that didn't work (bad selector, navigation
  timeout, etc.) — the browser itself is fine, but *this* action wasn't.
- **`tools/browser/fake.py::FakeBrowserAgent`** — a scripted double for tool-layer tests
  (`tests/test_browser_tools.py`): records every call, lets a test pre-set the next observation, and
  can be told to raise `BrowserActionFailed` for specific selectors. No real browser required.

`tools/browser/get_browser_agent()` is the one place a process gets its agent — unlike
`tools.computer.get_computer_agent()` (cheap to recompute every call), a browser session has to stay
the *same* open page across calls for "navigate, then click, then read the result" to mean anything,
so this is a process-level singleton. `reset_browser_agent()` tears it down and clears the singleton
(tests use this to start clean); every `browser_*` tool resolves its agent through
`get_browser_agent()` at call time unless a test injects its own.

## What was actually tested, and how

This build/CI sandbox doesn't have a browser pre-installed by default either — validating
`PlaywrightBrowserAgent` meant actually installing Playwright and a Chromium build and driving it for
real, not just shipping against `FakeBrowserAgent`. Against a real headless Chromium instance, backed
by small local HTML fixture files (`tests/test_browser_playwright.py`'s `page_path` fixture):

- **`navigate()`** — checked the returned `PageObservation`'s `title` and `text_excerpt` against the
  real page's actual `<title>` and body text, and that `url` reflects the real navigation.
- **`get_text()`** — read back the same page's real text through a separate call path.
- **`screenshot()`** — returned bytes checked for a valid PNG header (`\x89PNG\r\n\x1a\n`) and a
  plausible size.
- **`click()`** — the strongest test: clicked a real link wired to mutate the DOM
  (`onclick="...innerText='clicked'"`), then confirmed the mutation shows up in the *next*
  observation, not just that the call returned without raising. This is what "never assume a click
  succeeded" means in practice for a browser, the same way the Ctrl-D round-trip through a real
  `xterm` proves it for `FedoraAgent` in `docs/DEVICES.md`.
- **`fill()`** — set a real `<input>`'s value and read it back via `page.input_value()`.
- **`go_back()`** — navigated through two real pages and confirmed going back actually lands on the
  first page's title, not just that no exception was raised.
- **`current_url()`** — confirmed it reflects the most recent real navigation.
- **The honest-failure paths** — `click()`/`fill()` against a selector with no matching element raise
  `BrowserActionFailed` naming the selector, and `_ensure_page()` raises `BrowserUnavailable` when the
  `playwright` package itself can't be imported (simulated via `monkeypatch` on `__import__`, since
  the package genuinely is installed in this sandbox).

`tests/test_browser_playwright.py` skips per-test (via a fixture that attempts a real launch and calls
`pytest.skip()` on `BrowserUnavailable`) rather than a module-level `skipif`, because — unlike
`tools.computer.fedora.is_available()`'s cheap `shutil.which` check — whether a browser can actually
launch is only knowable by trying. CI now installs a real Chromium build
(`python -m playwright install --with-deps chromium` in `.github/workflows/tests.yml`, after
`pip install -e ".[dev]"`) so this file runs for real on every push instead of perpetually skipping.
Locally, without any browser installed at all, that one file's tests skip cleanly; the rest of the
suite is unaffected either way.

### The bug this found

The default `p.chromium.launch()` call (no `executable_path`) failed in this sandbox with
`Executable doesn't exist at /opt/pw-browsers/chromium_headless_shell-.../...` — the pip-installed
`playwright` package expected a different pre-staged browser build than what was actually staged at
`PLAYWRIGHT_BROWSERS_PATH`. This environment provides a `chromium` convenience symlink at the root of
that path specifically for this mismatch case. Fixed with `_find_chromium_executable()`, which checks
for that symlink and passes it as `executable_path` only when present, falling through to Playwright's
normal resolution otherwise — so a real deployment (where `playwright install` matched the installed
package version) is unaffected by this workaround.

## Permission levels for browser tools

Following the same "read/reversible is LOW, unpredictable-or-irreversible is REVIEW" split
`docs/DEVICES.md` and `docs/SECURITY.md` use elsewhere:

| Tool | Level | Why |
|---|---|---|
| `browser_navigate`, `browser_get_text`, `browser_screenshot`, `browser_go_back` | LOW | Read-only, or trivially reversible (loading a URL, going back) |
| `browser_click`, `browser_fill` | REVIEW | Kanna cannot know what a click or a filled-in value will actually do on an arbitrary page — it could submit a form, complete a purchase, or send something. Same reasoning as `computer_click`/`computer_type_text` in `docs/DEVICES.md` |

## What this means today

Browser tools are registered in the tool registry (`core/bootstrap.py`) and reachable through the
agent loop's normal plan-step path, where the REVIEW-level ones (`browser_click`, `browser_fill`)
require an `ApprovalGate` to say yes, same as everything else. There is no `kanna browser <...>` CLI
subcommand yet (unlike `kanna computer ...` / `kanna document ...`) — only the registry path exists
today. On a machine with no `playwright` package or no launchable browser binary, every call fails
cleanly with `browser_unavailable` — never a silent no-op reported as success.

## Known limitations

- One page at a time — no multi-tab/multi-context support.
- No cookie/session persistence across process restarts (the singleton dies with the process).
- `text_excerpt` is truncated (4000 chars) and is `inner_text("body")` — no structured DOM extraction
  beyond that; a page that needs specific-element text still goes through a CSS selector via a future
  `click`/`fill`-style "read this selector" tool, which doesn't exist yet.
- No file-upload/download handling.
- No CLI subcommand (`kanna browser ...`) yet — see above.

These are all reasonable follow-ups, not correctness gaps in what's shipped — see `docs/ROADMAP.md`.
