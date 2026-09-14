# Runtimes beyond Python

## Status: real, tested project scaffolding for the ecosystems with no scaffolding tool of their own

`tools/process/run_process.py::ProcessTool.DEFAULT_ALLOWED_EXECUTABLES` already let Kanna compile and
run C/C++ (`gcc`/`g++`), Rust (`rustc`/`cargo`), Java (`javac`/`java`), and Node (`node`/`npm`) when
they're installed. What was missing — the gap this pass closes — was a way to *set up* a new project
in the first place: `python3`/`pytest` have no equivalent need (a Python file just runs), but
`gcc`/`javac` have no build system or standard directory layout at all, and unlike Rust and Node they
have no single-command tool that creates one.

## Rust and Node: use their own real tool, not a hand-rolled copy

Rust (`cargo new <name>`) and Node (`npm init -y`) already have a real, standard scaffolding command,
and both `cargo` and `npm` are already in `ProcessTool`'s default allowlist. Reimplementing what
`cargo new` produces as a static template here would just drift from whatever the actual installed
`cargo` version generates — worse, not better, than delegating to the real tool. So `tools/scaffold/`
deliberately does **not** cover Rust or Node: ask the agent to run `process_run` with
`["cargo", "new", "<name>"]` or `["npm", "init", "-y"]` directly.

## C, C++, and plain Java: `project_scaffold`

C, C++, and Java have no such single-command tool built into the language itself. For Java
specifically, Maven and Gradle exist and would normally fill this role, but neither is assumed
installed (this build environment has neither, and there's no reason to assume an arbitrary user's
machine does either) — generating a `pom.xml` Kanna then couldn't actually build would violate the
same "never ship something untested/unusable" discipline as everything else in this project. So the
Java template is deliberately a plain `javac`/`java` skeleton, not a Maven project.

`tools/scaffold/templates.py` — pure functions, no I/O — generate:

- **C** (`c_project`): a `Makefile` (`gcc`, `-std=c11`, `-Wall -Wextra`), `src/main.c`, `.gitignore`,
  `README.md`.
- **C++** (`cpp_project`): the same shape with `g++`, `-std=c++17`, `src/main.cpp`.
- **Java** (`java_project`): `src/Main.java`, a `README.md` naming the exact `javac`/`java` commands
  to build and run it, `.gitignore`.

`tools/scaffold/tools.py::ProjectScaffoldTool` (`project_scaffold`, LOW-by-default like
`fs_write_file`/`document_generate_*` — REVIEW only when it would overwrite existing files) writes
these into a sandboxed directory. `overwrite=false` (the default) refuses to touch a directory that
already has any of the generated files present, listing exactly which ones; `overwrite=true` replaces
them and reports them as `files_modified` rather than `files_created`.

### `make` joined the process_run allowlist

A generated C/C++ project ships a `Makefile` — but `make` itself wasn't in
`ProcessTool.DEFAULT_ALLOWED_EXECUTABLES` before this pass, which meant Kanna could generate a
Makefile-based project but couldn't actually invoke `make` to build it through its own tools (only a
direct `gcc`/`g++` call, bypassing the Makefile entirely). Fixed by adding `make` to the default
allowlist — `tests/test_scaffold_build.py::test_scaffolded_c_project_is_buildable_through_process_run`
proves the whole path end to end: `project_scaffold` generates a project, then `process_run` (not a
raw `subprocess` call in the test — the actual tool Kanna itself would use) runs `make` and `make run`
against it for real.

## What was actually tested, and how

Every template is built and run for real, not just generated and assumed to compile:

- **`tests/test_scaffold_build.py`** — writes each language's generated files to a real `tmp_path`,
  then runs the real toolchain against them: `make`/`make run` for C and C++ (via real `subprocess`
  calls, mirroring how a user would build it by hand), `javac`+`java` for the Java skeleton, and a
  `make clean` check that the target binary is actually removed. Each test is `skipif`'d on the
  relevant compiler/`make` missing, the same pattern `tests/test_fedora_agent.py` uses for
  `xdotool`/`scrot`/`xclip` — this build/CI sandbox has `gcc`/`g++`/`make`/`javac`/`java` installed, so
  none of these skip here or in CI.
- **`tests/test_scaffold_tools.py`** — the tool layer: sandboxing, `project_name` defaulting to the
  destination directory's name vs. an explicit override, overwrite refusal/acceptance, rejecting a
  destination that's an existing non-directory file, rejecting an unsupported language, permission
  level. No compiler needed — these test the tool's own logic (file writing, path handling), not the
  generated code's correctness (that's `test_scaffold_build.py`'s job).
- **`tests/test_process_tool.py::test_default_allowlist_includes_make`** — a direct, minimal
  regression test that `make` is actually in the allowlist (the thing the whole integration proof in
  `test_scaffold_build.py` depends on).

## Known limitations

- No dependency resolution — every generated skeleton has zero external dependencies. Adding real
  dependency management (vcpkg/conan for C/C++, Maven/Gradle for Java) is future work, gated on those
  tools actually being confirmed installed somewhere Kanna can rely on, the same reasoning that kept
  Maven/Gradle out of the Java template for now.
- No `kanna scaffold ...` CLI subcommand yet (unlike `computer`/`document`/`trust`/`scheduler`) —
  registry/agent-loop path only.
- No project templates beyond a single "hello world" binary — no test-framework scaffolding
  (a CMake+CTest setup, a JUnit-wired Maven project once Maven scaffolding exists), no library-vs-
  binary distinction.
- C/C++ scaffolding assumes a Unix-like `make`; no MSVC/`nmake`/Visual Studio project generation.
