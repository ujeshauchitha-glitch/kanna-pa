"""Pure project-skeleton generators for languages with no single-command
ecosystem scaffolding tool of their own.

Rust (`cargo new <name>`) and Node (`npm init -y`) already have a real,
standard scaffolding command reachable through `tools.process.
run_process.ProcessTool`'s existing allowlist — hand-rolling equivalent
templates here would just drift from what those tools actually produce
and maintain, so this module deliberately does not cover them. C, C++,
and plain Java (no Maven/Gradle assumed installed — see the note on
`java_project` below) have no such tool, so these generate a minimal,
real, buildable skeleton directly, using only the compilers already in
`ProcessTool.DEFAULT_ALLOWED_EXECUTABLES` (`gcc`/`g++`/`javac`/`java`) to
build and run what gets generated — nothing here assumes an external
build tool is installed.

Every template was built and run for real (`make` / `make run` for C
and C++, `javac`+`java` for Java) against exactly what these functions
generate — see `tests/test_scaffold_build.py` — not just written and
assumed to compile.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScaffoldFile:
    path: str  # relative to the project directory
    content: str


def _c_family(project_name: str, *, cpp: bool) -> list[ScaffoldFile]:
    compiler = "CXX" if cpp else "CC"
    default_compiler = "g++" if cpp else "gcc"
    std = "-std=c++17" if cpp else "-std=c11"
    ext = "cpp" if cpp else "c"
    source_body = (
        f'#include <iostream>\n\nint main() {{\n    std::cout << "Hello from {project_name}!\\n";\n'
        "    return 0;\n}\n"
    ) if cpp else (
        f'#include <stdio.h>\n\nint main(void) {{\n    printf("Hello from {project_name}!\\n");\n'
        "    return 0;\n}\n"
    )

    makefile = f"""{compiler} ?= {default_compiler}
CFLAGS ?= -Wall -Wextra {std}

TARGET = {project_name}
SRC = src/main.{ext}

$(TARGET): $(SRC)
\t$({compiler}) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
\t./$(TARGET)

clean:
\trm -f $(TARGET)

.PHONY: run clean
"""

    return [
        ScaffoldFile("Makefile", makefile),
        ScaffoldFile(f"src/main.{ext}", source_body),
        ScaffoldFile(".gitignore", f"/{project_name}\n*.o\n"),
        ScaffoldFile(
            "README.md",
            f"# {project_name}\n\nBuild: `make`\nRun: `make run`\nClean: `make clean`\n",
        ),
    ]


def c_project(project_name: str) -> list[ScaffoldFile]:
    return _c_family(project_name, cpp=False)


def cpp_project(project_name: str) -> list[ScaffoldFile]:
    return _c_family(project_name, cpp=True)


def java_project(project_name: str) -> list[ScaffoldFile]:
    """A plain `javac`/`java` skeleton — no `pom.xml`/`build.gradle`.

    Maven/Gradle aren't in `ProcessTool.DEFAULT_ALLOWED_EXECUTABLES`
    (this sandbox doesn't have either installed, and neither is assumed
    present on an arbitrary user machine the way `javac`/`java`
    themselves are) — generating a `pom.xml` Kanna then can't actually
    build would violate the same "never ship something untested/
    unusable" discipline as everything else here. A real Maven/Gradle
    template is a reasonable follow-up once either tool is confirmed
    reachable — see `docs/ROADMAP.md`.
    """
    main_java = (
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        f'        System.out.println("Hello from {project_name}!");\n'
        "    }\n"
        "}\n"
    )
    readme = (
        f"# {project_name}\n\n"
        "No Maven/Gradle assumed — built directly with `javac`/`java`.\n\n"
        "Build: `javac -d build src/Main.java`\n"
        "Run:   `java -cp build Main`\n"
    )
    return [
        ScaffoldFile("src/Main.java", main_java),
        ScaffoldFile("README.md", readme),
        ScaffoldFile(".gitignore", "/build/\n"),
    ]


TEMPLATES = {"c": c_project, "cpp": cpp_project, "java": java_project}

# Shown to the caller in the tool's result so it's obvious how to actually
# use what was just generated, without needing to open README.md first.
NEXT_STEPS = {
    "c": "cd into the project and run: make && make run",
    "cpp": "cd into the project and run: make && make run",
    "java": "cd into the project and run: javac -d build src/Main.java && java -cp build Main",
}
