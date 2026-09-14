"""`project_scaffold` — the registry-exposed project scaffolding tool.

Sandboxed and overwrite-gated exactly like `fs_write_file` and the
`document_generate_*` tools: creating a brand-new project directory is
low-risk (auto-allowed by `core/bootstrap.py::default_policy`);
overwriting files that already exist there requires `overwrite=true`
and stays REVIEW-gated.
"""
from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, boolean, obj, string
from tools.scaffold.templates import NEXT_STEPS, TEMPLATES

_LANGUAGES = tuple(sorted(TEMPLATES))


class ProjectScaffoldTool:
    name = "project_scaffold"
    description = (
        "Generate a minimal, real, buildable project skeleton for a language with no "
        f"single-command scaffolding tool of its own ({', '.join(_LANGUAGES)}). For Rust or "
        "Node, use process_run with ['cargo', 'new', <name>] or ['npm', 'init', '-y'] instead — "
        "those ecosystems already have a real scaffolding tool; this one doesn't reimplement it."
    )
    permission = PermissionLevel.REVIEW
    input_schema = obj(
        {
            "language": string(enum=_LANGUAGES),
            "path": string(description="Directory to create the project in, within the sandbox"),
            "project_name": string(
                description="Used in generated file content (Makefile target, greeting, etc.); "
                             "defaults to the last path component", default=""),
            "overwrite": boolean(description="Allow overwriting files that already exist there",
                                  default=False),
        },
        required=("language", "path"),
    )
    output_schema = obj({
        "language": string(), "files_created": array(string()), "next_steps": string(),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if resolved.exists() and not resolved.is_dir():
            return ToolResult.fail("not_a_directory", f"path exists and is not a directory: {resolved}")

        language = args["language"]
        template = TEMPLATES.get(language)
        if template is None:
            return ToolResult.fail(
                "unsupported_language", f"no scaffold for {language!r}; supported: {_LANGUAGES}"
            )

        project_name = args.get("project_name") or resolved.name
        files = template(project_name)

        overwrite = bool(args.get("overwrite", False))
        already_exist = [f.path for f in files if (resolved / f.path).exists()]
        if already_exist and not overwrite:
            return ToolResult.fail(
                "would_overwrite",
                f"these files already exist under {resolved}: {already_exist}; "
                "pass overwrite=true to replace them",
            )

        created: list[str] = []
        modified: list[str] = []
        for scaffold_file in files:
            target = resolved / scaffold_file.path
            existed = target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(scaffold_file.content, encoding="utf-8")
            (modified if existed else created).append(str(target))

        result = ToolResult.ok({
            "language": language, "files_created": created + modified,
            "next_steps": NEXT_STEPS[language],
        })
        result.files_created = created
        result.files_modified = modified
        return result


ALL_TOOLS = [ProjectScaffoldTool()]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
