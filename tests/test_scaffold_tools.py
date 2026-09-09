from __future__ import annotations

from tools.scaffold.tools import ProjectScaffoldTool


def test_creates_c_project(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    result = tool.execute({"language": "c", "path": "myproj"}, ctx)

    assert result.success
    assert result.data["language"] == "c"
    assert (tmp_path / "myproj" / "Makefile").exists()
    assert (tmp_path / "myproj" / "src" / "main.c").exists()
    assert (tmp_path / "myproj" / ".gitignore").exists()
    assert (tmp_path / "myproj" / "README.md").exists()
    assert sorted(result.files_created) == sorted(str(p) for p in [
        tmp_path / "myproj" / "Makefile", tmp_path / "myproj" / "src" / "main.c",
        tmp_path / "myproj" / ".gitignore", tmp_path / "myproj" / "README.md",
    ])
    assert result.files_modified == []


def test_creates_cpp_project(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    result = tool.execute({"language": "cpp", "path": "myproj"}, ctx)
    assert result.success
    assert (tmp_path / "myproj" / "src" / "main.cpp").exists()


def test_creates_java_project(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    result = tool.execute({"language": "java", "path": "myproj"}, ctx)
    assert result.success
    assert (tmp_path / "myproj" / "src" / "Main.java").exists()
    assert "javac" in result.data["next_steps"]


def test_project_name_defaults_to_directory_name(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    tool.execute({"language": "c", "path": "hello_world"}, ctx)
    main_c = (tmp_path / "hello_world" / "src" / "main.c").read_text()
    assert "hello_world" in main_c


def test_project_name_override(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    tool.execute({"language": "c", "path": "myproj", "project_name": "custom_name"}, ctx)
    main_c = (tmp_path / "myproj" / "src" / "main.c").read_text()
    assert "custom_name" in main_c


def test_refuses_overwrite_without_flag(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    tool.execute({"language": "c", "path": "myproj"}, ctx)
    result = tool.execute({"language": "c", "path": "myproj"}, ctx)
    assert not result.success
    assert result.error.code == "would_overwrite"


def test_overwrite_with_flag_marks_modified(ctx, tmp_path):
    tool = ProjectScaffoldTool()
    tool.execute({"language": "c", "path": "myproj"}, ctx)
    result = tool.execute({"language": "c", "path": "myproj", "overwrite": True}, ctx)
    assert result.success
    assert sorted(result.files_modified) == sorted(str(p) for p in [
        tmp_path / "myproj" / "Makefile", tmp_path / "myproj" / "src" / "main.c",
        tmp_path / "myproj" / ".gitignore", tmp_path / "myproj" / "README.md",
    ])
    assert result.files_created == []


def test_rejects_sandbox_escape(ctx):
    tool = ProjectScaffoldTool()
    result = tool.execute({"language": "c", "path": "../../etc/myproj"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_rejects_path_that_is_an_existing_file(ctx, tmp_path):
    (tmp_path / "notadir").write_text("x")
    tool = ProjectScaffoldTool()
    result = tool.execute({"language": "c", "path": "notadir"}, ctx)
    assert not result.success
    assert result.error.code == "not_a_directory"


def test_rejects_unsupported_language(ctx):
    # Calling execute() directly (as this test does, bypassing
    # ToolRegistry.invoke()'s own input_schema enum check) still hits the
    # tool's own defense-in-depth check against TEMPLATES.
    tool = ProjectScaffoldTool()
    result = tool.execute({"language": "rust", "path": "myproj"}, ctx)
    assert not result.success
    assert result.error.code == "unsupported_language"


def test_registered_at_review_permission():
    assert ProjectScaffoldTool().permission.name == "REVIEW"
