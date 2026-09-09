from __future__ import annotations

from tools.filesystem.delete import DeleteFileTool
from tools.filesystem.info import FileInfoTool
from tools.filesystem.list_dir import ListDirectoryTool
from tools.filesystem.mkdir import CreateDirectoryTool
from tools.filesystem.read import ReadFileTool
from tools.filesystem.search import SearchFilesTool
from tools.filesystem.write import WriteFileTool


def test_write_then_read(ctx, tmp_path):
    write_result = WriteFileTool().execute({"path": "note.txt", "content": "hello kanna"}, ctx)
    assert write_result.success
    assert write_result.files_created == [str(tmp_path / "note.txt")]

    read_result = ReadFileTool().execute({"path": "note.txt"}, ctx)
    assert read_result.success
    assert read_result.data["content"] == "hello kanna"
    assert read_result.data["truncated"] is False


def test_write_refuses_overwrite_without_flag(ctx):
    WriteFileTool().execute({"path": "note.txt", "content": "v1"}, ctx)
    result = WriteFileTool().execute({"path": "note.txt", "content": "v2"}, ctx)
    assert not result.success
    assert result.error.code == "already_exists"


def test_write_overwrite_with_flag_marks_modified(ctx, tmp_path):
    WriteFileTool().execute({"path": "note.txt", "content": "v1"}, ctx)
    result = WriteFileTool().execute({"path": "note.txt", "content": "v2", "overwrite": True}, ctx)
    assert result.success
    assert result.files_modified == [str(tmp_path / "note.txt")]
    assert (tmp_path / "note.txt").read_text() == "v2"


def test_read_missing_file_fails(ctx):
    result = ReadFileTool().execute({"path": "missing.txt"}, ctx)
    assert not result.success
    assert result.error.code == "not_found"


def test_read_rejects_sandbox_escape(ctx):
    result = ReadFileTool().execute({"path": "../../etc/passwd"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_list_directory(ctx, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")

    result = ListDirectoryTool().execute({"path": "."}, ctx)
    assert result.success
    names = {e["name"] for e in result.data["entries"]}
    assert names == {"a.txt", "sub"}

    recursive = ListDirectoryTool().execute({"path": ".", "recursive": True}, ctx)
    paths = {e["path"] for e in recursive.data["entries"]}
    assert str(tmp_path / "sub" / "b.txt") in paths


def test_search_files(ctx, tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")

    result = SearchFilesTool().execute({"path": ".", "pattern": r"def \w+", "glob": "*.py"}, ctx)
    assert result.success
    assert len(result.data["matches"]) == 2


def test_search_invalid_regex(ctx):
    result = SearchFilesTool().execute({"path": ".", "pattern": "("}, ctx)
    assert not result.success
    assert result.error.code == "invalid_pattern"


def test_create_directory(ctx, tmp_path):
    result = CreateDirectoryTool().execute({"path": "a/b/c"}, ctx)
    assert result.success
    assert (tmp_path / "a" / "b" / "c").is_dir()


def test_file_info_reports_existence(ctx, tmp_path):
    missing = FileInfoTool().execute({"path": "nope.txt"}, ctx)
    assert missing.success
    assert missing.data["exists"] is False

    (tmp_path / "x.txt").write_text("hi")
    present = FileInfoTool().execute({"path": "x.txt"}, ctx)
    assert present.data["exists"] is True
    assert present.data["is_file"] is True
    assert present.data["size_bytes"] == 2


def test_delete_requires_recursive_for_directory(ctx, tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.txt").write_text("x")

    result = DeleteFileTool().execute({"path": "d"}, ctx)
    assert not result.success
    assert result.error.code == "is_a_directory"

    result = DeleteFileTool().execute({"path": "d", "recursive": True}, ctx)
    assert result.success
    assert not (tmp_path / "d").exists()


def test_delete_missing_path_fails(ctx):
    result = DeleteFileTool().execute({"path": "nope.txt"}, ctx)
    assert not result.success
    assert result.error.code == "not_found"
