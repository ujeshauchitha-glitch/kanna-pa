from __future__ import annotations

import pytest

from core.errors import SandboxViolation
from core.permissions.sandbox import Sandbox


def test_resolves_relative_path_within_root(tmp_path):
    (tmp_path / "sub").mkdir()
    sandbox = Sandbox([str(tmp_path)])
    resolved = sandbox.resolve("sub")
    assert resolved == (tmp_path / "sub").resolve()


def test_resolves_absolute_path_within_root(tmp_path):
    target = tmp_path / "file.txt"
    sandbox = Sandbox([str(tmp_path)])
    resolved = sandbox.resolve(str(target))
    assert resolved == target.resolve()


def test_rejects_parent_traversal(tmp_path):
    sandbox = Sandbox([str(tmp_path)])
    with pytest.raises(SandboxViolation):
        sandbox.resolve("../../etc/passwd")


def test_rejects_absolute_path_outside_root(tmp_path):
    sandbox = Sandbox([str(tmp_path)])
    with pytest.raises(SandboxViolation):
        sandbox.resolve("/etc/passwd")


def test_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret")

    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape"
    link.symlink_to(outside)

    sandbox = Sandbox([str(root)])
    with pytest.raises(SandboxViolation):
        sandbox.resolve("escape/secret.txt")


def test_is_within(tmp_path):
    sandbox = Sandbox([str(tmp_path)])
    assert sandbox.is_within("ok.txt") is True
    assert sandbox.is_within("../escape.txt") is False


def test_multiple_roots(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    sandbox = Sandbox([str(root_a), str(root_b)])
    assert sandbox.resolve(str(root_b / "x.txt")) == (root_b / "x.txt").resolve()
