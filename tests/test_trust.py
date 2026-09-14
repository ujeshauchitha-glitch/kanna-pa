"""Trusted-automation configuration: the pure matcher, the repository
(the audit trail of what's been pre-approved), and the gate that
consults both together.
"""
from __future__ import annotations

from core.memory.repositories.trust_rules import TrustRuleRepository
from core.permissions.gate import DenyAllGate, TrustStoreGate
from core.permissions.levels import PermissionLevel
from core.permissions.trust import rule_matches


# -- rule_matches: pure function --

def test_rule_matches_wrong_tool_never_matches():
    assert not rule_matches(rule_tool_name="computer_click", rule_args_pattern={},
                             call_tool_name="computer_key_press", call_args={})


def test_rule_matches_empty_pattern_matches_any_args():
    assert rule_matches(rule_tool_name="computer_open_application", rule_args_pattern={},
                         call_tool_name="computer_open_application", call_args={"name": "firefox"})


def test_rule_matches_requires_every_pattern_key_to_match():
    pattern = {"name": "firefox"}
    assert rule_matches(rule_tool_name="computer_open_application", rule_args_pattern=pattern,
                         call_tool_name="computer_open_application", call_args={"name": "firefox"})
    assert not rule_matches(rule_tool_name="computer_open_application", rule_args_pattern=pattern,
                             call_tool_name="computer_open_application", call_args={"name": "chrome"})


def test_rule_matches_missing_key_in_call_args_does_not_match():
    pattern = {"name": "firefox"}
    assert not rule_matches(rule_tool_name="computer_open_application", rule_args_pattern=pattern,
                             call_tool_name="computer_open_application", call_args={})


def test_rule_matches_ignores_extra_call_args_not_named_by_pattern():
    pattern = {"selector": "#submit"}
    assert rule_matches(rule_tool_name="browser_click", rule_args_pattern=pattern,
                         call_tool_name="browser_click",
                         call_args={"selector": "#submit", "extra": "ignored"})


# -- TrustRuleRepository: persistence + the audit trail --

def test_repository_add_and_list(db):
    repo = TrustRuleRepository(db)
    rule = repo.add(tool_name="computer_click", args_pattern={"x": 10}, note="testing")
    assert rule.id is not None
    assert rule.created_at

    rules = repo.list_all()
    assert len(rules) == 1
    assert rules[0].tool_name == "computer_click"
    assert rules[0].args_pattern == {"x": 10}
    assert rules[0].note == "testing"


def test_repository_add_defaults_to_empty_pattern_and_note(db):
    repo = TrustRuleRepository(db)
    rule = repo.add(tool_name="computer_screenshot")
    assert rule.args_pattern == {}
    assert rule.note == ""


def test_repository_get_returns_none_for_unknown_id(db):
    repo = TrustRuleRepository(db)
    assert repo.get(999) is None


def test_repository_remove_returns_true_only_if_a_row_was_deleted(db):
    repo = TrustRuleRepository(db)
    rule = repo.add(tool_name="computer_click")
    assert repo.remove(rule.id) is True
    assert repo.remove(rule.id) is False  # already gone
    assert repo.list_all() == []


def test_repository_list_all_is_ordered_by_creation(db):
    repo = TrustRuleRepository(db)
    first = repo.add(tool_name="a")
    second = repo.add(tool_name="b")
    assert [r.id for r in repo.list_all()] == [first.id, second.id]


# -- TrustStoreGate: the gate that actually consults the trust store --

def test_gate_approves_a_matching_rule_without_asking_fallback(db):
    TrustRuleRepository(db).add(tool_name="computer_click", args_pattern={"x": 5, "y": 5})
    calls = []

    class _RecordingFallback:
        def approve(self, **kwargs):
            calls.append(kwargs)
            return False

    gate = TrustStoreGate(TrustRuleRepository(db), fallback=_RecordingFallback())
    approved = gate.approve(tool_name="computer_click", args={"x": 5, "y": 5},
                             level=PermissionLevel.REVIEW, reason="r")

    assert approved is True
    assert calls == []  # never consulted — the trust store alone decided


def test_gate_falls_back_when_no_rule_matches(db):
    TrustRuleRepository(db).add(tool_name="computer_click", args_pattern={"x": 5})
    gate = TrustStoreGate(TrustRuleRepository(db), fallback=DenyAllGate())

    approved = gate.approve(tool_name="computer_click", args={"x": 999},
                             level=PermissionLevel.REVIEW, reason="r")

    assert approved is False


def test_gate_with_no_rules_delegates_entirely_to_fallback(db):
    gate = TrustStoreGate(TrustRuleRepository(db), fallback=DenyAllGate())
    assert gate.approve(tool_name="anything", args={}, level=PermissionLevel.REVIEW,
                         reason="r") is False


# -- bootstrap() integration: a rule granted via the repository actually
#    changes what the wired-up registry does, end to end --

def test_bootstrap_wires_trust_store_so_a_granted_rule_auto_approves(tmp_path, monkeypatch):
    from core.bootstrap import bootstrap

    # Default sandbox roots are cwd + KANNA_HOME (see core/config/settings.py)
    # — isolate both to tmp_path so the fs_delete call below stays sandboxed
    # and doesn't touch anything real.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KANNA_HOME", str(tmp_path / "kanna_home"))

    kanna = bootstrap(db_path=tmp_path / "kanna.db")
    try:
        # fs_delete is REVIEW and DenyAllGate (the default fallback) denies
        # it — confirm that baseline first.
        target = tmp_path / "doomed.txt"
        target.write_text("x")
        result = kanna.registry.invoke(
            "fs_delete", {"path": str(target)}, kanna.tool_context())
        assert not result.success
        assert result.error.code == "approval_denied"

        # Grant standing approval, then the identical call succeeds.
        kanna.trust_rules.add(tool_name="fs_delete", args_pattern={"path": str(target)})
        result = kanna.registry.invoke(
            "fs_delete", {"path": str(target)}, kanna.tool_context())
        assert result.success
    finally:
        kanna.close()
