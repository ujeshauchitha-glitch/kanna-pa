from __future__ import annotations

from core.permissions.gate import DenyAllGate, PreApprovedGate
from core.permissions.levels import Decision, PermissionLevel
from core.permissions.policy import PermissionPolicy, Rule


def test_low_defaults_to_allow():
    policy = PermissionPolicy()
    decision = policy.decide(tool_name="anything", args={}, level=PermissionLevel.LOW)
    assert decision == Decision.ALLOW


def test_review_defaults_to_require_approval():
    policy = PermissionPolicy()
    decision = policy.decide(tool_name="anything", args={}, level=PermissionLevel.REVIEW)
    assert decision == Decision.REQUIRE_APPROVAL


def test_restricted_defaults_to_deny():
    policy = PermissionPolicy()
    decision = policy.decide(tool_name="anything", args={}, level=PermissionLevel.RESTRICTED)
    assert decision == Decision.DENY


def test_rule_overrides_default():
    policy = PermissionPolicy()
    policy.add_rule(Rule(
        name="always deny fs_delete",
        predicate=lambda name, args: name == "fs_delete",
        decision=Decision.DENY,
    ))
    assert policy.decide(tool_name="fs_delete", args={}, level=PermissionLevel.REVIEW) == Decision.DENY
    assert policy.decide(tool_name="fs_read_file", args={}, level=PermissionLevel.LOW) == Decision.ALLOW


def test_deny_all_gate_never_approves():
    gate = DenyAllGate()
    assert gate.approve(tool_name="x", args={}, level=PermissionLevel.REVIEW, reason="r") is False


def test_preapproved_gate_only_approves_listed_tools():
    gate = PreApprovedGate({"fs_delete"})
    assert gate.approve(tool_name="fs_delete", args={}, level=PermissionLevel.REVIEW, reason="r") is True
    assert gate.approve(tool_name="fs_write_file", args={}, level=PermissionLevel.REVIEW, reason="r") is False
