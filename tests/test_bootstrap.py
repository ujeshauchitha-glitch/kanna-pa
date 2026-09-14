from __future__ import annotations

from core.bootstrap import build_registry, default_policy
from core.permissions.levels import Decision, PermissionLevel


def test_build_registry_includes_every_subsystem():
    registry = build_registry()
    names = set(registry.names())
    assert {"fs_read_file", "fs_write_file", "fs_delete"} <= names
    assert "process_run" in names
    assert {"finance_add_transaction", "finance_query", "finance_import_receipt",
            "finance_import_statement"} <= names
    assert {"document_generate_docx", "document_generate_pptx", "document_generate_pdf",
            "document_convert_to_pdf"} <= names
    assert {"computer_screenshot", "computer_click", "computer_inspect_screen"} <= names
    assert {"phone_screenshot", "phone_tap", "phone_inspect_screen"} <= names
    assert {"browser_navigate", "browser_click", "browser_fill"} <= names
    assert "vision_extract_structure" in names
    assert "project_scaffold" in names


def test_default_policy_downgrades_create_new_file_to_allow():
    policy = default_policy()
    for tool_name in ("fs_write_file", "document_generate_docx", "document_generate_pptx",
                       "document_generate_pdf", "document_convert_to_pdf", "project_scaffold"):
        decision = policy.decide(tool_name=tool_name, args={"overwrite": False},
                                  level=PermissionLevel.REVIEW)
        assert decision == Decision.ALLOW, tool_name

        decision = policy.decide(tool_name=tool_name, args={}, level=PermissionLevel.REVIEW)
        assert decision == Decision.ALLOW, tool_name  # overwrite absent defaults to False


def test_default_policy_keeps_overwrite_gated():
    policy = default_policy()
    for tool_name in ("fs_write_file", "document_generate_docx", "document_generate_pptx",
                       "document_generate_pdf", "document_convert_to_pdf", "project_scaffold"):
        decision = policy.decide(tool_name=tool_name, args={"overwrite": True},
                                  level=PermissionLevel.REVIEW)
        assert decision == Decision.REQUIRE_APPROVAL, tool_name


def test_default_policy_does_not_touch_unrelated_review_tools():
    policy = default_policy()
    decision = policy.decide(tool_name="fs_delete", args={"recursive": True},
                              level=PermissionLevel.REVIEW)
    assert decision == Decision.REQUIRE_APPROVAL
