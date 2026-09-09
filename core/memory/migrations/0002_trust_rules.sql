-- Standing tool-approval rules ("trusted automation" — see
-- core/permissions/gate.py::TrustStoreGate and docs/SECURITY.md).
-- Each row is a durable grant, so this table doubles as the audit trail
-- of what's been pre-approved, not just a runtime allowlist.
CREATE TABLE IF NOT EXISTS trust_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    args_pattern TEXT NOT NULL DEFAULT '{}',
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trust_rules_tool ON trust_rules(tool_name);
