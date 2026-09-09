-- Kanna initial schema.
-- Separate tables per concern (per KANNA spec) rather than one blob table.

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    project TEXT,
    due_date TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    request TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_steps (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    step_index INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    expected TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id);

CREATE TABLE IF NOT EXISTS execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    args TEXT NOT NULL,
    result TEXT NOT NULL,
    success INTEGER NOT NULL,
    duration_ms REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_execution_log_tool ON execution_log(tool_name);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    kind TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule_type TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    next_run_at TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL REFERENCES schedules(id),
    status TEXT NOT NULL,
    ran_at TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_jobs_schedule ON jobs(schedule_id);

-- Finance

CREATE TABLE IF NOT EXISTS finance_categories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    parent_category TEXT REFERENCES finance_categories(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS finance_category_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    category_id TEXT NOT NULL REFERENCES finance_categories(id)
);
CREATE INDEX IF NOT EXISTS idx_finance_rules_keyword ON finance_category_rules(keyword);

CREATE TABLE IF NOT EXISTS finance_transactions (
    id TEXT PRIMARY KEY,
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    category_id TEXT REFERENCES finance_categories(id),
    subcategory TEXT,
    merchant TEXT,
    description TEXT,
    payment_method TEXT,
    source TEXT NOT NULL DEFAULT 'manual',
    notes TEXT,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_finance_tx_date ON finance_transactions(occurred_at);
CREATE INDEX IF NOT EXISTS idx_finance_tx_category ON finance_transactions(category_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_finance_tx_hash ON finance_transactions(content_hash)
    WHERE content_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS finance_budgets (
    id TEXT PRIMARY KEY,
    category_id TEXT REFERENCES finance_categories(id),
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL,
    period TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_finance_budgets_category ON finance_budgets(category_id);

CREATE TABLE IF NOT EXISTS finance_recurring (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL,
    frequency TEXT NOT NULL,
    next_occurrence TEXT NOT NULL,
    category_id TEXT REFERENCES finance_categories(id),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_finance_recurring_next ON finance_recurring(next_occurrence);
