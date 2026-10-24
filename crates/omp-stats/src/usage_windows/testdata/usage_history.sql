-- usage_history rows for the usage_windows oracle test (same DDL as ai/src/auth/sqlite-credential-store.ts).
CREATE TABLE usage_history (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	recorded_at INTEGER NOT NULL,
	provider TEXT NOT NULL,
	account_key TEXT NOT NULL,
	email TEXT,
	account_id TEXT,
	limit_id TEXT NOT NULL,
	label TEXT NOT NULL,
	window_label TEXT,
	used_fraction REAL,
	status TEXT,
	resets_at INTEGER
);
CREATE INDEX idx_usage_history_series ON usage_history(provider, account_key, limit_id, recorded_at);
CREATE INDEX idx_usage_history_recorded ON usage_history(recorded_at);

-- Before the since cutoff (1700000000000): excluded by the query.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1699999000000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.9, NULL, NULL);

-- anthropic:7d, account A: growth, a reset drop, overage-by-fraction, explicit exhausted status.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1700000000000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.10, 'ok', NULL),
 (1700003600000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.35, 'ok', NULL),
 (1700007200000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.80, 'ok', NULL),
 (1700010800000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.999, NULL, NULL),
 (1700014400000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.05, 'ok', NULL),
 (1700018000000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.03, 'exhausted', NULL),
 (1700021600000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', NULL, 'ok', NULL),
 (1700025200000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.20, 'exhausted', NULL),
 (1700028800000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.22, 'ok', NULL);

-- anthropic:7d, account B: no email (account id label), overage above 1.0, a small jitter drop.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1700003600000, 'anthropic', 'key-b', NULL, 'org-b', 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.50, 'ok', NULL),
 (1700007200000, 'anthropic', 'key-b', NULL, 'org-b', 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.20, 'ok', NULL),
 (1700010800000, 'anthropic', 'key-b', NULL, 'org-b', 'anthropic:7d', 'Claude 7 Day', '7 Day', 1.20, 'ok', NULL),
 (1700014400000, 'anthropic', 'key-b', NULL, 'org-b', 'anthropic:7d', 'Claude 7 Day', '7 Day', 1.17, 'ok', NULL);

-- anthropic:7d, account C: neither email nor account id; empty-string email falls through to account id.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1700007200000, 'anthropic', 'key-c', NULL, NULL, 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.40, 'ok', NULL),
 (1700021600000, 'anthropic', 'key-c', '', 'acct-z', 'anthropic:7d', 'Claude 7 Day', '7 Day', 0.45, 'ok', NULL);

-- Same provider, different limit id with a model-scoped label that contains the window label (no suffix).
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1700000000000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d-opus', 'Claude 7 Day (Opus 7 Day)', '7 Day', 0.60, 'ok', NULL),
 (1700014400000, 'anthropic', 'key-a', 'a@x.io', NULL, 'anthropic:7d-opus', 'Claude 7 Day (Opus 7 Day)', '7 Day', 0.70, 'ok', NULL);

-- Same label family, distinct durations: window label suffix distinguishes them. Label change across snapshots.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1700000000000, 'google', 'g-1', 'g@example.com', NULL, 'google:daily', 'Usage (Google)', 'Daily', 0.30, 'ok', NULL),
 (1700003600000, 'google', 'g-1', 'g@example.com', NULL, 'google:daily', 'Usage (Google) v2', 'Daily', 0.45, 'ok', NULL),
 (1700007200000, 'google', 'g-1', 'g@example.com', NULL, 'google:daily', 'Usage (Google) v2', 'Daily', 0.95, 'ok', NULL),
 (1700000000000, 'google', 'g-2', NULL, NULL, 'google:weekly', 'Usage (Google)', 'Weekly', 0.10, NULL, NULL),
 (1700010800000, 'google', 'g-2', NULL, NULL, 'google:weekly', 'Usage (Google)', 'Weekly', 0.12, NULL, NULL);

-- Null window label and a second provider with a plain window.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at) VALUES
 (1700000000000, 'openai-codex', 'c-1', 'c@example.com', NULL, 'codex:primary', 'Codex 5h', NULL, 0.25, NULL, NULL),
 (1700003600000, 'openai-codex', 'c-1', 'c@example.com', NULL, 'codex:primary', 'Codex 5h', NULL, 0.55, NULL, NULL),
 (1700007200000, 'openai-codex', 'c-1', 'c@example.com', NULL, 'codex:primary', 'Codex 5h', NULL, 0.05, NULL, NULL);

-- Dense series (1000 points) to exercise downsampling to MAX_SERIES_POINTS.
INSERT INTO usage_history (recorded_at, provider, account_key, email, account_id, limit_id, label, window_label, used_fraction, status, resets_at)
WITH RECURSIVE seq(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM seq WHERE i < 999)
SELECT 1700000000000 + i * 60000, 'openai-codex', 'c-2', 'dense@example.com', NULL, 'codex:primary', 'Codex 5h', NULL,
       (i % 97) / 100.0, CASE WHEN i % 97 > 95 THEN 'exhausted' END, NULL
FROM seq;
