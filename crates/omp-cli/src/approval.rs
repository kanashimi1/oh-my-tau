//! Tool approval gate (TS `tools/approval.ts` + `ExtensionToolWrapper.execute` step 2).
//!
//! Every tool call is resolved against the active approval mode (`--approval-mode` >
//! `tools.approvalMode`, default `yolo`; `--auto-approve` forces `yolo`) and the user's
//! `tools.approval.<tool>` policies. A resolved `prompt` emits `tool_approval_requested`, asks the
//! UI (`select(prompt, ["Approve", "Deny"])`: a TUI dialog or an rpc `extension_ui_request`
//! frame) and emits `tool_approval_resolved`. Without a UI the call fails closed.

use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};

use omp_ext::lifecycle::ApprovalMode;
use pa_agent::abort::AbortSignal;
use pa_agent::types::AgentTool;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::runtime::Runtime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Read,
    Write,
    Exec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Allow,
    Deny,
    Prompt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Tool,
    User,
    Mode,
}

/// TS `ToolApprovalDecision`, normalized.
#[derive(Debug, Clone)]
pub struct Decision {
    pub tier: Tier,
    pub policy: Option<Policy>,
    pub reason: Option<String>,
    pub overridden: bool,
    pub policy_key: Option<String>,
}

impl Decision {
    #[must_use]
    pub fn tier(tier: Tier) -> Self {
        Self {
            tier,
            policy: None,
            reason: None,
            overridden: false,
            policy_key: None,
        }
    }
    fn with_policy(tier: Tier, policy: Policy, overridden: bool, reason: Option<String>) -> Self {
        Self {
            tier,
            policy: Some(policy),
            reason,
            overridden,
            policy_key: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub policy: Policy,
    pub reason: Option<String>,
    pub source: Source,
    pub policy_key: Option<String>,
}

#[must_use]
pub fn parse_mode(value: &str) -> Option<ApprovalMode> {
    match value {
        "always-ask" => Some(ApprovalMode::AlwaysAsk),
        "write" => Some(ApprovalMode::Write),
        "yolo" => Some(ApprovalMode::Yolo),
        _ => None,
    }
}

fn mode_max_tier(mode: ApprovalMode) -> Tier {
    match mode {
        ApprovalMode::AlwaysAsk => Tier::Read,
        ApprovalMode::Write => Tier::Write,
        ApprovalMode::Yolo => Tier::Exec,
    }
}

fn normalize_policy(value: &Value) -> Option<Policy> {
    match value.as_str()?.trim().to_lowercase().as_str() {
        "allow" => Some(Policy::Allow),
        "deny" => Some(Policy::Deny),
        "prompt" => Some(Policy::Prompt),
        _ => None,
    }
}

/// TS `resolveApproval`.
#[must_use]
pub fn resolve_approval(
    tool: &str,
    decision: &Decision,
    mode: ApprovalMode,
    user: &Map<String, Value>,
) -> Resolved {
    let policy_key = decision
        .policy_key
        .clone()
        .unwrap_or_else(|| tool.to_owned());
    let user_policy = user.get(&policy_key).and_then(normalize_policy);
    let fallback = if policy_key != tool && user_policy.is_none() {
        user.get(tool).and_then(normalize_policy)
    } else {
        None
    };
    let combined = user_policy.or(fallback);
    let combined_key = combined.map(|_| {
        if user_policy.is_some() {
            policy_key.clone()
        } else {
            tool.to_owned()
        }
    });
    let tool_key = decision.policy_key.clone();
    let from_tool = |policy: Policy| Resolved {
        policy,
        reason: decision.reason.clone(),
        source: Source::Tool,
        policy_key: tool_key.clone(),
    };

    if decision.policy == Some(Policy::Deny) {
        return from_tool(Policy::Deny);
    }
    if combined == Some(Policy::Deny) {
        return Resolved {
            policy: Policy::Deny,
            reason: None,
            source: Source::User,
            policy_key: combined_key,
        };
    }
    if mode == ApprovalMode::Yolo {
        if let Some(policy) = decision.policy {
            return from_tool(policy);
        }
        return Resolved {
            policy: combined.unwrap_or(Policy::Allow),
            reason: None,
            source: if combined.is_some() {
                Source::User
            } else {
                Source::Mode
            },
            policy_key: combined_key,
        };
    }
    if decision.overridden {
        let policy = if decision.policy == Some(Policy::Allow) {
            Policy::Allow
        } else {
            Policy::Prompt
        };
        return from_tool(policy);
    }
    if let Some(policy @ (Policy::Allow | Policy::Prompt)) = decision.policy {
        return from_tool(policy);
    }
    if let Some(policy) = combined {
        return Resolved {
            policy,
            reason: None,
            source: Source::User,
            policy_key: combined_key,
        };
    }
    if decision.tier <= mode_max_tier(mode) {
        return Resolved {
            policy: Policy::Allow,
            reason: None,
            source: Source::Mode,
            policy_key: None,
        };
    }
    Resolved {
        policy: Policy::Prompt,
        reason: decision.reason.clone(),
        source: Source::Mode,
        policy_key: None,
    }
}

/// TS `denyError` message.
#[must_use]
pub fn deny_message(resolved: &Resolved, tool: &str) -> String {
    if resolved.source == Source::Tool {
        return format!(
            "Tool \"{tool}\" is blocked by tool policy.{}",
            resolved
                .reason
                .as_ref()
                .map_or_else(String::new, |r| format!("\nReason: {r}"))
        );
    }
    let key = resolved.policy_key.as_deref().unwrap_or(tool);
    format!(
        "Tool \"{key}\" is blocked by user policy.\nTo allow: remove \"tools.approval.{key}: deny\" from config."
    )
}

/// TS `truncateForPrompt` (2000 UTF-16 units; counted in chars here).
#[must_use]
pub fn truncate_for_prompt(value: &str) -> String {
    const MAX: usize = 2000;
    let n = value.chars().count();
    if n <= MAX {
        return value.to_owned();
    }
    let head: String = value.chars().take(MAX).collect();
    format!("{head}[…{}ch elided…]", n - MAX)
}

// ---------------------------------------------------------------------------------------------
// Per-tool declarations (`approval` / `formatApprovalDetails` of each TS tool).
// ---------------------------------------------------------------------------------------------

fn strictest(decisions: Vec<Decision>) -> Decision {
    let mut tier = Tier::Read;
    for d in decisions {
        if d.policy == Some(Policy::Deny) {
            return d;
        }
        if d.tier > tier {
            tier = d.tier;
        }
    }
    Decision::tier(tier)
}

fn scheme_of(input: &str) -> Option<String> {
    let (scheme, _) = input.split_once("://")?;
    (!scheme.is_empty() && scheme.chars().all(|c| c.is_ascii_alphabetic()))
        .then(|| scheme.to_lowercase())
}

/// TS `InternalUrlRouter.writeTier`: non-URLs are `write`; a registered scheme's `spec.write.tier`;
/// a registered scheme without a write policy is denied as read-only.
fn write_tier(input: &str) -> Decision {
    let Some(scheme) = scheme_of(input) else {
        return Decision::tier(Tier::Write);
    };
    match scheme.as_str() {
        "vault" | "cfg" | "conflict" => Decision::tier(Tier::Write),
        "agent" | "local" => Decision::tier(Tier::Read),
        "proc" | "ssh" | "xd" => Decision::tier(Tier::Exec),
        "artifact" | "mcp" | "issue" | "pr" | "omp" | "history" | "rule" | "security" | "skill"
        | "attachment" | "memory" => Decision::with_policy(
            Tier::Write,
            Policy::Deny,
            false,
            Some(format!("{scheme}:// URLs are read-only")),
        ),
        _ => Decision::tier(Tier::Write),
    }
}

/// TS `InternalUrlRouter.readTier`: `ssh://` anywhere in the text is `exec`.
fn read_tier(text: &str) -> Tier {
    if text.to_lowercase().contains("ssh://") {
        Tier::Exec
    } else {
        Tier::Read
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn path_list(args: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for key in ["path", "paths"] {
        match args.get(key) {
            Some(Value::String(s)) => out.push(s.clone()),
            Some(Value::Array(a)) => {
                out.extend(a.iter().filter_map(Value::as_str).map(str::to_owned))
            }
            _ => {}
        }
    }
    out
}

/// Paths an `edit` payload writes: the `path` argument, per-edit paths, and hashline /
/// apply_patch section headers of the `input` payload (plus move destinations).
fn edit_targets(args: &Value) -> Vec<String> {
    let mut targets = Vec::new();
    if let Some(p) = str_arg(args, "path") {
        targets.push(p.to_owned());
    }
    if let Some(edits) = args.get("edits").and_then(Value::as_array) {
        for e in edits {
            if let Some(p) = str_arg(e, "path") {
                targets.push(p.to_owned());
            }
        }
    }
    if let Some(input) = str_arg(args, "input") {
        for line in input.lines() {
            let line = line.trim_end();
            if let Some(rest) = line.strip_prefix('[') {
                if let Some(body) = rest.strip_suffix(']') {
                    let path = body.rsplit_once('#').map_or(body, |(p, _)| p);
                    if !path.is_empty() {
                        targets.push(path.to_owned());
                    }
                }
            } else if let Some(p) = [
                "*** Add File: ",
                "*** Update File: ",
                "*** Delete File: ",
                "*** Move to: ",
            ]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
            {
                targets.push(p.trim().to_owned());
            }
        }
    }
    targets.dedup();
    targets
}

const LSP_READONLY_ACTIONS: &[&str] = &[
    "diagnostics",
    "definition",
    "type_definition",
    "implementation",
    "references",
    "hover",
    "symbols",
    "status",
    "capabilities",
];

/// Critical bash shapes (TS `CRITICAL_BASH_PATTERNS`).
static CRITICAL_BASH: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)\brm\s+(?:-\S+\s+)*(?:-[a-z]*[rRfF][a-z]*|--recursive|--force)\s+(?:-\S+\s+)*/",
        r"(?i)\brm\s+(?:-\S+\s+)*--no-preserve-root\b",
        r"(?i)\bsudo\s+rm\b",
        r"(?i)\bchmod\s+-R\s+[0-7]+\s+/",
        r"\bchmod\s+-R\s+[ugoa+\-=rwxXst,]+\s+/",
        r"(?i)\bchown\s+-R\s+\S+\s+/",
        r"(?i):\(\)\s*\{\s*:\s*\|\s*:",
        r"(?i)>\s*/dev/sd[a-z]",
        r"(?i)\bmkfs(\.|\b)",
        r"(?i)\bdd\s+if=.+of=/dev/",
        r"(?i)\bshred\s+/dev/",
        r"(?i)\bcryptsetup\b",
        r"(?i)>\s*/etc/(?:passwd|shadow|sudoers)\b",
        r"(?i)\btee\s+(?:-a\s+)?/etc/(?:passwd|shadow|sudoers)\b",
        r"(?i)\b(?:curl|wget|fetch)\b[^|]*\|\s*(?:bash|sh|zsh|fish)\b",
        r"(?i)(?:^|[\s;&|(])(?:bash|sh|zsh|source|\.)\s+<\(\s*(?:curl|wget|fetch)\b",
        r"(?i)\beval\s+[\x22'`]?\$\(\s*(?:curl|wget|fetch)\b|\beval\s+`\s*(?:curl|wget|fetch)\b",
        r"\bkill\s+-9\s+1\b",
        r"(?i)(?:^|[\s;&|(])(?:shutdown|poweroff|reboot|halt)(?:\s|$|[;|&])",
        r"(?i)(?:^|[\s;&|(])init\s+0\b",
        r"(?i)\bnc\b[^|;]*\s-[a-zA-Z]*[ec][a-zA-Z]*\s",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("critical bash pattern"))
    .collect()
});

static REINTERPRETED_ARG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[ \t])(?:-[^-]*[ce]|--(?:command|eval))(?:[= \t]|$)").expect("regex")
});

fn is_shell_control(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r' | ';' | '&' | '|' | '<' | '>' | '`' | '$' | '(' | ')'
    )
}

/// TS `hasBashApprovalShellControl`.
fn has_shell_control(command: &str) -> bool {
    let chars: Vec<char> = command.chars().collect();
    let mut quote: Option<char> = None;
    let mut reinterpretable = false;
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if quote == Some('\'') {
            if ch == '\'' {
                quote = None;
            } else if is_shell_control(ch) {
                reinterpretable = true;
            }
            i += 1;
            continue;
        }
        if ch == '\\' {
            if chars.get(i + 1).is_some_and(|c| is_shell_control(*c)) {
                reinterpretable = true;
            }
            i += 2;
            continue;
        }
        if quote == Some('"') {
            if ch == '"' {
                quote = None;
                i += 1;
                continue;
            }
            if ch == '`' || ch == '$' {
                return true;
            }
            if is_shell_control(ch) {
                reinterpretable = true;
            }
            i += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            i += 1;
            continue;
        }
        if is_shell_control(ch) {
            return true;
        }
        i += 1;
    }
    reinterpretable && REINTERPRETED_ARG.is_match(command)
}

/// TS `tokenizeShellSegments`, each segment's words joined by a space.
fn bash_segments(command: &str) -> Vec<String> {
    let chars: Vec<char> = command.chars().collect();
    let mut segments: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut buffer = String::new();
    let (mut in_single, mut in_double) = (false, false);
    fn push_buffer(buffer: &mut String, current: &mut Vec<String>) {
        if !buffer.is_empty() {
            current.push(std::mem::take(buffer));
        }
    }
    fn push_segment(
        buffer: &mut String,
        current: &mut Vec<String>,
        segments: &mut Vec<Vec<String>>,
    ) {
        push_buffer(buffer, current);
        if !current.is_empty() {
            segments.push(std::mem::take(current));
        }
    }
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                buffer.push(ch);
            }
            i += 1;
            continue;
        }
        if in_double {
            if ch == '\\' && i + 1 < chars.len() {
                let next = chars[i + 1];
                if matches!(next, '"' | '\\' | '$' | '`') {
                    buffer.push(next);
                    i += 2;
                    continue;
                }
            }
            if ch == '"' {
                in_double = false;
            } else {
                buffer.push(ch);
            }
            i += 1;
            continue;
        }
        match ch {
            '\'' => in_single = true,
            '"' => in_double = true,
            '\\' if i + 1 < chars.len() => {
                buffer.push(chars[i + 1]);
                i += 1;
            }
            ' ' | '\t' => push_buffer(&mut buffer, &mut current),
            '\n' | ';' | '&' | '|' | '(' | ')' => {
                push_segment(&mut buffer, &mut current, &mut segments)
            }
            _ => buffer.push(ch),
        }
        i += 1;
    }
    push_segment(&mut buffer, &mut current, &mut segments);
    segments
        .into_iter()
        .map(|s| s.join(" "))
        .filter(|s| !s.is_empty())
        .collect()
}

fn normalize_pattern(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn pattern_regex(pattern: &str) -> Option<Regex> {
    let escaped = normalize_pattern(pattern)
        .split('*')
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join(".*");
    Regex::new(&format!("^{escaped}$")).ok()
}

struct BashRule {
    matcher: String,
    approval: Policy,
}

fn bash_rules(settings: &omp_settings::Settings) -> Vec<BashRule> {
    let Value::Array(items) = settings.get("bash.patterns") else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let record = item.as_object()?;
            let matcher = normalize_pattern(record.get("match")?.as_str()?);
            let approval = normalize_policy(record.get("approval")?)?;
            (!matcher.is_empty()).then_some(BashRule { matcher, approval })
        })
        .collect()
}

fn rule_matches(command: &str, rule: &BashRule) -> bool {
    let Some(re) = pattern_regex(&rule.matcher) else {
        return false;
    };
    let normalized = normalize_pattern(command);
    if normalized.is_empty() {
        return false;
    }
    if rule.approval == Policy::Allow {
        return !has_shell_control(command) && re.is_match(&normalized);
    }
    re.is_match(&normalized) || bash_segments(command).iter().any(|s| re.is_match(s))
}

fn bash_decision(args: &Value, settings: &omp_settings::Settings) -> Decision {
    let command = str_arg(args, "command").unwrap_or("");
    let rules = bash_rules(settings);
    let rule = rules.iter().find(|r| rule_matches(command, r));
    if let Some(rule) = rule.filter(|r| r.approval == Policy::Deny) {
        return Decision::with_policy(
            Tier::Exec,
            Policy::Deny,
            true,
            Some(format!("Blocked by bash pattern: {}", rule.matcher)),
        );
    }
    if !command.is_empty() && CRITICAL_BASH.iter().any(|p| p.is_match(command)) {
        return Decision {
            tier: Tier::Exec,
            policy: None,
            reason: Some("Critical pattern detected".into()),
            overridden: true,
            policy_key: None,
        };
    }
    match rule {
        Some(r) if r.approval == Policy::Allow => {
            Decision::with_policy(Tier::Write, Policy::Allow, false, None)
        }
        Some(r) => Decision::with_policy(
            Tier::Exec,
            Policy::Prompt,
            true,
            Some(format!("Prompt required by bash pattern: {}", r.matcher)),
        ),
        None => Decision::tier(Tier::Exec),
    }
}

/// TS `tool.approval(args)` for the tools omp-rs ships; unknown tools default to `exec`.
#[must_use]
pub fn tool_decision(name: &str, args: &Value, settings: &omp_settings::Settings) -> Decision {
    match name {
        "bash" => bash_decision(args, settings),
        "read" => Decision::tier(read_tier(str_arg(args, "path").unwrap_or(""))),
        "grep" => Decision::tier(read_tier(&path_list(args).join("\n"))),
        "write" => match str_arg(args, "path") {
            Some(path) => write_tier(path),
            None => Decision::tier(Tier::Write),
        },
        "edit" => {
            let targets = edit_targets(args);
            if targets.is_empty() {
                Decision::tier(Tier::Write)
            } else {
                strictest(targets.iter().map(|t| write_tier(t)).collect())
            }
        }
        "ast_edit" => match args.get("paths").and_then(Value::as_array) {
            Some(paths) if !paths.is_empty() => {
                let mut decisions = Vec::new();
                for p in paths {
                    let Some(p) = p.as_str() else {
                        return Decision::tier(Tier::Exec);
                    };
                    decisions.push(write_tier(p));
                }
                strictest(decisions)
            }
            _ => Decision::tier(Tier::Write),
        },
        "lsp" => {
            let action = str_arg(args, "action")
                .map(str::to_lowercase)
                .unwrap_or_default();
            Decision::tier(if LSP_READONLY_ACTIONS.contains(&action.as_str()) {
                Tier::Read
            } else {
                Tier::Write
            })
        }
        "glob" | "find" | "ast_grep" | "web_search" | "ask" | "think" | "wait" | "yield"
        | "todo" | "checkpoint" | "rewind" | "context" => Decision::tier(Tier::Read),
        n if n.starts_with("mcp__") => Decision::tier(Tier::Write),
        _ => Decision::tier(Tier::Exec),
    }
}

/// TS `formatApprovalDetails`.
fn approval_details(name: &str, args: &Value) -> Vec<String> {
    let t = truncate_for_prompt;
    match name {
        "bash" => vec![format!(
            "Command: {}",
            t(str_arg(args, "command").unwrap_or("(missing)"))
        )],
        "write" => vec![
            format!("Path: {}", t(str_arg(args, "path").unwrap_or("(missing)"))),
            format!("Content:\n{}", t(str_arg(args, "content").unwrap_or(""))),
        ],
        "edit" => {
            let targets = edit_targets(args);
            if targets.is_empty() {
                vec!["File: (unknown)".into()]
            } else {
                targets.iter().map(|p| format!("File: {}", t(p))).collect()
            }
        }
        "eval" => {
            let language = str_arg(args, "language")
                .map_or_else(|| "javascript (default)".to_owned(), str::to_owned);
            vec![
                format!("Language: {language}"),
                format!("Code:\n{}", t(str_arg(args, "code").unwrap_or(""))),
            ]
        }
        "lsp" => {
            let mut lines = vec![format!(
                "Action: {}",
                str_arg(args, "action").unwrap_or("(missing)")
            )];
            if let Some(file) = str_arg(args, "file").filter(|f| !f.is_empty()) {
                lines.push(format!("File: {}", t(file)));
            }
            lines
        }
        "ast_edit" => {
            let mut lines = Vec::new();
            let ops = args
                .get("ops")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if let Some(first) = ops.first() {
                lines.push(format!(
                    "Pattern: {}",
                    t(str_arg(first, "pat").unwrap_or("undefined"))
                ));
                lines.push(format!(
                    "Replacement: {}",
                    t(str_arg(first, "out").unwrap_or("undefined"))
                ));
                if ops.len() > 1 {
                    lines.push(format!(
                        "+{} more op{}",
                        ops.len() - 1,
                        if ops.len() == 2 { "" } else { "s" }
                    ));
                }
            }
            if let Some(paths) = args
                .get("paths")
                .and_then(Value::as_array)
                .filter(|p| !p.is_empty())
            {
                let joined = paths
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!("Paths: {}", t(&joined)));
            }
            lines
        }
        "task" => task_details(args),
        _ => Vec::new(),
    }
}

fn task_details(args: &Value) -> Vec<String> {
    let t = truncate_for_prompt;
    let mut lines = Vec::new();
    if let Some(agent) = str_arg(args, "agent") {
        lines.push(format!("Agent: {}", t(agent)));
    }
    if let Some(name) = str_arg(args, "name").filter(|n| !n.trim().is_empty()) {
        lines.push(format!("Name: {}", t(name)));
    }
    if let Some(task) = str_arg(args, "task") {
        lines.push(format!("Task:\n{}", t(task)));
    }
    if let Some(context) = str_arg(args, "context").filter(|c| !c.trim().is_empty()) {
        lines.push(format!("Context:\n{}", t(context)));
    }
    let tasks = args
        .get("tasks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !tasks.is_empty() {
        let effective = |item: &Value| -> String {
            item.get("agent")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|a| !a.is_empty())
                .map_or_else(|| "task".to_owned(), str::to_owned)
        };
        let mut counts: Vec<(String, usize)> = Vec::new();
        for item in &tasks {
            let agent = effective(item);
            match counts.iter_mut().find(|(a, _)| *a == agent) {
                Some((_, n)) => *n += 1,
                None => counts.push((agent, 1)),
            }
        }
        let summary = counts
            .iter()
            .map(|(a, n)| format!("{a} ×{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("Batch agents: {}", t(&summary)));
        let first = &tasks[0];
        if first.is_object() {
            if let Some(name) = str_arg(first, "name").filter(|n| !n.trim().is_empty()) {
                lines.push(format!("Name: {}", t(name)));
            }
            lines.push(format!("Agent: {}", t(&effective(first))));
            if let Some(task) = str_arg(first, "task") {
                lines.push(format!("Task:\n{}", t(task)));
            }
        }
        if tasks.len() > 1 {
            lines.push(format!(
                "+{} more task{}",
                tasks.len() - 1,
                if tasks.len() == 2 { "" } else { "s" }
            ));
        }
    }
    lines
}

/// TS `formatApprovalPrompt`.
#[must_use]
pub fn format_prompt(name: &str, args: &Value, reason: Option<&str>) -> String {
    let mut lines = vec![format!("Allow tool: {name}")];
    if let Some(reason) = reason.filter(|r| !r.is_empty()) {
        lines.push(format!("Reason: {reason}"));
    }
    lines.extend(
        approval_details(name, args)
            .into_iter()
            .filter(|l| !l.is_empty()),
    );
    lines.join("\n")
}

// ---------------------------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------------------------

/// Where a prompt goes: the live runtime (its `extension_ui_request` frame stream) when the
/// session has a UI (tui, non-headless rpc), nothing in print/json mode.
#[derive(Default)]
struct Binding {
    rt: Option<Weak<Runtime>>,
    has_ui: bool,
}

pub struct Gate {
    /// `--approval-mode` (runtime override, not persisted).
    mode_override: Option<ApprovalMode>,
    /// `--auto-approve` / `--yolo`.
    auto_approve: bool,
    cwd: std::path::PathBuf,
    ext: Option<Arc<crate::ext::Extensions>>,
    binding: Mutex<Binding>,
    /// Calls the gate refused: TS throws before executing them, so no `tool_result` event follows.
    refused: Mutex<std::collections::HashSet<String>>,
}

static GATE: OnceLock<Arc<Gate>> = OnceLock::new();

impl Gate {
    /// Install the process gate (once; the first call wins).
    pub fn install(
        mode_override: Option<ApprovalMode>,
        auto_approve: bool,
        cwd: &std::path::Path,
        ext: Option<&Arc<crate::ext::Extensions>>,
    ) -> Arc<Self> {
        let gate = Arc::new(Self {
            mode_override,
            auto_approve,
            cwd: cwd.to_path_buf(),
            ext: ext.cloned(),
            binding: Mutex::new(Binding::default()),
            refused: Mutex::new(std::collections::HashSet::new()),
        });
        let _ = GATE.set(gate.clone());
        gate
    }

    /// The installed gate, if any.
    #[must_use]
    pub fn get() -> Option<Arc<Self>> {
        GATE.get().cloned()
    }

    /// Whether the gate refused call `id` (consumed once).
    #[must_use]
    pub fn take_refused(&self, id: &str) -> bool {
        self.refused
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id)
    }

    /// Bind the live runtime and whether the session has a UI (TS `hasUI()`).
    pub fn bind(&self, rt: &Arc<Runtime>, has_ui: bool) {
        let mut b = self
            .binding
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        b.rt = Some(Arc::downgrade(rt));
        b.has_ui = has_ui;
    }

    fn mode_and_policies(
        &self,
        settings: &omp_settings::Settings,
    ) -> (ApprovalMode, Map<String, Value>) {
        let policies = settings.get_record("tools.approval");
        if self.auto_approve {
            return (ApprovalMode::Yolo, policies);
        }
        let mode = self
            .mode_override
            .or_else(|| {
                settings
                    .get_string("tools.approvalMode")
                    .as_deref()
                    .and_then(parse_mode)
            })
            .unwrap_or(ApprovalMode::Yolo);
        (mode, policies)
    }

    /// Run the gate for one call on its effective arguments. `Err(message)` blocks the call (the
    /// message is the tool error TS throws).
    pub async fn check(
        &self,
        name: &str,
        call_id: &str,
        args: &Value,
        signal: &AbortSignal,
    ) -> Result<(), String> {
        let settings = omp_settings::Settings::load(&self.cwd);
        let (mode, policies) = self.mode_and_policies(&settings);
        let decision = tool_decision(name, args, &settings);
        let resolved = resolve_approval(name, &decision, mode, &policies);
        if resolved.policy == Policy::Deny {
            return Err(deny_message(&resolved, name));
        }
        if resolved.policy != Policy::Prompt {
            return Ok(());
        }
        let (rt, has_ui) = {
            let b = self
                .binding
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (b.rt.as_ref().and_then(Weak::upgrade), b.has_ui)
        };
        let session_id = rt.as_ref().map(|r| r.session_id()).unwrap_or_default();
        let observed = self.ext.as_ref().filter(|e| {
            e.host.has_handlers("tool_approval_requested")
                || e.host.has_handlers("tool_approval_resolved")
        });
        if let Some(ext) = observed {
            let r = ext
                .host
                .emit_tool_approval_requested(
                    session_id.clone(),
                    name.to_owned(),
                    call_id.to_owned(),
                    resolved.reason.clone(),
                    mode,
                    None,
                )
                .await;
            report(r);
        }
        let resolve_event = |approved: bool, reason: Option<String>| {
            let (ext, session_id) = (observed.cloned(), session_id.clone());
            let (name, call_id) = (name.to_owned(), call_id.to_owned());
            async move {
                if let Some(ext) = ext {
                    report(
                        ext.host
                            .emit_tool_approval_resolved(
                                session_id, name, call_id, approved, reason, None,
                            )
                            .await,
                    );
                }
            }
        };
        let (true, Some(rt)) = (has_ui, rt) else {
            resolve_event(false, Some("no interactive UI available".into())).await;
            return Err(format!(
                "Tool \"{name}\" requires approval but no interactive UI available.\nOptions:\n  1. Set tools.approvalMode: yolo in /settings\n  2. Add tools.approval.{name}: allow to config\n  3. Use an interactive UI to approve the tool call"
            ));
        };
        let prompt = format_prompt(name, args, resolved.reason.as_deref());
        let choice = tokio::select! {
            c = select(&rt, &prompt, &["Approve", "Deny"]) => Ok(c),
            () = signal.aborted() => Err(()),
        };
        match choice {
            Err(()) => {
                resolve_event(false, Some("approval aborted".into())).await;
                Err("Tool execution was cancelled while waiting for approval".to_owned())
            }
            Ok(choice) => {
                let approved = choice.as_deref() == Some("Approve");
                resolve_event(approved, (!approved).then(|| "denied by user".to_owned())).await;
                if approved {
                    Ok(())
                } else {
                    Err(format!("Tool call denied by user: {name}"))
                }
            }
        }
    }

    /// Wrap every tool with the gate (TS `ExtensionToolWrapper`): the call runs the approval check
    /// on its effective arguments (after the extension `tool_call` hook and `tool_execution_start`)
    /// right before the tool executes.
    #[must_use]
    pub fn wrap_tools(self: &Arc<Self>, tools: Vec<Arc<dyn AgentTool>>) -> Vec<Arc<dyn AgentTool>> {
        tools
            .into_iter()
            .map(|inner| {
                Arc::new(GatedTool {
                    inner,
                    gate: self.clone(),
                }) as Arc<dyn AgentTool>
            })
            .collect()
    }
}

struct GatedTool {
    inner: Arc<dyn AgentTool>,
    gate: Arc<Gate>,
}

impl AgentTool for GatedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn parameters(&self) -> &Value {
        self.inner.parameters()
    }
    fn label(&self) -> &str {
        self.inner.label()
    }
    fn prepare_arguments(&self, args: &Value) -> Option<Value> {
        self.inner.prepare_arguments(args)
    }
    fn execution_mode(&self) -> Option<pa_agent::types::ToolExecutionMode> {
        self.inner.execution_mode()
    }
    fn execute(
        self: Arc<Self>,
        tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        on_update: pa_agent::types::AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, anyhow::Result<pa_agent::types::AgentToolResult>> {
        Box::pin(async move {
            if let Err(message) = self
                .gate
                .check(self.inner.name(), &tool_call_id, &params, &signal)
                .await
            {
                self.gate
                    .refused
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(tool_call_id);
                return Err(anyhow::Error::msg(message));
            }
            self.inner
                .clone()
                .execute(tool_call_id, params, signal, on_update)
                .await
        })
    }
}

fn report(r: anyhow::Result<omp_ext::EmitResult>) {
    match r {
        Ok(r) => {
            for err in r.errors {
                eprintln!("omt: extension handler error: {err}");
            }
        }
        Err(e) => eprintln!("omt: extension emit: {e:#}"),
    }
}

/// `ctx.ui.select` through the session's frame stream: a TUI dialog or an rpc client's
/// `extension_ui_response`. `None` = cancelled.
async fn select(rt: &Arc<Runtime>, title: &str, options: &[&str]) -> Option<String> {
    let id = format!(
        "ext-ui-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    );
    let waiter = crate::rpc::register_pending_ui_request(&id);
    rt.send(json!({
        "type": "extension_ui_request",
        "id": id,
        "method": "select",
        "title": title,
        "options": options,
    }));
    let answer = waiter.await.ok()?;
    if answer.get("cancelled").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    answer
        .get("value")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> omp_settings::Settings {
        omp_settings::Settings::load_from(None, std::path::Path::new("/nonexistent"))
    }

    fn resolve(tool: &str, args: &Value, mode: ApprovalMode, user: &Value) -> Resolved {
        let d = tool_decision(tool, args, &settings());
        resolve_approval(tool, &d, mode, user.as_object().unwrap_or(&Map::new()))
    }

    #[test]
    fn yolo_allows_every_tier() {
        for (t, a) in [
            ("bash", json!({"command":"ls"})),
            ("write", json!({"path":"a","content":""})),
            ("read", json!({"path":"a"})),
        ] {
            assert_eq!(
                resolve(t, &a, ApprovalMode::Yolo, &json!({})).policy,
                Policy::Allow,
                "{t}"
            );
        }
    }

    #[test]
    fn modes_compare_tiers() {
        let w = json!({"path":"a","content":"x"});
        let b = json!({"command":"ls"});
        let r = json!({"path":"a"});
        assert_eq!(
            resolve("read", &r, ApprovalMode::AlwaysAsk, &json!({})).policy,
            Policy::Allow
        );
        assert_eq!(
            resolve("write", &w, ApprovalMode::AlwaysAsk, &json!({})).policy,
            Policy::Prompt
        );
        assert_eq!(
            resolve("write", &w, ApprovalMode::Write, &json!({})).policy,
            Policy::Allow
        );
        assert_eq!(
            resolve("bash", &b, ApprovalMode::Write, &json!({})).policy,
            Policy::Prompt
        );
        assert_eq!(
            resolve("bash", &b, ApprovalMode::AlwaysAsk, &json!({})).policy,
            Policy::Prompt
        );
    }

    #[test]
    fn user_policy_wins_over_mode_but_deny_wins_everywhere() {
        let b = json!({"command":"ls"});
        assert_eq!(
            resolve(
                "bash",
                &b,
                ApprovalMode::AlwaysAsk,
                &json!({"bash":"allow"})
            )
            .policy,
            Policy::Allow
        );
        assert_eq!(
            resolve("bash", &b, ApprovalMode::Yolo, &json!({"bash":"prompt"})).policy,
            Policy::Prompt
        );
        let d = resolve("bash", &b, ApprovalMode::Yolo, &json!({"bash":"deny"}));
        assert_eq!(d.policy, Policy::Deny);
        assert_eq!(
            deny_message(&d, "bash"),
            "Tool \"bash\" is blocked by user policy.\nTo allow: remove \"tools.approval.bash: deny\" from config."
        );
    }

    #[test]
    fn critical_bash_prompts_outside_yolo_only() {
        let rm = json!({"command":"rm -rf /"});
        assert_eq!(
            resolve("bash", &rm, ApprovalMode::Yolo, &json!({})).policy,
            Policy::Allow
        );
        let r = resolve("bash", &rm, ApprovalMode::Write, &json!({"bash":"allow"}));
        assert_eq!(r.policy, Policy::Prompt);
        assert_eq!(r.reason.as_deref(), Some("Critical pattern detected"));
    }

    #[test]
    fn read_only_scheme_write_is_denied_by_tool_policy() {
        let r = resolve(
            "write",
            &json!({"path":"skill://x","content":""}),
            ApprovalMode::Yolo,
            &json!({}),
        );
        assert_eq!(r.policy, Policy::Deny);
        assert_eq!(
            deny_message(&r, "write"),
            "Tool \"write\" is blocked by tool policy.\nReason: skill:// URLs are read-only"
        );
    }

    #[test]
    fn prompt_text_matches_ts_format() {
        assert_eq!(
            format_prompt(
                "bash",
                &json!({"command":"ls -la"}),
                Some("Critical pattern detected")
            ),
            "Allow tool: bash\nReason: Critical pattern detected\nCommand: ls -la"
        );
        assert_eq!(
            format_prompt("write", &json!({"path":"a.txt","content":"hi"}), None),
            "Allow tool: write\nPath: a.txt\nContent:\nhi"
        );
        let long = "x".repeat(2005);
        assert!(truncate_for_prompt(&long).ends_with("[…5ch elided…]"));
    }

    #[test]
    fn bash_segments_split_on_operators() {
        assert_eq!(
            bash_segments("cd x && rm -rf 'a b'; ls | wc"),
            vec!["cd x", "rm -rf a b", "ls", "wc"]
        );
        assert!(has_shell_control("ls; rm x"));
        assert!(!has_shell_control("git status"));
    }
}
