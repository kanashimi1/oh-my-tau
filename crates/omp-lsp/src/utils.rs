//! URI handling, language-id detection and output formatting (omp `lsp/utils.ts`
//! plus the theme-symbol defaults from pi-tui's symbol table).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::types::{Diagnostic, Location, SymbolInformation, SYMBOL_KIND_NAMES};

// =============================================================================
// Theme symbol defaults (pi-tui `theme/symbols.ts`)
// =============================================================================

pub const STATUS_SUCCESS: &str = "✔";
pub const STATUS_ERROR: &str = "✘";
pub const STATUS_WARNING: &str = "⚠";
pub const STATUS_INFO: &str = "ⓘ";
pub const NAV_CURSOR: &str = "❯";
pub const FORMAT_BULLET: &str = "•";
pub const FORMAT_DASH: &str = "—";
pub const ICON_FILE: &str = "📄";
pub const ICON_FOLDER: &str = "📁";
pub const ICON_PACKAGE: &str = "📦";
pub const ICON_MODEL: &str = "⬢";
pub const ICON_AUTO: &str = "⟲";

// =============================================================================
// URI Handling (Cross-Platform)
// =============================================================================

/// Convert a file path to a `file://` URI with percent-encoding.
#[must_use]
pub fn file_to_uri(file_path: &Path) -> String {
    let absolute = if file_path.is_absolute() {
        file_path.to_path_buf()
    } else {
        std::env::current_dir().map_or_else(|_| file_path.to_path_buf(), |cwd| cwd.join(file_path))
    };
    let mut s = absolute.to_string_lossy().into_owned();
    // Normalize Windows separators; keep the leading `/`.
    if std::path::MAIN_SEPARATOR == '\\' {
        s = s.replace('\\', "/");
    }
    if !s.starts_with('/') {
        s.insert(0, '/');
    }
    percent_encode_path(&s)
}

fn percent_encode_path(path: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("file://");
    for byte in path.bytes() {
        let c = byte as char;
        let keep = c.is_ascii_alphanumeric()
            || matches!(
                c,
                '-' | '.'
                    | '_'
                    | '~'
                    | '/'
                    | ':'
                    | '@'
                    | '+'
                    | '='
                    | ','
                    | '&'
                    | '$'
                    | '!'
                    | '\''
                    | '('
                    | ')'
                    | '*'
            );
        if keep {
            out.push(c);
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

/// Convert a `file://` URI to a file path, tolerating lax unencoded URIs.
#[must_use]
pub fn uri_to_file(uri: &str) -> PathBuf {
    if !uri.starts_with("file://") {
        return PathBuf::from(uri);
    }
    // Raw `#`/`?` would be treated as fragment/query and silently truncate.
    let mut rest = &uri[7..];
    if let Some(idx) = rest.find(['#', '?']) {
        rest = &rest[..idx];
    }
    let decoded = percent_decode(rest);
    // Windows: file:///C:/path → C:/path
    #[cfg(windows)]
    {
        if decoded.starts_with('/') && decoded.len() > 3 && decoded.as_bytes()[2] == b':' {
            return PathBuf::from(&decoded[1..]);
        }
    }
    PathBuf::from(decoded)
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Normalize a URI (or raw path) to a canonical filesystem key.
#[must_use]
pub fn uri_key(uri: &str) -> PathBuf {
    let path = if uri.starts_with("file://") {
        uri_to_file(uri)
    } else {
        PathBuf::from(uri)
    };
    normalize_path(&path)
}

/// Lexical normalization (`a/../b` collapse) without touching the filesystem.
#[must_use]
pub fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// =============================================================================
// Language-ID Detection (pi-tui `lang-from-path.ts`)
// =============================================================================

const EXTENSION_LANG: &[(&str, &str)] = &[
    ("ts", "typescript"),
    ("cts", "typescript"),
    ("mts", "typescript"),
    ("tsx", "typescriptreact"),
    ("js", "javascript"),
    ("jsx", "javascriptreact"),
    ("mjs", "javascript"),
    ("cjs", "javascript"),
    ("rs", "rust"),
    ("go", "go"),
    ("c", "c"),
    ("h", "c"),
    ("cpp", "cpp"),
    ("cc", "cpp"),
    ("cxx", "cpp"),
    ("hh", "cpp"),
    ("hpp", "cpp"),
    ("hxx", "cpp"),
    ("cu", "cpp"),
    ("cuh", "cpp"),
    ("ino", "cpp"),
    ("zig", "zig"),
    ("py", "python"),
    ("pyi", "python"),
    ("rb", "ruby"),
    ("rbw", "ruby"),
    ("gemspec", "ruby"),
    ("lua", "lua"),
    ("sh", "shellscript"),
    ("bash", "shellscript"),
    ("zsh", "shellscript"),
    ("ksh", "shellscript"),
    ("bats", "shellscript"),
    ("tmux", "shellscript"),
    ("cgi", "shellscript"),
    ("fcgi", "shellscript"),
    ("command", "shellscript"),
    ("tool", "shellscript"),
    ("fish", "fish"),
    ("pl", "perl"),
    ("pm", "perl"),
    ("perl", "perl"),
    ("php", "php"),
    ("java", "java"),
    ("kt", "kotlin"),
    ("ktm", "kotlin"),
];

/// LSP language identifier for a file path (`plaintext` when unknown).
#[must_use]
pub fn detect_language_id(file_path: &Path) -> String {
    let base_name = file_path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if base_name == "dockerfile"
        || base_name.starts_with("dockerfile.")
        || base_name == "containerfile"
    {
        return "dockerfile".to_string();
    }
    if base_name == ".emacs" {
        return "emacs-lisp".to_string();
    }
    if base_name == "makefile" || base_name == "gnumakefile" {
        return "makefile".to_string();
    }
    if base_name == "justfile" {
        return "just".to_string();
    }
    let ext = file_path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if base_name == "cmakelists.txt" || ext == "cmake" {
        return "cmake".to_string();
    }
    for (k, v) in EXTENSION_LANG {
        if *k == ext {
            return (*v).to_string();
        }
    }
    "plaintext".to_string()
}

// =============================================================================
// Path Display
// =============================================================================

/// Render a path relative to `cwd` when inside it (omp `formatPathRelativeToCwd`).
#[must_use]
pub fn format_path_relative_to_cwd(file_path: &Path, cwd: &Path) -> String {
    let resolved = normalize_path(file_path);
    let resolved_cwd = normalize_path(cwd);
    if let Ok(rel) = resolved.strip_prefix(&resolved_cwd) {
        let s = rel.to_string_lossy().replace('\\', "/");
        if s.is_empty() {
            ".".to_string()
        } else {
            s
        }
    } else {
        resolved.to_string_lossy().replace('\\', "/")
    }
}

/// Resolve a user-supplied path against the cwd (omp `resolveToCwd`).
#[must_use]
pub fn resolve_to_cwd(file_path: &str, cwd: &Path) -> PathBuf {
    let p = PathBuf::from(file_path);
    if p.is_absolute() {
        normalize_path(&p)
    } else {
        normalize_path(&cwd.join(p))
    }
}

// =============================================================================
// Diagnostic Formatting
// =============================================================================

fn severity_name(severity: Option<u32>) -> &'static str {
    match severity.unwrap_or(1) {
        2 => "warning",
        3 => "info",
        4 => "hint",
        _ => "error",
    }
}

/// Sort diagnostics by severity, then location, then message.
pub fn sort_diagnostics(diagnostics: &mut [Diagnostic]) {
    diagnostics.sort_by(|a, b| {
        let a_sev = a.severity.unwrap_or(1);
        let b_sev = b.severity.unwrap_or(1);
        a_sev
            .cmp(&b_sev)
            .then_with(|| a.range.start.line.cmp(&b.range.start.line))
            .then_with(|| a.range.start.character.cmp(&b.range.start.character))
            .then_with(|| a.message.cmp(&b.message))
    });
}

/// Strip noise from diagnostic messages (clippy URLs, override hints).
fn strip_diagnostic_noise(message: &str) -> String {
    message
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("for further information visit")
                && !trimmed.starts_with("http://")
                && !trimmed.starts_with("https://")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[allow(dead_code)]
fn severity_icon(severity: Option<u32>) -> &'static str {
    match severity.unwrap_or(1) {
        1 => STATUS_ERROR,
        2 => STATUS_WARNING,
        3 => STATUS_INFO,
        _ => FORMAT_BULLET,
    }
}

/// Format a diagnostic as `path:line:col [severity] [source] message (code)`.
#[must_use]
pub fn format_diagnostic(diagnostic: &Diagnostic, file_path: &str) -> String {
    let severity = severity_name(diagnostic.severity);
    let line = diagnostic.range.start.line + 1;
    let col = diagnostic.range.start.character + 1;
    let source = diagnostic
        .source
        .as_deref()
        .map(|s| format!("[{s}] "))
        .unwrap_or_default();
    let code = match &diagnostic.code {
        Some(Value::String(s)) => format!(" ({s})"),
        Some(Value::Number(n)) => format!(" ({n})"),
        _ => String::new(),
    };
    let message = strip_diagnostic_noise(&diagnostic.message);
    format!("{file_path}:{line}:{col} [{severity}] {source}{message}{code}")
}

/// Format a diagnostics summary grouped by severity (`1 error(s), 2 warning(s)`).
#[must_use]
pub fn format_diagnostics_summary(diagnostics: &[Diagnostic]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for d in diagnostics {
        *counts.entry(severity_name(d.severity)).or_default() += 1;
    }
    let mut parts = Vec::new();
    for sev in ["error", "warning", "info", "hint"] {
        if let Some(n) = counts.get(sev) {
            parts.push(format!("{n} {sev}(s)"));
        }
    }
    if parts.is_empty() {
        "no issues".to_string()
    } else {
        parts.join(", ")
    }
}

// =============================================================================
// Location Formatting
// =============================================================================

/// Format a location as `file:line:col` relative to cwd.
#[must_use]
pub fn format_location(location: &Location, cwd: &Path) -> String {
    let file = format_path_relative_to_cwd(&uri_to_file(&location.uri), cwd);
    let line = location.range.start.line + 1;
    let col = location.range.start.character + 1;
    format!("{file}:{line}:{col}")
}

// =============================================================================
// Symbol Formatting
// =============================================================================

fn symbol_kind_icon(kind: u32) -> &'static str {
    match kind {
        1 => ICON_FILE,
        2 | 3 | 23 => ICON_FOLDER,
        4 => ICON_PACKAGE,
        5 | 11 => ICON_MODEL,
        6 | 9 | 12 => ICON_AUTO,
        _ => FORMAT_BULLET,
    }
}

#[must_use]
pub fn symbol_kind_to_name(kind: u32) -> &'static str {
    SYMBOL_KIND_NAMES
        .get(kind as usize)
        .copied()
        .unwrap_or("Unknown")
}

/// Format a flat `SymbolInformation` row.
#[must_use]
pub fn format_symbol_information(symbol: &SymbolInformation, cwd: &Path) -> String {
    let icon = symbol_kind_icon(symbol.kind);
    let location = format_location(&symbol.location, cwd);
    let container = symbol
        .container_name
        .as_deref()
        .map(|c| format!(" ({c})"))
        .unwrap_or_default();
    format!("{icon} {}{container} @ {location}", symbol.name)
}

#[must_use]
pub fn filter_workspace_symbols(
    symbols: &[SymbolInformation],
    query: &str,
) -> Vec<SymbolInformation> {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        return symbols.to_vec();
    }
    symbols
        .iter()
        .filter(|s| {
            let container = s.container_name.clone().unwrap_or_default();
            let file = uri_to_file(&s.location.uri).to_string_lossy().into_owned();
            [s.name.clone(), container, file]
                .iter()
                .any(|f| f.to_lowercase().contains(&needle))
        })
        .cloned()
        .collect()
}

#[must_use]
pub fn dedupe_workspace_symbols(symbols: &[SymbolInformation]) -> Vec<SymbolInformation> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for symbol in symbols {
        let key = format!(
            "{}:{}:{}:{}:{}",
            symbol.name,
            symbol.container_name.clone().unwrap_or_default(),
            symbol.kind,
            symbol.location.uri,
            symbol.location.range.start.line
        );
        if seen.insert(key) {
            unique.push(symbol.clone());
        }
    }
    unique
}

// =============================================================================
// Code Action Formatting
// =============================================================================

/// Format a code action as `index: [kind] title (preferred)? (disabled: ..)?`.
#[must_use]
pub fn format_code_action(
    title: &str,
    kind: Option<&str>,
    is_preferred: bool,
    disabled_reason: Option<&str>,
    index: usize,
) -> String {
    let kind = kind.unwrap_or("action");
    let preferred = if is_preferred { " (preferred)" } else { "" };
    let disabled = disabled_reason
        .map(|r| format!(" (disabled: {r})"))
        .unwrap_or_default();
    format!("{index}: [{kind}] {title}{preferred}{disabled}")
}

// =============================================================================
// Grouped Diagnostic Output (pi-tui `formatGroupedDiagnosticMessages`)
// =============================================================================

/// Regex-equivalent split: `^(.+?):(\d+:\d+\s+.*)$` on the first `:d:d ` boundary.
fn split_diag_path(msg: &str) -> Option<(&str, &str)> {
    let bytes = msg.as_bytes();
    // Find the first `:` such that what follows is digits:digits+whitespace.
    for (i, &b) in bytes.iter().enumerate() {
        if b != b':' {
            continue;
        }
        let rest = &msg[i + 1..];
        // digits
        let digit_len = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digit_len == 0 {
            continue;
        }
        let after_digits = &rest[digit_len..];
        if !after_digits.starts_with(':') {
            continue;
        }
        let col_part = &after_digits[1..];
        let col_len = col_part.len()
            - col_part
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .len();
        if col_len == 0 {
            continue;
        }
        let after_col = &col_part[col_len..];
        if after_col.starts_with(char::is_whitespace) {
            return Some((&msg[..i], &msg[i + 1..]));
        }
    }
    None
}

struct PathNode {
    name: String,
    files: Vec<String>, // full path keys, in insertion order
    subdirs: Vec<PathNode>,
}

impl PathNode {
    fn new(name: String) -> Self {
        Self {
            name,
            files: Vec::new(),
            subdirs: Vec::new(),
        }
    }
}

fn build_path_tree(paths: &[String]) -> PathNode {
    let mut root = PathNode::new(String::new());
    for raw in paths {
        let normalized = raw.replace('\\', "/");
        let segments: Vec<&str> = normalized.split('/').collect();
        if segments.is_empty() {
            continue;
        }
        let mut node = &mut root;
        for seg in &segments[..segments.len() - 1] {
            let idx = if let Some(i) = node.subdirs.iter().position(|d| d.name == *seg) {
                i
            } else {
                node.subdirs.push(PathNode::new((*seg).to_string()));
                node.subdirs.len() - 1
            };
            node = &mut node.subdirs[idx];
        }
        node.files.push(raw.clone());
    }
    root
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

enum TreeEvent<'a> {
    File { depth: usize, path: &'a str },
    Dir { depth: usize, name: String },
}

fn walk_events<'a>(node: &'a PathNode, depth: usize, out: &mut Vec<TreeEvent<'a>>) {
    for file in &node.files {
        out.push(TreeEvent::File { depth, path: file });
    }
    for subdir in &node.subdirs {
        let mut parts = vec![subdir.name.clone()];
        let mut cur = subdir;
        while cur.files.is_empty() && cur.subdirs.len() == 1 {
            cur = &cur.subdirs[0];
            parts.push(cur.name.clone());
        }
        out.push(TreeEvent::Dir {
            depth,
            name: parts.join("/"),
        });
        walk_events(cur, depth + 1, out);
    }
}

/// Group `path:line:col [sev] msg` rows under `# dir/` / `## file` headers
/// (pi-tui `formatGroupedDiagnosticMessages` + `formatGroupedFiles`).
#[must_use]
pub fn format_grouped_diagnostic_messages(messages: &[String]) -> String {
    let mut order: Vec<String> = Vec::new();
    let mut by_file: Vec<(String, Vec<String>)> = Vec::new();
    let mut ungrouped: Vec<String> = Vec::new();
    for msg in messages {
        match split_diag_path(msg) {
            Some((raw_path, rest)) => {
                let file_path = raw_path.replace('\\', "/");
                if let Some(entry) = by_file.iter_mut().find(|(f, _)| *f == file_path) {
                    entry.1.push(format!("  {rest}"));
                } else {
                    by_file.push((file_path.clone(), vec![format!("  {rest}")]));
                    order.push(file_path);
                }
            }
            None => ungrouped.push(msg.clone()),
        }
    }
    if by_file.is_empty() {
        return ungrouped.join("\n");
    }
    let tree = build_path_tree(&order);
    let mut events = Vec::new();
    walk_events(&tree, 0, &mut events);
    let mut lines: Vec<String> = Vec::new();
    let mut emitted = false;
    for event in events {
        let (depth, header, body) = match event {
            TreeEvent::File { depth, path } => {
                let body = by_file
                    .iter()
                    .find(|(f, _)| f == path)
                    .map(|(_, b)| b.clone())
                    .unwrap_or_default();
                (depth, file_name(path).to_string(), body)
            }
            TreeEvent::Dir { depth, name } => (depth, format!("{name}/"), Vec::new()),
        };
        let is_dir = header.ends_with('/');
        let needs_separator = emitted && (depth == 0 || is_dir);
        if needs_separator {
            lines.push(String::new());
        }
        emitted = true;
        lines.push(format!("{} {header}", "#".repeat(depth + 1)));
        lines.extend(body);
    }
    if !ungrouped.is_empty() {
        if emitted {
            lines.push(String::new());
        }
        lines.extend(ungrouped);
    }
    lines.join("\n")
}

// =============================================================================
// Hover Content Extraction
// =============================================================================

/// Extract plain text from hover contents (string, `MarkupContent`, `MarkedString`[]).
pub fn extract_hover_text(contents: &Value) -> String {
    match contents {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(extract_hover_text)
            .collect::<Vec<_>>()
            .join("\n\n"),
        Value::Object(map) => map
            .get("value")
            .and_then(Value::as_str)
            .map_or_else(|| contents.to_string(), ToString::to_string),
        other => other.to_string(),
    }
}

// =============================================================================
// Symbol Column Resolution (omp `resolveSymbolColumn`)
// =============================================================================

fn first_non_whitespace_column(line_text: &str) -> usize {
    line_text.find(|c: char| !c.is_whitespace()).unwrap_or(0)
}

const IDENTIFIER_CHAR: fn(char) -> bool =
    |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$';

fn find_symbol_match_indexes(line_text: &str, symbol: &str, case_insensitive: bool) -> Vec<usize> {
    if symbol.is_empty() {
        return Vec::new();
    }
    let require_word_boundary = symbol
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && symbol.chars().all(IDENTIFIER_CHAR);
    let haystack: String = if case_insensitive {
        line_text.to_lowercase()
    } else {
        line_text.to_string()
    };
    let needle: String = if case_insensitive {
        symbol.to_lowercase()
    } else {
        symbol.to_string()
    };
    let haystack_char: Vec<char> = haystack.chars().collect();
    let needle_len = needle.chars().count();
    let mut indexes = Vec::new();
    let mut from = 0;
    while from + needle_len <= haystack_char.len() {
        let slice: String = haystack_char[from..from + needle_len].iter().collect();
        if slice != needle {
            from += 1;
            continue;
        }
        if require_word_boundary {
            let before_ok = from == 0 || !IDENTIFIER_CHAR(haystack_char[from - 1]);
            let after_idx = from + needle_len;
            let after_ok =
                after_idx >= haystack_char.len() || !IDENTIFIER_CHAR(haystack_char[after_idx]);
            if !before_ok || !after_ok {
                from += 1;
                continue;
            }
        }
        // Byte offset of the char index.
        let byte_offset: usize = haystack_char[..from].iter().map(|c| c.len_utf8()).sum();
        indexes.push(byte_offset);
        from += needle_len;
    }
    indexes
}

fn parse_symbol_spec(spec: &str) -> (String, usize) {
    if let Some(idx) = spec.rfind('#') {
        if let Ok(n) = spec[idx + 1..].parse::<usize>() {
            if idx + 1 < spec.len() && spec[idx + 1..].chars().all(|c| c.is_ascii_digit()) {
                return (spec[..idx].to_string(), n.max(1));
            }
        }
    }
    (spec.to_string(), 1)
}

/// Resolve the 0-indexed character column of `symbol` on 1-indexed `line`.
///
/// # Errors
///
/// File missing, symbol not found, or occurrence out of bounds.
pub async fn resolve_symbol_column(
    file_path: &Path,
    line: usize,
    symbol_spec: Option<&str>,
) -> anyhow::Result<usize> {
    let line_number = line.max(1);
    let file_text = tokio::fs::read_to_string(file_path)
        .await
        .map_err(|e| anyhow::anyhow!("File not found: {} ({e})", file_path.display()))?;
    let lines: Vec<&str> = file_text.split('\n').collect();
    let target_line = lines.get(line_number - 1).copied().unwrap_or("");
    let Some(spec) = symbol_spec else {
        return Ok(first_non_whitespace_column(target_line));
    };
    let (symbol, occurrence) = parse_symbol_spec(spec);
    let exact = find_symbol_match_indexes(target_line, &symbol, false);
    let fallback = if exact.is_empty() {
        find_symbol_match_indexes(target_line, &symbol, true)
    } else {
        exact
    };
    if fallback.is_empty() {
        anyhow::bail!("Symbol \"{symbol}\" not found on line {line_number}");
    }
    if occurrence > fallback.len() {
        anyhow::bail!(
            "Symbol \"{symbol}\" occurrence {occurrence} is out of bounds on line {line_number} (found {})",
            fallback.len()
        );
    }
    Ok(fallback[occurrence - 1])
}

/// Read `context_lines` of context around a 1-indexed line as `N: text` rows.
pub async fn read_location_context(
    file_path: &Path,
    line: usize,
    context_lines: usize,
) -> Vec<String> {
    let Ok(file_text) = tokio::fs::read_to_string(file_path).await else {
        return Vec::new();
    };
    let lines: Vec<&str> = file_text.split('\n').collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let target = line.max(1);
    let start = target.saturating_sub(context_lines).max(1);
    let end = (target + context_lines).min(lines.len());
    let mut out = Vec::new();
    for current in start..=end {
        out.push(format!(
            "{current}: {}",
            lines.get(current - 1).copied().unwrap_or("")
        ));
    }
    out
}
