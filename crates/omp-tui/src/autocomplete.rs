//! Slash-command and `@`-file autocomplete — the port of
//! `pi-tui/src/autocomplete.ts`.
//!
//! The [`Autocomplete`] widget owns the slash-command catalog (builtins +
//! extension/file commands delivered through `available_commands_update`
//! frames), computes suggestion lists for the editor's current buffer
//! (leading-slash commands, mid-prompt `skill:` tokens, `@` file references
//! and explicit-Tab path completion), applies accepted items back into the
//! buffer, and renders the popup above the editor.
//!
//! Ranking uses the ranked-tier `subsequenceScore` from autocomplete.ts
//! (100/80/60/40−gaps) for command text and `@` candidates; the word-local
//! engine in [`fuzzy`] (fuzzy.ts) is re-exported for overlay searches.

pub mod fuzzy;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::Frame;

pub use fuzzy::{fuzzy_filter, fuzzy_match, fuzzy_rank, FuzzyMatch, FuzzyText};

/// Back-compat ranking hook for overlays (e.g. the command palette):
/// higher is better, `0` means no match. Delegates to the word-local
/// [`fuzzy`] engine, mapping its lower-is-better scores onto a positive
/// integer scale.
#[must_use]
pub fn fuzzy_score(needle: &str, haystack: &str) -> i64 {
    if needle.is_empty() {
        return 1;
    }
    let m = fuzzy::fuzzy_match(needle, haystack);
    if !m.matches {
        return 0;
    }
    // Lower fuzzy score = better match = higher rank score.
    let rank = 1_000_000.0 - m.score.clamp(-100_000.0, 100_000.0) * 1000.0;
    i64::max(1, rank as i64)
}

/// The `skill:` namespace prefix (TS `SKILL_NAMESPACE`).
pub const SKILL_NAMESPACE: &str = "skill:";

const PATH_DELIMITERS: [char; 5] = [' ', '\t', '"', '\'', '='];

/// How many fuzzy file results to return (TS `maxResults: 100`).
const FUZZY_MAX_RESULTS: usize = 100;
/// Cap on directory entries visited during one fuzzy walk (safety bound the
/// native `fuzzyFind` gets from its own implementation).
const FUZZY_WALK_ENTRY_CAP: usize = 20_000;

/// One popup row (TS `AutocompleteItem`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AutocompleteItem {
    /// Text inserted into the buffer on accept.
    pub value: String,
    /// Display text (defaults to `value`).
    pub label: String,
    /// Dim description shown next to the label.
    pub description: Option<String>,
    /// Semantic icon name (`"folder"`, `"file"`, a command icon name).
    pub icon_name: Option<String>,
    /// Static detail (e.g. parent dir) when it differs from `description`.
    pub native_detail: Option<String>,
    /// Live value drawn right-aligned (e.g. current model).
    pub state: Option<String>,
    /// Dim inline hint shown after the cursor while selected.
    pub hint: Option<String>,
}

impl AutocompleteItem {
    #[cfg(test)]
    fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        let value = value.into();
        Self {
            label: label.into(),
            value,
            ..Self::default()
        }
    }

    /// Effective description: explicit, else the parent-dir detail.
    fn detail(&self) -> Option<&str> {
        self.description
            .as_deref()
            .or(self.native_detail.as_deref())
    }
}

/// Where a slash command came from in the catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSource {
    /// omp builtin (`source: "builtin"`).
    Builtin,
    /// `commands/*.md`-style file command (`source: "file"`).
    File,
    /// Extension-registered command (`source: "extension"`).
    Extension,
}

/// A slash command descriptor (TS `SlashCommand`, the catalog subset the
/// popup needs — argument completion providers are an editor-side concern).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlashCommand {
    /// Command name without the leading `/`.
    pub name: String,
    /// Alternate names.
    pub aliases: Vec<String>,
    /// Static description.
    pub description: Option<String>,
    /// Argument placeholder hint, e.g. `"[on|off]"`.
    pub argument_hint: Option<String>,
    /// Whether the command consumes argument text.
    pub allow_args: bool,
    /// Catalog origin.
    pub source: Option<CommandSource>,
}

impl SlashCommand {
    /// Parse one `available_commands_update` catalog entry.
    fn from_value(v: &Value) -> Option<Self> {
        let name = v.get("name")?.as_str()?.to_string();
        let aliases = v
            .get("aliases")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let description = v
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string);
        let argument_hint = v
            .get("input")
            .and_then(|i| i.get("hint"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let source = v
            .get("source")
            .and_then(Value::as_str)
            .and_then(|s| match s {
                "builtin" => Some(CommandSource::Builtin),
                "file" => Some(CommandSource::File),
                "extension" => Some(CommandSource::Extension),
                _ => None,
            });
        Some(Self {
            name,
            aliases,
            description,
            argument_hint,
            allow_args: true,
            source,
        })
    }
}

/// A suggestion list plus the buffer text it replaces on accept
/// (TS `{ items, prefix }`).
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestions {
    /// Ranked rows.
    pub items: Vec<AutocompleteItem>,
    /// The text before the cursor the items complete (slash token, `@`
    /// token, or path prefix).
    pub prefix: String,
}

/// New buffer state after applying a completion (TS `applyCompletion`).
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedCompletion {
    /// Full new buffer lines.
    pub lines: Vec<String>,
    /// Cursor line.
    pub cursor_line: usize,
    /// Cursor column (byte index within the line).
    pub cursor_col: usize,
}

fn find_last_delimiter(text: &str) -> Option<usize> {
    text.char_indices()
        .rev()
        .find(|(_, c)| PATH_DELIMITERS.contains(c))
        .map(|(i, _)| i)
}

fn find_unclosed_quote_start(text: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut quote_start = 0usize;
    for (i, c) in text.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
            if in_quotes {
                quote_start = i;
            }
        }
    }
    in_quotes.then_some(quote_start)
}

fn is_token_start(text: &str, index: usize) -> bool {
    index == 0
        || text[..index]
            .chars()
            .next_back()
            .is_some_and(|c| PATH_DELIMITERS.contains(&c))
}

/// Locate the slash that opens a slash command on the line, allowing leading
/// whitespace (TS `findLeadingSlashCommandStart`).
#[must_use]
pub fn find_leading_slash_command_start(text: &str) -> Option<usize> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('/') {
        return None;
    }
    Some(text.len() - trimmed.len())
}

/// Locate the last `/token` at the end of the line (TS
/// `findTrailingSlashCommandStart`): a `/` at the start or after whitespace,
/// followed by non-space, non-slash text to the cursor.
#[must_use]
pub fn find_trailing_slash_command_start(text: &str) -> Option<usize> {
    for (i, c) in text.char_indices().rev() {
        if c != '/' {
            continue;
        }
        let token_tail = &text[i + c.len_utf8()..];
        if token_tail.contains([' ', '/']) {
            continue;
        }
        if i == 0 {
            return Some(i);
        }
        if text[..i].ends_with(' ') || text[..i].ends_with('\t') {
            return Some(i);
        }
    }
    None
}

fn extract_quoted_prefix(text: &str) -> Option<String> {
    let quote_start = find_unclosed_quote_start(text)?;
    if quote_start > 0 && text[..quote_start].ends_with('@') {
        let at_index = quote_start - '@'.len_utf8();
        if !is_token_start(text, at_index) {
            return None;
        }
        return Some(text[at_index..].to_string());
    }
    if !is_token_start(text, quote_start) {
        return None;
    }
    Some(text[quote_start..].to_string())
}

/// The decomposition of a path-like prefix (TS `parsePathPrefix`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PathPrefix<'a> {
    raw: &'a str,
    is_at: bool,
    is_quoted: bool,
}

fn parse_path_prefix(prefix: &str) -> PathPrefix<'_> {
    if let Some(rest) = prefix.strip_prefix("@\"") {
        PathPrefix {
            raw: rest,
            is_at: true,
            is_quoted: true,
        }
    } else if let Some(rest) = prefix.strip_prefix('"') {
        PathPrefix {
            raw: rest,
            is_at: false,
            is_quoted: true,
        }
    } else if let Some(rest) = prefix.strip_prefix('@') {
        PathPrefix {
            raw: rest,
            is_at: true,
            is_quoted: false,
        }
    } else {
        PathPrefix {
            raw: prefix,
            is_at: false,
            is_quoted: false,
        }
    }
}

/// Whether an autocomplete value represents a directory: trailing slash or
/// backslash, optionally followed by a closing quote (TS
/// `isDirectoryCompletionValue`).
#[must_use]
pub fn is_directory_completion_value(value: &str) -> bool {
    let v = value.strip_suffix(['"', '\'']).unwrap_or(value);
    v.ends_with(['/', '\\'])
}

/// Build the insertable value for a path, adding `@`/quotes as needed
/// (TS `buildCompletionValue`).
fn build_completion_value(path: &str, is_at: bool, is_quoted: bool, is_directory: bool) -> String {
    let needs_quotes = is_quoted || path.contains(' ');
    let prefix = if is_at { "@" } else { "" };
    if !needs_quotes {
        return format!("{prefix}{path}");
    }
    let close_quote = if is_directory { "" } else { "\"" };
    format!("{prefix}\"{path}{close_quote}")
}

/// Ranked-tier subsequence match ("wig" ~ "skill:wig") (TS
/// `subsequenceMatch`).
#[must_use]
pub fn subsequence_match(query: &str, target: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    if query.len() > target.len() {
        return false;
    }
    let mut q = query.chars();
    let mut current = q.next();
    for tc in target.chars() {
        if current == Some(tc) {
            current = q.next();
            if current.is_none() {
                return true;
            }
        }
    }
    current.is_none()
}

/// Whether an `@` file completion `value` still fits the live `@` token
/// (TS `atCompletionMatches`): stale-list narrowing while a fresh search runs.
#[must_use]
pub fn at_completion_matches(token: &str, value: &str) -> bool {
    let query = parse_path_prefix(token)
        .raw
        .replace('\\', "/")
        .to_lowercase();
    let target = parse_path_prefix(value).raw.replace('\\', "/");
    let target = target.strip_suffix('"').unwrap_or(&target).to_lowercase();
    subsequence_match(&query, &target)
}

/// Ranked-tier subsequence score (100/80/60/40−5·gaps); higher is better,
/// 0 is no match (TS `subsequenceScore`).
#[must_use]
pub fn subsequence_score(query: &str, target: &str) -> i64 {
    if query.is_empty() {
        return 1;
    }
    if target == query {
        return 100;
    }
    if target.starts_with(query) {
        return 80;
    }
    if target.contains(query) {
        return 60;
    }
    let query: Vec<char> = query.chars().collect();
    let target: Vec<char> = target.chars().collect();
    let mut qi = 0usize;
    let mut gaps = 0i64;
    let mut last_match: Option<usize> = None;
    for (ti, &tc) in target.iter().enumerate() {
        if qi >= query.len() {
            break;
        }
        if query[qi] == tc {
            if let Some(prev) = last_match {
                if ti - prev > 1 {
                    gaps += 1;
                }
            }
            last_match = Some(ti);
            qi += 1;
        }
    }
    if qi != query.len() {
        return 0;
    }
    (40 - gaps * 5).max(1)
}

// ---------------------------------------------------------------------------
// Command scoring (TS scoreCommandTextMatch + skill namespace collapsing).
// ---------------------------------------------------------------------------

fn command_name(cmd: &SlashCommand) -> &str {
    &cmd.name
}

fn command_aliases(cmd: &SlashCommand) -> &[String] {
    &cmd.aliases
}

fn command_matches_name_or_alias(cmd: &SlashCommand, command_name: &str) -> bool {
    cmd.name == command_name || cmd.aliases.iter().any(|a| a == command_name)
}

/// Text-match scoring for command names/aliases/skills (TS
/// `scoreCommandTextMatch`). Flat prefix tier so same-prefix commands keep
/// registry order under the stable sort.
#[must_use]
pub fn score_command_text_match(lower_prefix: &str, lower_target: &str) -> i64 {
    if lower_prefix.is_empty() {
        return 1;
    }
    if lower_prefix == lower_target {
        return 1000;
    }
    if lower_target.starts_with(lower_prefix) {
        return 900;
    }
    if subsequence_match(lower_prefix, lower_target) {
        subsequence_score(lower_prefix, lower_target)
    } else {
        0
    }
}

/// Exact/leading-prefix tier for ordinary command names and aliases.
fn command_breakout_tier(lower_prefix: &str, lower_target: &str) -> i64 {
    if lower_prefix == lower_target {
        return 1000;
    }
    if lower_target.starts_with(lower_prefix) {
        return 900;
    }
    0
}

/// Match a bare skill name from the beginning of any hyphen-delimited
/// segment (TS `skillBareNameBreakoutTier`).
fn skill_bare_name_breakout_tier(lower_prefix: &str, lower_bare_name: &str) -> i64 {
    if lower_prefix.is_empty() {
        return 0;
    }
    if lower_prefix == lower_bare_name {
        return 1000;
    }
    if lower_bare_name.starts_with(lower_prefix) {
        return 900;
    }
    let bytes = lower_bare_name.as_bytes();
    let mut segment_start = 0usize;
    while segment_start < bytes.len() {
        while segment_start < bytes.len() && bytes[segment_start] != b'-' {
            segment_start += 1;
        }
        segment_start += 1;
        if segment_start >= bytes.len() {
            break;
        }
        if lower_bare_name[segment_start..].starts_with(lower_prefix) {
            let mut segment_end = segment_start;
            while segment_end < bytes.len() && bytes[segment_end] != b'-' {
                segment_end += 1;
            }
            return if lower_prefix.len() == segment_end - segment_start {
                1000
            } else {
                900
            };
        }
    }
    0
}

/// Whether a mid-prompt slash token is skill-shaped enough to surface `name`
/// (TS `midPromptSkillTokenMatches`).
#[must_use]
pub fn mid_prompt_skill_token_matches(
    lower_token: &str,
    name: &str,
    description: Option<&str>,
) -> bool {
    if SKILL_NAMESPACE.starts_with(lower_token) {
        return true;
    }
    let lower_name = name.to_lowercase();
    if lower_token.starts_with(SKILL_NAMESPACE) {
        if score_command_text_match(lower_token, &lower_name) > 0 {
            return true;
        }
        return description
            .is_some_and(|d| score_command_text_match(lower_token, &d.to_lowercase()) > 0);
    }
    lower_name.starts_with(SKILL_NAMESPACE)
        && skill_bare_name_breakout_tier(lower_token, &lower_name[SKILL_NAMESPACE.len()..]) > 0
}

/// Collapse `skill:*` commands into a single `/skill:` namespace row while
/// the typed prefix has not committed to the namespace (TS
/// `collapseSkillNamespace`). A skill breaks out only when its bare name
/// matches at a strictly stronger tier than every non-skill name/alias.
fn collapse_skill_namespace(commands: &[SlashCommand], lower_prefix: &str) -> Vec<SlashCommand> {
    if lower_prefix.starts_with(SKILL_NAMESPACE) {
        return commands.to_vec();
    }
    let approaches_namespace = SKILL_NAMESPACE.starts_with(lower_prefix);
    let mut command_tier = 0i64;
    if !approaches_namespace {
        for cmd in commands {
            let name = command_name(cmd);
            if name.is_empty() || name.starts_with(SKILL_NAMESPACE) {
                continue;
            }
            command_tier =
                command_tier.max(command_breakout_tier(lower_prefix, &name.to_lowercase()));
            for alias in command_aliases(cmd) {
                command_tier =
                    command_tier.max(command_breakout_tier(lower_prefix, &alias.to_lowercase()));
            }
            if command_tier == 1000 {
                break;
            }
        }
    }
    let mut skill_count = 0usize;
    let mut rest: Vec<SlashCommand> = commands
        .iter()
        .filter(|cmd| {
            let name = command_name(cmd);
            if !name.starts_with(SKILL_NAMESPACE) {
                return true;
            }
            skill_count += 1;
            !approaches_namespace
                && skill_bare_name_breakout_tier(
                    lower_prefix,
                    &name[SKILL_NAMESPACE.len()..].to_lowercase(),
                ) > command_tier
        })
        .cloned()
        .collect();
    if skill_count == 0 {
        return commands.to_vec();
    }
    if !SKILL_NAMESPACE.starts_with(lower_prefix) {
        return rest;
    }
    rest.push(SlashCommand {
        name: SKILL_NAMESPACE.to_string(),
        description: Some(format!(
            "{skill_count} skill{}",
            if skill_count == 1 { "" } else { "s" }
        )),
        ..SlashCommand::default()
    });
    rest
}

fn command_item(
    cmd: &SlashCommand,
    value: String,
    label: String,
    score: i64,
) -> (AutocompleteItem, i64) {
    let description = match (&cmd.argument_hint, &cmd.description) {
        (Some(hint), Some(desc)) => Some(format!("{hint} - {desc}")),
        (Some(hint), None) => Some(hint.clone()),
        (None, desc) => desc.clone(),
    };
    (
        AutocompleteItem {
            value,
            label,
            description,
            icon_name: cmd.source.map(command_icon),
            ..AutocompleteItem::default()
        },
        score,
    )
}

/// Icon name by catalog source (`folder`-style icon column).
fn command_icon(source: CommandSource) -> String {
    match source {
        CommandSource::Builtin => "command".to_string(),
        CommandSource::File => "file".to_string(),
        CommandSource::Extension => "extension".to_string(),
    }
}

/// Ranked slash-command completion rows (TS `buildSlashCommandCompletions`).
/// Equal text-match scores fall back to stable registry order.
fn build_slash_command_completions(
    commands: &[SlashCommand],
    lower_prefix: &str,
) -> Vec<AutocompleteItem> {
    let mut rows: Vec<(AutocompleteItem, i64)> = Vec::new();
    for cmd in commands {
        let name = command_name(cmd);
        if name.is_empty() {
            continue;
        }
        let lower_name = name.to_lowercase();
        let is_skill = name.starts_with(SKILL_NAMESPACE);

        let name_score = if lower_prefix.is_empty() && is_skill {
            950
        } else if is_skill {
            score_command_text_match(lower_prefix, &lower_name).max(skill_bare_name_breakout_tier(
                lower_prefix,
                &lower_name[SKILL_NAMESPACE.len()..],
            ))
        } else {
            score_command_text_match(lower_prefix, &lower_name)
        };
        let lower_desc = cmd.description.as_deref().unwrap_or("").to_lowercase();
        let desc_score = if !lower_desc.is_empty() && subsequence_match(lower_prefix, &lower_desc) {
            subsequence_score(lower_prefix, &lower_desc) / 2
        } else {
            0
        };
        let primary = name_score.max(desc_score);

        let mut best: Option<(AutocompleteItem, i64)> = None;
        if primary > 0 {
            best = Some(command_item(
                cmd,
                name.to_string(),
                name.to_string(),
                primary,
            ));
        }
        if !lower_prefix.is_empty() {
            for alias in command_aliases(cmd) {
                if alias == name {
                    continue;
                }
                let alias_score = score_command_text_match(lower_prefix, &alias.to_lowercase());
                if alias_score == 0 || best.as_ref().is_some_and(|(_, s)| alias_score <= *s) {
                    continue;
                }
                best = Some(command_item(cmd, alias.clone(), alias.clone(), alias_score));
            }
        }
        if let Some(row) = best {
            rows.push(row);
        }
    }
    // Stable sort: equal scores keep registry order (TS falls back to usage
    // counts, which we do not track yet).
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    rows.into_iter().map(|(item, _)| item).collect()
}

/// Mid-prompt `/word` skill lookup rows (TS `buildMidPromptSkillCompletions`).
fn build_mid_prompt_skill_completions(
    commands: &[SlashCommand],
    lower_prefix: &str,
) -> Vec<AutocompleteItem> {
    let skills: Vec<SlashCommand> = commands
        .iter()
        .filter(|cmd| {
            command_name(cmd).starts_with(SKILL_NAMESPACE)
                && mid_prompt_skill_token_matches(
                    lower_prefix,
                    command_name(cmd),
                    cmd.description.as_deref(),
                )
        })
        .cloned()
        .collect();
    build_slash_command_completions(&skills, lower_prefix)
}

/// Whether any non-command text precedes the slash (TS
/// `hasPromptTextBeforeSlash`).
fn has_prompt_text_before_slash(
    lines: &[String],
    cursor_line: usize,
    text_before_cursor: &str,
    slash_start: usize,
) -> bool {
    for line in &lines[..cursor_line.min(lines.len())] {
        if !line.trim().is_empty() {
            return true;
        }
    }
    !text_before_cursor[..slash_start].trim().is_empty()
}

// ---------------------------------------------------------------------------
// File completion (TS `#getFileSuggestions` / `#getFuzzyFileSuggestions`).
// ---------------------------------------------------------------------------

/// Per-directory readdir fast-path cache with TTL (TS `#dirCache`).
#[derive(Debug, Default)]
struct DirCache {
    entries: HashMap<PathBuf, (Instant, Vec<(String, FileKind)>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    File,
    Directory,
}

const DIR_CACHE_TTL: std::time::Duration = std::time::Duration::from_millis(2000);

impl DirCache {
    fn read_dir(&mut self, dir: &Path) -> Option<&Vec<(String, FileKind)>> {
        if self
            .entries
            .get(dir)
            .is_some_and(|(at, _)| at.elapsed() < DIR_CACHE_TTL)
        {
            return self.entries.get(dir).map(|(_, e)| e);
        }
        let mut entries = Vec::new();
        let rd = std::fs::read_dir(dir).ok()?;
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = match entry.file_type() {
                Ok(ft) if ft.is_dir() => FileKind::Directory,
                Ok(ft) if ft.is_symlink() => {
                    // Resolve symlink-to-directory; broken links are skipped.
                    match std::fs::metadata(entry.path()) {
                        Ok(md) if md.is_dir() => FileKind::Directory,
                        Ok(_) => FileKind::File,
                        Err(_) => continue,
                    }
                }
                Ok(_) => FileKind::File,
                Err(_) => continue,
            };
            entries.push((name, kind));
        }
        if self.entries.len() >= 100 {
            // Prune the 50 oldest (TS cache eviction).
            let mut keys: Vec<PathBuf> = self.entries.keys().cloned().collect();
            keys.sort_by_key(|k| {
                self.entries
                    .get(k)
                    .map(|(at, _)| *at)
                    .unwrap_or_else(Instant::now)
            });
            for key in keys.into_iter().take(50) {
                self.entries.remove(&key);
            }
        }
        self.entries
            .insert(dir.to_path_buf(), (Instant::now(), entries));
        self.entries.get(dir).map(|(_, e)| e)
    }

    fn invalidate(&mut self, dir: Option<&Path>) {
        match dir {
            Some(d) => {
                self.entries.remove(d);
            }
            None => self.entries.clear(),
        }
    }
}

/// Expand a leading `~/` to the home directory (TS `#expandHomePath`).
fn expand_home_path(file_path: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    if let Some(rest) = file_path.strip_prefix("~/") {
        let mut expanded = format!("{home}/{rest}");
        if file_path.ends_with('/') && !expanded.ends_with('/') {
            expanded.push('/');
        }
        expanded
    } else if file_path == "~" {
        home
    } else {
        file_path.to_string()
    }
}

/// Lexically resolve `..`/`.` components without touching the filesystem.
fn lexically_normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Lexically resolve `raw_prefix` and report whether it points outside
/// `base_path` (TS `#isOutsideCwd`) — outside-cwd roots use plain prefix
/// listing instead of a recursive fuzzy walk.
fn is_outside_cwd(base_path: &Path, raw_prefix: &str) -> bool {
    if raw_prefix.is_empty() {
        return false;
    }
    let target = if raw_prefix.starts_with('~') {
        PathBuf::from(expand_home_path(raw_prefix))
    } else if Path::new(raw_prefix).is_absolute() {
        PathBuf::from(raw_prefix)
    } else {
        base_path.join(raw_prefix)
    };
    let target = lexically_normalize(&target);
    let base = lexically_normalize(base_path);
    match target.strip_prefix(&base) {
        Ok(rel) => {
            // Inside (or equal to) the base — but a `..` head means outside.
            let _ = rel;
            false
        }
        Err(_) => true,
    }
}

/// Scoped fuzzy query: `@src/uti` searches `src/` for `uti` (TS
/// `#resolveScopedFuzzyQuery`). Returns (base_dir, query, display_base).
fn resolve_scoped_fuzzy_query(
    base_path: &Path,
    raw_query: &str,
) -> Option<(PathBuf, String, String)> {
    let slash_index = raw_query.rfind('/')?;
    let display_base = &raw_query[..slash_index + 1];
    let query = &raw_query[slash_index + 1..];
    let base_dir = if let Some(rest) = display_base.strip_prefix("~/") {
        let _ = rest;
        PathBuf::from(expand_home_path(display_base))
    } else if display_base.starts_with('/') {
        PathBuf::from(display_base)
    } else {
        base_path.join(display_base)
    };
    if !base_dir.is_dir() {
        return None;
    }
    Some((base_dir, query.to_string(), display_base.to_string()))
}

/// Parent dir in posix form for the detail column (TS `path.posix.dirname`).
fn parent_dir(path: &str) -> String {
    match path.rfind('/') {
        None | Some(0) => "/".to_string(),
        Some(i) => path[..i].to_string(),
    }
}

fn posix_basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Immediate-directory prefix listing for a path token (TS
/// `#getFileSuggestions`).
fn get_file_suggestions(
    base_path: &Path,
    dir_cache: &mut DirCache,
    prefix: &str,
) -> Vec<AutocompleteItem> {
    let PathPrefix {
        raw,
        is_at,
        is_quoted,
    } = parse_path_prefix(prefix);
    let expanded = raw.replace('\\', "/");
    let pre_expand = expanded.as_str();

    let mut expanded_prefix = expanded.clone();
    if expanded_prefix.starts_with('~') {
        expanded_prefix = expand_home_path(&expanded_prefix);
    }

    let is_root_prefix = pre_expand.is_empty()
        || matches!(pre_expand, "./" | "../" | "~" | "~/" | "/")
        || (is_at && pre_expand.is_empty());

    let (search_dir, search_prefix): (PathBuf, String) = if is_root_prefix
        || expanded_prefix.ends_with('/')
    {
        let dir = if expanded_prefix.starts_with('~') || Path::new(&expanded_prefix).is_absolute() {
            PathBuf::from(&expanded_prefix)
        } else {
            base_path.join(&expanded_prefix)
        };
        (dir, String::new())
    } else {
        // Split into directory and file prefix.
        let (dir, file) = match expanded_prefix.rfind('/') {
            Some(i) => (
                expanded_prefix[..i].to_string(),
                expanded_prefix[i + 1..].to_string(),
            ),
            None => (String::new(), expanded_prefix.clone()),
        };
        let search_dir =
            if expanded_prefix.starts_with('~') || Path::new(&expanded_prefix).is_absolute() {
                PathBuf::from(if dir.is_empty() { "/".to_string() } else { dir })
            } else {
                base_path.join(if dir.is_empty() { "." } else { &dir })
            };
        (search_dir, file)
    };

    let Some(entries) = dir_cache.read_dir(&search_dir) else {
        return Vec::new();
    };
    let mut suggestions: Vec<AutocompleteItem> = Vec::new();
    let display_prefix = raw.replace('\\', "/");
    let lower_search = search_prefix.to_lowercase();

    for (name, kind) in entries {
        if !name.to_lowercase().starts_with(&lower_search) {
            continue;
        }
        if name == ".git" {
            continue;
        }
        let is_directory = *kind == FileKind::Directory;

        let relative_path: String = if display_prefix.ends_with('/') {
            format!("{display_prefix}{name}")
        } else if display_prefix.contains('/') {
            if let Some(rest) = display_prefix.strip_prefix("~/") {
                let home_dir = match rest.rfind('/') {
                    Some(i) if i > 0 => &rest[..i],
                    _ => "",
                };
                if home_dir.is_empty() {
                    format!("~/{name}")
                } else {
                    format!("~/{home_dir}/{name}")
                }
            } else if display_prefix.starts_with('/') {
                let dir = match display_prefix.rfind('/') {
                    Some(0) => "",
                    Some(i) => &display_prefix[..i],
                    None => "",
                };
                if dir.is_empty() {
                    format!("/{name}")
                } else {
                    format!("{dir}/{name}")
                }
            } else {
                let dir = match display_prefix.rfind('/') {
                    Some(i) => &display_prefix[..i],
                    None => "",
                };
                let joined = if dir.is_empty() {
                    name.clone()
                } else {
                    format!("{dir}/{name}")
                };
                if display_prefix.starts_with("./") && !joined.starts_with("./") {
                    format!("./{joined}")
                } else {
                    joined
                }
            }
        } else if display_prefix.starts_with('~') {
            format!("~/{name}")
        } else {
            name.clone()
        };

        let path_value = if is_directory {
            format!("{relative_path}/")
        } else {
            relative_path.clone()
        };
        let value = build_completion_value(&path_value, is_at, is_quoted, is_directory);
        let parent = parent_dir(&relative_path);
        suggestions.push(AutocompleteItem {
            value,
            label: format!("{name}{}", if is_directory { "/" } else { "" }),
            icon_name: Some(if is_directory { "folder" } else { "file" }.to_string()),
            native_detail: (parent != "." && parent != "/").then_some(parent),
            ..AutocompleteItem::default()
        });
    }

    // Directories first, then alphabetically.
    suggestions.sort_by(|a, b| {
        let a_dir = is_directory_completion_value(&a.value);
        let b_dir = is_directory_completion_value(&b.value);
        b_dir.cmp(&a_dir).then_with(|| a.label.cmp(&b.label))
    });
    suggestions
}

/// Recursive fuzzy file walk inside the project (TS
/// `#getFuzzyFileSuggestions`, replacing the native `fuzzyFind` with a
/// bounded std::fs walk). Honors the TS contract: hidden + `.git` skipped
/// (unless the query's last segment starts with `.`), results filtered by
/// `subsequenceMatch`, capped at `FUZZY_MAX_RESULTS`.
fn get_fuzzy_file_suggestions(
    base_path: &Path,
    query: &str,
    is_quoted: bool,
) -> Vec<AutocompleteItem> {
    let scoped = resolve_scoped_fuzzy_query(base_path, query);
    let search_path: PathBuf = match &scoped {
        Some((dir, _, _)) => dir.clone(),
        None => base_path.to_path_buf(),
    };
    let fuzzy_query = scoped
        .as_ref()
        .map_or(query.to_string(), |(_, q, _)| q.clone());
    let display_base = scoped.as_ref().map(|(_, _, d)| d.clone());

    // Whether hidden files are admissible: the typed segment demands it.
    let allow_hidden = fuzzy_query.starts_with('.');

    let mut matches: Vec<(String, bool)> = Vec::new();
    let mut visited = 0usize;
    let mut stack: Vec<(PathBuf, String)> = vec![(search_path.clone(), String::new())];
    while let Some((dir, rel)) = stack.pop() {
        if visited >= FUZZY_WALK_ENTRY_CAP || matches.len() >= FUZZY_WALK_ENTRY_CAP {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            if visited >= FUZZY_WALK_ENTRY_CAP {
                break;
            }
            visited += 1;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == ".git" || (!allow_hidden && name.starts_with('.')) {
                continue;
            }
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            let is_dir = ft.is_dir()
                || (ft.is_symlink() && std::fs::metadata(entry.path()).is_ok_and(|m| m.is_dir()));
            let entry_rel = if rel.is_empty() {
                name.clone()
            } else {
                format!("{rel}/{name}")
            };
            if is_dir {
                stack.push((entry.path(), entry_rel.clone()));
            }
            let normalized = entry_rel.replace('\\', "/");
            let lower_query = fuzzy_query.to_lowercase();
            if lower_query.is_empty() || subsequence_match(&lower_query, &normalized.to_lowercase())
            {
                matches.push((normalized, is_dir));
            }
        }
    }

    // Rank: subsequence tier first, then path length; stable within a tier.
    let lower_query = fuzzy_query.to_lowercase();
    let mut ranked: Vec<(i64, usize, (String, bool))> = matches
        .into_iter()
        .map(|m| {
            let score = if lower_query.is_empty() {
                0
            } else {
                subsequence_score(&lower_query, &m.0.to_lowercase())
            };
            (score, m.0.len(), m)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    ranked.truncate(FUZZY_MAX_RESULTS);

    let mut suggestions = Vec::new();
    for (_, _, (entry_path, is_dir)) in ranked {
        let path_without_slash = entry_path.trim_end_matches('/').to_string();
        let display_path = match &display_base {
            Some(base) if base == "/" => format!("/{path_without_slash}"),
            Some(base) => format!("{base}{path_without_slash}"),
            None => path_without_slash,
        };
        let entry_name = posix_basename(&display_path).to_string();
        let completion_path = if is_dir {
            format!("{display_path}/")
        } else {
            display_path.clone()
        };
        let value = build_completion_value(&completion_path, true, is_quoted, is_dir);
        let parent = parent_dir(&display_path);
        suggestions.push(AutocompleteItem {
            value,
            label: format!("{entry_name}{}", if is_dir { "/" } else { "" }),
            description: Some(display_path.clone()),
            icon_name: Some(if is_dir { "folder" } else { "file" }.to_string()),
            native_detail: (parent != "." && parent != "/").then_some(parent),
            ..AutocompleteItem::default()
        });
    }
    suggestions
}

// ---------------------------------------------------------------------------
// The popup widget (TS `CombinedAutocompleteProvider` + SelectList hosting).
// ---------------------------------------------------------------------------

/// Slash/`@`/path autocomplete popup. Synchronous port of the TS
/// `CombinedAutocompleteProvider`: file I/O is plain `std::fs` (debounced by
/// the caller, see editor.md §9.2).
#[derive(Debug)]
pub struct Autocomplete {
    /// Whether the popup is open.
    pub open: bool,
    /// Current rows (already ranked).
    pub items: Vec<AutocompleteItem>,
    /// The buffer text the rows complete (TS `prefix`).
    pub prefix: String,
    /// Selected row index into `items`.
    pub selected: usize,
    /// Max visible popup rows (TS `setAutocompleteMaxVisible`).
    pub max_visible: usize,
    /// Slash-command catalog: builtins first, then extension/file commands.
    commands: Vec<SlashCommand>,
    /// Session working directory (fuzzy root).
    base_path: PathBuf,
    /// Immediate-dir listing cache.
    dir_cache: DirCache,
    /// Item accepted by the popup (Enter/Tab), awaiting pickup by the app
    /// which applies it to the editor buffer via [`Self::apply_completion`].
    confirmed: Option<(AutocompleteItem, String)>,
}

impl Default for Autocomplete {
    fn default() -> Self {
        Self {
            open: false,
            items: Vec::new(),
            prefix: String::new(),
            selected: 0,
            max_visible: 8,
            commands: Vec::new(),
            base_path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            dir_cache: DirCache::default(),
            confirmed: None,
        }
    }
}

impl Autocomplete {
    /// New popup rooted at `cwd` with no commands.
    #[must_use]
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            base_path: cwd,
            ..Self::default()
        }
    }

    /// Replace the catalog from an `available_commands_update` frame payload
    /// (`{"commands": [...]}`); extension and file commands ride the same
    /// list as builtins (TS `buildAvailableSlashCommands` order).
    pub fn update_commands(&mut self, frame: &Value) {
        let commands = frame
            .get("commands")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(SlashCommand::from_value).collect())
            .unwrap_or_default();
        self.set_commands(commands);
    }

    /// Replace the whole catalog.
    pub fn set_commands(&mut self, commands: Vec<SlashCommand>) {
        self.commands = commands;
    }

    /// Current catalog (read-only).
    #[must_use]
    pub fn commands(&self) -> &[SlashCommand] {
        &self.commands
    }

    /// Drop the immediate-dir cache (TS `invalidateDirCache`).
    pub fn invalidate_dir_cache(&mut self, dir: Option<&Path>) {
        self.dir_cache.invalidate(dir);
    }

    /// Close the popup and clear its rows.
    pub fn close(&mut self) {
        self.open = false;
        self.items.clear();
        self.prefix.clear();
        self.selected = 0;
    }

    /// Open the popup with fresh suggestions.
    fn show(&mut self, suggestions: Suggestions) {
        self.open = true;
        self.items = suggestions.items;
        self.prefix = suggestions.prefix;
        self.selected = 0;
    }

    /// Extract the live `@` token before the cursor, honoring quoting (TS
    /// `#extractAtPrefix`). Must follow a path delimiter or line start.
    fn extract_at_prefix(text: &str) -> Option<String> {
        if let Some(quoted) = extract_quoted_prefix(text) {
            if quoted.starts_with("@\"") {
                return Some(quoted);
            }
        }
        let token_start = find_last_delimiter(text).map_or(0, |i| i + 1);
        text[token_start..]
            .starts_with('@')
            .then(|| text[token_start..].to_string())
    }

    /// Extract a path-like prefix (TS `#extractPathPrefix`). Automatic
    /// updates complete only unambiguous path syntax; `force` (Tab) always
    /// returns the last token.
    fn extract_path_prefix(text: &str, force: bool) -> Option<String> {
        if let Some(quoted) = extract_quoted_prefix(text) {
            return Some(quoted);
        }
        let token_start = find_last_delimiter(text).map_or(0, |i| i + 1);
        let path_prefix = &text[token_start..];
        if force {
            return Some(path_prefix.to_string());
        }
        if path_prefix.starts_with('/')
            || path_prefix.starts_with("./")
            || path_prefix.starts_with("../")
            || path_prefix.starts_with("~/")
            || is_windows_drive_absolute(path_prefix)
        {
            return Some(path_prefix.to_string());
        }
        None
    }

    /// Compute suggestions for the buffer/cursor (TS `getSuggestions`).
    /// Slash commands first, then `@` files, then unambiguous path prefixes.
    /// Opens the popup and returns the suggestion set (also stored).
    pub fn suggestions(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> Option<Suggestions> {
        let current_line = lines.get(cursor_line).map_or("", String::as_str);
        let text_before_cursor = &current_line[..cursor_col.min(current_line.len())];
        let result = self.compute_suggestions(lines, cursor_line, text_before_cursor);
        if let Some(s) = &result {
            self.show(s.clone());
        } else {
            self.close();
        }
        result
    }

    fn compute_suggestions(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        text_before_cursor: &str,
    ) -> Option<Suggestions> {
        let leading_slash_start = find_leading_slash_command_start(text_before_cursor);
        let trailing_slash_start = find_trailing_slash_command_start(text_before_cursor);
        let has_text_before_trailing = trailing_slash_start.is_some_and(|s| {
            has_prompt_text_before_slash(lines, cursor_line, text_before_cursor, s)
        });
        let has_text_before_leading = leading_slash_start.is_some_and(|s| {
            has_prompt_text_before_slash(lines, cursor_line, text_before_cursor, s)
        });
        let slash_start = if has_text_before_trailing {
            trailing_slash_start
        } else if has_text_before_leading {
            None
        } else {
            leading_slash_start
        };

        if let Some(slash_start) = slash_start {
            let command_text = &text_before_cursor[slash_start..];
            let space_index = command_text.find(' ');
            let is_mid_prompt_skill_lookup = has_text_before_trailing;

            if space_index.is_none() {
                // No space yet — complete command names.
                let prefix = &command_text[1..];
                let lower_prefix = prefix.to_lowercase();
                let matches = if is_mid_prompt_skill_lookup {
                    build_mid_prompt_skill_completions(&self.commands, &lower_prefix)
                } else {
                    let collapsed = collapse_skill_namespace(&self.commands, &lower_prefix);
                    build_slash_command_completions(&collapsed, &lower_prefix)
                };
                if !matches.is_empty() {
                    // Leading slash keeps the full text-before-cursor so the
                    // Enter staleness check can re-apply; mid-prompt skill
                    // lookup replaces only the slash token.
                    let prefix = if is_mid_prompt_skill_lookup {
                        command_text.to_string()
                    } else {
                        text_before_cursor.to_string()
                    };
                    return Some(Suggestions {
                        items: matches,
                        prefix,
                    });
                }
                if !is_mid_prompt_skill_lookup
                    && Some(slash_start) == leading_slash_start
                    && !command_text[1..].contains('/')
                {
                    return None;
                }
                // A non-matching slash token may still be an absolute path.
            } else if !is_mid_prompt_skill_lookup {
                // Argument completion is an editor/extension concern; the
                // popup falls through to path completion below.
            }
        }

        // `@` file reference after a delimiter or at line start.
        if let Some(at_prefix) = Self::extract_at_prefix(text_before_cursor) {
            let PathPrefix { raw, is_quoted, .. } = parse_path_prefix(&at_prefix);
            if !raw.is_empty() && is_outside_cwd(&self.base_path, raw) {
                let items = get_file_suggestions(&self.base_path, &mut self.dir_cache, &at_prefix);
                return (!items.is_empty()).then_some(Suggestions {
                    items,
                    prefix: at_prefix,
                });
            }
            if raw.is_empty() {
                let items = get_file_suggestions(&self.base_path, &mut self.dir_cache, "@");
                return (!items.is_empty()).then_some(Suggestions {
                    items,
                    prefix: at_prefix,
                });
            }
            let items = get_fuzzy_file_suggestions(&self.base_path, raw, is_quoted);
            if items.is_empty() {
                let fallback =
                    get_file_suggestions(&self.base_path, &mut self.dir_cache, &at_prefix);
                return (!fallback.is_empty()).then_some(Suggestions {
                    items: fallback,
                    prefix: at_prefix,
                });
            }
            return Some(Suggestions {
                items,
                prefix: at_prefix,
            });
        }

        // Unambiguous path prefixes (`/abs`, `./`, `../`, `~/`, `C:/`).
        if let Some(path_match) = Self::extract_path_prefix(text_before_cursor, false) {
            let items = get_file_suggestions(&self.base_path, &mut self.dir_cache, &path_match);
            if items.is_empty() {
                return None;
            }
            return Some(Suggestions {
                items,
                prefix: path_match,
            });
        }

        None
    }

    /// Force file completion on Tab (TS `getForceFileSuggestions`).
    pub fn force_file_suggestions(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> Option<Suggestions> {
        let current_line = lines.get(cursor_line).map_or("", String::as_str);
        let text_before_cursor = &current_line[..cursor_col.min(current_line.len())];
        let trimmed = text_before_cursor.trim();
        if trimmed.starts_with('/') && !trimmed.contains(' ') {
            return None; // a slash command token, not a path
        }
        let path_match = Self::extract_path_prefix(text_before_cursor, true)?;
        let items = get_file_suggestions(&self.base_path, &mut self.dir_cache, &path_match);
        if items.is_empty() {
            return None;
        }
        let result = Suggestions {
            items,
            prefix: path_match,
        };
        self.show(result.clone());
        Some(result)
    }

    /// Whether Tab should trigger file completion (TS
    /// `shouldTriggerFileCompletion`).
    #[must_use]
    pub fn should_trigger_file_completion(
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        let current_line = lines.get(cursor_line).map_or("", String::as_str);
        let text_before_cursor = &current_line[..cursor_col.min(current_line.len())];
        let trimmed = text_before_cursor.trim();
        !(trimmed.starts_with('/') && !trimmed.contains(' '))
    }

    /// Sync slash completion for the Enter race (TS
    /// `trySyncSlashCompletion`): completes `/q` → `/quit` without I/O. The
    /// bare `/skill:` namespace row is excluded (it is not a command).
    #[must_use]
    pub fn try_sync_slash_completion(&self, text_before_cursor: &str) -> Option<Suggestions> {
        let slash_start = find_leading_slash_command_start(text_before_cursor)?;
        let command_text = &text_before_cursor[slash_start..];
        if command_text.len() <= 1 || command_text.contains(' ') {
            return None;
        }
        let lower_prefix = command_text[1..].to_lowercase();
        let collapsed = collapse_skill_namespace(&self.commands, &lower_prefix);
        let items: Vec<AutocompleteItem> =
            build_slash_command_completions(&collapsed, &lower_prefix)
                .into_iter()
                .filter(|item| item.value != SKILL_NAMESPACE)
                .collect();
        if items.is_empty() {
            return None;
        }
        Some(Suggestions {
            items,
            prefix: text_before_cursor.to_string(),
        })
    }

    /// Drain an item accepted through the popup keys (`item`, `prefix`).
    /// The app applies it to the editor buffer via
    /// [`Self::apply_completion`].
    #[must_use]
    pub fn take_confirmed(&mut self) -> Option<(AutocompleteItem, String)> {
        self.confirmed.take()
    }
}

/// `C:/` / `C:\` drive-absolute prefix check (TS regex `/^[A-Za-z]:[\\/]/`).
fn is_windows_drive_absolute(prefix: &str) -> bool {
    let bytes = prefix.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'/' || bytes[2] == b'\\')
}

impl Autocomplete {
    /// Apply an accepted item to the buffer (TS `applyCompletion`). `prefix`
    /// is the suggestion prefix captured when the list was shown; live
    /// tokens are re-derived from the buffer so acceptance never rewrites
    /// stale text.
    #[must_use]
    pub fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AppliedCompletion {
        let current_line = lines.get(cursor_line).map_or("", String::as_str);
        let cursor_col = cursor_col.min(current_line.len());
        let text_before_cursor = &current_line[..cursor_col];
        let after_cursor = &current_line[cursor_col..];

        let leading_slash_start = find_leading_slash_command_start(text_before_cursor);
        let trailing_slash_start = find_trailing_slash_command_start(text_before_cursor);
        let is_mid_prompt_skill_lookup = item.value.starts_with(SKILL_NAMESPACE)
            && trailing_slash_start.is_some_and(|s| {
                has_prompt_text_before_slash(lines, cursor_line, text_before_cursor, s)
            })
            && find_trailing_slash_command_start(prefix).is_some();

        if is_mid_prompt_skill_lookup {
            if let Some(slash) = trailing_slash_start {
                // Replace ONLY the slash token; keep surrounding prose (TS
                // mid-prompt skill apply, preserves the draft).
                let before_slash = &current_line[..slash];
                let insert = format!("/{} ", item.value);
                let new_line = format!("{before_slash}{insert}{after_cursor}");
                let mut new_lines = lines.to_vec();
                new_lines[cursor_line] = new_line;
                return AppliedCompletion {
                    lines: new_lines,
                    cursor_line,
                    cursor_col: before_slash.len() + insert.len(),
                };
            }
        }

        // Leading-slash command apply: replace the live command token. Path
        // items (values starting with `/` or `"`) take the path tail below.
        let is_path_item = item.value.starts_with('/') || item.value.starts_with('"');
        if !is_path_item && find_leading_slash_command_start(prefix).is_some() {
            let Some(slash) = leading_slash_start else {
                return self.apply_path_tail(
                    lines,
                    cursor_line,
                    current_line,
                    cursor_col,
                    item,
                    prefix,
                    after_cursor,
                );
            };
            let slash_token = &text_before_cursor[slash..];
            let is_known_command = self
                .commands
                .iter()
                .any(|c| command_matches_name_or_alias(c, &item.value));
            if !slash_token.contains(' ') && (is_known_command || !slash_token[1..].contains('/')) {
                let before_slash = &current_line[..slash];
                // The collapsed `/skill:` row completes to the namespace
                // itself (no trailing space) so the popup reopens with the
                // individual skills.
                let insert = if item.value == SKILL_NAMESPACE {
                    format!("/{}", item.value)
                } else {
                    format!("/{} ", item.value)
                };
                let new_line = format!("{before_slash}{insert}{after_cursor}");
                let mut new_lines = lines.to_vec();
                new_lines[cursor_line] = new_line;
                return AppliedCompletion {
                    lines: new_lines,
                    cursor_line,
                    cursor_col: before_slash.len() + insert.len(),
                };
            }
        }

        self.apply_path_tail(
            lines,
            cursor_line,
            current_line,
            cursor_col,
            item,
            prefix,
            after_cursor,
        )
    }

    /// Shared tail of [`Self::apply_completion`]: file-attachment and plain
    /// path/argument replacement.
    fn apply_path_tail(
        &self,
        lines: &[String],
        cursor_line: usize,
        current_line: &str,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
        after_cursor: &str,
    ) -> AppliedCompletion {
        let text_before_cursor = &current_line[..cursor_col];
        let mut before_prefix = &current_line[..cursor_col.saturating_sub(prefix.len())];

        // File attachment: re-derive the live `@` token.
        if prefix.starts_with('@') {
            if let Some(live_at) = Self::extract_at_prefix(text_before_cursor) {
                before_prefix = &current_line[..cursor_col.saturating_sub(live_at.len())];
            }
            let is_directory = is_directory_completion_value(&item.value);
            let suffix = if is_directory { "" } else { " " };
            let new_line = format!("{}{}{suffix}{after_cursor}", before_prefix, item.value);
            let mut new_lines = lines.to_vec();
            new_lines[cursor_line] = new_line;
            return AppliedCompletion {
                lines: new_lines,
                cursor_line,
                cursor_col: before_prefix.len() + item.value.len() + suffix.len(),
            };
        }

        // Slash-command argument and plain path completion: replace the
        // rendered prefix, preserving earlier arguments.
        let new_line = format!("{before_prefix}{}{after_cursor}", item.value);
        let mut new_lines = lines.to_vec();
        new_lines[cursor_line] = new_line;
        AppliedCompletion {
            lines: new_lines,
            cursor_line,
            cursor_col: before_prefix.len() + item.value.len(),
        }
    }

    /// The currently selected row.
    #[must_use]
    pub fn selected_item(&self) -> Option<&AutocompleteItem> {
        self.open.then(|| self.items.get(self.selected)).flatten()
    }

    /// Move the selection by `delta` rows (wrapping like the TS SelectList).
    fn move_selection(&mut self, delta: isize) {
        if self.items.is_empty() {
            return;
        }
        let len = self.items.len() as isize;
        self.selected = (((self.selected as isize + delta) % len + len) % len) as usize;
    }

    /// Page the selection by `pages` popup windows.
    fn page_selection(&mut self, pages: isize) {
        let page = self.max_visible.max(1) as isize;
        self.move_selection(pages * page);
    }
}

impl Component for Autocomplete {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if !self.open || self.items.is_empty() || area.height == 0 || area.width == 0 {
            return;
        }
        // Bottom-anchored popup list above the editor.
        // Keep the footer hint row: never show more rows than fit above it.
        let fit = usize::from(area.height.saturating_sub(1)).max(1);
        let visible = self.max_visible.min(self.items.len()).min(fit);
        let height = u16::min(area.height, visible as u16 + 1);
        let popup = Rect {
            x: area.x,
            y: area.y + area.height - height,
            width: area.width,
            height,
        };
        frame.render_widget(Clear, popup);

        let text_color = theme.color("text");
        let muted = theme.color("text_muted");
        let selected_color = theme.color("selected");
        let width = popup.width as usize;

        let mut rows: Vec<Line<'static>> = Vec::with_capacity(visible);
        // Scroll the window so the selected row stays visible.
        let first = self.selected.saturating_sub(visible - 1);
        for (i, item) in self.items.iter().enumerate().skip(first).take(visible) {
            let is_selected = i == self.selected;
            let base = if is_selected {
                Style::default()
                    .fg(selected_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(text_color)
            };
            let mut spans: Vec<Span<'static>> = vec![
                Span::styled(if is_selected { "❯ " } else { "  " }, base),
                Span::styled(item.label.clone(), base),
            ];
            // Description / detail column, truncated to fit.
            let used = 4 + display_width(&item.label);
            if let Some(detail) = item.detail() {
                let budget = width.saturating_sub(used + 8);
                if budget > 4 {
                    spans.push(Span::styled(
                        format!("  {}", truncate_to(&detail.to_string(), budget)),
                        Style::default().fg(muted),
                    ));
                }
            }
            // Right-aligned live state.
            if let Some(state) = &item.state {
                let row_width: usize = 4
                    + display_width(&item.label)
                    + item.detail().map_or(0, |d| display_width(d) + 2);
                let pad = width.saturating_sub(row_width + display_width(state) + 1);
                if pad > 0 {
                    spans.push(Span::styled(" ".repeat(pad), base));
                }
                spans.push(Span::styled(state.clone(), Style::default().fg(muted)));
            }
            rows.push(Line::from(spans));
        }
        // Footer hint row.
        rows.push(Line::from(Span::styled(
            "  ↑/↓ select · ⏎/tab accept · esc dismiss",
            Style::default().fg(muted),
        )));
        frame.render_widget(Paragraph::new(rows), popup);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        if !self.open {
            return EventOutcome::Ignored;
        }
        match key.code {
            KeyCode::Up => {
                self.move_selection(-1);
                EventOutcome::Consumed
            }
            KeyCode::Down => {
                self.move_selection(1);
                EventOutcome::Consumed
            }
            KeyCode::PageUp => {
                self.page_selection(-1);
                EventOutcome::Consumed
            }
            KeyCode::PageDown => {
                self.page_selection(1);
                EventOutcome::Consumed
            }
            KeyCode::Enter | KeyCode::Tab => {
                if let Some(item) = self.items.get(self.selected).cloned() {
                    let prefix = self.prefix.clone();
                    self.close();
                    self.confirmed = Some((item, prefix));
                }
                EventOutcome::Consumed
            }
            KeyCode::Esc => {
                self.close();
                EventOutcome::Consumed
            }
            KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.close();
                EventOutcome::Consumed
            }
            _ => EventOutcome::Ignored,
        }
    }

    fn wants_focus(&self) -> bool {
        self.open
    }
}

/// Display width of a string (counts chars; wide-char refinement lives in
/// the editor's text measurement).
fn display_width(s: &str) -> usize {
    s.chars().count()
}

/// Truncate `s` to at most `max` display columns, adding `…` when cut.
fn truncate_to(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let mut out: String = chars[..max.saturating_sub(1)].iter().collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::io::Write as _;

    fn cmd(name: &str, description: &str) -> SlashCommand {
        SlashCommand {
            name: name.to_string(),
            description: Some(description.to_string()),
            source: Some(CommandSource::Builtin),
            ..SlashCommand::default()
        }
    }

    fn sample_commands() -> Vec<SlashCommand> {
        vec![
            cmd("quit", "Exit the app"),
            cmd("model", "Show current model selection"),
            cmd("compact", "Compact the context"),
            cmd("skill:batch", "Run a batch skill"),
            cmd("skill:humanizer", "Humanize text"),
            cmd("skill:research-last30days", "Research the last 30 days"),
        ]
    }

    fn ac() -> Autocomplete {
        let mut a = Autocomplete::new(PathBuf::from("/nonexistent-cwd"));
        a.set_commands(sample_commands());
        a
    }

    // ---- scoring helpers ----

    #[test]
    fn subsequence_tiers() {
        assert_eq!(subsequence_score("wig", "wig"), 100);
        assert_eq!(subsequence_score("wig", "wiglet"), 80);
        assert_eq!(subsequence_score("wig", "skill:wig"), 60);
        assert!(subsequence_score("slw", "skill:wig") > 0);
        assert_eq!(subsequence_score("zzz", "skill:wig"), 0);
        assert!(subsequence_match("wig", "skill:wig"));
        assert!(!subsequence_match("xyz", "skill:wig"));
    }

    #[test]
    fn slash_prefix_scoring() {
        assert_eq!(score_command_text_match("q", "quit"), 900);
        assert_eq!(score_command_text_match("quit", "quit"), 1000);
        assert!(score_command_text_match("qt", "quit") > 0);
        assert_eq!(score_command_text_match("x", "quit"), 0);
    }

    // ---- slash completion ----

    #[test]
    fn slash_completes_command_names() {
        let mut a = ac();
        let lines = vec!["/qu".to_string()];
        let s = a.suggestions(&lines, 0, 3).expect("suggestions");
        assert_eq!(s.items[0].value, "quit");
        assert_eq!(s.prefix, "/qu");
        assert!(a.open);
    }

    #[test]
    fn slash_empty_prefix_lists_commands_and_skills_ranked() {
        let mut a = ac();
        let lines = vec!["/".to_string()];
        let s = a.suggestions(&lines, 0, 1).expect("suggestions");
        // Skills collapse into a single /skill: namespace row.
        assert!(s.items.iter().any(|i| i.value == "skill:"));
        assert!(!s.items.iter().any(|i| i.value == "skill:batch"));
        // Empty prefix ranks skills at 950: the namespace row is present but
        // ordinary commands still appear.
        assert!(s.items.iter().any(|i| i.value == "quit"));
    }

    #[test]
    fn skill_breakout_by_bare_prefix() {
        let mut a = ac();
        // `/hum` breaks skill:humanizer out of the namespace.
        let lines = vec!["/hum".to_string()];
        let s = a.suggestions(&lines, 0, 4).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value == "skill:humanizer"));
        assert!(!s.items.iter().any(|i| i.value == "skill:"));
    }

    #[test]
    fn skill_breakout_by_hyphen_segment() {
        let mut a = ac();
        let lines = vec!["/last".to_string()];
        let s = a.suggestions(&lines, 0, 5).expect("suggestions");
        assert!(s
            .items
            .iter()
            .any(|i| i.value == "skill:research-last30days"));
    }

    #[test]
    fn skill_namespace_prefix_expands() {
        let mut a = ac();
        let lines = vec!["/skill:ba".to_string()];
        let s = a.suggestions(&lines, 0, 9).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value == "skill:batch"));
        assert!(!s.items.iter().any(|i| i.value == "skill:"));
    }

    #[test]
    fn mid_prompt_skill_token() {
        let mut a = ac();
        let lines = vec!["please run /hum".to_string()];
        let s = a.suggestions(&lines, 0, 15).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value == "skill:humanizer"));
        // Prefix is only the slash token for mid-prompt lookups.
        assert_eq!(s.prefix, "/hum");
    }

    #[test]
    fn mid_prompt_plain_word_falls_through() {
        let mut a = ac();
        let lines = vec!["please run /zzz".to_string()];
        assert!(a.suggestions(&lines, 0, 15).is_none());
        assert!(!a.open);
    }

    #[test]
    fn absolute_path_falls_through_to_files() {
        let mut a = ac();
        let lines = vec!["/zzz".to_string()];
        // No matches + looks like an absolute path → falls through, and with
        // a nonexistent cwd there are no file suggestions either.
        assert!(a.suggestions(&lines, 0, 4).is_none());
    }

    // ---- sync completion ----

    #[test]
    fn sync_slash_completion_excludes_namespace_row() {
        let a = ac();
        let s = a.try_sync_slash_completion("/co").expect("sync");
        assert_eq!(s.items[0].value, "compact");
        let s = a.try_sync_slash_completion("/s").expect("sync");
        assert!(!s.items.iter().any(|i| i.value == "skill:"));
        assert!(a.try_sync_slash_completion("/").is_none());
        assert!(a.try_sync_slash_completion("/quit ").is_none());
    }

    // ---- apply ----

    #[test]
    fn apply_slash_replaces_live_token() {
        let a = ac();
        let lines = vec!["  /qu".to_string()];
        let applied =
            a.apply_completion(&lines, 0, 5, &AutocompleteItem::new("quit", "quit"), "/qu");
        assert_eq!(applied.lines[0], "  /quit ");
        assert_eq!(applied.cursor_col, 8);
    }

    #[test]
    fn apply_namespace_row_has_no_trailing_space() {
        let a = ac();
        let lines = vec!["/s".to_string()];
        let applied = a.apply_completion(
            &lines,
            0,
            2,
            &AutocompleteItem::new("skill:", "skill:"),
            "/s",
        );
        assert_eq!(applied.lines[0], "/skill:");
    }

    #[test]
    fn apply_mid_prompt_skill_preserves_prose() {
        let a = ac();
        let lines = vec!["please run /hum now".to_string()];
        let applied = a.apply_completion(
            &lines,
            0,
            15,
            &AutocompleteItem::new("skill:humanizer", "skill:humanizer"),
            "/hum",
        );
        assert_eq!(applied.lines[0], "please run /skill:humanizer  now");
    }

    #[test]
    fn apply_file_inserts_value_and_space() {
        let a = ac();
        let lines = vec!["see @src/ma".to_string()];
        let applied = a.apply_completion(
            &lines,
            0,
            11,
            &AutocompleteItem::new("@src/main.rs", "main.rs"),
            "@src/ma",
        );
        assert_eq!(applied.lines[0], "see @src/main.rs ");
    }

    #[test]
    fn apply_directory_chains_without_space() {
        let a = ac();
        let lines = vec!["@sr".to_string()];
        let applied =
            a.apply_completion(&lines, 0, 3, &AutocompleteItem::new("@src/", "src/"), "@sr");
        assert_eq!(applied.lines[0], "@src/");
        assert!(is_directory_completion_value("@src/"));
        assert!(is_directory_completion_value("@\"dir/\""));
        assert!(!is_directory_completion_value("@file"));
    }

    #[test]
    fn apply_quoted_path_keeps_quotes() {
        let a = ac();
        let lines = vec!["@\"my fi".to_string()];
        let applied = a.apply_completion(
            &lines,
            0,
            7,
            &AutocompleteItem::new("@\"my file.txt\"", "my file.txt"),
            "@\"my fi",
        );
        assert_eq!(applied.lines[0], "@\"my file.txt\" ");
    }

    #[test]
    fn completion_value_quoting_rules() {
        assert_eq!(
            build_completion_value("a b", true, false, false),
            "@\"a b\""
        );
        assert_eq!(build_completion_value("ab", true, false, false), "@ab");
        assert_eq!(build_completion_value("a b", true, true, true), "@\"a b");
        assert_eq!(build_completion_value("ab", false, true, false), "\"ab\"");
    }

    #[test]
    fn at_completion_matches_narrows_stale_lists() {
        assert!(at_completion_matches("@src/ma", "@src/main.rs"));
        assert!(!at_completion_matches("@src/zz", "@src/main.rs"));
        assert!(at_completion_matches("@", "@anything"));
    }

    // ---- popup keys ----

    #[test]
    fn popup_navigation_and_confirm() {
        let mut a = ac();
        let lines = vec!["/".to_string()];
        a.suggestions(&lines, 0, 1);
        let n = a.items.len();
        assert!(n >= 3);

        a.handle_key(KeyEvent::from(KeyCode::Down));
        assert_eq!(a.selected, 1);
        a.handle_key(KeyEvent::from(KeyCode::Up));
        assert_eq!(a.selected, 0);
        // Wraps around.
        a.handle_key(KeyEvent::from(KeyCode::Up));
        assert_eq!(a.selected, n - 1);
        a.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
        let expected = ((n as isize - 1 + a.max_visible as isize).rem_euclid(n as isize)) as usize;
        assert_eq!(a.selected, expected);

        a.handle_key(KeyEvent::from(KeyCode::Enter));
        assert!(!a.open);
        let (item, prefix) = a.take_confirmed().expect("confirmed");
        assert_eq!(prefix, "/");
        assert!(!item.value.is_empty());
        assert!(a.take_confirmed().is_none());
    }

    #[test]
    fn esc_dismisses_without_interrupt_side_effect() {
        let mut a = ac();
        let lines = vec!["/".to_string()];
        a.suggestions(&lines, 0, 1);
        assert_eq!(
            a.handle_key(KeyEvent::from(KeyCode::Esc)),
            EventOutcome::Consumed
        );
        assert!(!a.open);
        assert!(a.take_confirmed().is_none());
        // And with the popup closed keys pass through to the editor.
        assert_eq!(
            a.handle_key(KeyEvent::from(KeyCode::Char('a'))),
            EventOutcome::Ignored
        );
    }

    // ---- catalog from frames ----

    #[test]
    fn update_commands_parses_catalog_frame() {
        let mut a = Autocomplete::new(PathBuf::from("."));
        let frame = serde_json::json!({"type": "available_commands_update", "commands": [
            {"name": "model", "description": "Show model", "source": "builtin",
             "aliases": ["models"], "input": {"hint": "[model]"}},
            {"name": "mycmd", "source": "extension"},
            {}
        ]});
        a.update_commands(&frame);
        let commands = a.commands();
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].name, "model");
        assert_eq!(commands[0].aliases, vec!["models".to_string()]);
        assert_eq!(commands[0].argument_hint.as_deref(), Some("[model]"));
        assert_eq!(commands[0].source, Some(CommandSource::Builtin));
        assert_eq!(commands[1].source, Some(CommandSource::Extension));
    }

    // ---- real filesystem ----

    /// Build a unique temp dir; returns its path and registers cleanup.
    fn temp_tree() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "omp-tui-ac-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("src/deep")).unwrap();
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::File::create(dir.join("src/main.rs")).unwrap();
        std::fs::File::create(dir.join("src/lib.rs")).unwrap();
        std::fs::File::create(dir.join("src/deep/util.rs")).unwrap();
        std::fs::File::create(dir.join("README.md")).unwrap();
        let mut hidden = std::fs::File::create(dir.join(".env")).unwrap();
        writeln!(hidden, "SECRET=x").unwrap();
        dir
    }

    #[test]
    fn at_empty_query_lists_immediate_dir() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let lines = vec!["@".to_string()];
        let s = a.suggestions(&lines, 0, 1).expect("suggestions");
        let labels: Vec<&str> = s.items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"src/"));
        assert!(labels.contains(&"README.md"));
        // TS prefix listings skip only .git (fuzzy profile sets hidden:true).
        assert!(labels.contains(&".env"));
        assert!(!labels.contains(&".git/"));
        // Directories sort first.
        assert_eq!(s.items[0].label, "src/");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn at_fuzzy_query_matches_subsequence() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let lines = vec!["@man".to_string()];
        let s = a.suggestions(&lines, 0, 4).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value == "@src/main.rs"));
        // The tight match ranks first.
        assert_eq!(s.items[0].value, "@src/main.rs");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn at_scoped_query_lists_subdirectory() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let lines = vec!["@src/d".to_string()];
        let s = a.suggestions(&lines, 0, 6).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value == "@src/deep/"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn at_outside_cwd_uses_prefix_listing() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.join("src"));
        let lines = vec!["@../".to_string()];
        let s = a.suggestions(&lines, 0, 4).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value == "@../README.md"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tab_force_completes_bare_token() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let lines = vec!["see RE".to_string()];
        let s = a.force_file_suggestions(&lines, 0, 6).expect("suggestions");
        assert_eq!(s.items[0].value, "README.md");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tab_skips_leading_slash_command_token() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let lines = vec!["/qu".to_string()];
        assert!(a.force_file_suggestions(&lines, 0, 3).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unambiguous_path_prefix_completes() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let abs = format!("{}/src/li", dir.display());
        let lines = vec![abs.clone()];
        let s = a.suggestions(&lines, 0, abs.len()).expect("suggestions");
        assert!(s.items.iter().any(|i| i.value.ends_with("src/lib.rs")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- rendering ----

    #[test]
    fn renders_popup_above_anchor() {
        let mut a = ac();
        let lines = vec!["/".to_string()];
        a.suggestions(&lines, 0, 1);
        let backend = TestBackend::new(60, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| a.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut found = false;
        for y in 0..12 {
            let row: String = (0..60).map(|x| buf[(x, y)].symbol()).collect();
            if row.contains("accept") {
                found = true;
            }
        }
        assert!(found, "footer hint rendered");
        // Selected row is marked.
        let row0: String = (0..60).map(|x| buf[(x, 1)].symbol()).collect();
        assert!(
            row0.starts_with("❯") || row0.starts_with(' '),
            "rows aligned: {row0:?}"
        );
    }

    #[test]
    fn closed_popup_renders_nothing() {
        let a = ac();
        let backend = TestBackend::new(60, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| a.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let blank: String = (0..60).map(|x| buf[(x, 11)].symbol()).collect();
        assert!(blank.trim().is_empty());
    }

    #[test]
    fn dir_cache_invalidates() {
        let dir = temp_tree();
        let mut a = Autocomplete::new(dir.clone());
        let lines = vec!["@".to_string()];
        a.suggestions(&lines, 0, 1);
        // Add a file and invalidate: the listing picks it up.
        std::fs::File::create(dir.join("newfile.txt")).unwrap();
        a.invalidate_dir_cache(None);
        let s = a.suggestions(&lines, 0, 1).expect("suggestions");
        assert!(s.items.iter().any(|i| i.label == "newfile.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
