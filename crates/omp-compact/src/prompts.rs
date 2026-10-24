//! Prompt templates and the `prompt.render` post-render formatter.
//!
//! The templates are byte copies of omp's `pi-agent-core/src/compaction/prompts/*.md` (and the
//! coding-agent `context-notes.md`). omp renders them through Handlebars and then `format()`
//! (pi-utils `prompt.ts`): trailing whitespace trimmed, runs of blank lines collapsed. The
//! formatter matters beyond the static prompts: it also runs over the *substituted* summary text,
//! so a stored summary with blank-line runs reaches the model with them stripped.

use std::sync::OnceLock;

use regex::Regex;

pub const SUMMARIZATION: &str = include_str!("../prompts/compaction-summary.md");
pub const UPDATE_SUMMARIZATION: &str = include_str!("../prompts/compaction-update-summary.md");
pub const SHORT_SUMMARY: &str = include_str!("../prompts/compaction-short-summary.md");
pub const TURN_PREFIX: &str = include_str!("../prompts/compaction-turn-prefix.md");
pub const SUMMARIZATION_SYSTEM: &str = include_str!("../prompts/summarization-system.md");
pub const BRANCH_SUMMARY: &str = include_str!("../prompts/branch-summary.md");
pub const BRANCH_SUMMARY_PREAMBLE: &str = include_str!("../prompts/branch-summary-preamble.md");
pub const COMPACTION_SUMMARY_CONTEXT: &str =
    include_str!("../prompts/compaction-summary-context.md");
pub const BRANCH_SUMMARY_CONTEXT: &str = include_str!("../prompts/branch-summary-context.md");
pub const HANDOFF_SUMMARY_CONTEXT: &str = include_str!("../prompts/handoff-summary-context.md");
pub const HANDOFF_DOCUMENT: &str = include_str!("../prompts/handoff-document.md");
pub const CONTEXT_NOTES: &str = include_str!("../prompts/context-notes.md");

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("static regex"))
}

/// `prompt.render(template, vars)`: substitute `{{name}}` (unescaped) and run [`format`].
#[must_use]
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut text = template.to_owned();
    for (name, value) in vars {
        text = text.replace(&format!("{{{{{name}}}}}"), value);
    }
    format(&text)
}

/// The static prompts are rendered once per call; they have no variables.
#[must_use]
pub fn render_static(template: &str) -> String {
    format(template)
}

/// Port of pi-utils `format(content, { renderPhase: "post-render" })` (no ASCII-symbol
/// replacement, no RFC 2119 normalisation: `render` passes neither).
#[must_use]
pub fn format(content: &str) -> String {
    static OPEN: OnceLock<Regex> = OnceLock::new();
    static CLOSE: OnceLock<Regex> = OnceLock::new();
    static SEP: OnceLock<Regex> = OnceLock::new();
    static ROW: OnceLock<Regex> = OnceLock::new();
    let open = re(&OPEN, r"^<([a-z_-]+)(?:\s+[^>]*)?>$");
    let close = re(&CLOSE, r"^</([a-z_-]+)>$");
    let sep = re(&SEP, r"^\|[-:\s|]+\|$");
    let row = re(&ROW, r"^\|.*\|$");

    let lines: Vec<&str> = content.split('\n').collect();
    let mut result: Vec<String> = Vec::with_capacity(lines.len());
    let mut in_code_block = false;
    let mut top_level_tags: Vec<String> = Vec::new();
    let non_blank = |s: &str| s.chars().any(|c| !c.is_whitespace());

    let mut i = 0;
    while i < lines.len() {
        let mut line = lines[i].trim_end().to_owned();
        let mut s = line.len() - line.trim_start_matches([' ', '\t']).len();
        if line[s..].chars().next().is_some_and(|c| !c.is_ascii()) {
            s = line.len() - line.trim_start().len();
        }
        let first = line[s..].chars().next();

        if matches!(first, Some('`' | '~'))
            && (line[s..].starts_with("```") || line[s..].starts_with("~~~"))
        {
            in_code_block = !in_code_block;
            result.push(line);
            i += 1;
            continue;
        }
        if in_code_block {
            result.push(line);
            i += 1;
            continue;
        }

        let mut is_closing_line = false;
        if first == Some('<') {
            let trimmed = line[s..].to_owned();
            if trimmed.as_bytes().get(1) == Some(&b'/') {
                if let Some(caps) = close.captures(&trimmed) {
                    is_closing_line = true;
                    if top_level_tags.last().is_some_and(|t| *t == caps[1]) {
                        top_level_tags.pop();
                    }
                }
            } else if s == 0 && !trimmed.ends_with("/>") {
                if let Some(caps) = open.captures(&trimmed) {
                    top_level_tags.push(caps[1].to_owned());
                }
            }
        } else if first == Some('|') {
            let trimmed = line[s..].to_owned();
            if sep.is_match(&trimmed) {
                line = format!("{}{}", &line[..s], compact_table_sep(&trimmed));
            } else if row.is_match(&trimmed) {
                line = format!("{}{}", &line[..s], compact_table_row(&trimmed));
            }
        }

        if s >= line.len() {
            // Blank line: a run of 2+ is dropped entirely, a single one is kept.
            let next_blank = lines
                .get(i + 1)
                .is_none_or(|next| next.is_empty() || !non_blank(next));
            if next_blank {
                while result.last().is_some_and(String::is_empty) {
                    result.pop();
                }
                let mut j = i + 1;
                while j < lines.len() && (lines[j].is_empty() || !non_blank(lines[j])) {
                    j += 1;
                }
                i = j;
                continue;
            }
            if result.last().is_none_or(String::is_empty) {
                i += 1;
                continue;
            }
        }

        if is_closing_line {
            while result.last().is_some_and(String::is_empty) {
                result.pop();
            }
        }
        result.push(line);
        i += 1;
    }
    while result.last().is_some_and(String::is_empty) {
        result.pop();
    }
    result.join("\n")
}

fn compact_table_row(line: &str) -> String {
    line.split('|').map(str::trim).collect::<Vec<_>>().join("|")
}

fn compact_table_sep(line: &str) -> String {
    let cells: Vec<&str> = line.split('|').filter(|c| !c.trim().is_empty()).collect();
    let normalized: Vec<&str> = cells
        .iter()
        .map(|c| {
            let t = c.trim();
            match (t.starts_with(':'), t.ends_with(':')) {
                (true, true) => ":---:",
                (true, false) => ":---",
                (false, true) => "---:",
                (false, false) => "---",
            }
        })
        .collect();
    format!("|{}|", normalized.join("|"))
}
