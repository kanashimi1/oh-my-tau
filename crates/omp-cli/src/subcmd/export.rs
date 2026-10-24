//! `omp-rs export <session.jsonl> [-o out.html]` — standalone HTML export of
//! an omp session file, a port of the TS `omp --export <file>` launch flag
//! (`export/html/index.ts`): the same template with CSS/JS/marked/highlight
//! inlined and the session data base64-embedded. Subagent transcript embedding
//! (`subSessions`) and the TUI-theme palette are not ported; the web palette
//! matches TS exactly.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use base64::Engine as _;
use serde_json::{Map, Value};

const TEMPLATE_HTML: &str = include_str!("../../assets/export/template.html");
const TEMPLATE_CSS: &str = include_str!("../../assets/export/template.css");
const TEMPLATE_JS: &str = include_str!("../../assets/export/template.js");
const TOOL_VIEWS_JS: &str = include_str!("../../assets/export/tool-views.generated.js");
const MARKED_JS: &str = include_str!("../../assets/export/vendor/marked.min.js");
const HIGHLIGHT_JS: &str = include_str!("../../assets/export/vendor/highlight.min.js");

/// TS `COMMON_EXPORT_PALETTE` (`export/html/web-palette.ts`).
const COMMON_PALETTE: &[(&str, &str)] = &[
    (
        "--font-mono",
        r#"ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Mono", Menlo, Consolas, monospace"#,
    ),
    ("--body-bg", "var(--bg)"),
    ("--container-bg", "var(--bg-raised)"),
    ("--info-bg", "var(--bg-inset)"),
    ("--text", "var(--fg)"),
    ("--muted", "var(--fg-muted)"),
    ("--dim", "var(--fg-faint)"),
    ("--thinkingText", "var(--fg-muted)"),
    (
        "--borderMuted",
        "color-mix(in srgb, var(--border) 67%, transparent)",
    ),
    ("--borderAccent", "#945ff9"),
    ("--selectedBg", "var(--accent-muted)"),
    ("--success", "var(--ok)"),
    ("--error", "var(--err)"),
    ("--warning", "var(--warn)"),
    ("--userMessageBg", "var(--accent-muted)"),
    ("--userMessageText", "var(--fg)"),
    ("--customMessageBg", "var(--bg-overlay)"),
    ("--customMessageText", "var(--fg-muted)"),
    ("--customMessageLabel", "#945ff9"),
    ("--toolPendingBg", "var(--bg-raised)"),
    ("--toolSuccessBg", "var(--bg-inset)"),
    (
        "--toolErrorBg",
        "color-mix(in srgb, var(--err) 14%, transparent)",
    ),
    ("--toolTitle", "var(--fg)"),
    ("--toolOutput", "var(--fg-muted)"),
    ("--mdHeading", "var(--accent)"),
    ("--mdLink", "var(--link)"),
    ("--mdLinkUrl", "var(--fg-faint)"),
    ("--mdCode", "var(--fg)"),
    ("--mdCodeBlock", "var(--fg)"),
    ("--mdCodeBlockBorder", "var(--border)"),
    ("--mdQuote", "var(--fg-muted)"),
    ("--mdQuoteBorder", "var(--border-strong)"),
    ("--mdHr", "var(--border)"),
    ("--mdListBullet", "var(--accent)"),
    ("--toolDiffAdded", "var(--ok)"),
    ("--toolDiffRemoved", "var(--err)"),
    ("--toolDiffContext", "var(--fg-faint)"),
    ("--syntaxOperator", "var(--fg)"),
    ("--syntaxPunctuation", "var(--fg-muted)"),
    ("--thinkingOff", "var(--fg-faint)"),
    ("--thinkingMinimal", "var(--fg-faint)"),
    ("--thinkingLow", "#945ff9"),
    ("--thinkingMedium", "#b281d6"),
    ("--thinkingHigh", "var(--accent)"),
    ("--thinkingXhigh", "var(--warn)"),
    ("--bashMode", "#168b9a"),
    ("--pythonMode", "var(--warn)"),
    ("--statusLineBg", "var(--bg)"),
    ("--statusLineSep", "var(--fg-faint)"),
    ("--statusLineModel", "var(--accent)"),
    ("--statusLinePath", "#168b9a"),
    ("--statusLineGitClean", "var(--ok)"),
    ("--statusLineGitDirty", "var(--warn)"),
    ("--statusLineContext", "var(--fg-muted)"),
    ("--statusLineSpend", "#168b9a"),
    ("--statusLineStaged", "var(--ok)"),
    ("--statusLineDirty", "var(--warn)"),
    ("--statusLineUntracked", "#945ff9"),
    ("--statusLineOutput", "#b281d6"),
    ("--statusLineCost", "#b281d6"),
    ("--statusLineSubagents", "var(--accent)"),
];

const DARK_PALETTE: &[(&str, &str)] = &[
    ("--bg", "#0f0b14"),
    ("--bg-raised", "#16111c"),
    ("--bg-inset", "#09060c"),
    ("--bg-overlay", "#211b28"),
    ("--fg", "#e6e3ea"),
    ("--fg-muted", "#a49faa"),
    ("--fg-faint", "#6e6974"),
    ("--accent", "#ed4abf"),
    ("--accent-muted", "oklch(0.674 0.23 341 / 18%)"),
    ("--ok", "#68ca80"),
    ("--err", "#f05653"),
    ("--warn", "#e4b33f"),
    ("--border", "oklch(1 0 0 / 9%)"),
    ("--border-strong", "oklch(1 0 0 / 13%)"),
    ("--ring", "oklch(0.817 0.112 205 / 70%)"),
    ("--link", "#5ad8e5"),
    ("--syntaxComment", "#6e6974"),
    ("--syntaxKeyword", "#945ff9"),
    ("--syntaxFunction", "#e4b33f"),
    ("--syntaxVariable", "#5ad8e5"),
    ("--syntaxString", "#68ca80"),
    ("--syntaxNumber", "#ed4abf"),
    ("--syntaxType", "#b281d6"),
];

const LIGHT_PALETTE: &[(&str, &str)] = &[
    ("--bg", "oklch(0.985 0.004 307)"),
    ("--bg-raised", "oklch(1 0 0)"),
    ("--bg-inset", "oklch(0.95 0.006 307)"),
    ("--bg-overlay", "oklch(0.965 0.006 307)"),
    ("--fg", "oklch(0.26 0.03 307)"),
    ("--fg-muted", "oklch(0.46 0.03 307)"),
    ("--fg-faint", "oklch(0.58 0.025 307)"),
    ("--accent", "oklch(0.62 0.23 341)"),
    ("--accent-muted", "oklch(0.62 0.23 341 / 14%)"),
    ("--ok", "oklch(0.55 0.13 150)"),
    ("--err", "oklch(0.55 0.19 25)"),
    ("--warn", "oklch(0.6 0.13 85)"),
    ("--border", "oklch(0 0 0 / 10%)"),
    ("--border-strong", "oklch(0 0 0 / 15%)"),
    ("--ring", "oklch(0.58 0.13 230 / 70%)"),
    ("--link", "#0f7a88"),
    ("--syntaxComment", "#77717c"),
    ("--syntaxKeyword", "#6d35c7"),
    ("--syntaxFunction", "#8a6000"),
    ("--syntaxVariable", "#087f8c"),
    ("--syntaxString", "#247a3d"),
    ("--syntaxNumber", "#b51f88"),
    ("--syntaxType", "#7b3fa2"),
];

/// Entry point: `omp-rs export <session.jsonl> [--output <path>]`.
///
/// # Errors
/// Missing/unreadable session file or unwritable output path.
pub fn run(args: &[String]) -> Result<i32> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    // The `--export <file>` launch path calls run("export", &[path]) without
    // the name token; direct dispatch includes it. Skip it only when present.
    let mut it = args
        .iter()
        .skip(usize::from(args.first().is_some_and(|a| a == "export")))
        .peekable();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                output = Some(PathBuf::from(it.next().context("--output needs a value")?))
            }
            other if other.starts_with("--output=") => output = Some(PathBuf::from(&other[9..])),
            other if other.starts_with('-') => anyhow::bail!("unknown flag: {other}"),
            other => {
                if input.is_some() {
                    anyhow::bail!("export takes exactly one session file");
                }
                input = Some(PathBuf::from(other));
            }
        }
    }
    let input = input.context("Usage: omt export <session.jsonl> [-o out.html]")?;
    let data =
        session_data(&input).with_context(|| format!("read session file {}", input.display()))?;
    let html = generate_html(&data);
    let output = output.unwrap_or_else(|| {
        let stem = input
            .file_stem()
            .map_or("session", |s| s.to_str().unwrap_or("session"));
        PathBuf::from(format!("omp-session-{stem}.html"))
    });
    std::fs::write(&output, html).with_context(|| format!("write {}", output.display()))?;
    println!("{}", output.display());
    Ok(0)
}

/// The standalone HTML export of one session file (the `omp-rs web` export
/// button serves this directly).
///
/// # Errors
/// Missing or unreadable session file.
pub fn render_html(path: &Path) -> Result<String> {
    let data =
        session_data(path).with_context(|| format!("read session file {}", path.display()))?;
    Ok(generate_html(&data))
}

/// TS `buildSessionData` for the file-based path: header (minus
/// `previousSessionFiles`), raw entries, leaf id.
fn session_data(path: &Path) -> Result<Value> {
    let loaded = omp_session::load(path)?;
    let mut header: Option<Value> = None;
    let mut entries: Vec<Value> = Vec::new();
    let mut leaf_id: Option<String> = None;
    for line in &loaded.lines {
        let Ok(value) = serde_json::from_str::<Value>(&line.raw) else {
            continue;
        };
        let kind = value.get("type").and_then(Value::as_str);
        if kind == Some("session") {
            let mut h = value;
            if let Some(obj) = h.as_object_mut() {
                obj.remove("previousSessionFiles");
            }
            header = Some(h);
        } else if kind != Some("title") {
            // "title" lines are the fixed-width title slot, not transcript entries.
            leaf_id = value
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or(leaf_id);
            entries.push(value);
        }
    }
    Ok(serde_json::json!({
        "header": header,
        "entries": entries,
        "leafId": leaf_id,
    }))
}

/// TS `getTemplate` + `generateHtml`: inline every asset, then theme styles
/// and the base64 session payload.
fn generate_html(session_data: &Value) -> String {
    template_html()
        .replace(
            "<theme-vars/>",
            &format!("<style>{}</style>", theme_styles()),
        )
        .replace(
            "{{SESSION_DATA}}",
            &base64::engine::general_purpose::STANDARD
                .encode(serde_json::to_string(session_data).unwrap_or_default()),
        )
}

/// TS `getTemplate`: CSS minified like the TS pipeline, all scripts inlined.
fn template_html() -> String {
    let minified_css = minify_css(TEMPLATE_CSS);
    TEMPLATE_HTML
        .replace("<template-css/>", &format!("<style>{minified_css}</style>"))
        .replace(
            "<template-marked/>",
            &format!("<script>{MARKED_JS}</script>"),
        )
        .replace(
            "<template-highlight/>",
            &format!("<script>{HIGHLIGHT_JS}</script>"),
        )
        .replace(
            "<template-tool-views/>",
            &format!("<script>{TOOL_VIEWS_JS}</script>"),
        )
        .replace("<template-js/>", &format!("<script>{TEMPLATE_JS}</script>"))
}

/// TS minification: strip block comments, collapse whitespace, tighten braces.
fn minify_css(css: &str) -> String {
    let no_comments = {
        // `/* ... */` without regex: scan for comment markers.
        let mut out = String::with_capacity(css.len());
        let mut rest = css;
        while let Some(start) = rest.find("/*") {
            out.push_str(&rest[..start]);
            rest = match rest[start..].find("*/") {
                Some(end) => &rest[start + end + 2..],
                None => "",
            };
        }
        out.push_str(rest);
        out
    };
    let collapsed = no_comments.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::with_capacity(collapsed.len());
    let mut chars = collapsed.chars().peekable();
    while let Some(c) = chars.next() {
        if matches!(c, '{' | '}' | ':' | ';' | ',') {
            while out.ends_with(' ') {
                out.pop();
            }
            out.push(c);
            while chars.peek() == Some(&' ') {
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out.trim().to_string()
}

/// TS `generateThemeStyles` for the web palette.
fn theme_styles() -> String {
    [
        format!(":root, :root[data-theme=\"dark\"] {{ color-scheme: dark; {} }}", theme_vars(DARK_PALETTE)),
        format!(":root[data-theme=\"light\"] {{ color-scheme: light; {} }}", theme_vars(LIGHT_PALETTE)),
        format!(
            "@media (prefers-color-scheme: light) {{ :root:not([data-theme]) {{ color-scheme: light; {} }} }}",
            theme_vars(LIGHT_PALETTE)
        ),
    ]
    .join("\n")
}

/// TS `webExportThemeVars`: common entries overridden by the scheme's.
fn theme_vars(scheme: &[(&str, &str)]) -> String {
    let mut merged: Map<String, Value> = Map::new();
    for (key, value) in COMMON_PALETTE.iter().chain(scheme.iter()) {
        merged.insert((*key).to_string(), Value::String((*value).to_string()));
    }
    let mut out: Vec<String> = Vec::new();
    for (key, value) in merged {
        if let Some(text) = value.as_str() {
            out.push(format!("{key}: {text};"));
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review #6: TS writes the export with `Bun.write`, which silently
    /// replaces an existing file — the port's `fs::write` matches. Pin the
    /// behavior so a "protective" change cannot silently diverge from TS.
    #[test]
    fn export_overwrites_an_existing_output_like_ts() {
        let dir = std::env::temp_dir();
        let session = dir.join(format!("omp-rs-export-ow-{}.jsonl", std::process::id()));
        std::fs::write(
            &session,
            concat!(
                r#"{"type":"session","version":3,"id":"s1","timestamp":"t","cwd":"/x"}"#,
                "\n",
            ),
        )
        .unwrap();
        let out = dir.join(format!("omp-rs-export-ow-{}.html", std::process::id()));
        std::fs::write(&out, "previous contents").unwrap();
        let code = run(&[
            session.to_string_lossy().into_owned(),
            "-o".into(),
            out.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert_eq!(code, 0);
        let written = std::fs::read_to_string(&out).unwrap();
        assert!(
            written.starts_with("<!DOCTYPE html>"),
            "overwritten: {written:?}"
        );
        std::fs::remove_file(&session).ok();
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn minify_css_matches_ts_pipeline() {
        let css = "/* comment */ a {\n  color: red;\n}\n";
        assert_eq!(minify_css(css), "a{color:red;}");
    }

    #[test]
    fn template_has_no_leftover_placeholders() {
        let html =
            generate_html(&serde_json::json!({"header": null, "entries": [], "leafId": null}));
        assert!(!html.contains("<template-css/>"));
        assert!(!html.contains("{{SESSION_DATA}}"));
        assert!(html.contains("<style>"));
        assert!(html.contains("color-scheme: dark"));
        assert!(html.contains("color-scheme: light"));
    }

    #[test]
    fn session_data_extracts_header_entries_leaf() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("omp-rs-export-test-{}.jsonl", std::process::id()));
        std::fs::write(
            &path,
            concat!(
                r#"{"type":"session","version":3,"id":"s1","timestamp":"t","cwd":"/x","previousSessionFiles":["old.jsonl"]}"#,
                "\n",
                r#"{"type":"message","id":"e1","timestamp":1,"message":{"role":"user","content":"hi"}}"#,
                "\n",
            ),
        )
        .unwrap();
        let data = session_data(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let header = data.get("header").unwrap();
        assert!(header.get("previousSessionFiles").is_none());
        assert_eq!(data.get("leafId").unwrap().as_str(), Some("e1"));
        assert_eq!(data.get("entries").unwrap().as_array().unwrap().len(), 1);
    }
}
