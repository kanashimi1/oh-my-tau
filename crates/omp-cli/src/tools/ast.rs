//! `ast_grep` (structural search) and `ast_edit` (AST-aware rewrite) on the
//! vendored `pi-ast` engine, matching omp's TS tool semantics: `pat`/`path`/
//! `lang`/`skip` parameters, 50-match default limit with skip pagination,
//! parse-error normalization, and `ast_edit` op validation + per-replacement
//! diff preview lines. omp-rs has no staged-rewrite `resolve` surface, so
//! `ast_edit` applies rewrites in place instead of staging behind `xd://resolve`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use pa_agent::types::AgentToolResult;
use serde_json::{json, Value};

use super::util::{format_path_relative_to_cwd, path_list, resolve_to_cwd, str_arg};
use super::Env;
use pi_ast::language::SupportLang;
use pi_ast::ops::{
    collect_matches, compile_rewrite_rules, compile_search_patterns, resolve_supported_lang,
    rewrite_source, AstMatch,
};
use pi_ast::parse_cache::parse_cached;

pub(crate) const AST_GREP_DESCRIPTION: &str = "Structural code search via ast-grep. Use when syntax shape matters more than text (calls, declarations, language constructs).\n\n<instruction>\n- Narrow each call to one language. `pat` is ONE AST pattern; separate calls for unrelated patterns.\n- Set `lang` when extension inference is ambiguous (for example, `cpp` for `.h`); `.cu` and `.cuh` infer as C++.\n- `$NAME` captures one node; `$_` matches without binding; `$$$NAME` zero-or-more; `$$$` zero-or-more unbound.\n  - Use `$$$NAME`, NOT `$$NAME` (invalid). Names UPPERCASE, whole node — `prefix$VAR` fails.\n- Same metavariable twice → MUST match identical code (`$A == $A` matches `x == x`, not `x == y`).\n- Patterns MUST parse as single AST node. Non-standalone → wrap: `class $_ { … }`.\n- C++ expression-statement calls need trailing `;`: `ns::doThing($ARG);`, `$CALLEE($ARG);`.\n- TS: tolerate annotations — `async function $NAME($$$ARGS): $_ { $$$BODY }`.\n- Declaration forms are distinct — `function foo`, method `foo()`, `const foo = () => {}`; search the right form before concluding absence.\n- Loosest existence check: `pat: \"executeBash\"` with narrow `path`.\n</instruction>\n\n<critical>\n- AVOID repo-root scans — narrow `path` first.\n- Parse issues = query failure, not absence: fix pattern or tighten `path` before concluding \"no matches\".\n</critical>";
pub(crate) const AST_EDIT_DESCRIPTION: &str = "Structural AST-aware rewrites via ast-grep. Use for codemods where text replace is unsafe. Mixed-language paths are fine: each file is parsed in its own language, and a pattern only rewrites files it parses in.\n\n- Metavariables in `pat` (`$A`, `$$$ARGS`) substitute into `out`.\n- **Patterns match AST structure, not text.** `$NAME` = one node; `$_` = unbound; `$$$NAME` = zero-or-more.\n  - Use `$$$NAME`, NOT `$$NAME` (invalid). Names UPPERCASE, whole node — partial like `prefix$VAR` fails.\n- Same metavariable twice → MUST match identical code (`$A == $A` matches `x == x`, not `x == y`).\n- Rewrite patterns MUST parse as single AST node. Non-standalone → wrap: `class $_ { … }`.\n- TS: tolerate annotations — `async function $NAME($$$ARGS): $_ { $$$BODY }`. Delete with empty `out`: `{\"pat\":\"console.log($$$)\",\"out\":\"\"}`.\n- 1:1 substitution — no splitting/merging captures.\n- Changes are applied immediately.\n- Parse issues → malformed rewrite, not clean no-op. For one-off text edits, prefer the Edit tool.";

const DEFAULT_AST_LIMIT: usize = 50;
const MAX_AST_FILES: usize = 1000;
const MAX_PARSE_ERRORS: usize = 5;
const DIFF_LINE_MAX: usize = 120;

/// Parameter schema of `ast_grep` (matches the TS `astGrepSchema`).
#[must_use]
pub fn schema_ast_grep() -> Value {
    json!({
        "type": "object",
        "properties": {
            "pat": {"type": "string", "description": "ast pattern"},
            "path": {"type": "string", "description": "file, directory, glob, or internal URL to search; pass several as a semicolon-delimited list (\"src; tests\"). Omitted -> searches the workspace root (\".\")"},
            "lang": {"type": "string", "description": "language override, e.g. cpp for ambiguous .h files"},
            "skip": {"type": "number", "description": "matches to skip"}
        },
        "required": ["pat"],
        "additionalProperties": false
    })
}

/// Parameter schema of `ast_edit` (matches the TS `astEditSchema`).
#[must_use]
pub fn schema_ast_edit() -> Value {
    json!({
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "minItems": 1,
                "description": "rewrite ops",
                "items": {
                    "type": "object",
                    "properties": {
                        "pat": {"type": "string", "description": "ast pattern"},
                        "out": {"type": "string", "description": "replacement template"}
                    },
                    "required": ["pat", "out"],
                    "additionalProperties": false
                }
            },
            "paths": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "description": "file, directory, glob, or internal URL to rewrite"},
                "description": "files, directories, globs, or internal URLs to rewrite"
            }
        },
        "required": ["ops", "paths"],
        "additionalProperties": false
    })
}

/// Files under `entries` (files, dirs or globs, cwd-relative) that infer to a
/// supported language, sorted and capped at [`MAX_AST_FILES`].
fn scope_files(env: &Env, entries: &[String]) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for raw in entries {
        let resolved = resolve_to_cwd(raw, &env.cwd);
        if resolved.is_file() {
            files.push(resolved);
            continue;
        }
        let (base, glob): (PathBuf, String) = if resolved.is_dir() {
            (resolved, "**/*".to_string())
        } else {
            // Split a glob expression at its first wildcard character.
            let text = raw.replace('\\', "/");
            let cut = text.find(['*', '?', '[']).unwrap_or(text.len());
            let base_raw = text[..cut].trim_end_matches('/');
            let base = resolve_to_cwd(if base_raw.is_empty() { "." } else { base_raw }, &env.cwd);
            if !base.is_dir() {
                bail!("Path not found: {raw}");
            }
            (base, text[cut..].to_string())
        };
        for matched in pi_ast::ops::collect_matched_files(&base, &[glob])? {
            files.push(matched.absolute_path);
        }
    }
    files.retain(|file| SupportLang::from_path(file).is_some());
    files.sort();
    files.dedup();
    files.truncate(MAX_AST_FILES);
    Ok(files)
}

fn parse_error_note(file: &Path, env: &Env) -> String {
    format!(
        "{}: parse error (syntax tree contains error nodes)",
        format_path_relative_to_cwd(file, &env.cwd, false)
    )
}

fn cap_parse_errors(errors: Vec<String>) -> (Vec<String>, usize) {
    let total = errors.len();
    let mut capped: Vec<String> = errors.into_iter().take(MAX_PARSE_ERRORS).collect();
    if total > MAX_PARSE_ERRORS {
        capped.push(format!("… and {} more", total - MAX_PARSE_ERRORS));
    }
    (capped, total)
}

fn language_or_infer(explicit: Option<&str>, path: &Path) -> Result<SupportLang> {
    match explicit {
        Some(name) => resolve_supported_lang(name),
        None => SupportLang::from_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "Unable to infer language from file extension: {}. Specify `lang` explicitly.",
                path.display()
            )
        }),
    }
}

fn clamp_line(text: &str) -> String {
    text.split('\n')
        .next()
        .unwrap_or("")
        .chars()
        .take(DIFF_LINE_MAX)
        .collect()
}

/// `ast_grep`: structural search.
///
/// # Errors
/// Empty `pat`, invalid `skip`, unknown `lang`, unresolvable paths, or an
/// invalid pattern.
#[allow(clippy::too_many_lines)]
pub fn ast_grep(env: &Env, args: &Value) -> Result<AgentToolResult> {
    let pattern = str_arg(args, "pat")?.trim().to_string();
    if pattern.is_empty() {
        bail!("`pat` must be a non-empty pattern");
    }
    let skip = match args.get("skip") {
        None | Some(Value::Null) => 0usize,
        Some(v) => match v.as_f64() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Some(n) if n.is_finite() && n >= 0.0 => n.floor() as usize,
            _ => bail!("skip must be a non-negative number"),
        },
    };
    let lang = args.get("lang").and_then(Value::as_str);
    let mut entries = path_list(args.get("path").and_then(Value::as_str));
    if entries.is_empty() {
        entries.push(".".into());
    }
    let files = scope_files(env, &entries)?;

    let mut parse_errors: Vec<String> = Vec::new();
    // (relative path, absolute path, matches) kept in path order.
    let mut per_file: Vec<(String, PathBuf, Vec<AstMatch>)> = Vec::new();
    for file in &files {
        let Ok(source) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(language) = language_or_infer(lang, file) else {
            continue;
        };
        let patterns = compile_search_patterns(&pattern, language)
            .map_err(|error| anyhow::anyhow!("Invalid pattern: {error}"))?;
        if let Some(tree) = parse_cached(&source, language)? {
            if tree.root_node().has_error() {
                parse_errors.push(parse_error_note(file, env));
            }
        }
        let matches = collect_matches(&source, language, &patterns);
        if matches.is_empty() {
            continue;
        }
        per_file.push((
            format_path_relative_to_cwd(file, &env.cwd, false),
            file.clone(),
            matches,
        ));
    }
    per_file.sort_by(|a, b| a.0.cmp(&b.0));
    // TS flattens the visible window over all files, ordered by path then
    // position: skip globally, then take the default limit.
    let mut flat: Vec<(&str, &PathBuf, &AstMatch)> = Vec::new();
    for (relative, absolute, matches) in &per_file {
        for m in matches {
            flat.push((relative, absolute, m));
        }
    }
    let total_matches = flat.len();
    let visible = total_matches.saturating_sub(skip);
    let limit_reached = visible > DEFAULT_AST_LIMIT;

    let (capped_parse_errors, parse_errors_total) = cap_parse_errors(parse_errors);
    if visible == 0 {
        let mut text = if capped_parse_errors.is_empty() {
            "No matches found".to_string()
        } else {
            "No matches found. Parse issues mean the query may be mis-scoped; narrow `path` before concluding absence."
                .to_string()
        };
        if !capped_parse_errors.is_empty() {
            text.push_str("\n\nParse errors (showing ");
            text.push_str(&capped_parse_errors.len().to_string());
            text.push_str(" of ");
            text.push_str(&parse_errors_total.to_string());
            text.push(':');
            text.extend(capped_parse_errors.iter().map(|e| format!("\n{e}")));
        }
        return Ok(AgentToolResult::text(text));
    }

    // Render per file, mirroring the `grep` tool: hashline snapshot tags when
    // active, `*line:text` match rows, continuation lines space-marked.
    let mut model: Vec<String> = Vec::new();
    let window: Vec<(&str, &PathBuf, &AstMatch)> = flat
        .into_iter()
        .skip(skip)
        .take(DEFAULT_AST_LIMIT)
        .collect();
    let mut last_file: Option<&str> = None;
    for (relative, absolute, _m) in &window {
        if last_file == Some(relative) {
            continue;
        }
        last_file = Some(relative);
        if !model.is_empty() {
            model.push(String::new());
        }
        let tag = if env.hashline {
            env.store.record_file(absolute, None)
        } else {
            None
        };
        if let Some(t) = &tag {
            model.push(format!("[{relative}#{t}]"));
        } else {
            model.push(format!("{relative}:"));
        }
        let mut lines: Vec<String> = Vec::new();
        for (_, _, m) in window.iter().filter(|(r, _, _)| *r == *relative) {
            for (offset, line) in m.text.split('\n').enumerate() {
                let marker = if offset == 0 { '*' } else { ' ' };
                let row = m.line + offset;
                if tag.is_some() {
                    lines.push(format!("{marker}{row}:{line}"));
                } else {
                    lines.push(format!("{marker}{row}|{line}"));
                }
            }
        }
        if let Some(t) = &tag {
            env.store.record_seen_lines(
                absolute,
                t,
                &pi_edit::store::seen_lines_from_body(&lines.join("\n")),
            );
        }
        model.extend(lines);
    }
    if limit_reached {
        model.push(String::new());
        model.push("Result limit reached; narrow path or increase limit.".into());
    }
    if !capped_parse_errors.is_empty() {
        model.push(String::new());
        model.push(format!(
            "Parse errors (showing {} of {}):",
            capped_parse_errors.len(),
            parse_errors_total
        ));
        model.extend(capped_parse_errors.iter().cloned());
    }
    Ok(AgentToolResult::text(model.join("\n")))
}

/// `ast_edit`: AST-aware rewrite. Applies matching rewrites per file (each file
/// is parsed in its own language; a pattern only rewrites files it parses in).
///
/// # Errors
/// Missing/empty/duplicate `ops` patterns, unresolvable paths, or overlapping
/// replacements within one file.
#[allow(clippy::too_many_lines)]
pub fn ast_edit(env: &Env, args: &Value) -> Result<AgentToolResult> {
    let ops = args
        .get("ops")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("`ops` must include at least one op entry"))?;
    if ops.is_empty() {
        bail!("`ops` must include at least one op entry");
    }
    let mut rules: Vec<(String, String)> = Vec::with_capacity(ops.len());
    let mut seen: HashSet<String> = HashSet::new();
    for (index, entry) in ops.iter().enumerate() {
        let pat = entry.get("pat").and_then(Value::as_str).unwrap_or_default();
        if pat.is_empty() {
            bail!("`ops[{index}].pat` must be a non-empty pattern");
        }
        let out = entry.get("out").and_then(Value::as_str).unwrap_or_default();
        if !seen.insert(pat.to_string()) {
            bail!("Duplicate rewrite pattern: {pat}");
        }
        rules.push((pat.to_string(), out.to_string()));
    }
    let paths = args
        .get("paths")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("`paths` must include at least one path"))?;
    let entries: Vec<String> = paths
        .iter()
        .map(|p| {
            p.as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("`paths` entries must be strings"))
        })
        .collect::<Result<_>>()?;
    if entries.is_empty() {
        bail!("`paths` must include at least one path");
    }
    let files = scope_files(env, &entries)?;

    let mut total_replacements = 0u64;
    let mut parse_errors: Vec<String> = Vec::new();
    let mut sections: Vec<String> = Vec::new();
    let mut files_touched = 0usize;
    for file in &files {
        let Ok(source) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(language) = language_or_infer(None, file) else {
            continue;
        };
        let compiled = compile_rewrite_rules(&rules, language).map_err(|(index, error)| {
            anyhow::anyhow!("Invalid pattern in ops[{index}]: {error}")
        })?;
        if let Some(tree) = parse_cached(&source, language)? {
            if tree.root_node().has_error() {
                parse_errors.push(parse_error_note(file, env));
            }
        }
        // Replay one rule at a time: the first match on the working text gives
        // a line/column-accurate anchor for the `-`/`+` preview rows.
        let mut work = source.clone();
        let mut file_replacements = 0u64;
        let mut anchors: Vec<(usize, usize, String, String)> = Vec::new();
        for (rule, (pattern_text, _)) in compiled.iter().zip(rules.iter()) {
            let _ = pattern_text;
            let patterns = &rule.patterns;
            loop {
                let matches = collect_matches(&work, language, patterns);
                let Some(first) = matches.first() else {
                    break;
                };
                let mut line = 1usize;
                let mut col = 1usize;
                let mut offset = 0usize;
                while offset < first.byte_start {
                    match work[offset..].find('\n') {
                        Some(nl) if offset + nl < first.byte_start => {
                            line += 1;
                            col = 1;
                            offset += nl + 1;
                        }
                        _ => {
                            col += first.byte_start - offset;
                            offset = first.byte_start;
                        }
                    }
                }
                let (next, count) = rewrite_source(&work, language, std::slice::from_ref(rule))
                    .map_err(|e| anyhow::anyhow!(e))?;
                if count == 0 {
                    break;
                }
                // Before/after text of the first replacement of this pass.
                let after = extract_replacement(&work, &next, first);
                anchors.push((line, col, first.text.clone(), after));
                if next == work {
                    break; // idempotent rewrite; avoid spinning
                }
                work = next;
                file_replacements = file_replacements.saturating_add(u64::from(count));
            }
        }
        if file_replacements == 0 {
            continue;
        }
        std::fs::write(file, &work)?;
        total_replacements += file_replacements;
        files_touched += 1;
        let mut lines: Vec<String> = Vec::new();
        for (line, col, before, after) in &anchors {
            lines.push(format!("-{line}:{col} {}", clamp_line(before)));
            lines.push(format!("+{line}:{col} {}", clamp_line(after)));
        }
        sections.push(format!(
            "{}:",
            format_path_relative_to_cwd(file, &env.cwd, false)
        ));
        sections.extend(lines);
    }

    let (capped_parse_errors, parse_errors_total) = cap_parse_errors(parse_errors);
    if total_replacements == 0 {
        let mut text = "No replacements made".to_string();
        if !capped_parse_errors.is_empty() {
            text.push_str("\n\nParse errors (showing ");
            text.push_str(&capped_parse_errors.len().to_string());
            text.push_str(" of ");
            text.push_str(&parse_errors_total.to_string());
            text.push(':');
            text.extend(capped_parse_errors.iter().map(|e| format!("\n{e}")));
        }
        return Ok(AgentToolResult::text(text));
    }
    let mut model = vec![format!(
        "Applied {total_replacements} replacement{} in {files_touched} file{}.",
        if total_replacements == 1 { "" } else { "s" },
        if files_touched == 1 { "" } else { "s" }
    )];
    model.extend(sections);
    if !capped_parse_errors.is_empty() {
        model.push(String::new());
        model.push(format!(
            "Parse errors (showing {} of {}):",
            capped_parse_errors.len(),
            parse_errors_total
        ));
        model.extend(capped_parse_errors.iter().cloned());
    }
    Ok(AgentToolResult::text(model.join("\n")))
}

/// Replacement text at `first`'s span: `next` equals `work` with that span
/// substituted, so the after-text is `next` with the before-span removed.
fn extract_replacement(work: &str, next: &str, first: &AstMatch) -> String {
    let start = first.byte_start;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    let delta = first.byte_end as isize + next.len() as isize - work.len() as isize;
    let end = delta.max(0).cast_unsigned();
    let end = end.min(next.len());
    let end = if next.is_char_boundary(end) {
        end
    } else {
        work.len()
    };
    clamp_line(next.get(start..end).unwrap_or(""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Env;

    #[allow(clippy::match_wildcard_for_single_variants)] // wildcard is deliberate: forward-compatible with future content variants
    fn text_of(result: &AgentToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|c| match c {
                pa_agent::types::ToolResultContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn env(dir: &Path) -> Env {
        Env::new(dir, false)
    }

    /// Rust + TS fixtures: two functions in Rust, two calls in TS.
    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn alpha() { beta(1); }\nfn beta(x: i32) { x + 1 }\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/app.ts"),
            "function alpha() { beta(1); }\nfunction beta(x: number) { return x + 1 }\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn ast_grep_finds_matches_across_languages() {
        let dir = fixture();
        let result =
            ast_grep(&env(dir.path()), &json!({"pat": "beta($A)", "path": "src"})).unwrap();
        let text = text_of(&result);
        assert!(text.contains("src/app.ts:"), "{text}");
        assert!(text.contains("src/lib.rs:"), "{text}");
        assert!(text.contains("*1|beta(1)"), "{text}");
    }

    #[test]
    fn ast_grep_lang_override_scopes_to_one_language() {
        let dir = fixture();
        let result = ast_grep(
            &env(dir.path()),
            &json!({"pat": "function alpha() { beta($A); }", "path": "src", "lang": "typescript"}),
        )
        .unwrap();
        let text = text_of(&result);
        assert!(text.contains("src/app.ts:"), "{text}");
        assert!(text.contains("*1|function alpha() { beta(1); }"), "{text}");
        assert!(!text.contains("*1|fn alpha"), "{text}");
    }

    #[test]
    fn ast_grep_skip_paginates_matches() {
        let dir = fixture();
        let env = env(dir.path());
        let all = text_of(&ast_grep(&env, &json!({"pat": "$F($A)", "path": "src"})).unwrap());
        let skipped =
            text_of(&ast_grep(&env, &json!({"pat": "$F($A)", "path": "src", "skip": 1})).unwrap());
        assert!(
            all.lines().count() > skipped.lines().count(),
            "{all}\n---\n{skipped}"
        );
    }

    #[test]
    fn ast_grep_no_match_and_empty_pattern() {
        let dir = fixture();
        let env = env(dir.path());
        let miss = text_of(&ast_grep(&env, &json!({"pat": "nonexistent($$$)"})).unwrap());
        assert_eq!(miss, "No matches found");
        assert!(ast_grep(&env, &json!({"pat": "   "})).is_err());
    }

    #[test]
    fn ast_edit_rewrites_rust_and_ts_fixtures() {
        let dir = fixture();
        let env = env(dir.path());
        let result = ast_edit(
            &env,
            &json!({"ops": [{"pat": "beta($A)", "out": "gamma($A)"}], "paths": ["src"]}),
        )
        .unwrap();
        let text = text_of(&result);
        assert!(
            text.contains("Applied 2 replacements in 2 files."),
            "{text}"
        );
        assert!(text.contains("-1:14 beta(1)"), "{text}");
        assert!(text.contains("+1:14 gamma(1)"), "{text}");
        assert!(text.contains("+1:20 gamma(1)"), "{text}");
        let rs = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();
        assert!(rs.contains("gamma(1)"), "{rs}");
        let ts = std::fs::read_to_string(dir.path().join("src/app.ts")).unwrap();
        assert!(ts.contains("gamma(1)"), "{ts}");
    }

    #[test]
    fn ast_edit_delete_op_and_no_match() {
        let dir = fixture();
        let env = env(dir.path());
        let result = ast_edit(
            &env,
            &json!({"ops": [{"pat": "beta($A)", "out": ""}], "paths": ["src/lib.rs"]}),
        )
        .unwrap();
        assert!(text_of(&result).contains("Applied 1 replacement in 1 file."));
        let miss = ast_edit(
            &env,
            &json!({"ops": [{"pat": "nope($$$)", "out": "x"}], "paths": ["src"]}),
        )
        .unwrap();
        assert_eq!(text_of(&miss), "No replacements made");
    }

    #[test]
    fn ast_edit_validates_ops() {
        let dir = fixture();
        let env = env(dir.path());
        let dup = ast_edit(
            &env,
            &json!({"ops": [
                {"pat": "beta($A)", "out": "x"},
                {"pat": "beta($A)", "out": "y"}
            ], "paths": ["src"]}),
        )
        .unwrap_err();
        assert!(
            dup.to_string().contains("Duplicate rewrite pattern"),
            "{dup}"
        );
        let empty = ast_edit(
            &env,
            &json!({"ops": [{"pat": "", "out": "x"}], "paths": ["src"]}),
        )
        .unwrap_err();
        assert!(
            empty.to_string().contains("must be a non-empty pattern"),
            "{empty}"
        );
    }
}
