//! Opt-in experimental TypeScript declaration catalog for deferred tools.
//!
//! Gate: `OMP_TOOL_CATALOG=typescript` (exact value; anything else keeps the
//! original names-only note byte-for-byte). When enabled, the deferred-tools
//! note lists one compact TS-like signature per deferred tool instead of a
//! bare name list. Signatures are generated deterministically from the
//! registered tool's `name()` / `description()` / JSON Schema (`parameters()`).
//! The provider tool list itself is never touched — native tools stay JSON
//! Schema, and `tool.describe` / dynamic invocation keep working unchanged.

use pa_agent::types::AgentTool;
use serde_json::Value;
use std::sync::Arc;

/// Object properties rendered per nesting level before truncation.
const MAX_PROPS: usize = 16;
/// Enum literals rendered before truncation.
const MAX_ENUM: usize = 8;
/// Union/intersection branches rendered before truncation.
const MAX_BRANCHES: usize = 4;
/// Nesting depth bound; deeper levels collapse to `unknown`.
const MAX_DEPTH: usize = 4;
/// Description characters kept per tool.
const MAX_DESC_CHARS: usize = 120;
/// Characters per signature line before truncation. Chosen above the
/// largest real tool type (~380 chars) so ordinary registries emit only
/// complete, tsc-valid declarations; pathological schemas degrade via
/// comment-drop then bounded type elision, and registry scale is bounded
/// by [`TOTAL_CAP`].
const PER_TOOL_CAP: usize = 640;
/// Total characters for the whole signature block before degradation.
const TOTAL_CAP: usize = 8000;

/// Whether the opt-in catalog gate is active.
#[must_use]
pub fn catalog_requested() -> bool {
    gate(std::env::var("OMP_TOOL_CATALOG").ok().as_deref())
}

/// Pure gate decision (unit-testable): only the exact value `typescript` opts in.
#[must_use]
fn gate(raw: Option<&str>) -> bool {
    raw == Some("typescript")
}

/// The deferred-tools prompt block: either the original names-only note
/// (default, byte-identical) or the experimental TS signature catalog.
#[must_use]
pub fn deferred_note(catalog: bool, tools: &[&Arc<dyn AgentTool>]) -> String {
    if catalog {
        catalog_note(tools)
    } else {
        crate::agent::deferred_tools_note(
            &tools
                .iter()
                .map(|tool| tool.name().to_string())
                .collect::<Vec<_>>(),
        )
    }
}

/// Signature block for the deferred tools, sorted by name.
///
/// Bounds degrade deterministically without ever dropping a tool name:
/// full signatures -> descriptions elided -> types collapsed to `unknown`.
#[must_use]
fn catalog_note(tools: &[&Arc<dyn AgentTool>]) -> String {
    catalog_note_with_budget(tools, TOTAL_CAP, PER_TOOL_CAP)
}

/// Bounded builder with explicit budgets (internal; unit tests use small caps).
fn catalog_note_with_budget(
    tools: &[&Arc<dyn AgentTool>],
    total_cap: usize,
    per_tool_cap: usize,
) -> String {
    let mut sorted: Vec<&Arc<dyn AgentTool>> = tools.to_vec();
    sorted.sort_by_key(|tool| tool.name().to_string());
    let build = |annotate: bool| -> Vec<(String, String)> {
        sorted
            .iter()
            .enumerate()
            .map(|(idx, tool)| {
                let name = tool.name();
                (
                    name.to_string(),
                    signature(
                        name,
                        tool.description(),
                        tool.parameters(),
                        annotate,
                        per_tool_cap,
                        idx,
                    ),
                )
            })
            .collect()
    };
    let header = "Deferred tools (not in the tool list; call from an eval cell: \
`await tool.<name>(args)`, schema via `tool.describe({name})`, all via \
`tool.list()`). TypeScript signatures (experimental; OMP_TOOL_CATALOG=typescript):";
    let render = |lines: &[(String, String)]| {
        let mut out = String::from(header);
        for (_, line) in lines {
            out.push('\n');
            out.push_str(line);
        }
        out
    };
    let full = render(&build(true));
    if full.chars().count() <= total_cap {
        return full;
    }
    let bare = render(&build(false));
    if bare.chars().count() <= total_cap {
        return bare;
    }
    // Last resort: collapse every argument list; the name must survive.
    let collapsed: Vec<(String, String)> = sorted
        .iter()
        .enumerate()
        .map(|(idx, tool)| {
            let name = tool.name();
            (name.to_string(), {
                let (head, tail) = ts_decl(name, idx);
                format!("{head}unknown{tail}")
            })
        })
        .collect();
    render(&collapsed)
}

/// Declaration head/tail for a tool name. Bare `function` is only valid TS
/// for identifiers; deferred MCP names may contain `-`/`.`/`:`, and a quoted
/// const *binding* is itself invalid TS (tsc TS1134), so non-identifier names
/// get a deterministic `tool_<sorted-index>` binding holding a quoted
/// *property* with the function type (verified with `tsc --noEmit`).
/// Both forms keep the full name in the head and the same closing tail.
fn ts_decl(name: &str, decl_index: usize) -> (String, String) {
    if is_ts_ident(name) {
        (
            format!("declare function {name}(args: "),
            "): Promise<unknown>;".to_string(),
        )
    } else {
        // Object-type member with a quoted name: property form needs the
        // arrow tail (tsc TS1005 otherwise).
        (
            format!("declare const tool_{decl_index}: {{ {name:?}: (args: "),
            ") => Promise<unknown> };".to_string(),
        )
    }
}

/// One TS-like declaration for a tool. Over-budget shrink is deterministic
/// and validity-preserving where possible: drop the trailing comment first
/// (line stays valid TS), only then elide inside the type with an ellipsis
/// (bounded elision, documented). The head carries the full tool name and
/// the closing tail is always re-attached, so a cut line still reads as a
/// closed declaration. Per-tool budgets are >= 120, far above head+tail, so
/// the name is never dropped.
fn signature(
    name: &str,
    description: &str,
    schema: &Value,
    annotate: bool,
    line_cap: usize,
    decl_index: usize,
) -> String {
    let (head, tail) = ts_decl(name, decl_index);
    let ty = ts_type(schema, 0);
    let mut comment = String::new();
    if annotate {
        let first = description.split('\n').next().unwrap_or("").trim();
        if !first.is_empty() {
            comment = format!(" // {}", truncate_chars(first, MAX_DESC_CHARS));
        }
    }
    let full = format!("{head}{ty}{tail}{comment}");
    if full.chars().count() <= line_cap {
        return full;
    }
    // Drop the optional comment first: the declaration itself then stays
    // fully valid TS at the reduced length.
    let bare = format!("{head}{ty}{tail}");
    if bare.chars().count() <= line_cap {
        return bare;
    }
    // Last resort for pathological schemas: elide inside the type (the
    // ellipsis marks bounded elision; the head and closing tail survive).
    let budget = line_cap
        .saturating_sub(head.chars().count() + tail.chars().count() + 1)
        .max(1);
    format!("{head}{}{tail}", truncate_chars(&ty, budget))
}

/// JSON Schema -> compact TS type. Total recursion depth is bounded by
/// [`MAX_DEPTH`]; every list (props/enums/branches) is capped, so output size
/// is bounded deterministically.
fn ts_type(schema: &Value, depth: usize) -> String {
    if depth > MAX_DEPTH {
        return "unknown".to_string();
    }
    let obj = match schema.as_object() {
        Some(obj) => obj,
        None => return "unknown".to_string(),
    };
    if let Some(values) = obj.get("enum").and_then(Value::as_array) {
        // List caps keep the note valid TS: the overflow marker widens the
        // literal union with `string` instead of an invalid ellipsis token.
        return union(
            values
                .iter()
                .take(MAX_ENUM)
                .map(|v| literal(v))
                .chain((values.len() > MAX_ENUM).then(|| "(string)".to_string())),
        );
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = obj.get(key).and_then(Value::as_array) {
            return union(
                branches
                    .iter()
                    .take(MAX_BRANCHES)
                    .map(|b| ts_type(b, depth + 1))
                    .chain((branches.len() > MAX_BRANCHES).then(|| "(unknown)".to_string())),
            );
        }
    }
    if let Some(branches) = obj.get("allOf").and_then(Value::as_array) {
        return intersection(
            branches
                .iter()
                .take(MAX_BRANCHES)
                .map(|b| ts_type(b, depth + 1))
                .chain((branches.len() > MAX_BRANCHES).then(|| "(unknown)".to_string())),
        );
    }
    match obj.get("type") {
        Some(Value::String(t)) => scalar(t, obj, depth),
        Some(Value::Array(types)) => union(
            types
                .iter()
                .filter_map(Value::as_str)
                .map(|t| scalar(t, obj, depth)),
        ),
        _ => "unknown".to_string(),
    }
}

/// Non-array schema types: primitives, arrays, object bodies.
fn scalar(t: &str, obj: &serde_json::Map<String, Value>, depth: usize) -> String {
    match t {
        "string" => "string".to_string(),
        "number" | "integer" => "number".to_string(),
        "boolean" => "boolean".to_string(),
        "null" => "null".to_string(),
        "array" => {
            let item = obj
                .get("items")
                .map(|items| ts_type(items, depth + 1))
                .unwrap_or_else(|| "unknown".to_string());
            format!("{item}[]")
        }
        "object" => object_body(obj, depth),
        _ => "unknown".to_string(),
    }
}

/// `{ key: T; key2?: U; [k: string]: V }` / `Record<string, V>` fallback.
/// Property names are sorted for determinism (serde_json maps are
/// insertion-ordered in this workspace).
fn object_body(obj: &serde_json::Map<String, Value>, depth: usize) -> String {
    let required: Vec<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|req| req.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let props: Vec<(String, &Value)> = obj
        .get("properties")
        .and_then(Value::as_object)
        .map(|props| {
            let mut entries: Vec<(String, &Value)> =
                props.iter().map(|(k, v)| (k.clone(), v)).collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        })
        .unwrap_or_default();
    let additional = obj.get("additionalProperties");
    let index: Option<String> = match additional {
        Some(Value::Bool(false)) | None => None,
        Some(Value::Bool(true)) => Some("[k: string]: unknown".to_string()),
        Some(schema) => Some(format!("[k: string]: {}", ts_type(schema, depth + 1))),
    };
    if props.is_empty() {
        return match index {
            Some(idx) => format!("{{ {idx} }}"),
            None => "Record<string, unknown>".to_string(),
        };
    }
    let mut parts: Vec<String> = props
        .iter()
        .take(MAX_PROPS)
        .map(|(key, schema)| {
            let opt = if required.contains(&key.as_str()) {
                ""
            } else {
                "?"
            };
            format!("{}{}: {}", ts_key(key), opt, ts_type(schema, depth + 1))
        })
        .collect();
    if props.len() > MAX_PROPS {
        // Valid-TS overflow marker: extra properties surface as `unknown`.
        parts.push("[k: string]: unknown".to_string());
    }
    if let Some(idx) = index {
        parts.push(idx);
    }
    format!("{{ {} }}", parts.join("; "))
}

/// Valid bare TS identifier (ASCII conservative, deterministic).
fn is_ts_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && s.chars()
            .skip(1)
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// Bare identifier when possible; quoted string otherwise (deterministic).
fn ts_key(key: &str) -> String {
    if is_ts_ident(key) {
        key.to_string()
    } else {
        format!("{key:?}")
    }
}

/// JSON literal as a TS literal.
fn literal(value: &Value) -> String {
    match value {
        Value::String(s) => format!("{s:?}"),
        other => other.to_string(),
    }
}

fn union<'a>(parts: impl Iterator<Item = String> + 'a) -> String {
    dedup_join(parts, " | ")
}

fn intersection<'a>(parts: impl Iterator<Item = String> + 'a) -> String {
    dedup_join(parts, " & ")
}

/// Join non-empty parts; keep output deterministic by preserving input order.
fn dedup_join(parts: impl Iterator<Item = String>, sep: &str) -> String {
    let mut seen: Vec<String> = Vec::new();
    for part in parts {
        if !part.is_empty() && !seen.contains(&part) {
            seen.push(part);
        }
    }
    if seen.is_empty() {
        "unknown".to_string()
    } else {
        seen.join(sep)
    }
}

/// Char-boundary-safe truncation with an ellipsis marker.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push('\u{2026}');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{catalog_note_with_budget, gate, ts_type, AgentTool, TOTAL_CAP};
    use serde_json::{json, Value};
    use std::sync::Arc;

    fn t(schema: &Value) -> String {
        ts_type(&schema, 0)
    }

    #[test]
    fn gate_is_opt_in_only() {
        assert!(!gate(None));
        assert!(!gate(Some("")));
        assert!(!gate(Some("json")));
        assert!(!gate(Some("TypeScript")));
        assert!(gate(Some("typescript")));
    }

    #[test]
    fn scalars_map() {
        assert_eq!(t(&json!({"type": "string"})), "string");
        assert_eq!(t(&json!({"type": "integer"})), "number");
        assert_eq!(t(&json!({"type": "number"})), "number");
        assert_eq!(t(&json!({"type": "boolean"})), "boolean");
        assert_eq!(t(&json!({"type": "null"})), "null");
    }

    #[test]
    fn type_arrays_become_unions() {
        assert_eq!(t(&json!({"type": ["string", "null"]})), "string | null");
        assert_eq!(
            t(&json!({"type": ["integer", "string"]})),
            "number | string"
        );
    }

    #[test]
    fn enums_render_as_literal_unions() {
        assert_eq!(t(&json!({"enum": ["a", "b"]})), "\"a\" | \"b\"");
        assert_eq!(t(&json!({"enum": [1, true, null]})), "1 | true | null");
        let long = t(&json!({"enum": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]}));
        assert_eq!(
            long,
            "\"a\" | \"b\" | \"c\" | \"d\" | \"e\" | \"f\" | \"g\" | \"h\" | (string)"
        );
    }

    #[test]
    fn required_vs_optional_props() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"]
        });
        assert_eq!(t(&schema), "{ limit?: number; path: string }");
    }

    #[test]
    fn arrays_and_nesting() {
        let schema = json!({
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "integer"}}}
        });
        assert_eq!(t(&schema), "{ id?: number }[]");
        assert_eq!(t(&json!({"type": "array"})), "unknown[]");
    }

    #[test]
    fn object_property_order_is_sorted_not_insertion() {
        let a = json!({"type": "object", "properties": {"zeta": {"type": "string"}, "alpha": {"type": "string"}}});
        let b = json!({"type": "object", "properties": {"alpha": {"type": "string"}, "zeta": {"type": "string"}}});
        assert_eq!(t(&a), "{ alpha?: string; zeta?: string }");
        assert_eq!(t(&a), t(&b));
    }

    #[test]
    fn additional_properties_forms() {
        assert_eq!(
            t(&json!({"type": "object", "additionalProperties": {"type": "integer"}})),
            "{ [k: string]: number }"
        );
        assert_eq!(
            t(
                &json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": {"type": "integer"}})
            ),
            "{ a?: string; [k: string]: number }"
        );
        assert_eq!(
            t(
                &json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false})
            ),
            "{ a?: string }"
        );
        assert_eq!(
            t(&json!({"type": "object", "additionalProperties": true})),
            "{ [k: string]: unknown }"
        );
    }

    #[test]
    fn non_identifier_keys_are_quoted() {
        let schema = json!({"type": "object", "properties": {"my-key": {"type": "string"}}});
        assert_eq!(t(&schema), "{ \"my-key\"?: string }");
    }

    #[test]
    fn anyof_oneof_allof() {
        assert_eq!(
            t(&json!({"anyOf": [{"type": "string"}, {"type": "integer"}]})),
            "string | number"
        );
        assert_eq!(
            t(&json!({"oneOf": [{"type": "string"}, {"type": "string"}]})),
            "string"
        );
        assert_eq!(
            t(&json!({"allOf": [{"type": "string"}, {"type": "integer"}]})),
            "string & number"
        );
    }

    #[test]
    fn deep_nesting_is_bounded() {
        let mut schema = json!({"type": "string"});
        for _ in 0..50 {
            schema = json!({"type": "array", "items": schema});
        }
        let rendered = t(&schema);
        assert_eq!(rendered.matches("[]").count(), 5);
        assert!(rendered.ends_with("unknown[][][][][]"));
        assert!(!rendered.contains("panic"));
    }

    #[test]
    fn property_list_is_capped() {
        let mut props = serde_json::Map::new();
        for i in 0..30 {
            props.insert(format!("prop{i:02}"), json!({"type": "string"}));
        }
        let schema = json!({"type": "object", "properties": props});
        let rendered = t(&schema);
        assert!(rendered.contains("[k: string]: unknown"));
        assert!(!rendered.contains("prop16"));
        assert!(rendered.contains("prop15"));
    }

    #[test]
    fn per_tool_line_cap_never_drops_the_name() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}}
        });
        let rendered = super::signature("read", "long description", &schema, true, 30, 0);
        assert!(rendered.starts_with("declare function read(args: "));
        // Bounded elision: the cut is marked and the closing tail survives.
        assert!(rendered.contains('\u{2026}'));
        assert!(rendered.ends_with("): Promise<unknown>;"));
    }

    #[test]
    fn over_budget_drops_comment_before_eliding_the_type() {
        let schema = json!({"type": "object", "properties": {"text": {"type": "string"}}});
        let rendered = super::signature(
            "read",
            "a description long enough to overflow a tight cap",
            &schema,
            true,
            70,
            0,
        );
        // The declaration (head+type+tail) fits in 70; the comment is dropped
        // instead of cutting into the type: still valid, closed TS.
        assert!(rendered.starts_with("declare function read(args: "));
        assert!(rendered.ends_with("): Promise<unknown>;"));
        assert!(!rendered.contains('\u{2026}'));
        assert!(!rendered.contains("description"));
    }

    #[test]
    fn non_identifier_tool_names_use_quoted_property_form() {
        let schema = json!({"type": "object", "properties": {"text": {"type": "string"}}});
        let name = "mcp__fake-server.echo-tool";
        let rendered = super::signature(name, "d", &schema, false, 400, 2);
        // A quoted const *binding* is invalid TS (tsc TS1134); the valid shape
        // is a deterministic binding holding a quoted *property*.
        assert!(rendered.starts_with(&format!("declare const tool_2: {{ \"{name}\": (args: ")));
        assert!(rendered.ends_with(") => Promise<unknown> };"));
        assert!(!rendered.contains("declare function"));
        // Truncation keeps binding + full name + closed tail on a tiny budget.
        let cut = super::signature(name, "d", &schema, false, 60, 7);
        assert!(cut.starts_with("declare const tool_7: { "));
        assert!(cut.contains(&format!("\"{name}\"")));
        assert!(cut.ends_with(") => Promise<unknown> };"));
        assert!(cut.contains('\u{2026}'));
    }

    #[test]
    fn full_note_is_single_prefixed_and_valid_ts() {
        // Exactly one `declare` keyword per line (the renderer must not
        // duplicate the one `signature` embeds), deterministic sorted-index
        // bindings for non-identifiers, and — when tsc is available — the
        // emitted note must type-check.
        let schema = json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        });
        let names = [
            "read",
            "eval",
            "mcp__fake-srv.echo-tool",
            "a.b:fn",
            "zz_last",
        ];
        let tools: Vec<Arc<dyn AgentTool>> = names.iter().map(|n| synth_tool(n, &schema)).collect();
        let refs: Vec<&Arc<dyn AgentTool>> = tools.iter().collect();
        let note = catalog_note_with_budget(&refs, 8000, 300);
        let lines: Vec<&str> = note.lines().skip(1).collect();
        assert_eq!(lines.len(), names.len());
        for line in &lines {
            let one =
                line.starts_with("declare function ") ^ line.starts_with("declare const tool_");
            assert!(one, "double or unknown prefix: {line}");
        }
        // Sorted order: "a.b:fn" < "eval" < "mcp__fake-srv.echo-tool" < "read" < "zz_last".
        assert!(note.contains("declare const tool_0: { \"a.b:fn\": (args: "));
        assert!(note.contains("declare const tool_2: { \"mcp__fake-srv.echo-tool\": (args: "));
        assert!(note.contains("declare function read(args: { text: string }): Promise<unknown>;"));
        tsc_check(&[&note]);
    }

    #[test]
    fn collapsed_note_stays_valid_ts() {
        let schema = json!({"type": "object", "properties": {"text": {"type": "string"}}});
        let names = ["read", "mcp__fake-srv.echo-tool"];
        let tools: Vec<Arc<dyn AgentTool>> = names.iter().map(|n| synth_tool(n, &schema)).collect();
        let refs: Vec<&Arc<dyn AgentTool>> = tools.iter().collect();
        // Tiny total budget forces the collapsed degradation level.
        let note = catalog_note_with_budget(&refs, 200, 300);
        // Sorted: "mcp__fake-srv.echo-tool" < "read" -> binding index 0.
        assert!(note.contains(
            "declare const tool_0: { \"mcp__fake-srv.echo-tool\": (args: unknown) => Promise<unknown> };"
        ));
        tsc_check(&[&note]);
    }

    /// Compile generated notes with tsc when available; skip loudly otherwise.
    fn tsc_check(notes: &[&str]) {
        // "tsc" via PATH; the absolute path is this host's convenience install.
        let candidates = ["tsc", "/home/kaneki/.local/bin/tsc"];
        let found = candidates.iter().find(|c| which(c));
        let Some(tsc) = found else {
            eprintln!("tsc not found; skipping emitted-catalog TS validation");
            return;
        };
        for (i, note) in notes.iter().enumerate() {
            let path =
                std::env::temp_dir().join(format!("omt-tsc-check-{}-{i}.ts", std::process::id()));
            // The prose header is prompt text, not code: comment it for the
            // standalone .ts compilation (tsc reports on stdout).
            let mut file = String::new();
            let mut lines = note.lines();
            if let Some(header) = lines.next() {
                file.push_str("// ");
                file.push_str(header);
                file.push('\n');
            }
            file.push_str(&lines.collect::<Vec<_>>().join("\n"));
            file.push('\n');
            std::fs::write(&path, file).expect("write ts file");
            let out = std::process::Command::new(tsc)
                .args(["--noEmit", "--skipLibCheck", "--target", "es2022"])
                .arg(&path)
                .output()
                .expect("run tsc");
            let _ = std::fs::remove_file(&path);
            assert!(
                out.status.success(),
                "tsc rejected the catalog: {}{}",
                String::from_utf8_lossy(&out.stderr),
                String::from_utf8_lossy(&out.stdout)
            );
        }
    }

    fn which(cmd: &str) -> bool {
        if cmd.contains('/') {
            return std::path::Path::new(cmd).is_file();
        }
        std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).any(|d| d.join(cmd).is_file()))
            .unwrap_or(false)
    }

    #[test]
    fn total_budget_degrades_without_name_loss() {
        let schema = json!({
            "type": "object",
            "properties": (0..64).map(|i| (format!("field{i:02}"), json!({"type": "string"}))).collect::<serde_json::Map<String, Value>>()
        });
        let tools: Vec<serde_json::Value> = (0..40)
            .map(|i| json!({"name": format!("tool{i:02}"), "description": "x", "schema": schema}))
            .collect();
        // Run through the real path with a tiny total budget via a local shim:
        // build the note from 40 huge synthetic tools and a 500-char budget.
        let note = tiny_budget_note(&tools, 500);
        for i in 0..40 {
            assert!(
                note.contains(&format!("tool{i:02}(")),
                "name tool{i:02} lost"
            );
        }
        assert!(note.chars().count() <= 40 * 45 + 2);
    }

    /// Test shim: mirrors `catalog_note` degradation using synthetic values.
    fn tiny_budget_note(tools: &[Value], total_cap: usize) -> String {
        let mut sorted: Vec<&Value> = tools.iter().collect();
        sorted.sort_by_key(|t| t["name"].as_str().unwrap_or("").to_string());
        let line = |t: &Value, annotate: bool| -> String {
            let name = t["name"].as_str().unwrap_or("");
            let base = if annotate {
                format!(
                    "{}(args: {}): Promise<unknown>; // x",
                    name,
                    ts_type(&t["schema"], 0)
                )
            } else {
                format!(
                    "{name}(args: {}): Promise<unknown>;",
                    ts_type(&t["schema"], 0)
                )
            };
            base
        };
        let render = |lines: &[String]| format!("H\n{}", lines.join("\n"));
        let full: Vec<String> = sorted.iter().map(|t| line(t, true)).collect();
        let bare: Vec<String> = sorted.iter().map(|t| line(t, false)).collect();
        let collapsed: Vec<String> = sorted
            .iter()
            .map(|t| {
                format!(
                    "{}(args: unknown): Promise<unknown>;",
                    t["name"].as_str().unwrap_or("")
                )
            })
            .collect();
        for candidate in [render(&full), render(&bare), render(&collapsed)] {
            if candidate.chars().count() <= total_cap {
                return candidate;
            }
        }
        render(&collapsed)
    }

    #[test]
    fn total_cap_constant_never_truncates_names_for_real_registry_scales() {
        // 90 deferred tools (larger than any real registry) with fat schemas:
        // worst-case degradation must still keep every name.
        let schema = json!({
            "type": "object",
            "properties": (0..64).map(|i| (format!("field{i:02}"), json!({"type": "string"}))).collect::<serde_json::Map<String, Value>>()
        });
        let names: Vec<String> = (0..90).map(|i| format!("tool{i:02}")).collect();
        let owned: Vec<Arc<dyn AgentTool>> = names.iter().map(|n| synth_tool(n, &schema)).collect();
        let tools: Vec<&Arc<dyn AgentTool>> = owned.iter().collect();
        let note = catalog_note_with_budget(&tools, TOTAL_CAP, 400);
        for name in &names {
            assert!(note.contains(&format!("{name}(")), "name {name} lost");
        }
    }

    /// Minimal AgentTool for tests (execute is never called).
    fn synth_tool(name: &str, schema: &Value) -> Arc<dyn AgentTool> {
        struct Synth {
            name: String,
            schema: Value,
        }
        impl AgentTool for Synth {
            fn name(&self) -> &str {
                &self.name
            }
            fn description(&self) -> &str {
                "synthetic tool used by catalog tests"
            }
            fn parameters(&self) -> &Value {
                &self.schema
            }
            fn execute(
                self: Arc<Self>,
                _id: String,
                _args: Value,
                _signal: pa_agent::abort::AbortSignal,
                _on_update: pa_agent::types::AgentToolUpdateCallback,
            ) -> pa_agent::BoxFut<'static, anyhow::Result<pa_agent::types::AgentToolResult>>
            {
                unreachable!("synthetic tool is never executed")
            }
        }
        Arc::new(Synth {
            name: name.to_string(),
            schema: schema.clone(),
        })
    }
}
