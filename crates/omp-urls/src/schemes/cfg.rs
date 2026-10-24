//! `cfg://` protocol handler, read-only view (TS `internal-urls/cfg-protocol.ts`,
//! URL grammar per `pi-tui/tools/cfg-url.ts`).
//!
//! Read forms (`/` and `.` both separate segments):
//! - `cfg://` — every setting as a YAML-ish tree
//! - `cfg://<namespace>` — one namespace's subtree, e.g. `cfg://advisor`
//! - `cfg://<setting>` — one setting: value, type, default, source
//!
//! TS also supports approval-gated writes; those are not ported, so the
//! scheme stays read-only here (the trait's default `write` refuses).

use crate::{InternalResource, InternalUrl, UrlCompletion, UrlCtx, UrlProvider};

/// Canonical prefix for settings URLs (TS `CFG_URL_PREFIX`).
pub const CFG_URL_PREFIX: &str = "cfg://";

/// Trailing write segment that persists a change (TS `CFG_SAVE_SEGMENT`); parsed, never written here.
pub const CFG_SAVE_SEGMENT: &str = "save";

/// A setting addressed by a `cfg://` URL (TS `CfgTarget`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgTarget {
    /// Setting path segments; empty for the root. `/` and `.` both separate segments.
    pub segments: Vec<String>,
    /// Whether the URL ends in the `/save` suffix (removed from `segments`).
    pub save: bool,
}

/// Parse a `cfg://` URL (TS `parseCfgUrl`): `cfg://advisor/enabled`, `cfg://advisor.enabled/save`.
/// Returns `None` for other schemes; query and fragment are ignored.
pub fn parse_cfg_url(input: &str) -> Option<CfgTarget> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with(CFG_URL_PREFIX) {
        return None;
    }
    let mut segments: Vec<String> = trimmed[CFG_URL_PREFIX.len()..]
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .split(['/', '.'])
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let save = segments.len() > 1 && segments.last().is_some_and(|s| s == CFG_SAVE_SEGMENT);
    if save {
        segments.pop();
    }
    Some(CfgTarget { segments, save })
}

/// One row of the rendered config view.
#[derive(Debug, Clone)]
pub struct CfgSetting {
    /// Dotted id, e.g. `advisor.enabled`.
    pub id: String,
    /// TS type name (`boolean`, `number`, `string`, `enum`, `array`, `record`).
    pub ty: &'static str,
    /// Closed value set for enums.
    pub enum_values: Option<Vec<String>>,
    /// Effective value.
    pub value: serde_json::Value,
    /// Default value (`Null` when TS has none).
    pub default: serde_json::Value,
    /// Which layer owns the effective value.
    pub provenance: &'static str,
    /// TS `credential: true` — the effective value renders as `<redacted>`.
    pub credential: bool,
}

/// Collect the config view from a loaded settings scope (TS `allSettings()` +
/// per-setting `get`/`default`/`provenance`).
pub fn collect_settings(settings: &omp_settings::Settings) -> Vec<CfgSetting> {
    let mut rows: Vec<CfgSetting> = omp_settings::SETTINGS
        .iter()
        .map(|def| CfgSetting {
            id: def.id.to_string(),
            ty: def.ty.ts_name(),
            enum_values: match def.ty {
                omp_settings::SettingType::Enum(values) => {
                    Some(values.iter().map(|s| s.to_string()).collect())
                }
                _ => None,
            },
            value: settings.get(def.id),
            default: def.default_yaml.map_or_else(
                || serde_json::Value::Null,
                |yaml| {
                    serde_yaml_ng::from_str::<serde_json::Value>(yaml)
                        .unwrap_or(serde_json::Value::Null)
                },
            ),
            provenance: provenance_label(settings.provenance(def.id)),
            credential: def.credential,
        })
        .collect();
    // Ordered by id segment by segment so each namespace's members stay contiguous (TS sort).
    rows.sort_by(|a, b| {
        let left: Vec<&str> = a.id.split('.').collect();
        let right: Vec<&str> = b.id.split('.').collect();
        for i in 0..left.len().min(right.len()) {
            let order = left[i].to_lowercase().cmp(&right[i].to_lowercase());
            if order != std::cmp::Ordering::Equal {
                return order;
            }
        }
        left.len().cmp(&right.len())
    });
    rows
}

fn provenance_label(prov: Option<omp_settings::Provenance>) -> &'static str {
    match prov {
        Some(omp_settings::Provenance::Env) => "environment variable",
        Some(omp_settings::Provenance::Project) => "project config",
        Some(omp_settings::Provenance::Global) => "global config",
        _ => "default",
    }
}

/// TS `REDACTED` (cfg-protocol.ts).
const REDACTED: &str = "<redacted>";

/// TS `formatValue`: null → `null`, bool/number bare, everything else JSON;
/// credential settings redact any non-empty value.
fn format_value(setting: &CfgSetting, value: &serde_json::Value) -> String {
    if setting.credential && value.as_str().is_some_and(|s| !s.is_empty()) {
        return REDACTED.to_string();
    }
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

/// Canonicalize URL segments against the schema, case-insensitively (TS `resolveSegments`).
/// `leaf` is set when the path names a setting; `members` lists every setting beneath it.
#[derive(Debug)]
pub struct ResolvedSegments {
    pub path: String,
    pub leaf: Option<usize>,
    pub members: Vec<usize>,
}

/// Resolve `segments` against `rows`, returning indexes into `rows` (TS `resolveSegments`).
pub fn resolve_segments(
    rows: &[CfgSetting],
    segments: &[String],
) -> Result<ResolvedSegments, String> {
    if segments.is_empty() {
        return Ok(ResolvedSegments {
            path: String::new(),
            leaf: None,
            members: (0..rows.len()).collect(),
        });
    }
    let lower = segments.join(".").to_lowercase();
    let leaf = rows.iter().position(|r| r.id.to_lowercase() == lower);
    let members: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.id.to_lowercase().starts_with(&format!("{lower}.")))
        .map(|(i, _)| i)
        .collect();
    let path = leaf.map_or_else(
        || {
            members
                .first()
                .map(|&i| rows[i].id[..lower.len()].to_string())
        },
        |i| Some(rows[i].id.clone()),
    );
    let Some(path) = path else {
        let similar: Vec<String> = rows
            .iter()
            .filter(|r| {
                let id = r.id.to_lowercase();
                id.contains(&lower) || lower.contains(id.as_str())
            })
            .take(8)
            .map(|r| r.id.clone())
            .collect();
        let hint = if similar.is_empty() {
            String::new()
        } else {
            format!("\nSimilar: {}", similar.join(", "))
        };
        return Err(format!(
            "Unknown setting: {}{hint}\nRead {CFG_URL_PREFIX} for the full tree.",
            segments.join(".")
        ));
    };
    Ok(ResolvedSegments {
        path,
        leaf,
        members,
    })
}

/// Comment on a tree line: enum choices, non-default default, description (TS `treeComment`).
fn tree_comment(setting: &CfgSetting) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(choices) = &setting.enum_values {
        parts.push(choices.join("|"));
    }
    if setting.value != setting.default {
        parts.push(format!(
            "default {}",
            format_value(setting, &setting.default)
        ));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("  # {}", parts.join(" · "))
    }
}

/// YAML-ish tree of `members`, rooted below `prefix` (TS `renderTree`).
pub fn render_tree(rows: &[CfgSetting], members: &[usize], prefix: &str) -> (String, usize) {
    let mut lines: Vec<String> = Vec::new();
    let mut modified = 0;
    let mut opened: Vec<&str> = Vec::new();
    let strip = if prefix.is_empty() {
        0
    } else {
        prefix.len() + 1
    };
    for &mi in members {
        let setting = &rows[mi];
        let segments: Vec<&str> = setting.id[strip..].split('.').collect();
        let mut shared = 0;
        while shared < opened.len()
            && shared + 1 < segments.len()
            && opened[shared] == segments[shared]
        {
            shared += 1;
        }
        opened.truncate(shared);
        for (depth, segment) in segments
            .iter()
            .enumerate()
            .take(segments.len() - 1)
            .skip(shared)
        {
            lines.push(format!("{}{}:", "  ".repeat(depth), segment));
            opened.push(segment);
        }
        if setting.value != setting.default {
            modified += 1;
        }
        let indent = "  ".repeat(segments.len() - 1);
        lines.push(format!(
            "{}{}: {}{}",
            indent,
            segments.last().unwrap_or(&""),
            format_value(setting, &setting.value),
            tree_comment(setting)
        ));
    }
    (lines.join("\n"), modified)
}

/// One setting in full: value, type, default, source (TS `renderLeaf`).
pub fn render_leaf(setting: &CfgSetting) -> String {
    let mut lines = vec![
        format!("{}: {}", setting.id, format_value(setting, &setting.value)),
        format!("type: {}", setting.ty),
        format!("default: {}", format_value(setting, &setting.default)),
        format!("source: {}", setting.provenance),
    ];
    if let Some(choices) = &setting.enum_values {
        lines.push(format!("values: [{}]", choices.join(", ")));
    }
    lines.join("\n")
}

/// Resolve a `cfg://` URL against a loaded settings scope (TS `CfgProtocolHandler.resolve`).
pub fn resolve(
    settings: &omp_settings::Settings,
    url: &InternalUrl,
) -> Result<InternalResource, String> {
    let target = parse_cfg_url(&url.raw_href).unwrap_or(CfgTarget {
        segments: Vec::new(),
        save: false,
    });
    let rows = collect_settings(settings);
    let ResolvedSegments {
        path,
        leaf,
        members,
    } = resolve_segments(&rows, &target.segments)?;
    let mut sections: Vec<String> = Vec::new();
    if let Some(i) = leaf {
        sections.push(render_leaf(&rows[i]));
    }
    if !members.is_empty() {
        let (tree, _) = render_tree(&rows, &members, &path);
        sections.push(tree);
    }
    Ok(InternalResource {
        content: sections.join("\n\n"),
        is_directory: false,
        immutable: Some(true),
        path: None,
    })
}

/// Read-only `cfg://` provider over the layered settings view.
#[derive(Default)]
pub struct CfgProvider;

impl UrlProvider for CfgProvider {
    fn scheme(&self) -> &'static str {
        "cfg"
    }

    fn spec(&self) -> &crate::SchemeSpec {
        static SPEC: crate::SchemeSpec = crate::SchemeSpec {
            backing: crate::SchemeBacking::Virtual,
            selectors: crate::SchemeSelectors::Lines,
            immutable: true,
            single_slash_alias: false,
            linkable: false,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn read(&self, url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        let cwd = ctx.cwd.clone().unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        });
        let settings = omp_settings::Settings::load(&cwd);
        resolve(&settings, url).map_err(anyhow::Error::msg)
    }

    fn complete(&self, _query: &str, _ctx: &UrlCtx) -> Vec<UrlCompletion> {
        omp_settings::SETTINGS
            .iter()
            .map(|def| UrlCompletion {
                label: def.id.replace('.', "/"),
                detail: None,
            })
            .collect()
    }

    fn prompt_doc(&self) -> Option<String> {
        Some("`cfg://`: agent settings view.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setting(
        id: &str,
        ty: &'static str,
        value: serde_json::Value,
        default: serde_json::Value,
    ) -> CfgSetting {
        CfgSetting {
            id: id.to_string(),
            ty,
            enum_values: None,
            value,
            default,
            provenance: "default",
            credential: false,
        }
    }

    /// Review #1: credential settings (TS `credential: true`) must redact in
    /// leaf and tree renders, any non-empty value, while empty strings stay
    /// visible — matching TS `formatValue`'s `isCredential` guard.
    #[test]
    fn credential_values_are_redacted_like_ts() {
        let secret = "super-secret-broker-token";
        let mut cred = setting(
            "auth.broker.token",
            "string",
            serde_json::json!(secret),
            serde_json::Value::Null,
        );
        cred.credential = true;
        let leaf = render_leaf(&cred);
        assert!(leaf.contains("<redacted>"), "{leaf}");
        assert!(!leaf.contains(secret), "cleartext leak: {leaf}");
        // Tree render redacts too.
        let (tree, _) = render_tree(&[cred.clone()], &[0], "");
        assert!(tree.contains("<redacted>"), "{tree}");
        assert!(!tree.contains(secret), "cleartext leak: {tree}");
        // Empty value is NOT redacted (TS `value !== ""`).
        let mut empty = cred.clone();
        empty.value = serde_json::json!("");
        let leaf = render_leaf(&empty);
        assert!(leaf.starts_with("auth.broker.token: \"\""), "{leaf}");
        // Non-credential settings are untouched.
        let plain = setting(
            "model.name",
            "string",
            serde_json::json!("kimi"),
            serde_json::json!(""),
        );
        assert!(render_leaf(&plain).contains("kimi"));
    }

    /// The schema flags exactly the TS credential settings.
    #[test]
    fn credential_flags_match_ts_registry() {
        for id in [
            "auth.broker.token",
            "mnemopi.embeddingApiKey",
            "mnemopi.llmApiKey",
            "images.urls.credentials",
            "hindsight.apiToken",
            "dev.autoqaPush.token",
            "searxng.token",
            "searxng.basicPassword",
        ] {
            assert!(
                omp_settings::lookup(id).is_some_and(|d| d.credential),
                "{id}"
            );
        }
        assert!(!omp_settings::lookup("searxng.endpoint").unwrap().credential);
        assert!(!omp_settings::lookup("modelRoles").unwrap().credential);
    }

    fn rows() -> Vec<CfgSetting> {
        vec![
            setting(
                "advisor.enabled",
                "boolean",
                serde_json::Value::Bool(false),
                serde_json::Value::Bool(false),
            ),
            setting(
                "advisor.watchdog",
                "boolean",
                serde_json::Value::Bool(true),
                serde_json::Value::Bool(true),
            ),
            setting(
                "model.name",
                "string",
                serde_json::json!("kimi"),
                serde_json::json!(""),
            ),
        ]
    }

    #[test]
    fn parse_matches_ts() {
        assert_eq!(
            parse_cfg_url("cfg://advisor/enabled"),
            Some(CfgTarget {
                segments: vec!["advisor".into(), "enabled".into()],
                save: false
            })
        );
        assert_eq!(
            parse_cfg_url("cfg://advisor.enabled/save"),
            Some(CfgTarget {
                segments: vec!["advisor".into(), "enabled".into()],
                save: true
            })
        );
        assert_eq!(
            parse_cfg_url("cfg://"),
            Some(CfgTarget {
                segments: vec![],
                save: false
            })
        );
        // Single `save` alone is not stripped (TS: length > 1 guard).
        assert_eq!(
            parse_cfg_url("cfg://save"),
            Some(CfgTarget {
                segments: vec!["save".into()],
                save: false
            })
        );
        assert_eq!(parse_cfg_url("local://x"), None);
        assert_eq!(
            parse_cfg_url("cfg://a.b/c?query#frag"),
            Some(CfgTarget {
                segments: vec!["a".into(), "b".into(), "c".into()],
                save: false
            })
        );
    }

    #[test]
    fn segments_resolve_case_insensitively() {
        let rows = rows();
        let r = resolve_segments(&rows, &["ADVISOR".to_string()]).unwrap();
        assert_eq!(r.path, "advisor");
        assert_eq!(r.leaf, None);
        assert_eq!(r.members, vec![0, 1]);
        let r = resolve_segments(&rows, &["advisor".to_string(), "ENABLED".to_string()]).unwrap();
        assert_eq!(r.path, "advisor.enabled");
        assert_eq!(r.leaf, Some(0));
        assert!(r.members.is_empty());
    }

    #[test]
    fn unknown_setting_errors() {
        let rows = rows();
        let err = resolve_segments(&rows, &["zzz".to_string()]).unwrap_err();
        assert_eq!(err, "Unknown setting: zzz\nRead cfg:// for the full tree.");
        let err = resolve_segments(&rows, &["dviso".to_string()]).unwrap_err();
        assert!(err.starts_with("Unknown setting: dviso\nSimilar: advisor.enabled"));
    }

    #[test]
    fn leaf_rendering_matches_ts() {
        let rows = rows();
        let text = render_leaf(&rows[0]);
        assert_eq!(
            text,
            "advisor.enabled: false\ntype: boolean\ndefault: false\nsource: default"
        );
    }

    #[test]
    fn tree_rendering_matches_ts() {
        let mut rows = rows();
        rows[0].value = serde_json::Value::Bool(true); // modified → `default false` comment
        let members = vec![0, 1];
        let (text, modified) = render_tree(&rows, &members, "advisor");
        assert_eq!(text, "enabled: true  # default false\nwatchdog: true");
        assert_eq!(modified, 1);
        // Root scope opens namespaces.
        let members: Vec<usize> = (0..rows.len()).collect();
        let (text, _) = render_tree(&rows, &members, "");
        assert_eq!(
            text,
            "advisor:\n  enabled: true  # default false\n  watchdog: true\nmodel:\n  name: \"kimi\"  # default \"\""
        );
    }
}
