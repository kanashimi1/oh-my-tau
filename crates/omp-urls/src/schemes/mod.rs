//! Shared helpers for the read-only URL scheme providers.
//!
//! Mirrors the TS `internal-urls` handlers in
//! `pi-coding-agent/src/internal-urls/{skill,rule,omp,cfg,mcp,xd}-protocol.ts`.

pub mod agent;
pub mod artifact;
pub mod cfg;
pub mod core;
pub mod history;
pub mod local;
pub mod mcp;
pub mod omp;
pub mod rule;
pub mod skill;
pub mod xd;

use crate::Registry;

/// Register every read-only scheme provider owned by this module.
pub fn register_all(reg: &mut Registry) -> anyhow::Result<()> {
    reg.register(skill::SkillProvider)?;
    reg.register(rule::RuleProvider)?;
    reg.register(omp::OmpProvider)?;
    reg.register(cfg::CfgProvider)?;
    reg.register(mcp::McpProvider)?;
    reg.register(xd::XdProvider)?;
    core::register(reg)?;
    Ok(())
}

/// A skill discovered from a skills directory (TS: `extensibility/skills.ts`).
#[derive(Debug, Clone)]
pub struct SkillEntry {
    /// Skill name (frontmatter `name` or directory name).
    pub name: String,
    /// Absolute path of the skill's SKILL.md.
    pub file_path: String,
    /// Absolute path of the skill's base directory.
    pub base_dir: String,
    pub description: String,
    /// Agent-plugin skills (TS `skill.containRoot`): the package root every
    /// resolved target, including the bare instruction file and base
    /// directory, must canonically stay within. `None` = plain skill:
    /// lexical containment only, like TS.
    pub contain_root: Option<String>,
}

/// A rule from the project rulebook (TS: `capability/rule.ts`).
#[derive(Debug, Clone)]
pub struct RuleEntry {
    pub name: String,
    /// Absolute path of the rule file on disk (may be empty for virtual rules).
    pub path: String,
    /// Rule content with frontmatter stripped.
    pub content: String,
    pub description: String,
}

/// Raw resource payload returned by a scheme resolver (TS `InternalResource`).
#[derive(Debug, Clone)]
pub struct Resource {
    pub url: String,
    pub content: String,
    /// `text/markdown`, `application/json`, or `text/plain`.
    pub content_type: &'static str,
    /// On-disk path backing the resource, when file-backed.
    pub source_path: Option<String>,
}

/// Resource content type inferred from a file extension (TS `contentTypeForPath`).
pub fn content_type_for_path(path: &str) -> &'static str {
    let ext = path.rsplit_once('.').map_or("", |(_, e)| e);
    match ext {
        _ if ext.eq_ignore_ascii_case("md")
            || ext.eq_ignore_ascii_case("markdown")
            || ext.eq_ignore_ascii_case("mdc")
            || ext.eq_ignore_ascii_case("mdx") =>
        {
            "text/markdown"
        }
        _ if ext.eq_ignore_ascii_case("json") => "application/json",
        _ => "text/plain",
    }
}

/// Plain directory listing: directories first (`name/`), then files, by name.
pub fn format_directory_listing(mut entries: Vec<(String, bool)>) -> String {
    if entries.is_empty() {
        return "(empty directory)".to_string();
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    entries
        .into_iter()
        .map(|(name, is_dir)| if is_dir { format!("{name}/") } else { name })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a `---` fenced YAML-ish frontmatter block; returns `(meta, body)`.
pub fn parse_frontmatter(content: &str) -> (Vec<(String, String)>, String) {
    let Some(rest) = content.strip_prefix("---") else {
        return (Vec::new(), content.to_string());
    };
    let Some(end) = rest.find("\n---") else {
        return (Vec::new(), content.to_string());
    };
    let mut meta = Vec::new();
    for line in rest[..end].lines() {
        if let Some((k, v)) = line.split_once(':') {
            meta.push((
                k.trim().to_string(),
                v.trim().trim_matches('"').trim_matches('\'').to_string(),
            ));
        }
    }
    // Body starts after the closing `---` line.
    let after = &rest[end + 4..];
    let body = after.strip_prefix('\n').unwrap_or(after);
    (meta, body.trim().to_string())
}

/// Decode `%XX` escapes in a URL path segment (TS `decodeURIComponent`).
pub fn decode_component(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Reject `..` segments and absolute paths in a relative URL path (TS `validateRelativePath`).
pub fn validate_relative_path(rel: &str, scheme: &str) -> Result<(), String> {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\0') {
        return Err(format!("Invalid {} path: {rel}", scheme));
    }
    if rel.split(['/', '\\']).any(|seg| seg == "..") {
        return Err("Path traversal is not allowed".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_parses() {
        let (meta, body) =
            parse_frontmatter("---\nname: foo\ndescription: \"bar\"\n---\n\nBody here.\n");
        assert_eq!(meta[0], ("name".into(), "foo".into()));
        assert_eq!(meta[1], ("description".into(), "bar".into()));
        assert_eq!(body, "Body here.");
    }

    #[test]
    fn decode_component_works() {
        assert_eq!(decode_component("a%20b"), "a b");
        assert_eq!(decode_component("plain"), "plain");
    }

    #[test]
    fn split_host_path_works() {}
}
