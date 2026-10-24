//! `rule://` protocol handler (TS `internal-urls/rule-protocol.ts`).
//!
//! URL form: `rule://<name>` — reads rule content (frontmatter stripped).
//! Virtual backing: resolved content is the rule body, not the file's bytes.

use std::path::PathBuf;

use crate::{InternalResource, InternalUrl, UrlCompletion, UrlCtx, UrlProvider};

use super::RuleEntry;

/// Look up a rule by name; errors on a nameless URL (TS `lookupRule`/`findRule`).
fn find_rule<'a>(url: &InternalUrl, rules: &'a [RuleEntry]) -> Result<&'a RuleEntry, String> {
    if url.host.is_empty() {
        return Err("rule:// URL requires a rule name: rule://<name>".to_string());
    }
    match rules.iter().find(|r| r.name == url.host) {
        Some(rule) => Ok(rule),
        None => {
            let mut available: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
            available.sort();
            let available_str = if available.is_empty() {
                "none".to_string()
            } else {
                available.join(", ")
            };
            Err(format!(
                "Unknown rule: {}\nAvailable: {available_str}",
                url.host
            ))
        }
    }
}

/// Resolve a `rule://` URL to the rule body resource.
pub fn resolve(url: &InternalUrl, rules: &[RuleEntry]) -> Result<InternalResource, String> {
    let rule = find_rule(url, rules)?;
    Ok(InternalResource {
        content: rule.content.clone(),
        is_directory: false,
        immutable: Some(true),
        path: Some(rule.path.clone()),
    })
}

/// Rule source file (with frontmatter); `None` for unknown rules and rules with no file on disk.
pub fn locate(url: &InternalUrl, rules: &[RuleEntry]) -> Result<Option<PathBuf>, String> {
    if url.host.is_empty() {
        return Err("rule:// URL requires a rule name: rule://<name>".to_string());
    }
    Ok(rules
        .iter()
        .find(|r| r.name == url.host)
        .filter(|r| !r.path.is_empty() && PathBuf::from(&r.path).exists())
        .map(|r| PathBuf::from(&r.path)))
}

/// Autocomplete candidates over rule names + descriptions.
pub fn complete(rules: &[RuleEntry]) -> Vec<UrlCompletion> {
    rules
        .iter()
        .map(|r| UrlCompletion {
            label: r.name.clone(),
            detail: (!r.description.is_empty()).then(|| r.description.clone()),
        })
        .collect()
}

/// Read-only `rule://` provider over the session rulebook.
///
/// omp-rs has no TTSR rules subsystem yet, so the book is empty and every read
/// surfaces the TS "Unknown rule" error text.
#[derive(Default)]
pub struct RuleProvider;

impl UrlProvider for RuleProvider {
    fn scheme(&self) -> &'static str {
        "rule"
    }

    fn spec(&self) -> &crate::SchemeSpec {
        static SPEC: crate::SchemeSpec = crate::SchemeSpec {
            backing: crate::SchemeBacking::Virtual,
            selectors: crate::SchemeSelectors::Lines,
            immutable: true,
            single_slash_alias: false,
            linkable: true,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn read(&self, url: &InternalUrl, _ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        resolve(url, &active_rules()).map_err(anyhow::Error::msg)
    }

    fn locate(&self, url: &InternalUrl, _ctx: &UrlCtx) -> Option<PathBuf> {
        locate(url, &active_rules()).ok().flatten()
    }

    fn complete(&self, _query: &str, _ctx: &UrlCtx) -> Vec<UrlCompletion> {
        complete(&active_rules())
    }

    fn prompt_doc(&self) -> Option<String> {
        (active_rule_count() > 0).then(|| "`rule://<name>`: details.".to_string())
    }
}

/// Rules addressable by `rule://`. omp-rs does not port the TS rulebook
/// (Cursor/Windsurf/Cline discovery) yet; an empty book degrades to TS errors.
fn active_rules() -> Vec<RuleEntry> {
    Vec::new()
}

fn active_rule_count() -> usize {
    active_rules().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_internal_url;

    fn rule(name: &str, content: &str) -> RuleEntry {
        RuleEntry {
            name: name.to_string(),
            path: format!("/tmp/rules/{name}.md"),
            content: content.to_string(),
            description: String::new(),
        }
    }

    #[test]
    fn resolves_rule_body() {
        let rules = [rule("tdd", "Write tests first.")];
        let url = parse_internal_url("rule://tdd").unwrap();
        let res = resolve(&url, &rules).unwrap();
        assert_eq!(res.content, "Write tests first.");
        assert!(!res.is_directory);
    }

    #[test]
    fn errors_match_ts() {
        let url = parse_internal_url("rule://").unwrap();
        assert_eq!(
            resolve(&url, &[]).unwrap_err(),
            "rule:// URL requires a rule name: rule://<name>"
        );
        let rules = [rule("a", "x")];
        let url = parse_internal_url("rule://zzz").unwrap();
        assert_eq!(
            resolve(&url, &rules).unwrap_err(),
            "Unknown rule: zzz\nAvailable: a"
        );
        let url = parse_internal_url("rule://zzz").unwrap();
        assert_eq!(
            resolve(&url, &[]).unwrap_err(),
            "Unknown rule: zzz\nAvailable: none"
        );
    }

    #[test]
    fn locate_returns_existing_file_only() {
        let rules = [rule("ghost", "x")];
        let url = parse_internal_url("rule://ghost").unwrap();
        assert_eq!(locate(&url, &rules).unwrap(), None);
        let url = parse_internal_url("rule://").unwrap();
        assert_eq!(
            locate(&url, &rules).unwrap_err(),
            "rule:// URL requires a rule name: rule://<name>"
        );
    }
}
