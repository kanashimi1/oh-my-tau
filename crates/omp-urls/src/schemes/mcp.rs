//! `mcp://` protocol handler (TS `internal-urls/mcp-protocol.ts`).
//!
//! URL forms:
//! - `mcp://<resource-uri>` (e.g. `mcp://test://notes`, `mcp://ibkr://portfolio/positions`)
//! - a resource's native URI when its scheme has no OMP handler (e.g. `ags://capabilities/current-host`)
//!
//! Resource reads route through the MCP manager. omp-rs's `omp-mcp` exposes
//! tool calls only (no resource catalog yet), so provider reads surface the TS
//! "no manager" error text; the discovery/scoring helpers are ported verbatim
//! and unit-tested against the TS semantics for when the catalog lands.

use crate::{InternalResource, InternalUrl, UrlCtx, UrlProvider};

/// Resource catalog of one connected server (the slice of the MCP manager this scheme needs).
#[derive(Debug, Clone, Default)]
pub struct ServerResources {
    /// Concrete resource URIs.
    pub resources: Vec<String>,
    /// URI templates with `{...}` expressions.
    pub templates: Vec<String>,
}

/// Extract the server resource URI from any accepted URL spelling (TS `extractResourceUri`).
///
/// `scheme != "mcp"`: preserve the input byte-for-byte (server-advertised
/// native URI). `mcp`: reconstruct the wrapped URI from host + pathname,
/// eliding the bare trailing `/` URL parsing adds to host-only forms.
pub fn extract_resource_uri(url: &InternalUrl) -> Result<String, String> {
    let scheme = url.scheme.trim_end_matches(':').to_ascii_lowercase();
    if scheme != "mcp" {
        // Native URI form; keep it intact (TS preserves `url.rawHref ?? url.href`).
        return Ok(url.raw_href.clone());
    }
    // TS: `${host}${hasPath ? rawPathname : ""}${search}${hash}` — with our
    // parsed fields, `mcp://test://notes` has host `test:` and pathname
    // `//notes`, reconstructing `test://notes` exactly.
    let has_path = !url.pathname.is_empty() && url.pathname != "/";
    let search = if url.query.is_empty() {
        String::new()
    } else {
        format!("?{}", url.query)
    };
    let hash = if url.fragment.is_empty() {
        String::new()
    } else {
        format!("#{}", url.fragment)
    };
    let uri = format!(
        "{}{}{}{}",
        url.host,
        if has_path { url.pathname.as_str() } else { "" },
        search,
        hash
    )
    .trim()
    .to_string();
    if uri.is_empty() {
        return Err("mcp:// URL requires a resource URI: mcp://<resource-uri>".to_string());
    }
    Ok(uri)
}

/// Score a URI against a template: literal chars matched and expression count,
/// or `None` when the template does not match (TS `getUriTemplateMatchScore`).
pub fn uri_template_match_score(uri: &str, uri_template: &str) -> Option<(usize, usize)> {
    // Split the template at `{...}` expressions.
    let mut literal_segments: Vec<&str> = Vec::new();
    let mut expression_count = 0;
    let mut rest = uri_template;
    while let Some(start) = rest.find('{') {
        literal_segments.push(&rest[..start]);
        let end = rest[start..].find('}')? + start;
        expression_count += 1;
        rest = &rest[end + 1..];
    }
    literal_segments.push(rest);

    // Match: each literal segment must appear in order, with any text between.
    let mut remaining = uri;
    for (i, lit) in literal_segments.iter().enumerate() {
        if lit.is_empty() {
            continue;
        }
        match remaining.find(lit) {
            Some(idx) => remaining = &remaining[idx + lit.len()..],
            None => return None,
        }
        let _ = i;
    }
    let literal_chars: usize = literal_segments.iter().map(|s| s.len()).sum();
    Some((literal_chars, expression_count))
}

/// Pick the server owning `uri`: exact resource match first, else the best
/// template match — most literal chars, then fewest expressions, then lowest
/// server and template index (TS `resolveTargetServer`).
pub fn resolve_target_server(servers: &[(String, ServerResources)], uri: &str) -> Option<String> {
    for (name, res) in servers {
        if res.resources.iter().any(|r| r == uri) {
            return Some(name.clone());
        }
    }
    let mut best: Option<(usize, usize, usize, usize, String)> = None;
    for (server_index, (name, res)) in servers.iter().enumerate() {
        for (template_index, template) in res.templates.iter().enumerate() {
            let Some((literal_chars, expression_count)) = uri_template_match_score(uri, template)
            else {
                continue;
            };
            let better = match &best {
                None => true,
                Some((bl, be, bs, bt, _)) => {
                    literal_chars > *bl
                        || (literal_chars == *bl
                            && (expression_count < *be
                                || (expression_count == *be
                                    && (server_index < *bs
                                        || (server_index == *bs && template_index < *bt)))))
                }
            };
            if better {
                best = Some((
                    literal_chars,
                    expression_count,
                    server_index,
                    template_index,
                    name.clone(),
                ));
            }
        }
    }
    best.map(|(_, _, _, _, name)| name)
}

/// The TS "nothing has this resource" listing (TS `formatAvailableResources`).
pub fn format_available_resources(servers: &[(String, ServerResources)]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (name, res) in servers {
        for r in &res.resources {
            lines.push(format!("  {r} ({name})"));
        }
        for t in &res.templates {
            lines.push(format!("  {t} ({name}, template)"));
        }
    }
    if lines.is_empty() {
        "  (none)".to_string()
    } else {
        lines.join("\n")
    }
}

/// Join read contents like the TS handler: text parts with `\n---\n`,
/// `[Binary content: ...]` placeholders, `(empty resource)` fallback.
pub fn format_resource_contents(contents: &[ResourceContent]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for item in contents {
        match item {
            ResourceContent::Text(text) => parts.push(text.clone()),
            ResourceContent::Binary { mime, base64_len } => {
                parts.push(format!(
                    "[Binary content: {mime}, base64 length {base64_len}]"
                ));
            }
        }
    }
    if parts.is_empty() {
        "(empty resource)".to_string()
    } else {
        parts.join("\n---\n")
    }
}

/// One MCP resource content item (TS `MCPResourceReadResult.contents[]`).
#[derive(Debug, Clone)]
pub enum ResourceContent {
    Text(String),
    Binary { mime: String, base64_len: usize },
}

/// Resolve an `mcp://` URL against a live catalog (TS `McpProtocolHandler.resolve`,
/// minus the async manager handshake).
pub fn resolve(
    url: &InternalUrl,
    servers: &[(String, ServerResources)],
) -> Result<InternalResource, String> {
    let uri = extract_resource_uri(url)?;
    let Some(_server) = resolve_target_server(servers, &uri) else {
        return Err(format!(
            "No MCP server has resource \"{uri}\".\n\nAvailable resources:\n{}",
            format_available_resources(servers)
        ));
    };
    // Reading goes through the manager; with no content channel wired the TS
    // "no content" error is the honest answer.
    Err(format!(
        "Server \"{}\" returned no content for \"{uri}\".",
        _server
    ))
}

/// `mcp://` provider over MCP server resources.
///
/// `omp-mcp` exposes tool calls only — no resource catalog — so every read
/// surfaces the TS "no manager available" error text until resources land.
pub struct McpProvider;

/// Error text when no MCP manager is reachable (TS, verbatim).
pub const NO_MANAGER: &str = "No MCP manager available. MCP servers may not be configured.";

impl UrlProvider for McpProvider {
    fn scheme(&self) -> &'static str {
        "mcp"
    }

    fn spec(&self) -> &crate::SchemeSpec {
        static SPEC: crate::SchemeSpec = crate::SchemeSpec {
            backing: crate::SchemeBacking::Remote,
            selectors: crate::SchemeSelectors::Opaque,
            immutable: true,
            single_slash_alias: false,
            linkable: false,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn read(&self, url: &InternalUrl, _ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        let _ = url;
        anyhow::bail!("{NO_MANAGER}")
    }

    fn prompt_doc(&self) -> Option<String> {
        // TS `prompts/internal-urls/mcp.md`, trimmed.
        Some("`mcp://<uri>`: MCP resource.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_internal_url;

    fn server(name: &str, resources: &[&str], templates: &[&str]) -> (String, ServerResources) {
        (
            name.to_string(),
            ServerResources {
                resources: resources.iter().map(|s| s.to_string()).collect(),
                templates: templates.iter().map(|s| s.to_string()).collect(),
            },
        )
    }

    #[test]
    fn wrapped_uri_reconstruction() {
        let url = parse_internal_url("mcp://test://notes").unwrap();
        assert_eq!(extract_resource_uri(&url).unwrap(), "test://notes");
        let url = parse_internal_url("mcp://ibkr://portfolio/positions").unwrap();
        assert_eq!(
            extract_resource_uri(&url).unwrap(),
            "ibkr://portfolio/positions"
        );
    }

    #[test]
    fn empty_wrapper_errors() {
        let url = parse_internal_url("mcp://").unwrap();
        assert_eq!(
            extract_resource_uri(&url).unwrap_err(),
            "mcp:// URL requires a resource URI: mcp://<resource-uri>"
        );
    }

    #[test]
    fn native_uri_preserved() {
        let url = parse_internal_url("ags://capabilities/current-host").unwrap();
        assert_eq!(
            extract_resource_uri(&url).unwrap(),
            "ags://capabilities/current-host"
        );
    }

    #[test]
    fn exact_match_wins_over_template() {
        let servers = vec![
            server("templated", &[], &["test://{thing}"]),
            server("exact", &["test://notes"], &["test://notes"]),
        ];
        assert_eq!(
            resolve_target_server(&servers, "test://notes"),
            Some("exact".to_string())
        );
    }

    #[test]
    fn best_template_wins() {
        let servers = vec![
            server("a", &[], &["ibkr://{p}/positions"]),
            server("b", &[], &["ibkr://portfolio/{q}"]),
        ];
        // Both match `ibkr://portfolio/positions`: equal literal chars, equal
        // expression counts → lowest server index wins.
        assert_eq!(
            resolve_target_server(&servers, "ibkr://portfolio/positions"),
            Some("a".to_string())
        );
    }

    #[test]
    fn no_match_is_none() {
        let servers = vec![server("a", &["other://x"], &["foo://{x}"])];
        assert_eq!(resolve_target_server(&servers, "test://notes"), None);
    }

    #[test]
    fn available_resources_listing() {
        assert_eq!(format_available_resources(&[]), "  (none)");
        let servers = vec![
            server("main", &["test://notes"], &["test://{x}"]),
            server("aux", &[], &[]),
        ];
        assert_eq!(
            format_available_resources(&servers),
            "  test://notes (main)\n  test://{x} (main, template)"
        );
    }

    #[test]
    fn missing_resource_error_matches_ts() {
        let servers = vec![server("main", &["test://notes"], &[])];
        let url = parse_internal_url("mcp://test://missing").unwrap();
        assert_eq!(
            resolve(&url, &servers).unwrap_err(),
            "No MCP server has resource \"test://missing\".\n\nAvailable resources:\n  test://notes (main)"
        );
    }

    #[test]
    fn content_joining_matches_ts() {
        assert_eq!(
            format_resource_contents(&[
                ResourceContent::Text("a".into()),
                ResourceContent::Text("b".into())
            ]),
            "a\n---\nb"
        );
        assert_eq!(format_resource_contents(&[]), "(empty resource)");
        assert_eq!(
            format_resource_contents(&[ResourceContent::Binary {
                mime: "image/png".into(),
                base64_len: 42
            }]),
            "[Binary content: image/png, base64 length 42]"
        );
    }
}
