//! Internal URL parser (TS `internal-urls/parse.ts`).
//!
//! Handles hosts with colons (`agent://<id>`) that standard URL parsers
//! reject as bad ports. All URL-shaped input goes through here.

/// A parsed internal URL with the raw input preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalUrl {
    /// Raw input exactly as given.
    pub raw_href: String,
    /// Lowercased scheme without the colon (`"local"`).
    pub scheme: String,
    /// Decoded host segment between `://` and `/` (may contain `:`).
    pub host: String,
    /// Path after the host, percent-decoded lazily by callers that need it.
    pub pathname: String,
    /// Raw query string without the leading `?` (`""` when absent).
    pub query: String,
    /// Fragment without the leading `#` (`""` when absent).
    pub fragment: String,
}

impl InternalUrl {
    /// The URL without query/fragment: `scheme://host/path`.
    pub fn bare(&self) -> String {
        format!("{}://{}{}", self.scheme, self.host, self.pathname)
    }

    /// Query parameters parsed `?key=value&flag` style.
    pub fn query_params(&self) -> Vec<(String, String)> {
        if self.query.is_empty() {
            return Vec::new();
        }
        self.query
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|p| match p.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => (p.to_string(), String::new()),
            })
            .collect()
    }
}

fn is_scheme_char(c: u8, first: bool) -> bool {
    c.is_ascii_alphabetic() || (!first && (c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.')))
}

/// Byte index right after `scheme://` when `input` starts with a scheme + authority.
fn scheme_host_end(input: &str) -> Option<usize> {
    let b = input.as_bytes();
    if b.is_empty() || !b[0].is_ascii_alphabetic() {
        return None;
    }
    let mut i = 1;
    while i < b.len() && is_scheme_char(b[i], false) {
        i += 1;
    }
    if input[i..].starts_with("://") {
        Some(i + 3)
    } else {
        None
    }
}

/// Extract the lowercased scheme from a URI-shaped input, or `None`.
///
/// Accepts hierarchical (`scheme://…`) and opaque (`scheme:rest`) forms,
/// guarded against Windows drives (`C:\…`), dotted filenames (`foo.ts:50`),
/// and read-selector tails (`Makefile:12`, `README:raw:1-20`).
pub fn extract_uri_scheme(input: &str) -> Option<String> {
    if let Some(end) = scheme_host_end(input) {
        return Some(input[..end - 3].to_lowercase());
    }
    // Opaque form: scheme `:` rest, where rest is not a selector chain.
    let b = input.as_bytes();
    if b.is_empty() || !b[0].is_ascii_alphabetic() {
        return None;
    }
    let mut i = 1;
    while i < b.len() && is_scheme_char(b[i], false) {
        i += 1;
    }
    let scheme = &input[..i];
    if scheme.len() == 1 || scheme.contains('.') {
        return None;
    }
    let rest = input[i..].strip_prefix(':')?;
    if rest.is_empty() || is_selector_chain(rest) {
        return None;
    }
    Some(scheme.to_lowercase())
}

/// One selector chunk: `raw`, `conflicts`, or a line spec like `-?N([-+]N)?`.
fn is_selector_chunk(s: &str) -> bool {
    let lower = s.to_lowercase();
    if lower == "raw" || lower == "conflicts" {
        return true;
    }
    // -?\d+([-+]\d+)?(,…)* — used only as a guard, so accept the superset.
    !s.is_empty() && s.split(',').all(|part| parse_line_spec(part).is_some())
}

fn parse_line_spec(s: &str) -> Option<()> {
    let t = s.strip_prefix('-').unwrap_or(s);
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    let mut rest = &t[digits..];
    if let Some(r) = rest.strip_prefix(['-', '+']) {
        let d = r.len() - r.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if d == 0 {
            return None;
        }
        rest = &r[d..];
    }
    if rest.is_empty() {
        Some(())
    } else {
        None
    }
}

/// A full selector chain: chunks separated by `:` (`raw:2-4`, `1-20,30+5`).
fn is_selector_chain(s: &str) -> bool {
    s.split(':').all(is_selector_chunk)
}

/// Parse an internal URL; errors when the input has no `scheme://` shape.
pub fn parse_internal_url(input: &str) -> anyhow::Result<InternalUrl> {
    let end = scheme_host_end(input).ok_or_else(|| anyhow::anyhow!("Invalid URL: {input}"))?;
    let after = &input[end..];
    let host_len = after.find(['/', '?', '#']).unwrap_or(after.len());
    let raw_host = &after[..host_len];
    let host = url_decode(raw_host);
    let rest = &after[host_len..];
    let path_end = rest.find(['?', '#']).unwrap_or(rest.len());
    let pathname = &rest[..path_end];
    let mut query = String::new();
    let mut fragment = String::new();
    let mut tail = &rest[path_end..];
    if let Some(t) = tail.strip_prefix('?') {
        let q_end = t.find('#').unwrap_or(t.len());
        query = t[..q_end].to_string();
        tail = &t[q_end..];
    }
    if let Some(t) = tail.strip_prefix('#') {
        fragment = t.to_string();
    }
    let scheme = input[..end - 3].to_lowercase();
    Ok(InternalUrl {
        raw_href: input.to_string(),
        scheme,
        host,
        pathname: pathname.to_string(),
        query,
        fragment,
    })
}

/// Percent-decode, leaving malformed sequences untouched.
fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((h * 16 + l) as char);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn hex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_namespaced_host() {
        let url = parse_internal_url("skill://plugin:name/path").unwrap();
        assert_eq!(url.scheme, "skill");
        assert_eq!(url.host, "plugin:name");
        assert_eq!(url.pathname, "/path");
    }

    #[test]
    fn extracts_schemes() {
        assert_eq!(extract_uri_scheme("local://x").as_deref(), Some("local"));
        assert_eq!(extract_uri_scheme("C:\\temp"), None);
        assert_eq!(extract_uri_scheme("foo.ts:50"), None);
        assert_eq!(extract_uri_scheme("Makefile:12"), None);
        assert_eq!(extract_uri_scheme("README:raw:1-20"), None);
        assert_eq!(
            extract_uri_scheme("urn:example:doc").as_deref(),
            Some("urn")
        );
    }

    #[test]
    fn query_and_fragment() {
        // TS PATHNAME_RE: the host runs to the next `/`, `?`, or `#`, so a
        // host-only URL has an empty pathname.
        let url = parse_internal_url("local://f.txt?limit=3#frag").unwrap();
        assert_eq!(url.host, "f.txt");
        assert_eq!(url.pathname, "");
        assert_eq!(url.query, "limit=3");
        assert_eq!(url.fragment, "frag");
        assert_eq!(
            url.query_params(),
            vec![("limit".to_string(), "3".to_string())]
        );
    }
}
