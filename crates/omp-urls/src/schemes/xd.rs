//! `xd://` protocol handler (TS `internal-urls/xd-protocol.ts`, URLs per `pi-tui/tools/xd-url.ts`).
//!
//! Routes session-bound virtual tool devices. omp-rs does not thread a tool
//! session through [`UrlCtx`] yet, so every read reports the TS not-mounted
//! error; the URL grammar is ported verbatim for when a session lands.

use std::path::PathBuf;

use crate::{InternalResource, InternalUrl, UrlCtx, UrlProvider};

/// TS `NOT_MOUNTED`.
pub const NOT_MOUNTED: &str = "xd:// is not mounted in this session.";

/// Canonical prefix for virtual tool-device URLs (TS `XD_URL_PREFIX`).
pub const XD_URL_PREFIX: &str = "xd://";

/// The canonical tool name from a bare or `xd://`-prefixed spelling (TS `stripXdUrlPrefix`).
pub fn strip_xd_url_prefix(name: &str) -> &str {
    name.to_lowercase()
        .starts_with(XD_URL_PREFIX)
        .then_some(())
        .map_or(name, |_| &name[XD_URL_PREFIX.len()..])
}

/// Parse an `xd://` URL into its device target (TS `parseXdUrl`).
/// `None` for other or malformed URLs; `name: None` for the root.
pub fn parse_xd_url(input: &str) -> Option<Option<String>> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with(XD_URL_PREFIX) {
        return None;
    }
    let name = &trimmed[XD_URL_PREFIX.len()..];
    if name.is_empty() {
        return Some(None);
    }
    if name.contains(['/', '?', '#']) {
        return None;
    }
    Some(Some(name.to_string()))
}

/// Parse a read-only `xd://<tool>/<topic>` documentation URL (TS `parseXdTopicUrl`).
/// `None` for anything else, including bare device URLs.
pub fn parse_xd_topic_url(input: &str) -> Option<(String, String)> {
    let trimmed = input.trim();
    let rest = trimmed
        .strip_prefix(XD_URL_PREFIX)
        .or_else(|| trimmed.strip_prefix("xd://"))?;
    let (name, topic) = rest.split_once('/')?;
    if name.is_empty()
        || topic.is_empty()
        || name.contains(['?', '#'])
        || topic.contains(['/', '?', '#'])
    {
        return None;
    }
    Some((name.to_string(), topic.to_string()))
}

/// Resolve an `xd://` URL (TS `XdProtocolHandler.resolve`).
///
/// Without a mounted tool session every form yields the TS not-mounted error;
/// malformed URLs yield the TS invalid-URL error first.
pub fn resolve(url: &InternalUrl) -> Result<InternalResource, String> {
    let href = &url.raw_href;
    if parse_xd_topic_url(href).is_none() && parse_xd_url(href).is_none() {
        return Err(format!(
            "Invalid xd:// URL: {href}. Use xd://, xd://<tool>, or xd://<tool>/<topic>."
        ));
    }
    Err(NOT_MOUNTED.to_string())
}

/// `xd://` provider over the mounted tool devices (none in this build).
pub struct XdProvider;

impl UrlProvider for XdProvider {
    fn scheme(&self) -> &'static str {
        "xd"
    }

    fn spec(&self) -> &crate::SchemeSpec {
        static SPEC: crate::SchemeSpec = crate::SchemeSpec {
            backing: crate::SchemeBacking::Device,
            selectors: crate::SchemeSelectors::None,
            immutable: true,
            single_slash_alias: false,
            linkable: false,
            port_authority: false,
            write: None,
        };
        &SPEC
    }

    fn read(&self, url: &InternalUrl, _ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        resolve(url).map_err(anyhow::Error::msg)
    }

    fn locate(&self, _url: &InternalUrl, _ctx: &UrlCtx) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_internal_url;

    #[test]
    fn strip_prefix_works() {
        assert_eq!(strip_xd_url_prefix("xd://read"), "read");
        assert_eq!(strip_xd_url_prefix("read"), "read");
    }

    #[test]
    fn parse_device_matches_ts() {
        assert_eq!(parse_xd_url("xd://"), Some(None));
        assert_eq!(parse_xd_url("xd://read"), Some(Some("read".to_string())));
        assert_eq!(
            parse_xd_url(" xd://read \n"),
            Some(Some("read".to_string()))
        );
        assert_eq!(parse_xd_url("xd://read/extra"), None);
        assert_eq!(parse_xd_url("local://read"), None);
    }

    #[test]
    fn parse_topic_matches_ts() {
        assert_eq!(
            parse_xd_topic_url("xd://read/basics"),
            Some(("read".to_string(), "basics".to_string()))
        );
        assert_eq!(parse_xd_topic_url("xd://read"), None);
        assert_eq!(parse_xd_topic_url("xd://read/a/b"), None);
        assert_eq!(parse_xd_topic_url("xd:///topic"), None);
    }

    #[test]
    fn resolve_errors_match_ts() {
        let url = parse_internal_url("xd://read").unwrap();
        assert_eq!(resolve(&url).unwrap_err(), NOT_MOUNTED);
        let url = parse_internal_url("xd://read/a/b").unwrap();
        assert_eq!(
            resolve(&url).unwrap_err(),
            "Invalid xd:// URL: xd://read/a/b. Use xd://, xd://<tool>, or xd://<tool>/<topic>."
        );
    }
}
