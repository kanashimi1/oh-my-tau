//! Collab link formatting and parsing.
//!
//! Link format: `wss://<host[:port]>/r/<roomId>.<base64url-key>` where the
//! key is 32 bytes (view link) or 48 bytes (full link, `key || writeToken`).
//! Compact renderings: the default relay collapses to `<roomId>.<key>`,
//! other wss relays drop the scheme, plain-ws localhost links keep the full
//! URL. The secret is dot-joined rather than `#`-joined (RFC 3986 forbids a
//! raw `#` inside a fragment — strict URL stacks mangle it); parsers accept
//! the legacy `#` form and the `%23`-mangled form.
//!
//! Port of `formatCollabLink`/`formatCollabWebLink`/`parseCollabLink` from
//! `vendor-src/oh-my-pi/packages/coding-agent/src/collab/protocol.ts:194-339`
//! (the collab-web mirror link.ts:86-189 is byte-identical in behavior).

use url::Url;

use crate::protocol::{
    decode_base64_url, encode_base64_url, DEFAULT_RELAY_URL, ROOM_KEY_BYTES, WRITE_TOKEN_BYTES,
};

/// A successfully parsed collab link.
///
/// Field naming follows the port contract: `relay_url` is the
/// scheme-normalized relay *origin* (`wss://host[:port]` / `ws://…`), the
/// value `connect_relay` composes with `room_id`. (TS `ParsedCollabLink`
/// pre-joins them as `wsUrl = origin + "/r/" + roomId`, wire/src/index.ts:431;
/// the Rust relay client receives the two parts separately instead.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCollabLink {
    /// Normalized relay origin, e.g. `wss://my.omp.sh` or
    /// `ws://127.0.0.1:7466` (TS `wsUrl` minus the `/r/<roomId>` suffix).
    pub relay_url: String,
    /// 10-64 base64url chars, as emitted by
    /// [`crate::protocol::generate_room_id`].
    pub room_id: String,
    /// The 32-byte room key (first [`ROOM_KEY_BYTES`] bytes of the secret).
    pub key: Vec<u8>,
    /// Write token for full links; absent for read-only (view) links.
    pub write_token: Option<Vec<u8>>,
}

/// Render the shareable collab link (protocol.ts:238-248).
///
/// # Errors
/// When `relay_url` cannot be normalized — unparseable URL, unsupported
/// scheme, or plain `ws://` to a non-local host. Error strings are the TS
/// ones, asserted verbatim by the tests.
pub fn format_collab_link(
    relay_url: &str,
    room_id: &str,
    key: &[u8],
    write_token: Option<&[u8]>,
) -> Result<String, String> {
    let origin = normalize_relay_origin(relay_url)?;
    // TS: `writeToken ? Buffer.concat([key, writeToken]) : Buffer.from(key)` —
    // a zero-length token concatenates to the same bytes as no token.
    let mut secret = Vec::with_capacity(key.len() + write_token.map_or(0, <[u8]>::len));
    secret.extend_from_slice(key);
    if let Some(token) = write_token {
        secret.extend_from_slice(token);
    }
    let key_text = encode_base64_url(&secret);
    if origin == DEFAULT_RELAY_URL {
        return Ok(format!("{room_id}.{key_text}"));
    }
    // Compact form drops the scheme on wss; plain-ws localhost keeps its full
    // URL so parsing cannot mis-infer wss (protocol.ts:223-227).
    let compact = origin.strip_prefix("wss://").unwrap_or(&origin);
    Ok(format!("{compact}/r/{room_id}.{key_text}"))
}

/// Render the browser deep link: `<web base>/#<collab link>` so room secrets
/// stay out of HTTP path and query bytes (protocol.ts:284-292).
///
/// # Errors
/// Same cases as [`format_collab_link`], plus web-base validation errors for
/// an explicit `web_url` (must be http/https — http only toward localhost —
/// with no query or fragment).
pub fn format_collab_web_link(
    relay_url: &str,
    room_id: &str,
    key: &[u8],
    write_token: Option<&[u8]>,
    web_url: Option<&str>,
) -> Result<String, String> {
    // TS evaluates the web base first, so web-base errors win over relay
    // errors (protocol.ts:291).
    let base = normalize_collab_web_base_url(relay_url, web_url)?;
    let link = format_collab_link(relay_url, room_id, key, write_token)?;
    Ok(format!("{base}/#{link}"))
}

/// Parse a collab link in any accepted shape (protocol.ts:294-338):
/// bare `<roomId>.<key>` (legacy `<roomId>#<key>`, `%23`-mangled included),
/// scheme-less `host[:port]/r/…`, full `ws(s)://…/r/…` URLs, and http(s) web
/// deep links carrying any of those in the fragment.
///
/// # Errors
/// With the TS error strings when the URL is unparseable, the relay origin
/// is invalid, the path is not `/r/<roomId>`, or the key is not 32 (view) or
/// 48 (full) base64url bytes.
pub fn parse_collab_link(link: &str) -> Result<ParsedCollabLink, String> {
    // Lenient input: terminals that open OSC 8 links through strict URL stacks
    // percent-encode the legacy second `#` to `%23` (protocol.ts:295-297). The
    // `/gi` flag reduces to `/g` here: `%23` has no alphabetic hex digit.
    let mut text = link.trim().replace("%23", "#");
    // Bare `<roomId>.<key>` (legacy `<roomId>#<key>`) → default relay.
    if let Some((room, _, key)) = bare_link_split(&text) {
        text = format!("{DEFAULT_RELAY_URL}/r/{room}.{key}");
    // Scheme-less `host[:port]/r/…` → wss.
    } else if !text.contains("://") {
        text = format!("wss://{text}");
    }
    let url = Url::parse(&text).map_err(|_| format!("Invalid collab link: {link}"))?;
    // http(s) deep links may carry a complete collab link in the fragment;
    // invalid fragments fall through to direct relay validation instead of
    // double-recursing (protocol.ts:309-313,318-322).
    if matches!(url.scheme(), "http" | "https") {
        if let Some(inner) = url.fragment() {
            if let Ok(parsed) = parse_collab_link(inner) {
                return Ok(parsed);
            }
        }
    }
    // TS re-normalizes `url.origin` (dropping userinfo and any path), so an
    // opaque origin ("null") hits the unsupported-scheme error exactly as the
    // original scheme would not (protocol.ts:314-315).
    let origin_string = url.origin().ascii_serialization();
    let origin = normalize_relay_origin(&origin_string)?;
    let Some((room_id, dot_key)) = room_path_split(url.path()) else {
        // Non-http(s) deep links may also carry a complete collab link in the
        // fragment (protocol.ts:318-322); their parse errors ARE the result.
        if !matches!(url.scheme(), "http" | "https") {
            if let Some(inner) = url.fragment().filter(|inner| !inner.is_empty()) {
                return parse_collab_link(inner);
            }
        }
        return Err("Collab link must contain a /r/<roomId> path".to_string());
    };
    // Key rides dot-joined in the path (`/r/<roomId>.<key>`); legacy links
    // carry it in the fragment (`/r/<roomId>#<key>`) (protocol.ts:326-328).
    let fragment = match dot_key {
        Some(key) => key,
        None => url.fragment().unwrap_or_default(),
    };
    if fragment.is_empty() {
        return Err("Collab link is missing the <key> part".to_string());
    }
    let secret = decode_base64_url(fragment).filter(|secret| {
        secret.len() == ROOM_KEY_BYTES || secret.len() == ROOM_KEY_BYTES + WRITE_TOKEN_BYTES
    });
    let Some(secret) = secret else {
        return Err("Collab link key must be 32 (view) or 48 (full) base64url bytes".to_string());
    };
    let key = secret[..ROOM_KEY_BYTES].to_vec();
    let write_token = (secret.len() > ROOM_KEY_BYTES).then(|| secret[ROOM_KEY_BYTES..].to_vec());
    Ok(ParsedCollabLink {
        relay_url: origin,
        room_id: room_id.to_string(),
        key,
        write_token,
    })
}

/// Normalize a relay base URL (ws/wss/http/https) into a ws/wss origin
/// (protocol.ts:195-220): wss/https → `wss://`, ws/http → `ws://`, anything
/// else is rejected; plain `ws://` is only allowed for localhost.
fn normalize_relay_origin(relay_url: &str) -> Result<String, String> {
    let url = Url::parse(relay_url).map_err(|_| format!("Invalid relay URL: {relay_url}"))?;
    let scheme = match url.scheme() {
        "wss" | "https" => "wss",
        "ws" | "http" => "ws",
        other => return Err(format!("Unsupported relay URL scheme: {other}:")),
    };
    let hostname = url.host_str().unwrap_or_default();
    if scheme == "ws" && !is_local_hostname(hostname) {
        return Err(
            "relay link must be wss:// (plain ws:// is only allowed for localhost)".to_string(),
        );
    }
    // TS appends `:${port}` only when present; WHATWG omits default ports in
    // both implementations (protocol.ts:218-219).
    let port = url.port().map_or(String::new(), |port| format!(":{port}"));
    Ok(format!("{scheme}://{hostname}{port}"))
}

/// Web base for deep links (protocol.ts:250-277): derived from the relay
/// origin (wss→https, ws→http) unless an explicit `web_url` is given, which
/// must be http(s) — http only toward localhost — with no query or fragment;
/// trailing slashes are stripped from its path.
fn normalize_collab_web_base_url(relay_url: &str, web_url: Option<&str>) -> Result<String, String> {
    let Some(explicit) = web_url
        .map(str::trim)
        .filter(|explicit| !explicit.is_empty())
    else {
        let origin = normalize_relay_origin(relay_url)?;
        return Ok(if let Some(rest) = origin.strip_prefix("wss://") {
            format!("https://{rest}")
        } else {
            // `normalize_relay_origin` only yields wss:// or ws:// origins.
            format!("http://{}", origin.strip_prefix("ws://").unwrap_or(&origin))
        });
    };
    let url =
        Url::parse(explicit).map_err(|_| "collab.webUrl must start with http:// or https://")?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("collab.webUrl must start with http:// or https://".to_string());
    }
    if url.scheme() == "http" && !is_local_hostname(url.host_str().unwrap_or_default()) {
        return Err("collab.webUrl must use https:// unless it targets localhost".to_string());
    }
    // TS truthiness on `url.search`/`url.hash`: a bare `#` or `?` also counts
    // (protocol.ts:272); `fragment()`/`query()` are Some for those too.
    if url.query().is_some() || url.fragment().is_some() {
        return Err("collab.webUrl must not include a query string or fragment".to_string());
    }
    let origin = url.origin().ascii_serialization();
    // `url.pathname.replace(/\/+$/, "")` (protocol.ts:275).
    let path = url.path().trim_end_matches('/');
    Ok(format!("{origin}{path}"))
}

/// `LOCAL_HOSTNAMES` gate (protocol.ts:182-186): exactly localhost, the IPv4
/// loopback, or IPv6 `::1` (bracketed or not, as WHATWG serializes it).
fn is_local_hostname(hostname: &str) -> bool {
    matches!(hostname, "localhost" | "127.0.0.1" | "::1" | "[::1]")
}

/// `BARE_LINK_RE = ^([A-Za-z0-9_-]{10,64})[#.]([A-Za-z0-9_-]+)$`
/// (protocol.ts:180). The character class excludes `#` and `.`, so the first
/// separator fixes the split — no backtracking exists to model.
fn bare_link_split(text: &str) -> Option<(&str, u8, &str)> {
    let bytes = text.as_bytes();
    let sep_index = bytes
        .iter()
        .position(|&byte| byte == b'.' || byte == b'#')?;
    let room = &text[..sep_index];
    let tail = &text[sep_index + 1..];
    if !(10..=64).contains(&room.len())
        || !room.bytes().all(is_link_class)
        || tail.is_empty()
        || !tail.bytes().all(is_link_class)
    {
        return None;
    }
    Some((room, bytes[sep_index], tail))
}

/// `ROOM_PATH_RE = ^\/r\/([A-Za-z0-9_-]{10,64})(?:\.([A-Za-z0-9_-]+))?$`
/// (protocol.ts:179) → `(roomId, dot-joined key?)`.
fn room_path_split(path: &str) -> Option<(&str, Option<&str>)> {
    let rest = path.strip_prefix("/r/")?;
    match rest.split_once('.') {
        Some((room, key)) => {
            if !is_valid_room_id(room) || key.is_empty() || !key.bytes().all(is_link_class) {
                return None;
            }
            Some((room, Some(key)))
        }
        None => is_valid_room_id(rest).then_some((rest, None)),
    }
}

fn is_valid_room_id(room: &str) -> bool {
    (10..=64).contains(&room.len()) && room.bytes().all(is_link_class)
}

fn is_link_class(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::generate_room_id;

    // TS-oracle fixtures (bun, oracle-t1.mts, /tmp/oracle-t1.json).
    const KEY: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];
    const TOKEN: [u8; 16] = [
        0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe,
        0xff,
    ];
    const KEY_TEXT: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    const KEY_TOKEN_TEXT: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh_w8fLz9PX29_j5-vv8_f7_";
    const ROOM: &str = "room1234567890abcd";

    #[test]
    fn format_default_relay_collapses_to_bare() {
        // TS oracle: formatCollabLink("wss://my.omp.sh", ROOM, KEY).
        assert_eq!(
            format_collab_link(DEFAULT_RELAY_URL, ROOM, &KEY, None).expect("valid relay"),
            format!("{ROOM}.{KEY_TEXT}")
        );
    }

    #[test]
    fn format_default_relay_full_link_appends_token() {
        // TS oracle: formatCollabLink("wss://my.omp.sh", ROOM, KEY, TOKEN).
        assert_eq!(
            format_collab_link(DEFAULT_RELAY_URL, ROOM, &KEY, Some(&TOKEN)).expect("valid relay"),
            format!("{ROOM}.{KEY_TOKEN_TEXT}")
        );
    }

    #[test]
    fn format_local_relay_keeps_full_ws_url() {
        // TS oracle: "ws://127.0.0.1:7466/r/<room>.<key>" — plain-ws keeps the
        // scheme so parsing cannot mis-infer wss (protocol.ts:223-227).
        assert_eq!(
            format_collab_link("ws://127.0.0.1:7466", ROOM, &KEY, None).expect("valid relay"),
            format!("ws://127.0.0.1:7466/r/{ROOM}.{KEY_TEXT}")
        );
    }

    #[test]
    fn format_http_relay_upgrades_to_ws() {
        // TS oracle: http://127.0.0.1:9224 → "ws://127.0.0.1:9224/r/…".
        assert_eq!(
            format_collab_link("http://127.0.0.1:9224", ROOM, &KEY, None).expect("valid relay"),
            format!("ws://127.0.0.1:9224/r/{ROOM}.{KEY_TEXT}")
        );
    }

    #[test]
    fn format_custom_wss_drops_scheme() {
        // TS oracle: "relay.example.com:8443/r/<room>.<key>".
        assert_eq!(
            format_collab_link("wss://relay.example.com:8443", ROOM, &KEY, None)
                .expect("valid relay"),
            format!("relay.example.com:8443/r/{ROOM}.{KEY_TEXT}")
        );
    }

    #[test]
    fn format_https_relay_drops_scheme_and_default_port() {
        // TS oracle: https://relay.example.com → "relay.example.com/r/…".
        assert_eq!(
            format_collab_link("https://relay.example.com", ROOM, &KEY, Some(&TOKEN))
                .expect("valid relay"),
            format!("relay.example.com/r/{ROOM}.{KEY_TOKEN_TEXT}")
        );
    }

    #[test]
    fn format_rejects_bad_relays_with_ts_messages() {
        // TS oracles: exact thrown messages (protocol.ts:200,213,216).
        assert_eq!(
            format_collab_link("ftp://x", ROOM, &KEY, None),
            Err("Unsupported relay URL scheme: ftp:".to_string())
        );
        assert_eq!(
            format_collab_link("ws://my.omp.sh", ROOM, &KEY, None),
            Err(
                "relay link must be wss:// (plain ws:// is only allowed for localhost)".to_string()
            )
        );
        assert_eq!(
            format_collab_link("not a url", ROOM, &KEY, None),
            Err("Invalid relay URL: not a url".to_string())
        );
    }

    #[test]
    fn format_web_link_derives_https_from_relay() {
        // TS oracles for the derived-web-base rule (protocol.ts:252-258).
        assert_eq!(
            format_collab_web_link("ws://127.0.0.1:7466", ROOM, &KEY, None, None).expect("valid"),
            format!("http://127.0.0.1:7466/#ws://127.0.0.1:7466/r/{ROOM}.{KEY_TEXT}")
        );
        assert_eq!(
            format_collab_web_link(DEFAULT_RELAY_URL, ROOM, &KEY, Some(&TOKEN), None)
                .expect("valid"),
            format!("https://my.omp.sh/#{ROOM}.{KEY_TOKEN_TEXT}")
        );
        assert_eq!(
            format_collab_web_link("https://relay.example.com", ROOM, &KEY, None, None)
                .expect("valid"),
            format!("https://relay.example.com/#relay.example.com/r/{ROOM}.{KEY_TEXT}")
        );
    }

    #[test]
    fn format_web_link_explicit_base_rules() {
        // TS oracles (protocol.ts:260-276): localhost http allowed, trailing
        // slashes stripped, default port elided.
        assert_eq!(
            format_collab_web_link(
                DEFAULT_RELAY_URL,
                ROOM,
                &KEY,
                None,
                Some("http://127.0.0.1:7466")
            )
            .expect("valid"),
            format!("http://127.0.0.1:7466/#{ROOM}.{KEY_TEXT}")
        );
        assert_eq!(
            format_collab_web_link(
                DEFAULT_RELAY_URL,
                ROOM,
                &KEY,
                None,
                Some("https://web.example.com/ui/")
            )
            .expect("valid"),
            format!("https://web.example.com/ui/#{ROOM}.{KEY_TEXT}")
        );
        assert_eq!(
            format_collab_web_link(
                DEFAULT_RELAY_URL,
                ROOM,
                &KEY,
                None,
                Some("https://web.example.com:443")
            )
            .expect("valid"),
            format!("https://web.example.com/#{ROOM}.{KEY_TEXT}")
        );
    }

    #[test]
    fn format_web_link_rejects_bad_explicit_bases_with_ts_messages() {
        // TS oracles (protocol.ts:264-274).
        assert_eq!(
            format_collab_web_link(
                DEFAULT_RELAY_URL,
                ROOM,
                &KEY,
                None,
                Some("http://web.example.com")
            ),
            Err("collab.webUrl must use https:// unless it targets localhost".to_string())
        );
        assert_eq!(
            format_collab_web_link(
                DEFAULT_RELAY_URL,
                ROOM,
                &KEY,
                None,
                Some("ftp://web.example.com")
            ),
            Err("collab.webUrl must start with http:// or https://".to_string())
        );
        assert_eq!(
            format_collab_web_link(
                DEFAULT_RELAY_URL,
                ROOM,
                &KEY,
                None,
                Some("https://web.example.com/?q=1")
            ),
            Err("collab.webUrl must not include a query string or fragment".to_string())
        );
        assert_eq!(
            format_collab_web_link(DEFAULT_RELAY_URL, ROOM, &KEY, None, Some("not a url")),
            Err("collab.webUrl must start with http:// or https://".to_string())
        );
    }

    #[test]
    fn parse_bare_forms_all_resolve_to_default_relay() {
        let expected = ParsedCollabLink {
            relay_url: "wss://my.omp.sh".to_string(),
            room_id: ROOM.to_string(),
            key: KEY.to_vec(),
            write_token: None,
        };
        // TS oracles: dot form, legacy `#` form, and the `%23`-mangled form all
        // parse identically (protocol.ts:295-300).
        assert_eq!(
            parse_collab_link(&format!("{ROOM}.{KEY_TEXT}")),
            Ok(expected.clone())
        );
        assert_eq!(
            parse_collab_link(&format!("{ROOM}#{KEY_TEXT}")),
            Ok(expected.clone())
        );
        assert_eq!(
            parse_collab_link(&format!("{ROOM}%23{KEY_TEXT}")),
            Ok(expected)
        );
    }

    #[test]
    fn parse_full_and_schemeless_forms() {
        // TS oracles: explicit ws URL, scheme-less wss default, custom wss
        // port with write token.
        let parsed =
            parse_collab_link(&format!("ws://127.0.0.1:7466/r/{ROOM}.{KEY_TEXT}")).expect("parses");
        assert_eq!(parsed.relay_url, "ws://127.0.0.1:7466");
        assert_eq!(parsed.room_id, ROOM);
        assert_eq!(parsed.key, KEY);
        assert_eq!(parsed.write_token, None);

        let parsed = parse_collab_link(&format!("my.omp.sh/r/{ROOM}.{KEY_TEXT}")).expect("parses");
        assert_eq!(parsed.relay_url, "wss://my.omp.sh");

        let parsed =
            parse_collab_link(&format!("relay.example.com:8443/r/{ROOM}.{KEY_TOKEN_TEXT}"))
                .expect("parses");
        assert_eq!(parsed.relay_url, "wss://relay.example.com:8443");
        assert_eq!(parsed.key, KEY);
        assert_eq!(parsed.write_token, Some(TOKEN.to_vec()));
    }

    #[test]
    fn parse_web_deep_links_recurse_into_fragment() {
        // TS oracles: derived local web link, default web link, explicit-path
        // web link, and a deeply nested https link — all yield the inner
        // collab link (protocol.ts:309-313).
        for (link, relay, token) in [
            (
                format!("http://127.0.0.1:7466/#ws://127.0.0.1:7466/r/{ROOM}.{KEY_TEXT}"),
                "ws://127.0.0.1:7466",
                None,
            ),
            (
                format!("https://my.omp.sh/#{ROOM}.{KEY_TOKEN_TEXT}"),
                "wss://my.omp.sh",
                Some(TOKEN.to_vec()),
            ),
            (
                format!("https://web.example.com/ui/#{ROOM}.{KEY_TEXT}"),
                "wss://my.omp.sh",
                None,
            ),
            (
                format!("https://portal.example.com/sessions/https://my.omp.sh/#{ROOM}.{KEY_TOKEN_TEXT}"),
                "wss://my.omp.sh",
                Some(TOKEN.to_vec()),
            ),
        ] {
            let parsed = parse_collab_link(&link).unwrap_or_else(|err| panic!("{link}: {err}"));
            assert_eq!(parsed.relay_url, relay);
            assert_eq!(parsed.room_id, ROOM);
            assert_eq!(parsed.key, KEY);
            assert_eq!(parsed.write_token, token);
        }
    }

    #[test]
    fn parse_legacy_hash_and_pct23_in_path() {
        // TS oracles: key in the fragment after the path, raw and %23-mangled.
        for sep in ["#", "%23"] {
            let parsed = parse_collab_link(&format!("wss://my.omp.sh/r/{ROOM}{sep}{KEY_TEXT}"))
                .expect("parses");
            assert_eq!(parsed.relay_url, "wss://my.omp.sh");
            assert_eq!(parsed.key, KEY);
            assert_eq!(parsed.write_token, None);
        }
    }

    #[test]
    fn parse_room_id_length_bounds_come_from_the_regex() {
        // TS oracles: 10-char room ids pass the {10,64} bound, 9-char ones do
        // not match BARE_LINK_RE and end up host-parsed → /r/ path error.
        assert_eq!(
            parse_collab_link(&format!("room123456.{KEY_TEXT}"))
                .expect("parses")
                .room_id,
            "room123456"
        );
        assert_eq!(
            parse_collab_link(&format!("room12345.{KEY_TEXT}")),
            Err("Collab link must contain a /r/<roomId> path".to_string())
        );
    }

    #[test]
    fn parse_ignores_query_strings() {
        // TS oracle: `…key?x=1` parses, query dropped (URL parser semantics).
        let parsed =
            parse_collab_link(&format!("wss://my.omp.sh/r/{ROOM}.{KEY_TEXT}?x=1")).expect("parses");
        assert_eq!(parsed.room_id, ROOM);
        assert_eq!(parsed.key, KEY);
    }

    #[test]
    fn parse_error_messages_match_ts() {
        // TS oracles (protocol.ts:307,317-323,329-335 and normalize).
        assert_eq!(
            parse_collab_link(&format!("wss://my.omp.sh/other/{ROOM}.{KEY_TEXT}")),
            Err("Collab link must contain a /r/<roomId> path".to_string())
        );
        assert_eq!(
            parse_collab_link(&format!("wss://my.omp.sh/r/{ROOM}")),
            Err("Collab link is missing the <key> part".to_string())
        );
        let short_key = crate::protocol::encode_base64_url(&KEY[..16]);
        assert_eq!(
            parse_collab_link(&format!("wss://my.omp.sh/r/{ROOM}.{short_key}")),
            Err("Collab link key must be 32 (view) or 48 (full) base64url bytes".to_string())
        );
        assert_eq!(
            parse_collab_link(&format!("ws://my.omp.sh/r/{ROOM}.{KEY_TEXT}")),
            Err(
                "relay link must be wss:// (plain ws:// is only allowed for localhost)".to_string()
            )
        );
        // http(s) web link whose fragment is NOT a collab link falls through
        // to direct validation → /r/ path error (protocol.ts:309-313,317).
        assert_eq!(
            parse_collab_link("http://127.0.0.1:7466/#garbage"),
            Err("Collab link must contain a /r/<roomId> path".to_string())
        );
        assert_eq!(
            parse_collab_link(":::"),
            Err("Invalid collab link: :::".to_string())
        );
    }

    #[test]
    fn format_parse_round_trip_all_shapes() {
        for relay in [
            DEFAULT_RELAY_URL,
            "ws://127.0.0.1:7466",
            "http://127.0.0.1:9224",
            "wss://relay.example.com:8443",
            "https://relay.example.com",
        ] {
            for token in [None, Some(TOKEN.as_slice())] {
                let room = generate_room_id();
                let link = format_collab_link(relay, &room, &KEY, token).expect("valid relay");
                let parsed = parse_collab_link(&link).expect("round-trips");
                assert_eq!(parsed.relay_url, normalize(relay), "{link}");
                assert_eq!(parsed.room_id, room);
                assert_eq!(parsed.key, KEY);
                assert_eq!(
                    parsed.write_token.map(Vec::into_boxed_slice),
                    token.map(Box::from)
                );
            }
        }
    }

    fn normalize(relay: &str) -> String {
        match relay {
            "http://127.0.0.1:9224" => "ws://127.0.0.1:9224".to_string(),
            "https://relay.example.com" => "wss://relay.example.com".to_string(),
            other => other.to_string(),
        }
    }

    #[test]
    fn parse_accepts_ts_generated_room_ids() {
        // 100 generateRoomId() samples from the bun oracle session: all parse
        // as room ids against the default relay.
        for id in [
            "TVRND9HgcxNGyXI19BWE1A",
            "efjL8UiuAf6cd8T4gRN9fw",
            "nAGillisdn93rAhbU3TNmg",
        ] {
            let parsed = parse_collab_link(&format!("{id}.{KEY_TEXT}")).expect("sample id parses");
            assert_eq!(parsed.room_id, id);
            assert_eq!(parsed.relay_url, "wss://my.omp.sh");
        }
    }
}
