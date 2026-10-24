//! Selector peeling for internal URLs (TS `pi-tui/src/tools/read.ts`
//! `splitInternalUrlSel` + `url-scheme-host.ts`).

use crate::spec::SchemeSelectors;

/// Split a hierarchical `scheme://rest` URL into its lowercased scheme and the text after `://`.
pub fn split_url_scheme(input: &str) -> Option<(String, &str)> {
    let bytes = input.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    let mut i = 1;
    while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'+' | b'-' | b'.'))
    {
        i += 1;
    }
    let scheme = &input[..i];
    let rest = input[i..].strip_prefix("://")?;
    Some((scheme.to_lowercase(), rest))
}

/// One permissive selector chunk: `raw`, `conflicts`, `img`, a line-range
/// list (`1-20,30+5`), or a (possibly malformed) tail like `-10-5`.
fn is_selector_chunk(chunk: &str) -> bool {
    let lower = chunk.to_lowercase();
    if matches!(lower.as_str(), "raw" | "conflicts" | "img") {
        return true;
    }
    if let Some(body) = chunk.strip_prefix('-') {
        // `-N` or the malformed `-N-M` the TS grammar accepts for peeling.
        let digits = body.len() - body.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits == 0 {
            return false;
        }
        let rest = &body[digits..];
        return rest.is_empty()
            || rest
                .strip_prefix(['-', '+'])
                .is_some_and(|r| !r.is_empty() && r.chars().all(|c: char| c.is_ascii_digit()));
    }
    // `L?N(..|[-+])L?M?` chunks joined by commas.
    !chunk.is_empty()
        && chunk.split(',').all(|part| {
            let t = part
                .strip_prefix('L')
                .or_else(|| part.strip_prefix('l'))
                .unwrap_or(part);
            match t.split_once(['-', '+', '.']) {
                Some((start, end)) => {
                    let start = start
                        .strip_prefix('L')
                        .or_else(|| start.strip_prefix('l'))
                        .unwrap_or(start);
                    !start.is_empty()
                        && start.chars().all(|c: char| c.is_ascii_digit())
                        && (end.is_empty()
                            || end
                                .strip_prefix('L')
                                .or_else(|| end.strip_prefix('l'))
                                .unwrap_or(end)
                                .chars()
                                .all(|c: char| c.is_ascii_digit()))
                }
                None => !t.is_empty() && t.chars().all(|c: char| c.is_ascii_digit()),
            }
        })
}

/// Peel a trailing read-selector chain off an internal URL.
///
/// Only schemes whose spec declares `lines` selectors peel; `opaque`/
/// `none`/unregistered schemes pass through verbatim. A trailing `:N` on a
/// `portAuthority` scheme with no path is a port, not a selector.
pub fn split_selectors(
    raw_path: &str,
    spec_of: impl Fn(&str) -> Option<SchemeSelectors>,
    port_authority: impl Fn(&str) -> bool,
) -> (String, Option<String>) {
    let Some((scheme, rest)) = split_url_scheme(raw_path) else {
        return (raw_path.to_string(), None);
    };
    let Some(selectors) = spec_of(&scheme) else {
        return (raw_path.to_string(), None);
    };
    if selectors != SchemeSelectors::Lines {
        return (raw_path.to_string(), None);
    }
    if port_authority(&scheme) && !rest.contains('/') {
        return (raw_path.to_string(), None);
    }

    let scheme_end = raw_path.len() - rest.len();
    let mut path = raw_path;
    let mut chunks: Vec<&str> = Vec::new();
    while let Some(colon) = path.rfind(':') {
        if colon < scheme_end {
            break;
        }
        let tail = &path[colon + 1..];
        if !is_selector_chunk(tail) {
            break;
        }
        chunks.insert(0, tail);
        path = &path[..colon];
    }
    if chunks.is_empty() {
        return (raw_path.to_string(), None);
    }
    (path.to_string(), Some(chunks.join(":")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(_: &str) -> Option<SchemeSelectors> {
        Some(SchemeSelectors::Lines)
    }
    fn no_port(_: &str) -> bool {
        false
    }

    #[test]
    fn peels_chain() {
        let (path, sel) = split_selectors("artifact://3:raw:-100-5", lines, no_port);
        assert_eq!(path, "artifact://3");
        assert_eq!(sel.as_deref(), Some("raw:-100-5"));
    }

    #[test]
    fn peels_iteratively() {
        let (path, sel) = split_selectors("local://f.txt:1-20:raw", lines, no_port);
        assert_eq!(path, "local://f.txt");
        assert_eq!(sel.as_deref(), Some("1-20:raw"));
    }

    #[test]
    fn no_match_passes_through() {
        let (path, sel) = split_selectors("local://f.txt:abc", lines, no_port);
        assert_eq!(path, "local://f.txt:abc");
        assert_eq!(sel, None);
    }

    #[test]
    fn port_not_selector() {
        let port = |_: &str| true;
        let (path, sel) = split_selectors("ssh://host:2222", lines, port);
        assert_eq!(path, "ssh://host:2222");
        assert_eq!(sel, None);
    }
}
