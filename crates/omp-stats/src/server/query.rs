//! Minimal `URL` / `URLSearchParams` equivalent for the router.

/// A parsed request target: decoded path plus the decoded query pairs.
pub(crate) struct Target {
    pub path: String,
    pairs: Vec<(String, String)>,
}

impl Target {
    /// Split `path?query`. The path is percent-decoded like `URL.pathname`
    /// is *not* (TS compares the raw pathname), so it stays raw.
    pub fn parse(path_and_query: &str) -> Self {
        let without_fragment = path_and_query.split('#').next().unwrap_or("");
        let (path, query) = match without_fragment.split_once('?') {
            Some((p, q)) => (p, q),
            None => (without_fragment, ""),
        };
        let pairs = query
            .split('&')
            .filter(|s| !s.is_empty())
            .map(|pair| match pair.split_once('=') {
                Some((k, v)) => (decode(k), decode(v)),
                None => (decode(pair), String::new()),
            })
            .collect();
        Self {
            path: path.to_string(),
            pairs,
        }
    }

    /// `searchParams.get(name)`: first match, `None` when absent.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// `application/x-www-form-urlencoded` decoding (`+` is a space, bad escapes stay literal).
fn decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len()
                && hex(bytes[i + 1]).is_some()
                && hex(bytes[i + 2]).is_some() =>
            {
                out.push(hex(bytes[i + 1]).unwrap_or(0) << 4 | hex(bytes[i + 2]).unwrap_or(0));
                i += 2;
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// JS `parseInt(s, 10)`: optional sign, leading digits, rest ignored; `None` = NaN.
pub(crate) fn parse_int(s: &str) -> Option<i64> {
    let t = s.trim_start();
    let (neg, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    if end == 0 {
        return None;
    }
    let n: i64 = digits[..end].parse().ok()?;
    Some(if neg { -n } else { n })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_pairs() {
        let t = Target::parse("/api/x?range=7d&q=a%20b+c&flag&file=%2Fa%2Fb");
        assert_eq!(t.path, "/api/x");
        assert_eq!(t.get("range"), Some("7d"));
        assert_eq!(t.get("q"), Some("a b c"));
        assert_eq!(t.get("flag"), Some(""));
        assert_eq!(t.get("file"), Some("/a/b"));
        assert_eq!(t.get("nope"), None);
    }

    #[test]
    fn parse_int_like_js() {
        assert_eq!(parse_int("12abc"), Some(12));
        assert_eq!(parse_int("-3"), Some(-3));
        assert_eq!(parse_int("abc"), None);
        assert_eq!(parse_int(""), None);
    }
}
