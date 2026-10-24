//! Python-compatible text helpers (`json.dumps(ensure_ascii=False)`, `str()`, truthiness,
//! char slicing) so the Rust port reproduces the Python builders byte for byte.

use serde_json::Value;

/// `s[:n]` on characters.
#[must_use]
pub fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Number of characters (`len(s)`).
#[must_use]
pub fn clen(s: &str) -> usize {
    s.chars().count()
}

/// `re.sub(r"\s+", " ", s).strip()`.
#[must_use]
pub fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Python truthiness of a JSON value.
#[must_use]
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn esc(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))
}

/// `json.dumps(v, ensure_ascii=False)` (separators `", "` / `": "`).
#[must_use]
pub fn dumps(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => esc(s),
        Value::Array(a) => format!("[{}]", a.iter().map(dumps).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", esc(k), dumps(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn py_quote(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_quote(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", py_quote(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `str(v)` of a decoded JSON value.
#[must_use]
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn char_slice_is_char_based() {
        assert_eq!(take_chars("héllo", 2), "hé");
        assert_eq!(take_chars("ab", 5), "ab");
    }

    #[test]
    fn py_str_matches_python() {
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(
            py_str(&json!(["a", 1, {"k": false}])),
            "['a', 1, {'k': False}]"
        );
        assert_eq!(py_str(&json!("it's")), "it's");
        assert_eq!(py_str(&json!(["it's"])), "[\"it's\"]");
    }

    #[test]
    fn dumps_matches_python() {
        assert_eq!(
            dumps(&json!({"a": [1, "é"], "b": null})),
            "{\"a\": [1, \"é\"], \"b\": null}"
        );
    }
}
