//! Budgeted, lossless structured output for Code Mode tool results.
//!
//! Arrays of homogeneous flat objects become typed TSV. Other JSON values use
//! compact JSON. Default elision is never inferred from truthiness: callers
//! must explicitly enable it and supply JSON Schema defaults.

use serde_json::Value;

use super::util::{
    cut_bytes, save_artifact, truncate_head_lines, truncate_tail_lines, SpillConfig,
};

const TABLE_HEADER: &str = "# omt.table.v1";

/// Optional schema-directed projection. No fields are omitted by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOptions<'a> {
    pub schema: Option<&'a Value>,
    pub omit_schema_defaults: bool,
}

/// Encode without dropping any values, including `false`, `0`, and `null`.
#[must_use]
pub fn encode(value: &Value) -> String {
    encode_with_options(value, EncodeOptions::default())
}

/// Encode with explicit, schema-directed default omission when requested.
#[must_use]
pub fn encode_with_options(value: &Value, options: EncodeOptions<'_>) -> String {
    let projected = options
        .schema
        .filter(|_| options.omit_schema_defaults)
        .map(|schema| omit_schema_defaults(value, schema));
    let value = projected.as_ref().unwrap_or(value);
    table(value).unwrap_or_else(|| value.to_string())
}

/// Encode and, above the configured spill threshold, save the complete encoded
/// value as an artifact and return a bounded JSON envelope.
#[must_use]
pub fn render(value: &Value, cfg: &SpillConfig) -> String {
    budget(encode(value), cfg)
}

fn budget(full: String, cfg: &SpillConfig) -> String {
    if full.len() <= cfg.threshold {
        return full;
    }
    let full_output = save_artifact(&full);
    let (head, head_lines) = head_window(&full, cfg);
    let remaining = full.len().saturating_sub(head.len());
    let (tail, tail_lines) = tail_window(&full, remaining, cfg);
    let full_lines = count_lines(&full);
    let omitted_bytes = full.len().saturating_sub(head.len() + tail.len());
    let envelope = serde_json::json!({
        "head": head,
        "tail": tail,
        "omitted": {
            "bytes": omitted_bytes,
            "lines": full_lines.saturating_sub(head_lines + tail_lines),
        },
        "retained": {
            "bytes": head.len() + tail.len(),
            "lines": head_lines + tail_lines,
        },
        "full_output": full_output,
        "full_bytes": full.len(),
        "full_lines": full_lines,
    });
    envelope.to_string()
}

fn head_window(full: &str, cfg: &SpillConfig) -> (String, usize) {
    if cfg.head_bytes == 0 || cfg.tail_lines == 0 {
        return (String::new(), 0);
    }
    let truncated = truncate_head_lines(full, cfg.head_bytes, cfg.tail_lines);
    if truncated.output_lines > 0 {
        return (truncated.content, truncated.output_lines);
    }
    let head = cut_bytes(full, cfg.head_bytes).to_owned();
    let lines = count_lines(&head);
    (head, lines)
}

fn tail_window(full: &str, remaining: usize, cfg: &SpillConfig) -> (String, usize) {
    if cfg.tail_bytes == 0 || cfg.tail_lines == 0 || remaining == 0 {
        return (String::new(), 0);
    }
    let truncated = truncate_tail_lines(full, cfg.tail_bytes.min(remaining), cfg.tail_lines);
    (truncated.content, truncated.output_lines)
}

fn count_lines(text: &str) -> usize {
    text.bytes().filter(|byte| *byte == b'\n').count() + 1
}

fn omit_schema_defaults(value: &Value, schema: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let properties = schema.get("properties").and_then(Value::as_object);
            map.iter()
                .filter_map(|(name, value)| {
                    let property = properties.and_then(|properties| properties.get(name));
                    if property.and_then(|schema| schema.get("default")) == Some(value) {
                        return None;
                    }
                    let value = property.map_or_else(
                        || value.clone(),
                        |schema| omit_schema_defaults(value, schema),
                    );
                    Some((name.clone(), value))
                })
                .collect::<serde_json::Map<_, _>>()
                .into()
        }
        Value::Array(rows) => schema.get("items").map_or_else(
            || value.clone(),
            |schema| {
                Value::Array(
                    rows.iter()
                        .map(|row| omit_schema_defaults(row, schema))
                        .collect(),
                )
            },
        ),
        _ => value.clone(),
    }
}

/// A self-describing TSV table with a marker line, escaped column names, and a
/// scalar type line. Mixed column types, nested values, and shape mismatches
/// fall back to JSON rather than losing type information.
fn table(value: &Value) -> Option<String> {
    let rows = value.as_array()?;
    let first = rows.first()?.as_object()?;
    if first.is_empty() {
        return None;
    }
    let mut columns = Vec::with_capacity(first.len());
    for (name, value) in first {
        columns.push((name.as_str(), scalar_kind(value)?));
    }
    for row in rows {
        let object = row.as_object()?;
        if object.len() != columns.len()
            || !columns.iter().all(|(name, _)| object.contains_key(*name))
        {
            return None;
        }
        for (name, kind) in &columns {
            if scalar_kind(object.get(*name)?)? != *kind {
                return None;
            }
        }
    }
    let mut out = String::from(TABLE_HEADER);
    out.push('\n');
    for (index, (name, _)) in columns.iter().enumerate() {
        if index > 0 {
            out.push('\t');
        }
        push_escaped(&mut out, name);
    }
    out.push('\n');
    for (index, (_, kind)) in columns.iter().enumerate() {
        if index > 0 {
            out.push('\t');
        }
        out.push_str(kind);
    }
    for row in rows {
        out.push('\n');
        let object = row.as_object()?;
        for (index, (name, kind)) in columns.iter().enumerate() {
            if index > 0 {
                out.push('\t');
            }
            let cell = object.get(*name)?;
            match *kind {
                "string" => push_escaped(&mut out, cell.as_str()?),
                "null" => out.push_str("null"),
                _ => out.push_str(&cell.to_string()),
            }
        }
    }
    Some(out)
}

fn scalar_kind(value: &Value) -> Option<&'static str> {
    match value {
        Value::Null => Some("null"),
        Value::Bool(_) => Some("boolean"),
        Value::Number(_) => Some("number"),
        Value::String(_) => Some("string"),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn push_escaped(out: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::util::SESSION_FILE_TEST_LOCK;
    use serde_json::json;

    fn unescape(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(character) = chars.next() {
            if character != '\\' {
                out.push(character);
                continue;
            }
            match chars.next().expect("escape") {
                '\\' => out.push('\\'),
                't' => out.push('\t'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                other => panic!("unknown escape: {other}"),
            }
        }
        out
    }

    fn decode_table(encoded: &str) -> Option<Value> {
        let mut lines = encoded.split('\n');
        if lines.next()? != TABLE_HEADER {
            return None;
        }
        let names: Vec<String> = lines.next()?.split('\t').map(unescape).collect();
        let kinds: Vec<&str> = lines.next()?.split('\t').collect();
        if names.len() != kinds.len() {
            panic!("column/type count differs");
        }
        let mut rows = Vec::new();
        for line in lines {
            let cells: Vec<String> = line.split('\t').map(unescape).collect();
            if cells.len() != columns_len(&names) {
                panic!("row width differs");
            }
            let mut row = serde_json::Map::new();
            for ((name, kind), cell) in names.iter().zip(&kinds).zip(cells) {
                let value = match *kind {
                    "null" if cell == "null" => Value::Null,
                    "boolean" => Value::Bool(cell.parse().expect("boolean")),
                    "number" => serde_json::from_str(&cell).expect("number"),
                    "string" => Value::String(cell),
                    _ => panic!("invalid typed cell"),
                };
                row.insert(name.clone(), value);
            }
            rows.push(Value::Object(row));
        }
        Some(Value::Array(rows))
    }

    fn columns_len(names: &[String]) -> usize {
        names.len()
    }

    #[test]
    fn typed_tsv_is_lossless_for_defaults_column_names_and_escaping() {
        let original = json!([
            {"false": false, "zero": 0, "null": null, "text": "a\tb\nc\\d\r"},
            {"false": true, "zero": 0.0, "null": null, "text": "\"\\\n\t\r"}
        ]);
        let encoded = encode(&original);
        assert!(encoded.starts_with(TABLE_HEADER));
        assert_eq!(decode_table(&encoded), Some(original.clone()));
        assert!(encoded.len() < original.to_string().len());
    }

    #[test]
    fn nested_heterogeneous_and_mixed_type_rows_use_json_fallback() {
        for original in [
            json!([{"id": 1, "nested": {"x": false}}, {"id": 2, "nested": {"x": true}}]),
            json!([{"id": 1}, {"id": 2, "extra": 3}]),
            json!([{"value": null}, {"value": "set"}]),
            json!([]),
            json!({"false": false, "zero": 0, "null": null, "empty": []}),
        ] {
            assert_eq!(encode(&original), original.to_string());
        }
    }

    #[test]
    fn defaults_are_omitted_only_with_schema_and_explicit_opt_in() {
        let original = json!({
            "false": false,
            "zero": 0,
            "null": null,
            "nested": {"drop": 0, "keep": false}
        });
        let schema = json!({
            "type": "object",
            "properties": {
                "false": {"default": false},
                "zero": {"default": 0},
                "null": {"default": null},
                "nested": {"properties": {"drop": {"default": 0}, "keep": {"default": true}}}
            }
        });
        assert_eq!(encode(&original), original.to_string());
        assert_eq!(
            encode_with_options(
                &original,
                EncodeOptions {
                    schema: Some(&schema),
                    omit_schema_defaults: false
                }
            ),
            original.to_string()
        );
        assert_eq!(
            encode_with_options(
                &original,
                EncodeOptions {
                    schema: Some(&schema),
                    omit_schema_defaults: true
                }
            ),
            json!({"nested": {"keep": false}}).to_string()
        );
        assert_eq!(
            encode_with_options(
                &original,
                EncodeOptions {
                    schema: None,
                    omit_schema_defaults: true
                }
            ),
            original.to_string()
        );
    }

    #[test]
    fn oversized_table_spills_lossless_artifact_and_reports_measurements() {
        let _lock = SESSION_FILE_TEST_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let previous = omp_urls::global_session_file();
        omp_urls::set_global_session_file(Some(dir.path().join("session.jsonl")));
        let rows = Value::Array(
            (0..3000)
                .map(|id| json!({"id": id, "ok": false, "zero": 0, "nil": null, "text": "a\tb\nc\\d"}))
                .collect(),
        );
        let encoded = encode(&rows);
        let inline = render(&rows, &SpillConfig::default());
        let envelope: Value = serde_json::from_str(&inline).expect("bounded envelope");
        assert_eq!(envelope["full_output"], "artifact://0");
        assert!(envelope["head"]
            .as_str()
            .is_some_and(|head| !head.is_empty()));
        assert!(envelope["tail"]
            .as_str()
            .is_some_and(|tail| !tail.is_empty()));
        assert!(envelope["omitted"]["bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0));
        assert_eq!(envelope["full_bytes"], encoded.len());
        assert_eq!(
            envelope["retained"]["bytes"].as_u64(),
            Some(
                (envelope["head"].as_str().expect("head").len()
                    + envelope["tail"].as_str().expect("tail").len()) as u64
            )
        );
        let artifact = std::fs::read_to_string(dir.path().join("session/0.log")).expect("artifact");
        assert_eq!(artifact, encoded);
        assert_eq!(decode_table(&artifact), Some(rows.clone()));
        omp_urls::set_global_session_file(previous);

        let baseline = serde_json::to_string_pretty(&rows).expect("baseline JSON");
        let token_proxy = |bytes: usize| bytes.div_ceil(4);
        println!(
            "CONTAINMENT_MEASURE baseline_bytes={} baseline_tokens={} encoded_bytes={} encoded_tokens={} inline_bytes={} inline_tokens={}",
            baseline.len(),
            token_proxy(baseline.len()),
            encoded.len(),
            token_proxy(encoded.len()),
            inline.len(),
            token_proxy(inline.len()),
        );
        assert!(encoded.len() < baseline.len());
        assert!(inline.len() < encoded.len());
    }
}
