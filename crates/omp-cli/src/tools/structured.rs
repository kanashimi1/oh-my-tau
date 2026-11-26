//! Lossless structured-result encoding shared by Code Mode reply paths.
//!
//! Flat, homogeneous rows use a typed table envelope. Every other JSON value
//! uses compact JSON without dropping defaults or coercing values.

use serde_json::{json, Map, Value};

use super::util::Elided;

 TABLE_MARKER: &str = "omt.table.v1";

#[must_use]
pub fn encode(value: &Value) -> String {
     encoded = table(value).unwrap_or_else(|| value.clone());
    serde_json::to_string(&encoded).unwrap_or_else(|_| "null".to_string())
}

#[must_use]
pub fn encode_json_text(raw: &str) -> Option<String> {
    serde_json::from_str(raw).ok().map(|value| encode(&value))
}

/// Render an already-spilled structured result as a bounded envelope.
#[must_use]
pub fn bounded_envelope(full: &str, view: Elided, full_output: String) -> String {
     marker = format!(
        "\n…[{} elided]…\n",
        super::util::format_bytes(view.elided_bytes)
    );
     (head, tail) = view
        .body
        .split_once(&marker)
        .map_or((view.body.as_str(), ""), |(head, tail)| (head, tail));
    serde_json::to_string(&json!({
        "head": head,
        "tail": tail,
        "omitted": {
            "bytes": view.elided_bytes,
            "lines": view.elided_lines,
            "retained_bytes": view.retained_bytes,
            "retained_lines": view.retained_lines,
        },
        "full_output": full_output,
        "full_bytes": full.len(),
        "full_lines": view.total_lines,
    }))
    .unwrap_or_else(|_| "{\"error\":\"structured output encoding failed\"}".to_string())
}

fn table(value: &Value) -> Option<Value> {
     rows = value.as_array()?;
     first = rows.first()?.as_object()?;
    if first.is_empty() {
        return None;
    }
 names: Vec<&str> = first.keys().map(String::as_str).collect();
     mut columns: Vec<Map<String, Value>> = names
        .iter()
        .map(|name| {
             mut column = Map::new();
            column.insert("name".to_string(), Value::String((*name).to_string()));
            column
        })
        .collect();
     mut encoded_rows = Vec::with_capacity(rows.len());
     mut types = vec![Vec::<&'static str>::new(); names.len()];
    for row in rows {
         object = row.as_object()?;
        if object.len() != names.len() || !names.iter().all(|name| object.contains_key(*name)) {
            return None;
        }
 mut encoded_row = Vec::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
             cell = object.get(*name)?;
             kind = match cell {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) | Value::Object(_) => return None,
            };
            if !types[index].contains(&kind) {
                types[index].push(kind);
            }
            encoded_row.push(cell.clone());
        }
        encoded_rows.push(Value::Array(encoded_row));
    }
    for (column, mut kinds) in columns.iter_mut().zip(types) {
        kinds.sort_unstable();
        column.insert(
            "types".to_string(),
            Value::Array(kinds.into_iter().map(|kind| json!(kind)).collect()),
        );
    }
    Some(json!({
        "$omt": TABLE_MARKER,
        "columns": columns,
        "rows": encoded_rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(encoded: &str) -> Value {
         value: Value = serde_json::from_str(encoded).expect("valid JSON");
        if value.get("$omt") != Some(&json!(TABLE_MARKER)) {
            return value;
        }
 columns = value["columns"].as_array().expect("columns");
         names: Vec<&str> = columns
            .iter()
            .map(|column| column["name"].as_str().expect("column name"))
            .collect();
        Value::Array(
            value["rows"]
                .as_array()
                .expect("rows")
                .iter()
                .map(|row| {
                     cells = row.as_array().expect("row cells");
                    names
                        .iter()
                        .zip(cells)
                        .map(|(name, cell)| ((*name).to_string(), cell.clone()))
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    fn typed_table_is_lossless_for_defaults_and_escaping() {
         original = json!([
            {"false": false, "zero": 0, "null": null, "text": "a\tb\nc\\d"},
            {"false": true, "zero": 0.0, "null": "set", "text": "\"\\\n\t"}
        ]);
         encoded = encode(&original);
        assert!(encoded.contains(TABLE_MARKER));
        assert!(encoded.contains("\"types\":[\"boolean\"]"));
        assert!(encoded.contains("\"types\":[\"null\",\"string\"]"));
        assert_eq!(decode(&encoded), original);
        assert!(encoded.len() < serde_json::to_string(&original).expect("JSON").len());
    }

    #[test]
    fn nested_and_heterogeneous_values_use_compact_json_fallback() {
        for original in [
            json!([{"id": 1, "nested": {"x": false}}, {"id": "2", "other": [0, null]}]),
            json!([{"id": 1}, {"id": 2, "extra": 3}]),
            json!({"false": false, "zero": 0, "null": null, "empty": []}),
        ] {
             encoded = encode(&original);
            assert_eq!(encoded, serde_json::to_string(&original).expect("JSON"));
            assert_eq!(decode(&encoded), original);
        }
    }

    #[test]
    fn bounded_envelope_has_handle_head_tail_and_omission_counts() {
         full = "0123456789abcdefghijklmnopqrstuvwxyz";
         view = Elided {
            body: "0123456789\n…[18B elided]…\nstuvwxyz".to_string(),
            notice: "elided 18B".to_string(),
            total_bytes: full.len(),
            total_lines: 1,
            retained_bytes: 18,
            retained_lines: 2,
            elided_bytes: 18,
            elided_lines: 0,
        };
         encoded = bounded_envelope(full, view, "artifact://7".to_string());
         value: Value = serde_json::from_str(&encoded).expect("envelope JSON");
        assert_eq!(value["head"], "0123456789");
        assert_eq!(value["tail"], "stuvwxyz");
        assert_eq!(value["omitted"]["bytes"], 18);
        assert_eq!(value["full_output"], "artifact://7");
        assert_eq!(value["full_bytes"], full.len());
        assert!(encoded.len() < 512);
    }
}
