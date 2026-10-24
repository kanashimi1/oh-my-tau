//! Input-schema passthrough: a port of pi-ai's `normalizeSchemaForMCP`.
//!
//! omp registers MCP tools with their server-provided schema after this
//! normalization, which — for the MCP option set — is nearly a passthrough:
//!   - recursively strips `$schema` keys (`isMcpUnsupportedSchemaField`),
//!   - strips the `nullable` keyword (`stripNullableKeyword: true`),
//!   - folds an `anyOf`/`oneOf` whose branches are all bare `const` into a
//!     single `enum` (with an inferred scalar `type` when unambiguous),
//!   - dereferences same-document `$ref`s (omp upgrades to draft 2020-12 and
//!     inlines refs first; this port inlines local `#/...` refs and leaves
//!     external refs untouched, which covers every server config in the
//!     field — external refs are vanishingly rare in MCP `tools/list`).
//!
//! Everything else — `properties`, `required`, formats, defaults — is passed
//! through verbatim (the MCP option set disables field renaming, combiner
//! merging and description spilling).

use serde_json::Value;

/// Normalize an MCP tool input schema for registration, per omp.
#[must_use]
pub fn normalize_schema_for_mcp(schema: &Value) -> Value {
    let mut doc = schema.clone();
    inline_local_refs(&mut doc, schema);
    strip_and_fold(&doc)
}

fn inline_local_refs(node: &mut Value, root: &Value) {
    match node {
        Value::Object(map) => {
            if let Some(Value::String(r)) = map.get("$ref") {
                if let Some(target) = resolve_pointer(root, r) {
                    let mut replacement = target.clone();
                    inline_local_refs(&mut replacement, root);
                    *node = replacement;
                    return;
                }
            }
            for value in map.values_mut() {
                inline_local_refs(value, root);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                inline_local_refs(item, root);
            }
        }
        _ => {}
    }
}

fn resolve_pointer<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix('#')?;
    root.pointer(pointer)
}

const COMBINERS: [&str; 2] = ["anyOf", "oneOf"];

fn strip_and_fold(node: &Value) -> Value {
    match node {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            let folded_enum = const_fold(map);
            for (key, value) in map {
                if key == "$schema"
                    || key == "nullable"
                    || (folded_enum.is_some() && COMBINERS.contains(&key.as_str()))
                {
                    continue;
                }
                out.insert(key.clone(), strip_and_fold(value));
            }
            if let Some(enum_values) = folded_enum {
                out.insert("enum".to_string(), Value::Array(enum_values.clone()));
                if let Some(ty) = infer_enum_type(&enum_values) {
                    out.insert("type".to_string(), Value::String(ty.to_string()));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(strip_and_fold).collect()),
        other => other.clone(),
    }
}

/// Fold an `anyOf`/`oneOf` whose branches are all bare `{const}` into the
/// enum value list (omp's all-const combiner rule for the MCP option set).
fn const_fold(map: &serde_json::Map<String, Value>) -> Option<Vec<Value>> {
    for combiner in COMBINERS {
        if let Some(Value::Array(variants)) = map.get(combiner) {
            if !variants.is_empty()
                && variants.iter().all(|v| {
                    v.as_object()
                        .is_some_and(|o| o.len() == 1 && o.contains_key("const"))
                })
            {
                return Some(
                    variants
                        .iter()
                        .filter_map(|v| v.get("const"))
                        .cloned()
                        .collect(),
                );
            }
        }
    }
    None
}

fn infer_enum_type(values: &[Value]) -> Option<&'static str> {
    let mut ty: Option<&'static str> = None;
    for value in values {
        let this = match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        match ty {
            None => ty = Some(this),
            Some(t) if t != this => return None,
            _ => {}
        }
    }
    ty
}
