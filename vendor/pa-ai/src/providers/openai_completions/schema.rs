//! OpenAI-style strict tool-schema adaptation (TS `adaptSchemaForStrict` →
//! `tryEnforceStrictSchema`: `hasUnrepresentableStrictObjectMap`, `sanitizeSchemaForStrictMode`,
//! `enforceStrictSchema`, `ai/src/utils/schema/normalize.ts`).
//!
//! Input schemas are already draft 2020-12 wire schemas (the TS `toolWireSchema` stage), so the
//! draft-07 upgrade is not repeated. Key order follows the TS object construction (the strict
//! output is order-sensitive on the wire).

use serde_json::{json, Map, Value};

const NON_STRUCTURAL: [&str; 38] = [
    "format",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "multipleOf",
    "$schema",
    "examples",
    "default",
    "title",
    "$comment",
    "if",
    "then",
    "else",
    "not",
    "unevaluatedProperties",
    "unevaluatedItems",
    "patternProperties",
    "propertyNames",
    "contains",
    "minContains",
    "maxContains",
    "dependentRequired",
    "dependentSchemas",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "deprecated",
    "readOnly",
    "writeOnly",
    "minProperties",
    "maxProperties",
];
const COMBINATORS: [&str; 3] = ["anyOf", "allOf", "oneOf"];

fn is_unrepresentable_branch(v: &Value) -> bool {
    v.is_boolean() || v.as_object().is_some_and(Map::is_empty)
}

/// Schemas strict mode cannot represent (open key sets, boolean/empty branches).
fn has_unrepresentable_strict_object_map(schema: &Map<String, Value>) -> bool {
    if schema
        .get("patternProperties")
        .and_then(Value::as_object)
        .is_some_and(|m| !m.is_empty())
    {
        return true;
    }
    match schema.get("additionalProperties") {
        Some(Value::Bool(true)) => return true,
        Some(Value::Object(_)) => return true,
        _ => {}
    }
    let check = |v: &Value| {
        is_unrepresentable_branch(v)
            || v.as_object()
                .is_some_and(has_unrepresentable_strict_object_map)
    };
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        if props.values().any(check) {
            return true;
        }
    }
    if let Some(items) = schema.get("items") {
        if is_unrepresentable_branch(items) {
            return true;
        }
        match items {
            Value::Object(o) => {
                if has_unrepresentable_strict_object_map(o) {
                    return true;
                }
            }
            Value::Array(a) => {
                if a.iter().any(check) {
                    return true;
                }
            }
            _ => {}
        }
    }
    if let Some(a) = schema.get("prefixItems").and_then(Value::as_array) {
        if a.iter().any(check) {
            return true;
        }
    }
    for key in COMBINATORS {
        if let Some(a) = schema.get(key).and_then(Value::as_array) {
            if a.iter().any(check) {
                return true;
            }
        }
    }
    for key in ["$defs", "definitions"] {
        if let Some(d) = schema.get(key).and_then(Value::as_object) {
            if d.values().any(check) {
                return true;
            }
        }
    }
    false
}

fn resolve_strict_ref<'a>(
    root: &'a Map<String, Value>,
    reference: &str,
) -> Option<&'a Map<String, Value>> {
    let path = reference.strip_prefix("#/")?;
    let mut cursor: &Map<String, Value> = root;
    for raw in path.split('/') {
        let segment = raw.replace("~1", "/").replace("~0", "~");
        cursor = cursor.get(&segment)?.as_object()?;
    }
    Some(cursor)
}

fn primitive_type_of(v: &Value) -> Option<&'static str> {
    match v {
        Value::Null => Some("null"),
        Value::String(_) => Some("string"),
        Value::Number(_) => Some("number"),
        Value::Bool(_) => Some("boolean"),
        _ => None,
    }
}

fn schema_type_accepts(ty: &str, v: &Value) -> bool {
    match ty {
        "null" => v.is_null(),
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.as_f64().is_some_and(|f| f.fract() == 0.0) && v.is_number(),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        _ => true,
    }
}

fn narrow_enum_to_type(schema: &mut Map<String, Value>, ty: &str) -> bool {
    let Some(values) = schema.get("enum").and_then(Value::as_array) else {
        return true;
    };
    let narrowed: Vec<Value> = values
        .iter()
        .filter(|v| schema_type_accepts(ty, v))
        .cloned()
        .collect();
    if narrowed.is_empty() {
        return false;
    }
    if narrowed.len() != values.len() {
        schema.insert("enum".into(), Value::Array(narrowed));
    }
    true
}

fn infer_strict_primitive_type(node: &Map<String, Value>) -> Option<&'static str> {
    let values: Vec<&Value> = match node.get("enum").and_then(Value::as_array) {
        Some(a) => a.iter().collect(),
        None => node.get("const").map(|c| vec![c]).unwrap_or_default(),
    };
    if values.is_empty() {
        return None;
    }
    let mut inferred: Option<&'static str> = None;
    for v in values {
        let t = primitive_type_of(v)?;
        match inferred {
            None => inferred = Some(t),
            Some(i) if i != t => return None,
            _ => {}
        }
    }
    inferred
}

fn sanitize_value(v: &Value, root: &Map<String, Value>) -> Value {
    match v {
        Value::Object(o) => Value::Object(sanitize(o, root)),
        other => other.clone(),
    }
}

fn sanitize_array(v: &[Value], root: &Map<String, Value>) -> Value {
    Value::Array(v.iter().map(|e| sanitize_value(e, root)).collect())
}

/// First pass of strict-mode preparation (TS `sanitizeSchemaForStrictMode`).
fn sanitize(schema: &Map<String, Value>, root: &Map<String, Value>) -> Map<String, Value> {
    // `$ref` with siblings: inline the resolved definition (siblings win).
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        if schema.keys().any(|k| k != "$ref") {
            if let Some(resolved) = resolve_strict_ref(root, reference) {
                let mut merged = resolved.clone();
                for (k, v) in schema {
                    if k != "$ref" {
                        merged.insert(k.clone(), v.clone());
                    }
                }
                return sanitize(&merged, root);
            }
        }
    }
    // Collapse a single-element `allOf` (the inlined entry's keys win).
    if let Some(Value::Array(all_of)) = schema.get("allOf") {
        if let [Value::Object(sole)] = all_of.as_slice() {
            let mut merged = schema.clone();
            merged.shift_remove("allOf");
            for (k, v) in sole {
                merged.insert(k.clone(), v.clone());
            }
            return sanitize(&merged, root);
        }
    }

    if let Some(Value::Array(type_array)) = schema.get("type") {
        let variants_types: Vec<&str> = type_array.iter().filter_map(Value::as_str).collect();
        let mut without_type = schema.clone();
        without_type.shift_remove("type");
        let sanitized_without_type = sanitize(&without_type, root);
        if variants_types.is_empty() {
            return sanitized_without_type;
        }
        let mut variant_base = sanitized_without_type.clone();
        let description = variant_base.shift_remove("description");
        let mut variants: Vec<Value> = Vec::new();
        for vt in variants_types {
            let mut variant = variant_base.clone();
            variant.insert("type".into(), json!(vt));
            if vt != "object" {
                variant.shift_remove("properties");
                variant.shift_remove("required");
                variant.shift_remove("additionalProperties");
            }
            if vt != "array" {
                variant.shift_remove("items");
            }
            if !narrow_enum_to_type(&mut variant, vt) {
                continue;
            }
            variants.push(Value::Object(sanitize(&variant, root)));
        }
        if variants.is_empty() {
            return sanitized_without_type;
        }
        if variants.len() == 1 {
            let Value::Object(mut sole) = variants.remove(0) else {
                unreachable!("variants are objects")
            };
            if let Some(d) = description {
                if !sole.contains_key("description") {
                    sole.insert("description".into(), d);
                }
            }
            return sole;
        }
        let mut result = Map::new();
        result.insert("anyOf".into(), Value::Array(variants));
        if let Some(d) = description {
            result.insert("description".into(), d);
        }
        return result;
    }

    // Scalar `type`: walk the keys, rewriting or stripping per strict-mode rules.
    let mut sanitized = Map::new();
    for (key, value) in schema {
        if NON_STRUCTURAL.contains(&key.as_str())
            || key == "type"
            || key == "const"
            || key == "nullable"
        {
            continue;
        }
        match (key.as_str(), value) {
            ("properties", Value::Object(props)) => {
                let mut out = Map::new();
                for (name, p) in props {
                    out.insert(name.clone(), sanitize_value(p, root));
                }
                sanitized.insert("properties".into(), Value::Object(out));
            }
            ("items", Value::Object(o)) => {
                sanitized.insert("items".into(), Value::Object(sanitize(o, root)));
            }
            ("items", Value::Array(a)) => {
                sanitized.insert("items".into(), sanitize_array(a, root));
            }
            ("prefixItems", Value::Array(a)) => {
                sanitized.insert("prefixItems".into(), sanitize_array(a, root));
            }
            (k, Value::Array(a)) if COMBINATORS.contains(&k) => {
                sanitized.insert(k.to_string(), sanitize_array(a, root));
            }
            ("$defs" | "definitions", Value::Object(defs)) => {
                let mut out = Map::new();
                for (name, d) in defs {
                    out.insert(name.clone(), sanitize_value(d, root));
                }
                sanitized.insert(key.clone(), Value::Object(out));
            }
            ("additionalProperties", _) => {}
            ("description", Value::String(text)) if schema.contains_key("default") => {
                let default = &schema["default"];
                let formatted = match default {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let description = if text.contains("(default:") {
                    text.clone()
                } else {
                    format!("{text} (default: {formatted})")
                };
                sanitized.insert("description".into(), Value::String(description));
            }
            _ => {
                sanitized.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(const_val) = schema.get("const") {
        let mut existing = sanitized
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !existing.contains(const_val) {
            existing.push(const_val.clone());
        }
        sanitized.insert("enum".into(), Value::Array(existing));
    }
    if let Some(Value::String(ty)) = schema.get("type") {
        sanitized.insert("type".into(), Value::String(ty.clone()));
    }
    if !sanitized.contains_key("type") && sanitized.get("properties").is_some_and(Value::is_object)
    {
        sanitized.insert("type".into(), json!("object"));
    }
    if !sanitized.contains_key("type")
        && (sanitized.contains_key("items") || sanitized.contains_key("prefixItems"))
    {
        sanitized.insert("type".into(), json!("array"));
    }
    if !sanitized.contains_key("type") {
        if let Some(inferred) = infer_strict_primitive_type(&sanitized) {
            sanitized.insert("type".into(), json!(inferred));
        }
    }
    if schema.get("nullable") == Some(&Value::Bool(true)) {
        let mut without = sanitized;
        let description = without.shift_remove("description");
        let mut wrapper = Map::new();
        wrapper.insert(
            "anyOf".into(),
            json!([Value::Object(without), { "type": "null" }]),
        );
        if let Some(d) = description {
            wrapper.insert("description".into(), d);
        }
        return wrapper;
    }
    sanitized
}

fn is_pure_any_of(v: &Value) -> bool {
    v.as_object().is_some_and(|o| {
        o.get("anyOf").is_some_and(Value::is_array)
            && o.keys().all(|k| k == "anyOf" || k == "description")
    })
}

struct NotRepresentable;

fn enforce_value(v: &Value) -> Result<Value, NotRepresentable> {
    match v {
        Value::Object(o) => Ok(Value::Object(enforce(o)?)),
        other => Ok(other.clone()),
    }
}

/// TS `enforceStrictSchema`: closed objects, every property required, optional ones nullable.
fn enforce(schema: &Map<String, Value>) -> Result<Map<String, Value>, NotRepresentable> {
    let mut result = schema.clone();
    if result.get("type").and_then(Value::as_str) == Some("object") {
        result.insert("additionalProperties".into(), json!(false));
        let props = result
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let original_required: Vec<String> = result
            .get("required")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let mut strict_props = Map::new();
        for (key, value) in &props {
            let processed = match value {
                Value::Object(_) => enforce_value(value)?,
                other => other.clone(),
            };
            if !original_required.contains(key) {
                let already_nullable = processed.as_object().is_some_and(|o| {
                    o.get("anyOf").and_then(Value::as_array).is_some_and(|a| {
                        a.iter().any(|v| {
                            v.as_object()
                                .is_some_and(|o| o.get("type") == Some(&json!("null")))
                        })
                    })
                });
                if already_nullable {
                    strict_props.insert(key.clone(), processed);
                    continue;
                }
                if is_pure_any_of(&processed) {
                    let mut o = processed.as_object().cloned().unwrap_or_default();
                    let mut any_of = o
                        .get("anyOf")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    any_of.push(json!({ "type": "null" }));
                    o.insert("anyOf".into(), Value::Array(any_of));
                    strict_props.insert(key.clone(), Value::Object(o));
                    continue;
                }
                if let Some(o) = processed.as_object() {
                    if o.get("description").is_some_and(Value::is_string) {
                        let mut without = o.clone();
                        let description =
                            without.shift_remove("description").unwrap_or(Value::Null);
                        let mut wrapper = Map::new();
                        wrapper.insert(
                            "anyOf".into(),
                            json!([Value::Object(without), { "type": "null" }]),
                        );
                        wrapper.insert("description".into(), description);
                        strict_props.insert(key.clone(), Value::Object(wrapper));
                        continue;
                    }
                }
                strict_props.insert(
                    key.clone(),
                    json!({ "anyOf": [processed, { "type": "null" }] }),
                );
                continue;
            }
            strict_props.insert(key.clone(), processed);
        }
        let required: Vec<Value> = strict_props.keys().map(|k| json!(k)).collect();
        result.insert("properties".into(), Value::Object(strict_props));
        result.insert("required".into(), Value::Array(required));
    }
    if let Some(items) = result.get("items").cloned() {
        match &items {
            Value::Array(a) => {
                let mapped: Result<Vec<Value>, _> = a.iter().map(enforce_value).collect();
                result.insert("items".into(), Value::Array(mapped?));
            }
            Value::Object(_) => {
                result.insert("items".into(), enforce_value(&items)?);
            }
            _ => {}
        }
    }
    if let Some(Value::Array(a)) = result.get("prefixItems").cloned() {
        let mapped: Result<Vec<Value>, _> = a.iter().map(enforce_value).collect();
        result.insert("prefixItems".into(), Value::Array(mapped?));
    }
    for key in COMBINATORS {
        if let Some(Value::Array(a)) = result.get(key).cloned() {
            let mapped: Result<Vec<Value>, _> = a.iter().map(enforce_value).collect();
            result.insert(key.into(), Value::Array(mapped?));
        }
    }
    // Splice nested pure unions into the parent `anyOf`.
    if let Some(Value::Array(any_of)) = result.get("anyOf").cloned() {
        if any_of.iter().any(is_pure_any_of) {
            let mut flattened: Vec<Value> = Vec::new();
            for branch in any_of {
                if !is_pure_any_of(&branch) {
                    flattened.push(branch);
                    continue;
                }
                let b = branch.as_object().expect("pure anyOf node is an object");
                if let Some(Value::Array(inner)) = b.get("anyOf") {
                    flattened.extend(inner.iter().cloned());
                }
                if let Some(d) = b.get("description").filter(|d| d.is_string()) {
                    if !result.contains_key("description") {
                        result.insert("description".into(), d.clone());
                    }
                }
            }
            result.insert("anyOf".into(), Value::Array(flattened));
        }
    }
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(defs)) = result.get(key).cloned() {
            let mut next = Map::new();
            for (name, def) in &defs {
                next.insert(
                    name.clone(),
                    match def {
                        Value::Object(_) => enforce_value(def)?,
                        other => other.clone(),
                    },
                );
            }
            result.insert(key.into(), Value::Object(next));
        }
    }
    if !result.contains_key("type") {
        if let Some(inferred) = infer_strict_primitive_type(&result) {
            result.insert("type".into(), json!(inferred));
        }
    }
    if !result.contains_key("type")
        && !result.contains_key("$ref")
        && !COMBINATORS
            .iter()
            .any(|k| result.get(*k).is_some_and(Value::is_array))
        && !result.get("not").is_some_and(Value::is_object)
    {
        return Err(NotRepresentable);
    }
    Ok(result)
}

/// TS `adaptSchemaForStrict`: the (possibly strict-enforced) schema and whether strict mode
/// applies. A schema strict mode cannot represent comes back unchanged with `strict: false`.
pub(crate) fn adapt_schema_for_strict(schema: &Value, want_strict: bool) -> (Value, bool) {
    if !want_strict {
        return (schema.clone(), false);
    }
    let Some(obj) = schema.as_object() else {
        return (schema.clone(), false);
    };
    if has_unrepresentable_strict_object_map(obj) {
        return (schema.clone(), false);
    }
    let sanitized = sanitize(obj, obj);
    match enforce(&sanitized) {
        Ok(enforced) => (Value::Object(enforced), true),
        Err(NotRepresentable) => (schema.clone(), false),
    }
}

// ---------------------------------------------------------------------------
// Tool wire schema (TS `toolWireSchema` post-processing) and host flavors
// ---------------------------------------------------------------------------

const NULLABLE_SCALAR_TYPES: [&str; 4] = ["string", "number", "integer", "boolean"];
const SCHEMA_DEFINING_SIBLINGS: [&str; 19] = [
    "$ref",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "contains",
    "enum",
    "if",
    "items",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "properties",
    "propertyNames",
    "then",
    "else",
    "unevaluatedItems",
    "unevaluatedProperties",
];
const WIRE_VALUE_KEYS: [&str; 10] = [
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "items",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
    "not",
];
const WIRE_MAP_KEYS: [&str; 4] = ["properties", "patternProperties", "$defs", "definitions"];
const WIRE_ARRAY_KEYS: [&str; 4] = ["anyOf", "oneOf", "allOf", "prefixItems"];

fn has_schema_defining_sibling(o: &Map<String, Value>) -> bool {
    o.keys()
        .any(|k| k != "anyOf" && SCHEMA_DEFINING_SIBLINGS.contains(&k.as_str()))
}

fn is_null_variant(o: &Map<String, Value>) -> bool {
    o.get("type") == Some(&json!("null")) && o.len() == 1
}

fn rewrite_nullable_scalar_any_of(o: &mut Map<String, Value>) {
    if has_schema_defining_sibling(o) {
        return;
    }
    let Some(Value::Array(variants)) = o.get("anyOf") else {
        return;
    };
    if variants.len() != 2 {
        return;
    }
    let mut scalar: Option<Map<String, Value>> = None;
    let mut scalar_type: Option<String> = None;
    let mut saw_null = false;
    for variant in variants {
        let Some(v) = variant.as_object() else { return };
        if is_null_variant(v) {
            if saw_null {
                return;
            }
            saw_null = true;
            continue;
        }
        let ty = v.get("type").and_then(Value::as_str);
        match ty {
            Some(t) if NULLABLE_SCALAR_TYPES.contains(&t) && scalar.is_none() => {
                scalar = Some(v.clone());
                scalar_type = Some(t.to_string());
            }
            _ => return,
        }
    }
    let (Some(scalar), Some(scalar_type), true) = (scalar, scalar_type, saw_null) else {
        return;
    };
    o.shift_remove("anyOf");
    for (k, v) in &scalar {
        if k == "type" || k == "enum" || k == "const" || o.contains_key(k) {
            continue;
        }
        o.insert(k.clone(), v.clone());
    }
    if let Some(c) = scalar.get("const") {
        o.insert("enum".into(), json!([c, null]));
    } else if let Some(Value::Array(e)) = scalar.get("enum") {
        let mut e = e.clone();
        if !e.contains(&Value::Null) {
            e.push(Value::Null);
        }
        o.insert("enum".into(), Value::Array(e));
    }
    o.insert("type".into(), json!([scalar_type, "null"]));
}

fn homogeneous_enum_scalar_type(values: &[Value]) -> Option<&'static str> {
    let mut inferred: Option<&'static str> = None;
    for v in values {
        let t = match v {
            Value::String(_) => "string",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            _ => return None,
        };
        match inferred {
            None => inferred = Some(t),
            Some(i) if i != t => return None,
            _ => {}
        }
    }
    inferred
}

fn infer_bare_enum_scalar_type(o: &mut Map<String, Value>) {
    if o.contains_key("type") {
        return;
    }
    let Some(Value::Array(e)) = o.get("enum") else {
        return;
    };
    if let Some(t) = homogeneous_enum_scalar_type(e) {
        o.insert("type".into(), json!(t));
    }
}

fn collapse_const_union_any_of(o: &mut Map<String, Value>) {
    if has_schema_defining_sibling(o) || o.contains_key("type") {
        return;
    }
    let Some(Value::Array(variants)) = o.get("anyOf") else {
        return;
    };
    if variants.len() < 2 {
        return;
    }
    let mut values = Vec::new();
    let mut branch_description: Option<String> = None;
    let mut described = 0usize;
    for variant in variants {
        let Some(v) = variant.as_object() else { return };
        let Some(c) = v.get("const") else { return };
        if v.keys().any(|k| k != "const" && k != "description") {
            return;
        }
        if let Some(d) = v.get("description").and_then(Value::as_str) {
            if described == 0 {
                branch_description = Some(d.to_string());
            } else if branch_description.as_deref() != Some(d) {
                return;
            }
            described += 1;
        }
        values.push(c.clone());
    }
    if described != 0 && described != variants.len() {
        return;
    }
    if described == variants.len() {
        if let Some(d) = o.get("description").and_then(Value::as_str) {
            if Some(d) != branch_description.as_deref() {
                return;
            }
        }
    }
    let Some(scalar_type) = homogeneous_enum_scalar_type(&values) else {
        return;
    };
    o.shift_remove("anyOf");
    o.insert("type".into(), json!(scalar_type));
    o.insert("enum".into(), Value::Array(values));
    if !o.get("description").is_some_and(Value::is_string) {
        if let Some(d) = branch_description {
            o.insert("description".into(), json!(d));
        }
    }
}

fn walk_wire(node: &mut Value) {
    match node {
        Value::Array(a) => a.iter_mut().for_each(walk_wire),
        Value::Object(o) => {
            rewrite_nullable_scalar_any_of(o);
            infer_bare_enum_scalar_type(o);
            collapse_const_union_any_of(o);
            for v in o.values_mut() {
                walk_wire(v);
            }
        }
        _ => {}
    }
}

fn is_empty_object(v: &Value) -> bool {
    v.as_object().is_some_and(Map::is_empty)
}

/// TS `normalizeEmptySchemas`: `{}` (unconstrained) becomes `true` in every schema position.
fn normalize_empty_schemas(node: &mut Value) {
    match node {
        Value::Array(a) => a.iter_mut().for_each(normalize_empty_schemas),
        Value::Object(o) => {
            for key in WIRE_VALUE_KEYS {
                if o.get(key).is_some_and(is_empty_object) {
                    o.insert(key.into(), Value::Bool(true));
                }
            }
            for key in WIRE_MAP_KEYS {
                if let Some(Value::Object(m)) = o.get_mut(key) {
                    for v in m.values_mut() {
                        if is_empty_object(v) {
                            *v = Value::Bool(true);
                        }
                    }
                }
            }
            for key in WIRE_ARRAY_KEYS {
                if let Some(Value::Array(a)) = o.get_mut(key) {
                    for v in a.iter_mut() {
                        if is_empty_object(v) {
                            *v = Value::Bool(true);
                        }
                    }
                }
            }
            for v in o.values_mut() {
                normalize_empty_schemas(v);
            }
        }
        _ => {}
    }
}

/// TS `toolWireSchema` for a raw JSON-schema tool: nullable-scalar `anyOf` collapse, bare-enum
/// typing, const-union collapse, and `{}` → `true`. (Parameters are already draft 2020-12.)
pub(crate) fn tool_wire_schema(schema: &Value) -> Value {
    let mut wire = schema.clone();
    walk_wire(&mut wire);
    normalize_empty_schemas(&mut wire);
    wire
}

/// TS `flattenExclusiveRequiredRootUnion` (xAI): drop a root union made only of typeless
/// required-key fragments.
pub(crate) fn flatten_exclusive_required_root_union(schema: &Value) -> Value {
    let Some(obj) = schema.as_object() else {
        return schema.clone();
    };
    let union_key = if obj.get("anyOf").is_some_and(Value::is_array) {
        "anyOf"
    } else if obj.get("oneOf").is_some_and(Value::is_array) {
        "oneOf"
    } else {
        return schema.clone();
    };
    let Some(Value::Array(union)) = obj.get(union_key) else {
        return schema.clone();
    };
    if union.is_empty() {
        return schema.clone();
    }
    let typed_object = match obj.get("type") {
        Some(Value::String(s)) => s == "object",
        Some(Value::Array(a)) => a.iter().any(|v| v == "object"),
        _ => false,
    };
    if !typed_object && !obj.get("properties").is_some_and(Value::is_object) {
        return schema.clone();
    }
    let exclusive = |branch: &Value| {
        let Some(b) = branch.as_object() else {
            return false;
        };
        if b.contains_key("type") {
            return false;
        }
        let Some(Value::Array(req)) = b.get("required") else {
            return false;
        };
        !req.is_empty()
            && req
                .iter()
                .all(|n| n.as_str().is_some_and(|s| !s.is_empty()))
            && b.keys()
                .all(|k| k == "required" || k == "description" || k == "title")
    };
    if !union.iter().all(exclusive) {
        return schema.clone();
    }
    let mut flattened = obj.clone();
    flattened.shift_remove(union_key);
    Value::Object(flattened)
}

// ---- Moonshot (MFJS) normalization — TS `normalizeSchemaForMoonshot`

const TYPE_KEYS_ARRAY: [&str; 9] = [
    "items",
    "prefixItems",
    "contains",
    "minContains",
    "maxContains",
    "minItems",
    "maxItems",
    "uniqueItems",
    "unevaluatedItems",
];
const TYPE_KEYS_OBJECT: [&str; 10] = [
    "properties",
    "required",
    "additionalProperties",
    "patternProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
];
const TYPE_KEYS_STRING: [&str; 6] = [
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "contentEncoding",
    "contentMediaType",
];
const TYPE_KEYS_NUMBER: [&str; 5] = [
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

fn type_specific_keys(ty: &str) -> &'static [&'static str] {
    match ty {
        "array" => &TYPE_KEYS_ARRAY,
        "object" => &TYPE_KEYS_OBJECT,
        "string" => &TYPE_KEYS_STRING,
        "number" | "integer" => &TYPE_KEYS_NUMBER,
        _ => &[],
    }
}

fn is_cca_type_specific(key: &str) -> bool {
    TYPE_KEYS_ARRAY.contains(&key)
        || TYPE_KEYS_OBJECT.contains(&key)
        || TYPE_KEYS_STRING.contains(&key)
        || TYPE_KEYS_NUMBER.contains(&key)
}

const LIFTABLE_TO_DESCRIPTION: [&str; 16] = [
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "default",
    "examples",
];

fn moonshot_unsupported(key: &str) -> bool {
    key != "default" && (NON_STRUCTURAL.contains(&key) || key == "prefixItems")
}

#[derive(Clone, Copy, PartialEq)]
enum ChildKind {
    Schema,
    Map,
}

fn classify_schema_child(key: &str, value: &Value, inside_map: bool) -> Option<ChildKind> {
    if inside_map {
        return Some(ChildKind::Schema);
    }
    match key {
        "properties" | "patternProperties" | "dependencies" | "dependentSchemas" | "$defs"
        | "definitions" => Some(ChildKind::Map),
        "items" | "additionalItems" | "unevaluatedItems" | "not" | "if" | "then" | "else"
        | "contains" | "propertyNames" | "contentSchema" | "anyOf" | "oneOf" | "allOf"
        | "prefixItems" => Some(ChildKind::Schema),
        "additionalProperties" | "unevaluatedProperties" if value.is_object() => {
            Some(ChildKind::Schema)
        }
        _ => None,
    }
}

fn infer_json_schema_type(v: &Value) -> Option<&'static str> {
    Some(match v {
        Value::Null => "null",
        Value::Array(_) => "array",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Object(_) => "object",
    })
}

fn push_enum_value(values: &mut Vec<Value>, v: &Value) {
    if !values.contains(v) {
        values.push(v.clone());
    }
}

fn split_type_array_into_any_of(obj: Map<String, Value>) -> Map<String, Value> {
    let Some(Value::Array(types_raw)) = obj.get("type") else {
        return obj;
    };
    if obj.get("anyOf").is_some_and(Value::is_array) {
        return obj;
    }
    let mut types: Vec<String> = Vec::new();
    for t in types_raw {
        if let Some(s) = t.as_str() {
            if s != "null" && !types.iter().any(|x| x == s) {
                types.push(s.to_string());
            }
        }
    }
    if types.len() < 2 {
        return obj;
    }
    let mut branches: Vec<Map<String, Value>> = types
        .iter()
        .map(|t| {
            let mut m = Map::new();
            m.insert("type".into(), json!(t));
            m
        })
        .collect();
    let has_null = types_raw.iter().any(|t| t == "null");
    let mut out = Map::new();
    for (key, value) in &obj {
        if key == "type" {
            continue;
        }
        if !is_cca_type_specific(key) {
            out.insert(key.clone(), value.clone());
            continue;
        }
        for (i, t) in types.iter().enumerate() {
            if type_specific_keys(t).contains(&key.as_str()) {
                branches[i].insert(key.clone(), value.clone());
            }
        }
    }
    let _ = has_null; // `nullable` is stripped for Moonshot (stripNullableKeyword)
    out.insert(
        "anyOf".into(),
        Value::Array(branches.into_iter().map(Value::Object).collect()),
    );
    out
}

fn drop_foreign_type_keywords(schema: &mut Map<String, Value>, ty: &str) {
    let allowed = type_specific_keys(ty);
    let foreign: Vec<String> = schema
        .keys()
        .filter(|k| is_cca_type_specific(k) && !allowed.contains(&k.as_str()))
        .cloned()
        .collect();
    for k in foreign {
        schema.shift_remove(&k);
    }
}

pub(crate) fn spill_to_description(node: &mut Map<String, Value>, spill: &[(String, Value)]) {
    if spill.is_empty() {
        return;
    }
    let existing = node
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let formatted = format!(
        "{{{}}}",
        spill
            .iter()
            .map(|(k, v)| format!("{k}: {}", serde_json::to_string(v).unwrap_or_default()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let description = if existing.is_empty() {
        formatted
    } else {
        format!("{existing}\n\n{formatted}")
    };
    node.insert("description".into(), Value::String(description));
}

fn moonshot_post(mut schema: Map<String, Value>) -> Map<String, Value> {
    if let Some(Value::Array(one_of)) = schema.get("oneOf").cloned() {
        schema.shift_remove("oneOf");
        let mut any_of = schema
            .get("anyOf")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        any_of.extend(one_of);
        schema.insert("anyOf".into(), Value::Array(any_of));
    }
    if let Some(Value::Array(e)) = schema.get("enum") {
        if !e.iter().all(|v| v.is_string() || v.is_number()) {
            schema.shift_remove("enum");
        }
    }
    schema
}

fn moonshot_node(value: &Value, inside_map: bool, boolean_is_subschema: bool) -> Value {
    match value {
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|e| moonshot_node(e, inside_map, boolean_is_subschema))
                .collect(),
        ),
        Value::Bool(_) if boolean_is_subschema => json!({}),
        Value::Object(o) => Value::Object(moonshot_object(o, inside_map)),
        other => other.clone(),
    }
}

fn moonshot_child(key: &str, entry: &Value, inside_map: bool) -> Value {
    match classify_schema_child(key, entry, inside_map) {
        Some(kind) => moonshot_node(entry, kind == ChildKind::Map, kind == ChildKind::Schema),
        None => entry.clone(),
    }
}

fn moonshot_object(value: &Map<String, Value>, inside_map: bool) -> Map<String, Value> {
    let obj = if inside_map {
        value.clone()
    } else {
        split_type_array_into_any_of(value.clone())
    };
    // anyOf/oneOf whose every branch is a bare `const` collapses to an enum.
    for combiner in ["anyOf", "oneOf"] {
        let Some(Value::Array(variants)) = obj.get(combiner) else {
            continue;
        };
        let all_const = !variants.is_empty()
            && variants
                .iter()
                .all(|v| v.as_object().is_some_and(|o| o.contains_key("const")));
        if !all_const {
            continue;
        }
        let mut result = Map::new();
        let mut deduped: Vec<Value> = Vec::new();
        for v in variants {
            push_enum_value(&mut deduped, &v["const"]);
        }
        result.insert("enum".into(), Value::Array(deduped.clone()));
        let explicit: Vec<&str> = variants
            .iter()
            .filter_map(|v| v.get("type").and_then(Value::as_str))
            .collect();
        if explicit.len() == variants.len() && explicit.iter().all(|t| *t == explicit[0]) {
            result.insert("type".into(), json!(explicit[0]));
        } else {
            let inferred: Vec<&str> = deduped.iter().filter_map(infer_json_schema_type).collect();
            let distinct: std::collections::BTreeSet<&str> = inferred.iter().copied().collect();
            if distinct.len() == 1 {
                result.insert("type".into(), json!(inferred[0]));
            } else {
                let non_null: Vec<&str> =
                    inferred.iter().copied().filter(|t| *t != "null").collect();
                let non_null_set: std::collections::BTreeSet<&str> =
                    non_null.iter().copied().collect();
                if inferred.contains(&"null") && non_null_set.len() == 1 {
                    result.insert("type".into(), json!(non_null[0]));
                }
            }
        }
        let mut spill: Vec<(String, Value)> = Vec::new();
        for (key, entry) in &obj {
            if key == combiner || result.contains_key(key) {
                continue;
            }
            if !inside_map && moonshot_unsupported(key) {
                if LIFTABLE_TO_DESCRIPTION.contains(&key.as_str()) {
                    spill.push((key.clone(), entry.clone()));
                }
                continue;
            }
            if key == "nullable" {
                continue;
            }
            result.insert(key.clone(), moonshot_child(key, entry, inside_map));
        }
        spill_to_description(&mut result, &spill);
        return moonshot_post(result);
    }

    let mut result = Map::new();
    let mut spill: Vec<(String, Value)> = Vec::new();
    let mut const_value: Option<Value> = None;
    for (key, entry) in &obj {
        if !inside_map && moonshot_unsupported(key) {
            if LIFTABLE_TO_DESCRIPTION.contains(&key.as_str()) {
                spill.push((key.clone(), entry.clone()));
            }
            continue;
        }
        if key == "nullable" {
            continue;
        }
        if key == "const" {
            const_value = Some(entry.clone());
            continue;
        }
        result.insert(key.clone(), moonshot_child(key, entry, inside_map));
    }
    if let Some(Value::Array(types)) = result.get("type").cloned() {
        let names: Vec<&str> = types.iter().filter_map(Value::as_str).collect();
        let first = names
            .iter()
            .copied()
            .find(|t| *t != "null")
            .or_else(|| names.first().copied());
        match first {
            Some(t) => {
                result.insert("type".into(), json!(t));
                drop_foreign_type_keywords(&mut result, t);
            }
            None => {
                result.shift_remove("type");
            }
        }
    }
    if let Some(c) = const_value {
        let mut existing = result
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        push_enum_value(&mut existing, &c);
        result.insert("enum".into(), Value::Array(existing));
        if !result
            .get("type")
            .is_some_and(|t| t.as_str().is_some_and(|s| !s.is_empty()))
        {
            if let Some(t) = infer_json_schema_type(&c) {
                result.insert("type".into(), json!(t));
            }
        }
    }
    let type_missing = !result
        .get("type")
        .is_some_and(|t| t.as_str().is_some_and(|s| !s.is_empty()));
    if type_missing
        && !result.get("anyOf").is_some_and(Value::is_array)
        && !result.get("oneOf").is_some_and(Value::is_array)
    {
        if let Some(Value::Array(e)) = result.get("enum") {
            if !e.is_empty() {
                let types: Vec<Option<&str>> = e.iter().map(infer_json_schema_type).collect();
                if types.iter().all(Option::is_some)
                    && types
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        == 1
                {
                    let t = types[0].expect("checked");
                    result.insert("type".into(), json!(t));
                }
            }
        }
    }
    spill_to_description(&mut result, &spill);
    moonshot_post(result)
}

/// TS `normalizeSchemaForMoonshot`: the stricter MFJS subset Moonshot/Kimi native hosts validate
/// against (type arrays scalarized, `const` → `enum`, validators stripped into the description).
pub(crate) fn normalize_schema_for_moonshot(schema: &Value) -> Value {
    moonshot_node(schema, false, true)
}

// ---- Grammar-constrained local backends — TS `sanitizeSchemaForGrammar`

fn open_subschema_widening() -> Value {
    json!({ "anyOf": [
        { "type": "string" }, { "type": "number" }, { "type": "boolean" },
        { "type": "object" }, { "type": "array" }, { "type": "null" }
    ] })
}

fn grammar_node(value: &Value, is_subschema: bool) -> Value {
    match value {
        Value::Bool(true) if is_subschema => open_subschema_widening(),
        Value::Bool(false) if is_subschema => json!({ "not": open_subschema_widening() }),
        Value::Array(a) => Value::Array(a.iter().map(|i| grammar_node(i, is_subschema)).collect()),
        Value::Object(o) => {
            let mut out = Map::new();
            for (key, child) in o {
                let next = match key.as_str() {
                    "properties" | "patternProperties" | "dependencies" | "dependentSchemas"
                    | "$defs" | "definitions"
                        if child.is_object() =>
                    {
                        let m = child.as_object().expect("checked");
                        Value::Object(
                            m.iter()
                                .map(|(k, v)| (k.clone(), grammar_node(v, true)))
                                .collect(),
                        )
                    }
                    "anyOf" | "oneOf" | "allOf" | "prefixItems" if child.is_array() => {
                        grammar_node(child, true)
                    }
                    "items" | "additionalItems" | "contains" | "contentSchema"
                    | "propertyNames" | "if" | "then" | "else" | "not" | "unevaluatedItems" => {
                        grammar_node(child, true)
                    }
                    "additionalProperties" | "unevaluatedProperties" if !child.is_boolean() => {
                        grammar_node(child, true)
                    }
                    _ => child.clone(),
                };
                out.insert(key.clone(), next);
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// TS `sanitizeSchemaForGrammar`: widen bare boolean subschemas llama.cpp/LM Studio/vLLM cannot
/// compile to a grammar.
pub(crate) fn sanitize_schema_for_grammar(schema: &Value) -> Value {
    grammar_node(schema, true)
}
