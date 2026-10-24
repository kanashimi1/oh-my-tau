//! outputSchema validation for child results.
//!
//! Ported from the task tool's `outputSchema` / `schemaMode` handling: a
//! declared schema describes the child's final output; `schemaMode` decides
//! whether a violation fails the run (`strict`) or is only reported
//! (`permissive`, the default).

use serde_json::Value;

use crate::params::SchemaMode;

/// Outcome of validating a child output against an outputSchema.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaValidation {
    /// No schema was declared; nothing to check.
    NoSchema,
    /// Output parsed as JSON and satisfies the schema.
    Valid,
    /// Output is not JSON and could not be checked. Fails only in strict mode.
    NotJson,
    /// Output is JSON but violates the schema. Fails only in strict mode.
    Invalid(Vec<String>),
}

impl SchemaValidation {
    /// Whether this outcome fails the run under `mode`.
    #[must_use]
    pub fn is_failure(&self, mode: SchemaMode) -> bool {
        mode == SchemaMode::Strict && matches!(self, Self::NotJson | Self::Invalid(_))
    }

    /// Human-readable summary, for envelope error fields.
    #[must_use]
    pub fn summary(&self) -> Option<String> {
        match self {
            Self::NoSchema | Self::Valid => None,
            Self::NotJson => Some("output is not valid JSON (outputSchema declared)".to_string()),
            Self::Invalid(errors) => Some(format!(
                "output does not satisfy outputSchema: {}",
                errors.join("; ")
            )),
        }
    }
}

/// Validate `output` against `schema`.
///
/// The returned outcome decides failure via [`SchemaValidation::is_failure`]
/// with the caller's mode (permissive by default).
///
/// `schema` accepts the same coarse shapes as the TS wire contract
/// (`object | boolean | string | null`): JSON-schema objects validate
/// directly; `true`/absent validates anything; `false` validates nothing;
/// strings are treated as JSON-schema text (parsed leniently; unparseable
/// strings mean "no schema").
///
/// # Errors
/// Returns `SchemaValidation` (never panics); an uncompileable schema object
/// yields `NotJson`-adjacent behavior via [`SchemaValidation::Invalid`] with
/// the compiler message.
#[must_use]
pub fn validate_output(output: &str, schema: Option<&Value>) -> SchemaValidation {
    let Some(schema) = schema else {
        return SchemaValidation::NoSchema;
    };
    let compiled = match compile_schema(schema) {
        Ok(Some(c)) => c,
        Ok(None) => return SchemaValidation::NoSchema,
        Err(message) => return SchemaValidation::Invalid(vec![message]),
    };
    let value: Value = match serde_json::from_str(output.trim()) {
        Ok(v) => v,
        Err(_) => return SchemaValidation::NotJson,
    };
    if compiled.validate(&value).is_ok() {
        return SchemaValidation::Valid;
    }
    let errors: Vec<String> = compiled
        .iter_errors(&value)
        .map(|e| format!("{}: {}", e.instance_path, e))
        .collect();
    SchemaValidation::Invalid(errors)
}

type Compiled = jsonschema::Validator;

fn compile_schema(schema: &Value) -> Result<Option<Compiled>, String> {
    match schema {
        // `true` = any output accepted; absent/null handled by caller.
        Value::Bool(true) | Value::Null => Ok(None),
        Value::Bool(false) => {
            // `false` = no output satisfies it.
            let c = jsonschema::Validator::new(&serde_json::json!(false))
                .map_err(|e| format!("uncompileable outputSchema: {e}"))?;
            Ok(Some(c))
        }
        Value::Object(_) => jsonschema::Validator::new(schema)
            .map(Some)
            .map_err(|e| format!("uncompileable outputSchema: {e}")),
        // String form: JSON-schema text.
        Value::String(s) => {
            let parsed: Value = match serde_json::from_str(s) {
                Ok(v) => v,
                Err(_) => return Ok(None),
            };
            jsonschema::Validator::new(&parsed)
                .map(Some)
                .map_err(|e| format!("uncompileable outputSchema: {e}"))
        }
        other => Err(format!("unsupported outputSchema kind: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str =
        r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}"#;

    #[test]
    fn no_schema_is_noop() {
        assert_eq!(
            validate_output("plain prose", None),
            SchemaValidation::NoSchema
        );
    }

    #[test]
    fn valid_json_passes() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        assert_eq!(
            validate_output(r#"{"ok":true}"#, Some(&schema)),
            SchemaValidation::Valid
        );
    }

    #[test]
    fn permissive_tolerates_violations() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        let v = validate_output("not json at all", Some(&schema));
        assert_eq!(v, SchemaValidation::NotJson);
        assert!(!v.is_failure(SchemaMode::Permissive));
        assert!(v.is_failure(SchemaMode::Strict));
        let v = validate_output(r#"{"ok":"nope"}"#, Some(&schema));
        assert!(matches!(v, SchemaValidation::Invalid(_)));
        assert!(!v.is_failure(SchemaMode::Permissive));
        assert!(v.is_failure(SchemaMode::Strict));
        let v = validate_output(r#"{"ok":"nope"}"#, Some(&schema));
        assert!(v.summary().unwrap().contains("outputSchema"));
    }

    #[test]
    fn boolean_and_string_schema_forms() {
        assert_eq!(
            validate_output("prose", Some(&Value::Bool(true))),
            SchemaValidation::NoSchema
        );
        let fail = validate_output("{}", Some(&Value::Bool(false)));
        assert!(matches!(fail, SchemaValidation::Invalid(_)));
        let s = Value::String(SCHEMA.to_string());
        assert_eq!(
            validate_output(r#"{"ok":false}"#, Some(&s)),
            SchemaValidation::Valid
        );
        let bad_string = Value::String("{not json".to_string());
        assert_eq!(
            validate_output("prose", Some(&bad_string)),
            SchemaValidation::NoSchema
        );
    }

    #[test]
    fn uncompileable_schema_reported() {
        let schema = serde_json::json!({"type": 42});
        let v = validate_output("{}", Some(&schema));
        assert!(matches!(v, SchemaValidation::Invalid(_)));
    }
}
