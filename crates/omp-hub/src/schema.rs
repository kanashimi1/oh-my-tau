//! `answer` validation against the original envelope's `reply_schema` with
//! the workspace `jsonschema` 0.29 crate (same one `omp-task` uses).

use serde_json::Value;

/// Outcome of validating an answer payload against a reply schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplyValidation {
    /// No schema on the original: any payload accepted.
    NoSchema,
    Valid,
    /// Payload is not JSON at all (`SPEC.md`: invalid answers return the
    /// validation errors to the answerer).
    NotJson(String),
    /// JSON, but does not satisfy the schema; the strings are
    /// `instance_path: detail`.
    Invalid(Vec<String>),
}

impl ReplyValidation {
    /// Error detail for the answerer when invalid.
    #[must_use]
    pub fn error_detail(&self) -> Option<Vec<String>> {
        match self {
            Self::Invalid(errors) => Some(errors.clone()),
            Self::NotJson(e) => Some(vec![format!("payload is not JSON: {e}")]),
            _ => None,
        }
    }
}

/// Validate `payload` (a JSON text) against `schema`.
#[must_use]
pub fn validate_reply(schema: Option<&Value>, payload: &str) -> ReplyValidation {
    let Some(schema) = schema else {
        return ReplyValidation::NoSchema;
    };
    if matches!(schema, Value::Bool(true) | Value::Null) {
        return ReplyValidation::NoSchema;
    }
    let compiled = match jsonschema::Validator::new(schema) {
        Ok(c) => c,
        Err(e) => {
            return ReplyValidation::Invalid(vec![format!("uncompileable reply_schema: {e}")])
        }
    };
    let value: Value = match serde_json::from_str(payload.trim()) {
        Ok(v) => v,
        Err(e) => return ReplyValidation::NotJson(e.to_string()),
    };
    if compiled.validate(&value).is_ok() {
        return ReplyValidation::Valid;
    }
    ReplyValidation::Invalid(
        compiled
            .iter_errors(&value)
            .map(|e| format!("{}: {}", e.instance_path, e))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str =
        r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}"#;

    fn schema() -> Value {
        serde_json::from_str(SCHEMA).unwrap()
    }

    #[test]
    fn no_schema_accepts_anything() {
        assert_eq!(validate_reply(None, "prose"), ReplyValidation::NoSchema);
        assert_eq!(
            validate_reply(Some(&Value::Bool(true)), "prose"),
            ReplyValidation::NoSchema
        );
    }

    #[test]
    fn valid_json_passes() {
        assert_eq!(
            validate_reply(Some(&schema()), r#"{"ok":true}"#),
            ReplyValidation::Valid
        );
    }

    #[test]
    fn invalid_json_reports_errors() {
        let v = validate_reply(Some(&schema()), r#"{"ok":"yes"}"#);
        match v {
            ReplyValidation::Invalid(errors) => {
                assert_eq!(errors.len(), 1);
                assert!(errors[0].contains("ok"), "errors: {errors:?}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn non_json_payload_is_notjson() {
        assert!(matches!(
            validate_reply(Some(&schema()), "not json at all"),
            ReplyValidation::NotJson(_)
        ));
    }
}
