use super::*;

#[test]
fn classifies_claude_opus_wire_id() {
    let id = classify_model("anthropic", "claude-opus-5-5", None).unwrap();
    assert_eq!(id.class, "anthropic");
    assert_eq!(id.family.as_deref(), Some("opus"));
    assert_eq!(id.revision.as_deref(), Some("5.5.0"));
    assert_eq!(id.effort, None);
    assert_eq!(id.thinking_variant, None);
    assert_eq!(id.logical_id, None);
}

#[test]
fn classifies_glm_flash_wire_id() {
    let id = classify_model("zai", "glm-5.3-flash", None).unwrap();
    assert_eq!(id.class, "glm");
    assert_eq!(id.family.as_deref(), Some("flash"));
    assert_eq!(id.revision.as_deref(), Some("5.3.0"));
}

#[test]
fn thinking_suffix_collapses_to_logical_id() {
    let id = classify_model("anthropic", "claude-opus-4-1-thinking", None).unwrap();
    assert_eq!(id.class, "anthropic");
    assert_eq!(id.family.as_deref(), Some("opus"));
    assert_eq!(id.revision.as_deref(), Some("4.1.0"));
    assert_eq!(id.thinking_variant, Some(true));
    assert_eq!(id.logical_id.as_deref(), Some("claude-opus-4-1"));
}

#[test]
fn effort_suffix_collapses_to_logical_id() {
    let id = classify_model("openai", "gpt-5.3-codex-high", None).unwrap();
    assert_eq!(id.class, "openai");
    assert_eq!(id.family.as_deref(), Some("codex"));
    assert_eq!(id.revision.as_deref(), Some("5.3.0"));
    assert_eq!(id.effort, Some(Effort::High));
    assert_eq!(id.logical_id.as_deref(), Some("gpt-5.3-codex"));
}

#[test]
fn unrecognized_model_is_unknown() {
    let id = classify_model("openai", "totally-unknown-model", None).unwrap();
    assert_eq!(id.class, "unknown");
    assert_eq!(id.family, None);
    assert_eq!(id.revision, None);
}

#[test]
fn parses_explicit_and_prefix_revisions() {
    assert_eq!(parse_revision("4.6"), Some((4, 6, 0)));
    assert_eq!(parse_revision("4-6-1"), Some((4, 6, 1)));
    assert_eq!(parse_revision("4.6.1.2"), None);
    assert_eq!(parse_revision("v4"), None);
    assert_eq!(parse_revision_prefix("4-6-turbo"), Some((4, 6, 0)));
    assert_eq!(parse_revision_prefix("3.3-70b"), Some((3, 3, 0)));
    assert_eq!(parse_revision_prefix("32b"), None);
}

#[test]
fn compares_and_formats_revisions() {
    assert!(compare_revision((4, 6, 0), (4, 10, 0)) < 0);
    assert!(compare_revision((5, 0, 0), (4, 9, 9)) > 0);
    assert_eq!(compare_revision((1, 2, 3), (1, 2, 3)), 0);
    assert_eq!(format_revision((4, 6, 0)), "4.6.0");
}

/// Review regression: a suffix longer than the id skips to the next suffix
/// (TS `continue`), it does not end the search.
#[test]
fn billing_variant_plain_skips_suffixes_longer_than_the_id() {
    use super::collapse::billing_variant_plain;
    // "-contributor" (12) is longer than "x-free" (6); "-free" must still match.
    assert_eq!(billing_variant_plain("x-free").as_deref(), Some("x"));
    assert_eq!(
        billing_variant_plain("model-contributor").as_deref(),
        Some("model")
    );
    assert_eq!(billing_variant_plain("plain"), None);
}

/// Review regression: non-ASCII ids whose Unicode lowercase changes byte
/// length must not panic or mis-slice. Deliberate divergence: TS returns
/// `İİ-model-t` here (JS `toLowerCase` turns `İ` into two UTF-16 units and
/// shifts its indices); no real model id contains such characters.
#[test]
fn strip_thinking_variant_suffix_handles_non_ascii() {
    use super::collapse::strip_thinking_variant_suffix;
    assert_eq!(
        strip_thinking_variant_suffix("İİ-model-thinking").as_deref(),
        Some("İİ-model")
    );
    assert_eq!(
        strip_thinking_variant_suffix("модель-thinking").as_deref(),
        Some("модель")
    );
    assert_eq!(strip_thinking_variant_suffix("model-non-thinking"), None);
}
