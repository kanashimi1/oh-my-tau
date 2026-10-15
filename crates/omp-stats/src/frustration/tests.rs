use super::*;
use crate::types::FrustrationCounts;

fn row(provider: &str, model: &str, first_seen: i64, messages: i64) -> FrustrationModelRow {
    FrustrationModelRow {
        model: model.to_owned(),
        provider: Some(provider.to_owned()),
        first_seen,
        counts: FrustrationCounts { messages, judged: 0, annoyed: 1, at_assistant: 1, angry: 0 },
    }
}

#[test]
fn questions_parse_like_judge_questions() {
    let questions: &Questions = &FRUSTRATION_QUESTIONS;
    match &questions["annoyed"] {
        Question::Score { criteria, .. } => assert_eq!(criteria.len(), 4),
        other => panic!("annoyed should be a score question, got {other:?}"),
    }
    match &questions["target"] {
        Question::Choice { criteria, .. } => {
            let labels: Vec<&str> = criteria.keys().map(String::as_str).collect();
            assert_eq!(labels, ["assistant", "none", "other"]);
        }
        other => panic!("target should be a choice question, got {other:?}"),
    }
}

#[test]
fn judge_questions_rejects_single_level_score() {
    let text = r#"{"q": {"instructions": "x", "criteria": ["only"]}}"#;
    assert!(judge_questions(text).is_err());
}

#[test]
fn judge_questions_true_false_labels_make_a_rubric() {
    let text = r#"{"q": {"instructions": "x", "criteria": {"true": "yes", "false": "no"}},
                   "e": {"instructions": "y", "criteria": {}}}"#;
    let questions = judge_questions(text).expect("parse");
    assert!(matches!(questions["q"], Question::Noul { criteria: Some(_), .. }));
    assert!(matches!(questions["e"], Question::Choice { .. }));
}

#[test]
fn merge_merges_spelling_variants_and_sums_counts() {
    let rows = [
        row("anthropic", "claude-opus-4-5", 200, 3),
        row("anthropic", "claude-opus-4-5-thinking", 100, 2),
    ];
    let merged = merge_frustration_rows(&rows);
    assert_eq!(merged.len(), 1, "variants of one identity share a row");
    let stats = &merged[0];
    assert_eq!(stats.counts.messages, 5);
    assert_eq!(stats.first_seen, 100);
    assert_eq!(stats.models.len(), 2);
    assert_eq!(stats.family.as_deref(), Some("opus"));
    assert_eq!(stats.label, "opus 4.5");
}

#[test]
fn merge_keeps_unclassified_rows_by_raw_model_id_in_first_use_order() {
    let rows = [
        row("nowhere", "mystery-b", 300, 1),
        row("nowhere", "mystery-a", 100, 1),
        row("nowhere", "mystery-b", 50, 1),
    ];
    let merged = merge_frustration_rows(&rows);
    let keys: Vec<&str> = merged.iter().map(|m| m.key.as_str()).collect();
    assert_eq!(keys, ["mystery-b", "mystery-a"]);
    assert_eq!(merged[0].label, "mystery-b");
    assert_eq!(merged[0].revision, None);
    assert_eq!(merged[0].first_seen, 50);
    assert_eq!(merged[0].counts.messages, 2);
}

#[test]
fn merge_sorts_classified_before_unclassified() {
    let rows = [row("nowhere", "mystery", 1, 1), row("anthropic", "claude-opus-4-5", 999, 1)];
    let merged = merge_frustration_rows(&rows);
    assert_eq!(merged[0].model_class, "anthropic");
    assert_eq!(merged[1].key, "mystery");
}

#[test]
fn estimate_reports_no_judge_without_touching_db() {
    let value = serde_json::to_value(estimate_frustration_run(Some("all"))).expect("serialize");
    assert_eq!(value["available"], false);
    assert!(value["reason"].as_str().is_some_and(|r| r.contains("without a judge")));
}

#[test]
fn start_refuses_with_503_when_no_judge() {
    match start_frustration_run(None) {
        StartFrustrationRunResult::Refused { status, error } => {
            assert_eq!(status, 503);
            assert!(error.contains("without a judge"));
        }
        other => panic!("expected refusal, got {other:?}"),
    }
}

#[test]
fn cancel_when_idle_leaves_status_idle() {
    let job = cancel_frustration_run();
    assert_eq!(job.state, FrustrationJobState::Idle);
    assert_eq!(job.finished_at, None);
    assert_eq!(job.concurrency, 0);
}

#[cfg(feature = "db_ingest")]
mod golden {
    use super::*;
    use crate::testutil::golden::{assert_golden, fixture_home, golden_requests};

    #[test]
    fn dashboard_stats_match_golden_for_every_range() {
        let Some(home) = fixture_home() else { return };
        let requests = golden_requests("/api/stats/frustration");
        assert_eq!(requests.len(), 6);
        for request in &requests {
            let range = request.split("range=").nth(1);
            let stats = get_frustration_dashboard_stats(range).expect("dashboard stats");
            let actual = serde_json::to_value(stats).expect("serialize");
            assert_golden(request, &actual, &home);
        }
    }

    #[test]
    fn estimate_on_fixture_is_unavailable() {
        let Some(_home) = fixture_home() else { return };
        let value = serde_json::to_value(estimate_frustration_run(Some("all"))).expect("serialize");
        assert_eq!(value["available"], false);
    }
}
