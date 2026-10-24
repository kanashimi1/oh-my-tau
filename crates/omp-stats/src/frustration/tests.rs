use super::*;
use crate::types::FrustrationCounts;

fn row(provider: &str, model: &str, first_seen: i64, messages: i64) -> FrustrationModelRow {
    FrustrationModelRow {
        model: model.to_owned(),
        provider: Some(provider.to_owned()),
        first_seen,
        counts: FrustrationCounts {
            messages,
            judged: 0,
            annoyed: 1,
            at_assistant: 1,
            angry: 0,
        },
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
    assert!(matches!(
        questions["q"],
        Question::Noul {
            criteria: Some(_),
            ..
        }
    ));
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
    let rows = [
        row("nowhere", "mystery", 1, 1),
        row("anthropic", "claude-opus-4-5", 999, 1),
    ];
    let merged = merge_frustration_rows(&rows);
    assert_eq!(merged[0].model_class, "anthropic");
    assert_eq!(merged[1].key, "mystery");
}

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;

use crate::db_query::{get_pending_frustration_prose, get_pending_frustration_totals};
use crate::testutil::TestHome;
use crate::types::{FrustrationEstimate, FrustrationJobState};

/// Holds the shared db/HOME lock and leaves judge + job state idle on drop.
struct Isolated {
    _home: TestHome,
}

impl Isolated {
    fn new(home: TestHome) -> Self {
        reset_for_tests();
        Self { _home: home }
    }

    #[cfg(feature = "db_ingest")]
    fn home(&self) -> &TestHome {
        &self._home
    }
}

impl Drop for Isolated {
    fn drop(&mut self) {
        reset_for_tests();
    }
}

/// Judge answering deterministically from the text: length mod 3 picks angry-at-assistant,
/// annoyed-at-other, or neutral. `quick` answers settle at once; later calls hang until cancelled.
struct FakeJudge {
    model: Option<JudgeModel>,
    quick: AtomicUsize,
    calls: AtomicUsize,
    in_flight: Arc<AtomicUsize>,
    fail: bool,
}

struct InFlight(Arc<AtomicUsize>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, AtomicOrdering::SeqCst);
    }
}

impl FakeJudge {
    fn new(quick: usize) -> Self {
        Self {
            model: Some(JudgeModel {
                provider: "fake".into(),
                id: "fake-1".into(),
                cost_input: 2.0,
                cost_output: 10.0,
            }),
            quick: AtomicUsize::new(quick),
            calls: AtomicUsize::new(0),
            in_flight: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }
    }

    fn answer(text: &str) -> JudgeAnswer {
        let (levels, target): (&[(&str, f64)], &str) = match text.len() % 3 {
            0 => (
                &[("0", 0.0), ("1", 0.1), ("2", 0.1), ("3", 0.8)],
                "assistant",
            ),
            1 => (&[("0", 0.1), ("1", 0.2), ("2", 0.7), ("3", 0.0)], "other"),
            _ => (&[("0", 0.9), ("1", 0.1), ("2", 0.0), ("3", 0.0)], "none"),
        };
        JudgeAnswer {
            provider: "fake".into(),
            model: "fake-1".into(),
            annoyed_levels: levels.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
            target: target.into(),
            cost: 0.001,
        }
    }
}

impl StatsJudge for FakeJudge {
    fn primary_model(&self) -> Option<JudgeModel> {
        self.model.clone()
    }

    fn judge<'a>(&'a self, text: &'a str, questions: &'a Questions) -> JudgeFuture<'a> {
        Box::pin(async move {
            assert_eq!(questions.len(), 2);
            self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            if self.fail {
                anyhow::bail!("rate limited");
            }
            self.in_flight.fetch_add(1, AtomicOrdering::SeqCst);
            let _guard = InFlight(Arc::clone(&self.in_flight));
            let quick = self
                .quick
                .fetch_update(AtomicOrdering::SeqCst, AtomicOrdering::SeqCst, |n| {
                    n.checked_sub(1)
                })
                .is_ok();
            if !quick {
                std::future::pending::<()>().await;
            }
            Ok(Self::answer(text))
        })
    }
}

fn provide(judge: &Arc<FakeJudge>) {
    let judge = Arc::clone(judge);
    set_stats_judge_provider(Some(Arc::new(move || {
        Ok(Arc::clone(&judge) as Arc<dyn StatsJudge>)
    })));
}

fn refusal(result: StartFrustrationRunResult) -> (u16, String) {
    match result {
        StartFrustrationRunResult::Refused { status, error } => (status, error),
        other => panic!("expected refusal, got {other:?}"),
    }
}

#[test]
fn standalone_reports_no_judge() {
    let _iso = Isolated::new(TestHome::new());
    let value = serde_json::to_value(estimate_frustration_run(Some("all")).expect("estimate"))
        .expect("serialize");
    assert_eq!(value["available"], false);
    assert!(value["reason"]
        .as_str()
        .is_some_and(|r| r.contains("without a judge")));
    let (status, error) = refusal(start_frustration_run(None).expect("start"));
    assert_eq!(status, 503);
    assert!(error.contains("without a judge"));
    assert!(!judge_available());
}

#[test]
fn provider_failure_and_missing_model_refuse_with_503() {
    let _iso = Isolated::new(TestHome::new());
    set_stats_judge_provider(Some(Arc::new(|| anyhow::bail!("no credentials"))));
    assert!(judge_available());
    let (status, error) = refusal(start_frustration_run(None).expect("start"));
    assert_eq!(
        (status, error.as_str()),
        (503, "The judge could not be loaded: no credentials")
    );
    let estimate =
        serde_json::to_value(estimate_frustration_run(None).expect("estimate")).expect("serialize");
    assert_eq!(
        estimate["reason"],
        "The judge could not be loaded: no credentials"
    );

    let mut judge = FakeJudge::new(0);
    judge.model = None;
    provide(&Arc::new(judge));
    let (status, error) = refusal(start_frustration_run(None).expect("start"));
    assert_eq!(status, 503);
    assert!(error.contains("`judge` model role"), "{error}");
}

#[test]
fn cancel_when_idle_leaves_status_idle() {
    let _iso = Isolated::new(TestHome::new());
    let job = cancel_frustration_run();
    assert_eq!(job.state, FrustrationJobState::Idle);
    assert_eq!(job.finished_at, None);
    assert_eq!(job.concurrency, 0);
}

#[test]
fn adaptive_limit_slow_starts_then_halves_once_per_cooldown() {
    let mut limit = run::AdaptiveLimit::new();
    assert_eq!(limit.value(), 32);
    limit.succeeded();
    assert_eq!(limit.value(), 33);
    limit.failed();
    assert_eq!(limit.value(), 16);
    limit.failed();
    assert_eq!(
        limit.value(),
        16,
        "second failure inside the cooldown does not halve again"
    );
    limit.succeeded();
    assert!(
        limit.limit < 17.0,
        "congestion avoidance grows by 4/limit, not 1"
    );
}

#[cfg(feature = "db_ingest")]
mod golden {
    use super::*;
    use crate::testutil::golden::{assert_golden, fixture_home, golden_requests};

    #[test]
    fn dashboard_stats_match_golden_for_every_range() {
        let Some(home) = fixture_home() else { return };
        let iso = Isolated::new(home);
        let requests = golden_requests("/api/stats/frustration");
        assert_eq!(requests.len(), 6);
        for request in &requests {
            let range = request.split("range=").nth(1);
            let stats = get_frustration_dashboard_stats(range).expect("dashboard stats");
            let actual = serde_json::to_value(stats).expect("serialize");
            assert_golden(request, &actual, iso.home());
        }
    }

    fn pending_total() -> usize {
        get_pending_frustration_prose(None).expect("pending").len()
    }

    fn wait_until(what: &str, mut ready: impl FnMut() -> bool) {
        for _ in 0..400 {
            if ready() {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("timed out waiting for {what}");
    }

    fn verdict_rows() -> Vec<(String, f64, f64, String, String)> {
        crate::db::with_db(|db| {
            let mut stmt = db.prepare(
                "SELECT prose_hash, p_annoyed, p_angry, target, judge FROM frustration_verdicts ORDER BY prose_hash",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .expect("verdict rows")
    }

    #[test]
    fn fake_judge_run_over_fixture_writes_verdicts_the_dashboard_reports() {
        let Some(home) = fixture_home() else { return };
        let _iso = Isolated::new(home);
        let judge = Arc::new(FakeJudge::new(usize::MAX));
        provide(&judge);

        let before = get_frustration_dashboard_stats(Some("all")).expect("stats");
        assert!(before.judge_available);
        assert_eq!(before.overall.judged, 0);
        let pending = get_pending_frustration_prose(None).expect("pending");
        assert!(
            pending.len() >= 4,
            "fixture needs unjudged prose, got {}",
            pending.len()
        );

        let estimate = estimate_frustration_run(Some("all")).expect("estimate");
        let totals = get_pending_frustration_totals(None).expect("totals");
        let FrustrationEstimate::Available {
            messages,
            chars,
            input_tokens,
            cost,
            judge: label,
            ..
        } = estimate
        else {
            panic!("estimate should be available");
        };
        assert_eq!((messages, chars), (totals.messages, totals.chars));
        assert_eq!(
            input_tokens,
            messages * 561 + (chars as f64 / 5.9).ceil() as i64
        );
        let expected_cost = input_tokens as f64 * 2.0 / 1e6 + (messages * 8) as f64 * 10.0 / 1e6;
        assert!((cost - expected_cost).abs() < 1e-12);
        assert_eq!(label, "fake/fake-1");

        let started = start_frustration_run(Some("all")).expect("start");
        let StartFrustrationRunResult::Started { job } = started else {
            panic!("expected start, got {started:?}")
        };
        assert_eq!(job.state, FrustrationJobState::Running);
        assert_eq!(job.total as usize, pending.len());
        assert!(wait_frustration_run_finished(Duration::from_secs(30)));

        let status = get_frustration_job_status();
        assert_eq!(status.state, FrustrationJobState::Done);
        assert_eq!((status.done as usize, status.failed), (pending.len(), 0));
        assert!((status.cost - 0.001 * pending.len() as f64).abs() < 1e-9);
        assert_eq!(status.judge.as_deref(), Some("fake/fake-1"));
        assert!(status.finished_at.is_some() && status.error.is_none());
        assert_eq!(judge.calls.load(AtomicOrdering::SeqCst), pending.len());

        let rows = verdict_rows();
        assert_eq!(rows.len(), pending.len());
        let by_hash: std::collections::HashMap<_, _> = pending
            .iter()
            .map(|p| (p.hash.clone(), p.prose.clone()))
            .collect();
        for (hash, p_annoyed, p_angry, target, label) in &rows {
            let expected = FakeJudge::answer(&by_hash[hash]);
            assert_eq!(label, "fake/fake-1");
            assert_eq!(target, &expected.target);
            assert!((p_angry - expected.annoyed_levels["3"]).abs() < 1e-12);
            assert!(
                (p_annoyed - (expected.annoyed_levels["2"] + expected.annoyed_levels["3"])).abs()
                    < 1e-12
            );
        }

        // Nothing left to judge, and the dashboard now classifies by verdict.
        assert_eq!(pending_total(), 0);
        let after = get_frustration_dashboard_stats(Some("all")).expect("stats");
        assert_eq!(after.overall.judged, after.overall.messages);
        assert!(after.overall.judged > 0);
        assert_eq!(after.job.state, FrustrationJobState::Done);
        let annoyed_rows = crate::db::with_db(|db| {
            Ok(db.query_row(
                "SELECT COUNT(*) FROM user_messages u JOIN frustration_verdicts v ON v.prose_hash = u.prose_hash \
                 WHERE u.prose != '' AND v.p_annoyed >= 0.5",
                [],
                |r| r.get::<_, i64>(0),
            )?)
        })
        .expect("annoyed rows");
        assert_eq!(after.overall.annoyed, annoyed_rows);
        assert!(
            after.overall.angry <= after.overall.at_assistant
                && after.overall.at_assistant <= after.overall.annoyed
        );

        // A second run has nothing to do and finishes immediately.
        let StartFrustrationRunResult::Started { job } =
            start_frustration_run(Some("all")).expect("start")
        else {
            panic!("second start");
        };
        assert_eq!((job.state, job.total), (FrustrationJobState::Done, 0));
        assert_eq!(job.finished_at, job.started_at);
    }

    #[test]
    fn double_start_gets_409_and_cancel_keeps_paid_verdicts_and_stops_in_flight() {
        let Some(home) = fixture_home() else { return };
        let _iso = Isolated::new(home);
        let judge = Arc::new(FakeJudge::new(3));
        provide(&judge);
        let total = pending_total();
        assert!(total > 3);

        let StartFrustrationRunResult::Started { .. } =
            start_frustration_run(Some("all")).expect("start")
        else {
            panic!("expected start");
        };
        let (status, error) = refusal(start_frustration_run(Some("all")).expect("second start"));
        assert_eq!(status, 409);
        assert!(error.contains("already in progress"));

        wait_until("three quick verdicts", || {
            get_frustration_job_status().done == 3
        });
        wait_until("in-flight requests", || {
            judge.in_flight.load(AtomicOrdering::SeqCst) > 0
        });
        assert!(get_frustration_job_status().concurrency >= 4);

        let cancelled = cancel_frustration_run();
        assert_eq!(cancelled.state, FrustrationJobState::Cancelled);
        assert!(cancelled.finished_at.is_some());
        assert!(wait_frustration_run_finished(Duration::from_secs(10)));
        assert_eq!(
            judge.in_flight.load(AtomicOrdering::SeqCst),
            0,
            "in-flight judge requests were dropped"
        );

        let status = get_frustration_job_status();
        assert_eq!(
            status.state,
            FrustrationJobState::Cancelled,
            "a finished worker does not overwrite a cancel"
        );
        assert_eq!((status.done, status.failed), (3, 0));
        assert_eq!(
            verdict_rows().len(),
            3,
            "verdicts already paid for are flushed on cancel"
        );
        assert_eq!(pending_total(), total - 3);

        // After a cancel a new run is accepted and resumes with the rest.
        let fresh = Arc::new(FakeJudge::new(usize::MAX));
        provide(&fresh);
        let StartFrustrationRunResult::Started { job } =
            start_frustration_run(Some("all")).expect("restart")
        else {
            panic!("restart");
        };
        assert_eq!(job.total as usize, total - 3);
        assert!(wait_frustration_run_finished(Duration::from_secs(30)));
        assert_eq!(
            get_frustration_job_status().state,
            FrustrationJobState::Done
        );
        assert_eq!(verdict_rows().len(), total);
    }

    #[test]
    fn failing_judge_trips_the_circuit_breaker() {
        let Some(home) = fixture_home() else { return };
        let _iso = Isolated::new(home);
        let mut judge = FakeJudge::new(0);
        judge.fail = true;
        let judge = Arc::new(judge);
        provide(&judge);
        let total = pending_total();
        if total < 25 {
            eprintln!("fixture has only {total} pending texts; circuit breaker needs 25");
            return;
        }
        let StartFrustrationRunResult::Started { .. } =
            start_frustration_run(Some("all")).expect("start")
        else {
            panic!("expected start");
        };
        assert!(wait_frustration_run_finished(Duration::from_secs(30)));
        let status = get_frustration_job_status();
        assert_eq!(
            status.state,
            FrustrationJobState::Failed,
            "total={total} status={status:?}"
        );
        assert_eq!(status.error.as_deref(), Some("rate limited"));
        assert_eq!(status.done, 0);
        assert!(status.failed >= 25);
        assert!(verdict_rows().is_empty());
    }

    #[cfg(feature = "server")]
    #[test]
    fn dashboard_endpoints_drive_a_run() {
        use crate::server::{handle_api, STATS_ACTION_HEADER};
        let Some(home) = fixture_home() else { return };
        let _iso = Isolated::new(home);
        let judge = Arc::new(FakeJudge::new(usize::MAX));
        provide(&judge);
        let plain = hyper::HeaderMap::new();
        let mut action = hyper::HeaderMap::new();
        action.insert(STATS_ACTION_HEADER, "1".parse().expect("header"));

        let estimate =
            handle_api("GET", "/api/frustration/estimate?range=all", &plain).expect("estimate");
        let body = estimate.body_json().expect("json");
        assert_eq!(body["available"], true);
        assert_eq!(body["judge"], "fake/fake-1");
        assert_eq!(
            handle_api("POST", "/api/frustration/judge?range=all", &plain)
                .expect("no header")
                .status,
            403
        );

        let started =
            handle_api("POST", "/api/frustration/judge?range=all", &action).expect("start");
        assert_eq!(started.status, 202);
        assert_eq!(started.body_json().expect("json")["state"], "running");
        assert!(wait_frustration_run_finished(Duration::from_secs(30)));

        let stats = handle_api("GET", "/api/stats/frustration?range=all", &plain).expect("stats");
        let body = stats.body_json().expect("json");
        assert_eq!(body["judgeAvailable"], true);
        assert_eq!(body["job"]["state"], "done");
        assert_eq!(body["overall"]["judged"], body["overall"]["messages"]);

        let cancel = handle_api("POST", "/api/frustration/cancel", &action).expect("cancel");
        assert_eq!(
            cancel.body_json().expect("json")["state"],
            "done",
            "cancel of a finished run is a no-op"
        );
    }
}
