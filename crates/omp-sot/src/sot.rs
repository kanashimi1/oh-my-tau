//! The closed `SoT` loop (Algorithm 1) and its two injected surfaces: the
//! text-generating [`Backend`] and the state-producing [`StateTracker`].
//!
//! [`run`] executes the feedback controller
//! `m_t → ỹ_{<t} = 𝒮(y_{<t}; m_t) → y_t → m_{t+1}, z_t = 𝒯(m_t)` (Eq. 11):
//! generate a sentence-level unit from the active evidence, read the
//! endogenous state, re-select the active evidence under budget `B`, decide
//! stop, and on stop emit the final answer. The loop is path-dependent by
//! construction: the evidence carried at one step changes the next unit,
//! and the new unit changes the next state.
//!
//! The loop itself is pure control flow over the injected traits — the only
//! I/O is the [`Backend`], and the only state semantics live in the
//! [`StateTracker`].

use crate::controller::{TrainingFreeEvidence, TrainingFreeStop};
use crate::embed::{NearestPool, Pca, SentenceEncoder};
use crate::state::{raw_from_observation, Normalizer, Observation, Trajectory};

/// Text-generation surface of the frozen backbone `f` (Eq. 3, 4).
///
/// The backend decides how the problem and the selected evidence are
/// materialized into a prompt (and how its own token budget is enforced);
/// the `SoT` loop only passes the selected sentence units.
pub trait Backend {
    /// Generate the next sentence-level reasoning unit
    /// `y_t ~ ℙ(· | p, ỹ_{<t})` (Eq. 3, Algorithm 1 line 3).
    fn generate_step(&mut self, problem: &str, evidence: &[&str]) -> String;

    /// Generate the final answer `a ~ ℙ(· | p, ỹ_{<t+1}, y_t)` (Eq. 4,
    /// Algorithm 1 line 8). `last` is the reasoning unit that triggered the
    /// stop; it is passed separately from `evidence`, which contains the
    /// earlier active steps only.
    fn generate_answer(&mut self, problem: &str, evidence: &[&str], last: &str) -> String;
}

/// State source: maps a generated sentence unit to its [`Observation`]
/// (sentence center `u_t` plus the two snapshot coordinates the dynamics
/// cannot derive — dispersion δ and predictive entropy H, exact from
/// hidden-state telemetry or the SoT-Embed proxies).
///
/// The loop keeps the center trajectory itself and derives the dynamical
/// coordinates `v`/`c` (Eq. 15-16) plus the raw signature (Eq. 13), so a
/// tracker is stateless beyond what its source requires.
pub trait StateTracker {
    /// Read out the observation for one generated reasoning unit.
    fn observe(&mut self, unit: &str) -> Observation;
}

impl<F: FnMut(&str) -> Observation> StateTracker for F {
    fn observe(&mut self, unit: &str) -> Observation {
        self(unit)
    }
}

/// SoT-Embed state tracker (App. D.2, Eq. 28): the fixed sentence encoder
/// produces `u_t = Enc(y_t)`, and the two snapshot coordinates come from
/// the embedding-space proxies — δ as the squared PCA reconstruction
/// residual against the training distribution's principal subspace, H as the
/// mean cosine distance to the `k` nearest training-pool embeddings. With no
/// PCA fit / empty pool the corresponding proxy is `0.0`.
pub struct EmbeddingTracker<E> {
    encoder: E,
    pca: Option<Pca>,
    pool: NearestPool,
}

impl<E: SentenceEncoder> EmbeddingTracker<E> {
    /// Build a tracker from a fixed encoder plus the fitted trajectory
    /// summaries (both optional; `None`/empty degrade the corresponding
    /// proxy to `0.0`).
    #[must_use]
    pub fn new(encoder: E, pca: Option<Pca>, pool: NearestPool) -> Self {
        Self { encoder, pca, pool }
    }

    /// Mutable access to the wrapped encoder (warm-up, batching, caches).
    #[must_use]
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }
}

impl<E: SentenceEncoder> StateTracker for EmbeddingTracker<E> {
    fn observe(&mut self, unit: &str) -> Observation {
        let u = self.encoder.encode(unit);
        let dispersion = self.pca.as_ref().map_or(0.0, |p| p.residual(&u));
        let entropy = self.pool.uncertainty(&u);
        Observation::new(u, dispersion, entropy)
    }
}

/// Why the loop terminated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    /// The stopping operator fired (Eq. 27 / Algorithm 1 line 7).
    Operator,
    /// The `max_steps` safety cap was reached before the operator fired
    /// (Table 11: `T_max` is a safety cap only); the answer is generated
    /// from the last state anyway.
    StepCap,
}

/// Result of one [`run`]: the full recorded trajectory, the final active
/// evidence, and the generated answer.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Every reasoning unit with its raw and normalized states.
    pub trajectory: Trajectory,
    /// Indices (chronological) of the steps active in the final evidence
    /// selection `ỹ` under budget `B`.
    pub active: Vec<usize>,
    /// The generated final answer.
    pub answer: String,
    /// Number of reasoning steps taken (units generated).
    pub steps: usize,
    /// Why the loop terminated.
    pub reason: StopReason,
}

/// Execute the `SoT` closed loop (Algorithm 1) with the training-free
/// operators.
///
/// * `evidence_op` / `stop_op` — the fixed `𝒮_tf` / `𝒯_tf` rules (Eq. 25-27).
/// * `normalizer` — the componentwise mean/std fit on offline training
///   trajectories (Eq. 18); use [`Normalizer::identity`] to run on raw
///   signatures.
/// * `budget` — `B`, the maximum number of historical steps kept active.
///   The paper's deployment budget is a token budget (Table 11: 1536);
///   this crate counts reasoning units and leaves token budgeting to the
///   backend's prompt materialization.
/// * `max_steps` — `T_max`, the safety cap on reasoning steps (Table 11:
///   32). Reaching it ends the loop with [`StopReason::StepCap`] and still
///   generates an answer from the last state.
///
/// Per-step flow (Algorithm 1): select `ỹ_{<t}` from the recorded states →
/// generate `y_t` → read the observation and append the state → re-select
/// `ỹ_{<t+1}` conditioned on `m_t` → stop check → on stop, answer from
/// `ỹ_{<t+1}` and `y_t` (the current unit is never duplicated inside
/// `evidence`).
#[allow(clippy::too_many_arguments)]
pub fn run(
    backend: &mut impl Backend,
    tracker: &mut impl StateTracker,
    evidence_op: &TrainingFreeEvidence,
    stop_op: &TrainingFreeStop,
    normalizer: &Normalizer,
    problem: &str,
    budget: usize,
    max_steps: usize,
) -> Outcome {
    let mut trajectory = Trajectory::new();
    let mut centers: Vec<Vec<f64>> = Vec::new();

    for steps in 1..=max_steps {
        // Algorithm 1 line 3: y_t ~ ℙ(· | p, ỹ_{<t}).
        let active = evidence_op.select(&trajectory, budget);
        let unit = backend.generate_step(problem, &units(&trajectory, &active));

        // Line 4: read out the endogenous state from y_t.
        let obs = tracker.observe(&unit);
        centers.push(obs.center.clone());
        let raw = raw_from_observation(&centers, &obs);
        trajectory.push(unit, raw, normalizer);

        // Line 5: ỹ_{<t+1} ← 𝒮(y_{≤t}; m_t) under budget B.
        let active = evidence_op.select(&trajectory, budget);

        // Lines 6-8: z_t ← 𝒯(m_t); on stop, answer from (ỹ_{<t+1}, y_t).
        if stop_op.should_stop(trajectory.states()) {
            let answer = answer_with(backend, &trajectory, &active, problem);
            return Outcome {
                trajectory,
                active,
                answer,
                steps,
                reason: StopReason::Operator,
            };
        }
    }

    // Safety cap: answer from the last state regardless.
    let active = evidence_op.select(&trajectory, budget);
    let answer = answer_with(backend, &trajectory, &active, problem);
    Outcome {
        trajectory,
        active,
        answer,
        steps: max_steps,
        reason: StopReason::StepCap,
    }
}

/// Materialize selected units as string slices for the backend.
fn units<'a>(trajectory: &'a Trajectory, active: &[usize]) -> Vec<&'a str> {
    let all = trajectory.units();
    active.iter().map(|&i| all[i].as_str()).collect()
}

/// Answer generation (Eq. 4): the current unit `y_t` is passed separately
/// and never duplicated inside the evidence list (Algorithm 1 line 8).
fn answer_with(
    backend: &mut impl Backend,
    trajectory: &Trajectory,
    active: &[usize],
    problem: &str,
) -> String {
    let last = trajectory.units().last().map_or("", String::as_str);
    let mut evidence = units(trajectory, active);
    if let Some(pos) = evidence.iter().position(|u| *u == last) {
        evidence.remove(pos);
    }
    backend.generate_answer(problem, &evidence, last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RawState;

    /// Scripted backend: fixed step lines, fixed answer, records every call.
    struct Scripted {
        steps: Vec<String>,
        answer: String,
        step_calls: Vec<Vec<String>>,
        answer_calls: Vec<(Vec<String>, String)>,
    }

    impl Scripted {
        fn new(steps: &[&str], answer: &str) -> Self {
            Self {
                steps: steps.iter().map(|s| (*s).to_owned()).collect(),
                answer: answer.to_owned(),
                step_calls: Vec::new(),
                answer_calls: Vec::new(),
            }
        }
    }

    impl Backend for Scripted {
        fn generate_step(&mut self, _problem: &str, evidence: &[&str]) -> String {
            self.step_calls
                .push(evidence.iter().map(|s| (*s).to_owned()).collect());
            self.steps
                .get(self.step_calls.len() - 1)
                .cloned()
                .unwrap_or_else(|| "overflow.".into())
        }

        fn generate_answer(&mut self, _problem: &str, evidence: &[&str], last: &str) -> String {
            self.answer_calls.push((
                evidence.iter().map(|s| (*s).to_owned()).collect(),
                last.to_owned(),
            ));
            self.answer.clone()
        }
    }

    /// Tracker producing distinct deterministic centers plus fixed proxies:
    /// center `[len, words]`, dispersion `0.25·k`, entropy `0.1` per unit.
    fn test_tracker() -> impl StateTracker {
        let mut k = 0usize;
        move |unit: &str| {
            k += 1;
            #[allow(clippy::cast_precision_loss)] // test fixture: text lengths ≪ 2^52.
            let len = unit.len() as f64;
            #[allow(clippy::cast_precision_loss)] // test fixture: word counts ≪ 2^52.
            let words = unit.split_whitespace().count() as f64;
            #[allow(clippy::cast_precision_loss)] // test fixture: step indices ≪ 2^52.
            let dispersion = 0.25 * k as f64;
            Observation::new(vec![len, words], dispersion, 0.1)
        }
    }

    /// Permissive stop: one stable state is enough at any threshold.
    fn eager_stop() -> TrainingFreeStop {
        TrainingFreeStop {
            r_stop: 1,
            h_max: f64::INFINITY,
            v_max: f64::INFINITY,
            c_min: f64::NEG_INFINITY,
        }
    }

    /// Impossible stop: never fires, so the loop always hits the cap.
    fn never_stop() -> TrainingFreeStop {
        TrainingFreeStop {
            r_stop: 1,
            h_max: f64::NEG_INFINITY,
            ..TrainingFreeStop::new()
        }
    }

    #[test]
    fn run_stops_via_operator_and_answers() {
        let mut backend = Scripted::new(&["one step.", "two steps."], "42");
        let mut tracker = test_tracker();
        let out = run(
            &mut backend,
            &mut tracker,
            &TrainingFreeEvidence::new(),
            &eager_stop(),
            &Normalizer::identity(),
            "what is 6*7?",
            4,
            32,
        );
        assert_eq!(out.reason, StopReason::Operator);
        assert_eq!(out.steps, 1);
        assert_eq!(out.answer, "42");
        assert_eq!(out.trajectory.len(), 1);
        assert_eq!(out.trajectory.units(), ["one step."]);
        // First step generates from empty evidence.
        assert_eq!(backend.step_calls, vec![Vec::<String>::new()]);
        // Answer got the current unit separately, no duplication.
        assert_eq!(
            backend.answer_calls,
            vec![(Vec::<String>::new(), "one step.".to_owned())]
        );
    }

    #[test]
    fn run_hits_step_cap_and_answers_from_last_state() {
        let mut backend = Scripted::new(&["one.", "two.", "three."], "done");
        let mut tracker = test_tracker();
        let out = run(
            &mut backend,
            &mut tracker,
            &TrainingFreeEvidence::new(),
            &never_stop(),
            &Normalizer::identity(),
            "p",
            8,
            2,
        );
        assert_eq!(out.reason, StopReason::StepCap);
        assert_eq!(out.steps, 2);
        assert_eq!(out.answer, "done");
        assert_eq!(out.trajectory.units(), ["one.", "two."]);
        assert_eq!(backend.answer_calls.len(), 1);
    }

    #[test]
    fn run_feeds_back_selected_evidence_under_budget() {
        let mut backend = Scripted::new(&["one.", "two.", "three."], "done");
        let mut tracker = test_tracker();
        let out = run(
            &mut backend,
            &mut tracker,
            &TrainingFreeEvidence::new(),
            &never_stop(),
            &Normalizer::identity(),
            "p",
            1,
            3,
        );
        assert_eq!(out.steps, 3);
        // Step 1: empty history. Steps 2-3: exactly one active unit each,
        // drawn from the recorded history.
        assert_eq!(backend.step_calls.len(), 3);
        assert!(backend.step_calls[0].is_empty());
        for call in &backend.step_calls[1..] {
            assert_eq!(call.len(), 1, "budget B = 1 must cap active evidence");
        }
        assert!(backend.step_calls[1].contains(&"one.".to_owned()));
        let last_evidence = backend
            .answer_calls
            .first()
            .expect("answer generated at the cap")
            .0
            .clone();
        assert!(
            last_evidence.iter().all(|u| u != "three."),
            "the triggering unit is passed separately, not duplicated: {last_evidence:?}"
        );
    }

    #[test]
    fn run_with_zero_max_steps_answers_immediately() {
        let mut backend = Scripted::new(&["unused."], "empty");
        let mut tracker = test_tracker();
        let out = run(
            &mut backend,
            &mut tracker,
            &TrainingFreeEvidence::new(),
            &never_stop(),
            &Normalizer::identity(),
            "p",
            4,
            0,
        );
        assert_eq!(out.reason, StopReason::StepCap);
        assert_eq!(out.steps, 0);
        assert!(out.trajectory.is_empty());
        assert!(backend.step_calls.is_empty());
        assert_eq!(backend.answer_calls.len(), 1);
    }

    #[test]
    fn run_records_raw_and_normalized_states() {
        let mut backend = Scripted::new(&["one."], "a");
        let mut tracker = test_tracker();
        let out = run(
            &mut backend,
            &mut tracker,
            &TrainingFreeEvidence::new(),
            &eager_stop(),
            &Normalizer::identity(),
            "p",
            4,
            8,
        );
        // Identity normalizer: raw == normalized; boundary rules v_1 = c_1 = 0.
        let raw = out.trajectory.last_raw().expect("one step recorded");
        approx_eq(raw.delta, 0.25);
        approx_eq(raw.v, 0.0);
        approx_eq(raw.c, 0.0);
        approx_eq(raw.h, 0.1);
    }

    /// Deterministic test encoder: `[char count, word count]`.
    struct LenEncoder;

    impl SentenceEncoder for LenEncoder {
        fn encode(&mut self, sentence: &str) -> Vec<f64> {
            #[allow(clippy::cast_precision_loss)] // test fixture: text lengths ≪ 2^52.
            let chars = sentence.chars().count() as f64;
            #[allow(clippy::cast_precision_loss)] // test fixture: word counts ≪ 2^52.
            let words = sentence.split_whitespace().count() as f64;
            vec![chars, words]
        }
    }

    #[test]
    fn embedding_tracker_wires_encoder_and_proxies() {
        let train: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![f64::from(i), 0.5 * f64::from(i)])
            .collect();
        let pca = Pca::fit(&train, 1);
        let pool = NearestPool::new(train.clone(), 3);
        let mut tracker = EmbeddingTracker::new(LenEncoder, pca, pool);

        let obs = tracker.observe("alpha beta.");
        assert_eq!(obs.center, vec![11.0, 2.0]);

        let pca = Pca::fit(&train, 1).expect("fit succeeds");
        approx_eq(obs.dispersion, pca.residual(&obs.center));
        let pool = NearestPool::new(train, 3);
        approx_eq(obs.entropy, pool.uncertainty(&obs.center));

        // Without summaries both proxies degrade to 0.0.
        let mut bare = EmbeddingTracker::new(LenEncoder, None, NearestPool::default());
        let obs = bare.observe("alpha beta.");
        approx_eq(obs.dispersion, 0.0);
        approx_eq(obs.entropy, 0.0);
    }

    #[test]
    fn embedding_tracker_drives_full_loop() {
        let train: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i), 1.0]).collect();
        let pca = Pca::fit(&train, 1);
        let pool = NearestPool::new(train, 2);
        let mut tracker = EmbeddingTracker::new(LenEncoder, pca, pool);
        // First unit's center [30, 6] sits off the training line y = 1, so
        // the PCA-residual δ-proxy is strictly positive.
        let mut backend = Scripted::new(&["a much longer first step here.", "two."], "ok");
        let out = run(
            &mut backend,
            &mut tracker,
            &TrainingFreeEvidence::new(),
            &never_stop(),
            &Normalizer::identity(),
            "p",
            4,
            2,
        );
        assert_eq!(out.steps, 2);
        // Proxies actually flow into the recorded raw states.
        let raws = out.trajectory.raw_states();
        assert!(raws[0].delta > 0.0, "PCA residual proxy recorded");
        assert!(raws[0].h > 0.0, "pool-distance proxy recorded");
        assert!(raws[1].v > 0.0, "dynamics from the embedding trajectory");
    }

    fn approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {a} ≈ {b}");
    }

    #[test]
    fn raw_state_builder_used_by_run_matches_manual() {
        // Guard the Eq. 13-17 wiring the loop relies on: observation
        // snapshots plus trajectory-derived dynamics.
        let centers = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![6.0, 4.0]];
        let obs = Observation::new(centers[2].clone(), 0.5, 0.7);
        let raw = raw_from_observation(&centers, &obs);
        let eq = RawState {
            delta: 0.5,
            v: 3.0, // ‖(6,4) − (3,4)‖
            c: 0.6, // cos((3,0),(3,4)) = 9/(3·5)
            h: 0.7,
        };
        approx_eq(raw.delta, eq.delta);
        approx_eq(raw.v, eq.v);
        approx_eq(raw.c, eq.c);
        approx_eq(raw.h, eq.h);
    }
}
