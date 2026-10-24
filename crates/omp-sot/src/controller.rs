//! Training-free state-conditioned operators `𝒮_tf` and `𝒯_tf` (App. D.1,
//! Eq. 25-27).
//!
//! Both operators replace the paper's learned 582-parameter controller heads
//! with fixed rules on the *same* endogenous state interface `m_t`: no
//! controller parameters are fitted. The evidence rule scores each
//! historical step by state compatibility with the current reasoning
//! regime; the stop rule terminates only after the state has stayed in a
//! stable low-uncertainty regime for a trailing window of steps, so the
//! loop cannot stop on transient confidence alone.

use crate::state::{State, Trajectory, C, H, V};

/// Evidence-organization operator `𝒮_tf` (Eq. 25-26): fixed state-compatibility
/// scoring of historical reasoning steps.
///
/// Each historical step `y_j` paired with state `m_j` is scored against the
/// current state `m_t`:
///
/// `s_tf_{t,j} = α1·⟨m_t, m_j⟩ − α2·‖m_t − m_j‖₂ + α3·c_j − α4·H_j`  (Eq. 25)
///
/// where `⟨m_t, m_j⟩` encourages compatibility between the current and
/// historical reasoning regimes, `‖m_t − m_j‖₂` penalizes mismatched states,
/// `c_j` favors directionally stable historical steps, and `H_j` downweights
/// locally uncertain ones. Under budget `B`, the deterministic operator keeps
/// the top-scoring historical steps (Eq. 26), inserted back in chronological
/// order.
///
/// All states are the normalized controller-facing states (Eq. 19), so the
/// dot-product and distance terms live in units of the offline training
/// distribution. The paper publishes no numeric α values; the default is a
/// uniform unit weight on all four terms.
#[derive(Clone, Debug)]
pub struct TrainingFreeEvidence {
    /// α1 — regime-compatibility (dot product) weight, Eq. 25.
    pub alpha_dot: f64,
    /// α2 — state-mismatch (Euclidean distance) weight, Eq. 25.
    pub alpha_dist: f64,
    /// α3 — directional-stability (`c_j`) weight, Eq. 25.
    pub alpha_c: f64,
    /// α4 — local-uncertainty (`H_j`) weight, Eq. 25.
    pub alpha_h: f64,
}

impl Default for TrainingFreeEvidence {
    /// Uniform unit weights on all four Eq. 25 terms (the paper publishes
    /// no numeric α values).
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingFreeEvidence {
    /// Uniform unit weights on all four Eq. 25 terms.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha_dot: 1.0,
            alpha_dist: 1.0,
            alpha_c: 1.0,
            alpha_h: 1.0,
        }
    }

    /// Eq. 25 score of one historical state `m_j` against the current state
    /// `m_t`. Pure function of the two normalized states.
    #[must_use]
    pub fn score(&self, m_t: &State, m_j: &State) -> f64 {
        let dot: f64 = m_t.iter().zip(m_j).map(|(a, b)| a * b).sum();
        let dist = m_t
            .iter()
            .zip(m_j)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        self.alpha_dot * dot - self.alpha_dist * dist + self.alpha_c * m_j[C]
            - self.alpha_h * m_j[H]
    }

    /// Eq. 26: keep the top-`budget` scoring historical steps, conditioned
    /// on the *last* recorded state (the current reasoning regime), and
    /// return their indices in chronological order.
    ///
    /// An empty trajectory or `budget == 0` selects nothing. Every recorded
    /// unit is a candidate — including the unit whose state is conditioned
    /// on (it typically scores highest: identical regimes); callers that
    /// materialize the context decide how the current unit is presented
    /// (see [`crate::sot::run`], which passes it separately to answer
    /// generation per Algorithm 1 line 8).
    #[must_use]
    pub fn select(&self, traj: &Trajectory, budget: usize) -> Vec<usize> {
        let states = traj.states();
        if budget == 0 || states.is_empty() {
            return Vec::new();
        }
        let m_t = states[states.len() - 1];
        // Stable sort keeps chronological order among equal scores.
        let mut scored: Vec<(usize, f64)> = (0..states.len())
            .map(|j| (j, self.score(&m_t, &states[j])))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(budget);
        let mut kept: Vec<usize> = scored.into_iter().map(|(j, _)| j).collect();
        kept.sort_unstable();
        kept
    }
}

/// Stopping operator `𝒯_tf` (Eq. 27): deterministic trailing-window stop.
///
/// Terminates only when the state has remained in a stable low-uncertainty
/// regime for `r_stop` consecutive steps — jointly low `H_t`, sufficiently
/// small movement `v_t`, and positive directional consistency `c_t` over the
/// whole trailing window `m_{t−r_stop+1:t}`, so the controller does not
/// terminate on transient confidence alone.
///
/// All thresholds apply to *normalized* coordinates (Eq. 19), i.e. they are
/// expressed relative to the offline training distribution. The paper
/// publishes no absolute values; the defaults sit at the normalization
/// origin: `H` and `v` at-or-below the training mean, `c` at-or-above it.
#[derive(Clone, Debug)]
pub struct TrainingFreeStop {
    /// `r_stop` — trailing window length: number of consecutive stable
    /// states required before stopping (Eq. 27). `0` disables
    /// operator-driven stopping entirely (the step cap still applies).
    pub r_stop: usize,
    /// Normalized-`H` ceiling: `H_t ≤ h_max` counts as low uncertainty.
    pub h_max: f64,
    /// Normalized-`v` ceiling: `v_t ≤ v_max` counts as sufficiently small
    /// state movement.
    pub v_max: f64,
    /// Normalized-`c` floor: `c_t ≥ c_min` counts as positive directional
    /// consistency.
    pub c_min: f64,
}

impl Default for TrainingFreeStop {
    /// `r_stop = 3` with all thresholds at the normalization origin
    /// (at/below the training mean for `H` and `v`, at/above it for `c`).
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingFreeStop {
    /// `r_stop = 3` with all thresholds at the normalization origin (the
    /// paper publishes no absolute values).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            r_stop: 3,
            h_max: 1.0,
            v_max: 1.0,
            c_min: 0.0,
        }
    }

    /// Eq. 27: stop once *every* state in the trailing window of `r_stop`
    /// states is stable (low `H`, small `v`, positive `c`). Fewer than
    /// `r_stop` states have been observed, or `r_stop == 0` → never stop.
    #[must_use]
    pub fn should_stop(&self, states: &[State]) -> bool {
        if self.r_stop == 0 || states.len() < self.r_stop {
            return false;
        }
        states[states.len() - self.r_stop..]
            .iter()
            .all(|m| m[H] <= self.h_max && m[V] <= self.v_max && m[C] >= self.c_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Normalizer, RawState};

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {a} ≈ {b}");
    }

    fn state(delta: f64, v: f64, c: f64, h: f64) -> State {
        // Identity normalizer: normalized state == raw signature.
        Normalizer::identity().normalize(RawState { delta, v, c, h })
    }

    fn traj_with(states: &[State]) -> Trajectory {
        let mut traj = Trajectory::new();
        let n = Normalizer::identity();
        for (i, m) in states.iter().enumerate() {
            traj.push(
                format!("unit {i}."),
                RawState {
                    delta: m[0],
                    v: m[1],
                    c: m[2],
                    h: m[3],
                },
                &n,
            );
        }
        traj
    }

    #[test]
    fn score_matches_eq25() {
        let op = TrainingFreeEvidence::new();
        let m_t = state(0.0, 0.0, 0.0, 0.5);
        let m_j = state(0.0, 0.0, 0.5, 0.25);
        // dot = 0.5·0.25 = 0.125; dist = √(0.5² + 0.25²) = √0.3125;
        // +c_j −H_j = +0.5 −0.25.
        let expected = 0.125 - 0.3125f64.sqrt() + 0.5 - 0.25;
        approx(op.score(&m_t, &m_j), expected);
    }

    #[test]
    fn score_weights_apply() {
        let op = TrainingFreeEvidence {
            alpha_dot: 0.0,
            alpha_dist: 0.0,
            alpha_c: 2.0,
            alpha_h: 1.0,
        };
        let m_t = state(0.0, 0.0, 0.0, 0.0);
        let m_j = state(0.0, 0.0, 0.3, 0.4);
        // 2·c_j − H_j = 0.6 − 0.4.
        approx(op.score(&m_t, &m_j), 0.2);
    }

    #[test]
    fn select_empty_and_zero_budget() {
        let op = TrainingFreeEvidence::new();
        assert!(op.select(&Trajectory::new(), 3).is_empty());
        let traj = traj_with(&[state(0.0, 0.0, 0.0, 0.0)]);
        assert!(op.select(&traj, 0).is_empty());
    }

    #[test]
    fn select_keeps_top_scoring_in_chronological_order() {
        // Condition on m_t = [0,0,0,0]: dot = 0 for all, so the score is
        // −‖m_j‖₂ + c_j − H_j over all four coordinates:
        //   j0 = (−√0.25+0.81) + 0.9        ≈ −0.1296
        //   j1 = (−√(4+1+25)) + (−1) − 5    ≈ −11.477
        //   j2 = (−√(0.36+0.01)) + 0.6 − 0.1 ≈ −0.1083
        //   j3 = 0 (m_t itself)
        let m_t = state(0.0, 0.0, 0.0, 0.0);
        let m1 = state(0.5, 0.0, 0.9, 0.0);
        let m2 = state(0.0, 2.0, -1.0, 5.0);
        let m3 = state(0.0, 0.0, 0.6, 0.1);
        let traj = traj_with(&[m1, m2, m3, m_t]);
        let op = TrainingFreeEvidence::new();
        assert_eq!(op.select(&traj, 2), vec![2, 3]);
        assert_eq!(op.select(&traj, 3), vec![0, 2, 3]);
        assert_eq!(op.select(&traj, 10), vec![0, 1, 2, 3]);
    }

    #[test]
    fn select_conditions_on_last_state() {
        // The last recorded state is the current regime and scores highest
        // against itself (dist = 0), so with budget 1 the current unit is
        // its own best evidence.
        let op = TrainingFreeEvidence::new();
        let mut traj = traj_with(&[state(1.0, 1.0, 1.0, 1.0)]);
        traj.push(
            "current.".into(),
            RawState {
                delta: 0.0,
                v: 0.0,
                c: 0.0,
                h: 0.0,
            },
            &Normalizer::identity(),
        );
        assert_eq!(op.select(&traj, 1), vec![1]);
    }

    #[test]
    fn stop_requires_full_stable_window() {
        let op = TrainingFreeStop {
            r_stop: 2,
            h_max: 0.3,
            v_max: 0.3,
            c_min: 0.4,
        };
        // Two trailing stable states → stop.
        let stable = [state(0.0, 0.1, 0.5, 0.1), state(0.0, 0.2, 0.6, 0.2)];
        assert!(op.should_stop(&stable));
        // Only one state so far → cannot establish stability.
        assert!(!op.should_stop(&stable[1..]));
        // An unstable state inside the window blocks stopping.
        let unstable_tail = [state(0.0, 0.1, 0.5, 0.1), state(0.0, 0.9, -0.5, 0.8)];
        assert!(!op.should_stop(&unstable_tail));
        // High H blocks even with small v and positive c.
        let uncertain_tail = [state(0.0, 0.1, 0.5, 0.1), state(0.0, 0.1, 0.5, 0.9)];
        assert!(!op.should_stop(&uncertain_tail));
        // r_stop = 0 disables operator-driven stopping.
        let disabled = TrainingFreeStop {
            r_stop: 0,
            ..TrainingFreeStop::new()
        };
        assert!(!disabled.should_stop(&stable));
    }

    #[test]
    fn stop_defaults_at_normalization_origin() {
        let op = TrainingFreeStop::new();
        // Exactly at the origin (training mean) counts as stable.
        let at_mean = [state(0.0, 0.0, 0.0, 0.0); 3];
        assert!(op.should_stop(&at_mean));
        // Any normalized coordinate on the unstable side blocks it.
        let hot = [
            state(0.0, 0.0, 0.0, 0.0),
            state(0.0, 0.0, 0.0, 0.0),
            state(0.0, 2.0, 0.0, 0.0),
        ];
        assert!(!op.should_stop(&hot));
    }
}
