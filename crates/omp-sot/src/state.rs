//! Endogenous reasoning state `m_t` (App. B.1, Eq. 13-19).
//!
//! The raw controller signature `s_t = [δ_t, v_t, c_t, H_t]` combines geometry,
//! dynamics, and uncertainty:
//!
//! * `δ` — within-sentence dispersion around the sentence center (Eq. 14);
//! * `v` — displacement between consecutive sentence centers (Eq. 15);
//! * `c` — cosine alignment between consecutive displacements (Eq. 16);
//! * `H` — mean next-token predictive entropy over the sentence (Eq. 17).
//!
//! Boundary steps follow the paper: `v_1 = c_1 = 0`, `c_2 = 0`. The signature
//! is normalized componentwise with offline mean/std (Eq. 18); the result is
//! the controller-facing state `m_t` (Eq. 19).
//!
//! `v` and `c` depend only on the sentence-center *trajectory*, so they are
//! computed here from `u_t` vectors supplied by the caller — hidden-state
//! telemetry for the training-free variant, sentence embeddings for
//! SoT-Embed. `δ` and `H` must be supplied per observation: exact values from
//! hidden states, or the proxies documented in [`crate::embed`].

/// Index of the geometric coordinate δ (within-unit dispersion).
pub const DELTA: usize = 0;
/// Index of the movement coordinate v (step displacement).
pub const V: usize = 1;
/// Index of the directional coordinate c (cosine alignment).
pub const C: usize = 2;
/// Index of the uncertainty coordinate H (predictive entropy).
pub const H: usize = 3;

/// Numerical-stability constant for the `c` denominator (Eq. 16).
pub const EPSILON: f64 = 1e-8;

/// Normalized controller-facing endogenous state `m_t ∈ ℝ⁴` (Eq. 19).
pub type State = [f64; 4];

/// Raw controller signature `s_t` before normalization (Eq. 13).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawState {
    /// Within-unit dispersion δ (Eq. 14).
    pub delta: f64,
    /// Step displacement v (Eq. 15).
    pub v: f64,
    /// Directional consistency c (Eq. 16).
    pub c: f64,
    /// Predictive entropy H (Eq. 17).
    pub h: f64,
}

impl RawState {
    /// Componentwise view as an array `[δ, v, c, H]`.
    #[must_use]
    pub fn as_array(&self) -> State {
        [self.delta, self.v, self.c, self.h]
    }
}

/// Per-step observation a state source supplies for one reasoning unit:
/// the sentence center `u_t` plus the two snapshot coordinates the dynamics
/// cannot derive from the center trajectory.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Sentence center `u_t` — mean final-layer hidden state (Eq. 12) or a
    /// sentence embedding (Eq. 28).
    pub center: Vec<f64>,
    /// Within-unit dispersion δ (Eq. 14), or the SoT-Embed proxy.
    pub dispersion: f64,
    /// Mean predictive entropy H (Eq. 17), or the SoT-Embed proxy.
    pub entropy: f64,
}

impl Observation {
    /// Build an observation from raw parts.
    #[must_use]
    pub fn new(center: Vec<f64>, dispersion: f64, entropy: f64) -> Self {
        Self {
            center,
            dispersion,
            entropy,
        }
    }
}

/// Componentwise mean/std normalization fitted on offline trajectories
/// (Eq. 18). The identity normalizer (default) passes raw signatures through.
#[derive(Clone, Debug)]
pub struct Normalizer {
    mean: State,
    std: State,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::identity()
    }
}

impl Normalizer {
    /// Normalizer that leaves signatures unchanged.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            mean: [0.0; 4],
            std: [1.0; 4],
        }
    }

    /// Fit `μ` and `σ` componentwise on raw signatures from offline training
    /// trajectories. Uses the population std (`1/n`), matching the paper's
    /// "componentwise mean and standard deviation estimated from the offline
    /// training trajectories". Needs at least two samples; returns the
    /// identity normalizer for shorter inputs.
    #[must_use]
    pub fn fit(samples: &[RawState]) -> Self {
        let n = samples.len();
        if n < 2 {
            return Self::identity();
        }
        #[allow(clippy::cast_precision_loss)] // n is a trajectory length, far below 2^52.
        let nf = n as f64;
        let mut mean = [0.0; 4];
        for s in samples {
            let a = s.as_array();
            for (m, x) in mean.iter_mut().zip(a) {
                *m += x / nf;
            }
        }
        let mut std = [0.0; 4];
        for s in samples {
            let a = s.as_array();
            for ((sd, mu), x) in std.iter_mut().zip(mean).zip(a) {
                *sd += (x - mu).powi(2) / nf;
            }
        }
        for sd in &mut std {
            *sd = sd.sqrt().max(EPSILON);
        }
        Self { mean, std }
    }

    /// Componentwise mean `μ` of the fitted normalization (Eq. 18).
    #[must_use]
    pub fn mean(&self) -> State {
        self.mean
    }

    /// Componentwise standard deviation `σ` of the fitted normalization
    /// (Eq. 18).
    #[must_use]
    pub fn std(&self) -> State {
        self.std
    }

    /// Build a normalizer from a previously persisted `(μ, σ)` fit
    /// (deserialized offline statistics).
    #[must_use]
    pub fn from_mean_std(mean: State, std: State) -> Self {
        Self { mean, std }
    }

    /// Apply Eq. 18: `(s − μ) / max(σ, ε)`. The denominator is floored at
    /// ε rather than shifted by it, so an already-stable scale (e.g. the
    /// identity normalizer's unit std) passes values through unchanged.
    #[must_use]
    pub fn normalize(&self, s: RawState) -> State {
        let a = s.as_array();
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = (a[i] - self.mean[i]) / self.std[i].max(EPSILON);
        }
        out
    }
}

/// The evolving reasoning trajectory `(y_1, m_1), …, (y_t, m_t)`: sentence
/// units paired with their (raw and normalized) endogenous states.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    units: Vec<String>,
    raw: Vec<RawState>,
    states: Vec<State>,
}

impl Trajectory {
    /// Empty trajectory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded reasoning units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Whether no reasoning unit has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Append a unit and its raw signature, storing both the raw and the
    /// normalized state.
    pub fn push(&mut self, unit: String, raw: RawState, normalizer: &Normalizer) {
        self.units.push(unit);
        self.raw.push(raw);
        self.states.push(normalizer.normalize(raw));
    }

    /// Sentence units `y_1..y_t` in order.
    #[must_use]
    pub fn units(&self) -> &[String] {
        &self.units
    }

    /// Raw signatures `s_1..s_t`.
    #[must_use]
    pub fn raw_states(&self) -> &[RawState] {
        &self.raw
    }

    /// Normalized controller-facing states `m_1..m_t` (Eq. 19).
    #[must_use]
    pub fn states(&self) -> &[State] {
        &self.states
    }

    /// The raw signature of step `t` (last step), if any.
    #[must_use]
    pub fn last_raw(&self) -> Option<&RawState> {
        self.raw.last()
    }
}

/// Compute the dynamical coordinates `v_t` and `c_t` (Eq. 15-16) for the
/// next step from the trailing sentence centers and the snapshot coordinates
/// of an [`Observation`].
///
/// `centers` must be the full center trajectory *including* the new
/// observation's center (i.e., after appending `obs.center`).
#[must_use]
pub fn dynamics(centers: &[Vec<f64>], _obs: &Observation) -> (f64, f64) {
    let t = centers.len();
    if t < 2 {
        // Boundary: v_1 = c_1 = 0.
        return (0.0, 0.0);
    }
    let u_t = &centers[t - 1];
    let u_prev = &centers[t - 2];
    let v = l2(u_t, u_prev);
    if t < 3 {
        // Boundary: c_2 = 0.
        return (v, 0.0);
    }
    let u_prev2 = &centers[t - 3];
    let d1 = sub(u_t, u_prev);
    let d2 = sub(u_prev, u_prev2);
    let num = dot(&d1, &d2);
    let den = l2_norm(&d1) * l2_norm(&d2);
    let c = num / (den + EPSILON);
    (v, c)
}

/// Build the raw signature for one step from the center trajectory and an
/// observation: exact Eq. 13-17 semantics for any center source.
#[must_use]
pub fn raw_from_observation(centers: &[Vec<f64>], obs: &Observation) -> RawState {
    let (v, c) = dynamics(centers, obs);
    RawState {
        delta: obs.dispersion,
        v,
        c,
        h: obs.entropy,
    }
}

/// Euclidean distance between two equal-length vectors.
#[must_use]
pub fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// L2 norm of a vector.
#[must_use]
pub fn l2_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// Dot product of two equal-length vectors.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity of two non-zero vectors (0 if either has ~zero norm).
#[must_use]
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let den = l2_norm(a) * l2_norm(b);
    if den < EPSILON {
        0.0
    } else {
        dot(a, b) / den
    }
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {a} ≈ {b}");
    }

    #[test]
    fn boundary_rules() {
        let obs = Observation::new(vec![1.0, 0.0], 0.5, 0.7);
        let (v, c) = dynamics(&[vec![1.0, 0.0]], &obs);
        approx(v, 0.0);
        approx(c, 0.0);
    }

    #[test]
    fn v_and_c_match_equations() {
        // u1 -> u2 -> u3: u3 continues the direction of u2->u3? Construct a
        // straight line: u1=(0,0), u2=(1,0), u3=(2,0): v=1, c≈1.
        let centers = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let obs = Observation::new(centers[2].clone(), 0.1, 0.2);
        let (v, c) = dynamics(&centers, &obs);
        approx(v, 1.0);
        approx(c, 1.0 / (1.0 + EPSILON));
        // Right-angle turn: u3=(1,1): v=1, c≈0.
        let centers = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let obs = Observation::new(centers[2].clone(), 0.1, 0.2);
        let (v, c) = dynamics(&centers, &obs);
        approx(v, 1.0);
        approx(c, 0.0);
        // Reversal: u3 = u1: c ≈ -1.
        let centers = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let obs = Observation::new(centers[2].clone(), 0.1, 0.2);
        let (_v, c) = dynamics(&centers, &obs);
        assert!(c < -0.99, "expected reversal c ≈ -1, got {c}");
        // c_2 boundary with three centers where third is the new one:
        // t=2 means centers has 2 entries → covered above; t=3 checks c.
    }

    #[test]
    fn c2_boundary_is_zero() {
        let centers = vec![vec![0.0], vec![5.0]];
        let obs = Observation::new(centers[1].clone(), 0.0, 0.0);
        let (v, c) = dynamics(&centers, &obs);
        approx(v, 5.0);
        approx(c, 0.0);
    }

    #[test]
    fn normalizer_zero_mean_unit_std() {
        let samples = vec![
            RawState {
                delta: 1.0,
                v: 2.0,
                c: -1.0,
                h: 0.5,
            },
            RawState {
                delta: 3.0,
                v: 4.0,
                c: 1.0,
                h: 1.5,
            },
            RawState {
                delta: 5.0,
                v: 6.0,
                c: 0.0,
                h: 2.5,
            },
        ];
        let n = Normalizer::fit(&samples);
        let mut mean = [0.0; 4];
        for s in &samples {
            let m = n.normalize(*s);
            for i in 0..4 {
                mean[i] += m[i] / 3.0;
            }
        }
        for m in mean {
            approx(m, 0.0);
        }
        // First sample normalized to [-1.22..] under population std of
        // equally spaced values: (1-3)/1.633 ≈ -1.2247.
        let m0 = n.normalize(samples[0]);
        approx(m0[DELTA], -2.0 / (8.0f64 / 3.0).sqrt());
    }

    #[test]
    fn normalizer_short_input_is_identity() {
        let n = Normalizer::fit(&[RawState::default()]);
        let m = n.normalize(RawState {
            delta: 2.0,
            v: 0.0,
            c: 0.0,
            h: 0.0,
        });
        approx(m[DELTA], 2.0);
    }

    #[test]
    fn trajectory_roundtrip() {
        let mut traj = Trajectory::new();
        assert!(traj.is_empty());
        let n = Normalizer::identity();
        traj.push(
            "a".into(),
            RawState {
                delta: 0.0,
                v: 0.0,
                c: 0.0,
                h: 0.1,
            },
            &n,
        );
        traj.push(
            "b".into(),
            RawState {
                delta: 1.0,
                v: 0.5,
                c: 0.5,
                h: 0.2,
            },
            &n,
        );
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.units(), ["a", "b"]);
        approx(traj.last_raw().unwrap().h, 0.2);
        approx(traj.states()[1][C], 0.5);
    }
}
