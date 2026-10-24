//! SoT-Embed building blocks (App. D.2, Eq. 28): a fixed sentence encoder, a
//! PCA subspace fit on the training distribution, and a training-pool
//! proximity readout.
//!
//! The paper specifies only that control inputs are built "directly from the
//! embedding trajectory": the controller conditions on `u_t` and compact
//! trajectory summaries "such as a low-dimensional subspace fit on the
//! training distribution (PCA coordinates per step)". The snapshot proxies
//! used here, replacing the hidden-state-only coordinates δ and H, are:
//!
//! * δ-proxy — squared PCA reconstruction residual of `u_t`: how far the
//!   step sits from the training distribution's principal subspace (the
//!   embedding analogue of "internally concentrated or diffuse").
//! * H-proxy — mean cosine distance from `u_t` to its `k` nearest neighbors
//!   in the training-pool embeddings: how resolved the step is against seen
//!   material (the embedding analogue of predictive entropy).
//!
//! The dynamics coordinates `v` and `c` are computed on the embedding
//! trajectory exactly as in Eq. 15-16 (see [`crate::state::dynamics`]).

use crate::state::{cosine, dot, l2_norm, EPSILON};

/// Fixed sentence encoder `Enc: y_t ↦ u_t ∈ ℝ^d` (Eq. 28). Implementations
/// wrap whatever model is at hand (local ONNX/runtime embedding, a provider
/// embeddings endpoint, a tokenizer-level mean pooler); the controller only
/// sees the vector.
pub trait SentenceEncoder {
    /// Encode one sentence-level reasoning unit into an embedding.
    fn encode(&mut self, sentence: &str) -> Vec<f64>;
}

/// A PCA subspace fit on the training distribution: `mean` plus the top-`k`
/// eigenvectors of the embedding covariance, eigenvalues in descending
/// order.
#[derive(Clone, Debug)]
pub struct Pca {
    mean: Vec<f64>,
    components: Vec<Vec<f64>>,
    eigenvalues: Vec<f64>,
}

impl Pca {
    /// Fit the subspace on training embeddings. Returns `None` when the fit
    /// is underdetermined (fewer than two points, zero dimensions, or `k = 0`).
    ///
    /// The eigendecomposition is deterministic power iteration with explicit
    /// symmetric deflation (`cov ← cov − λvvᵀ`); no external linear-algebra
    /// dependency is pulled in for what is an offline, small-`k` fit.
    #[allow(clippy::many_single_char_names)] // covariance eigendecomposition in the paper's notation (n, d, k, v, w)
    #[must_use]
    pub fn fit(points: &[Vec<f64>], k: usize) -> Option<Self> {
        let n = points.len();
        let d = points.first().map_or(0, Vec::len);
        if n < 2 || d == 0 || k == 0 || points.iter().any(|p| p.len() != d) {
            return None;
        }
        #[allow(clippy::cast_precision_loss)] // n is a training-pool size, far below 2^52.
        let nf = n as f64;
        let mut mean = vec![0.0; d];
        for p in points {
            for (m, x) in mean.iter_mut().zip(p) {
                *m += x / nf;
            }
        }
        // Covariance (population, 1/n — consistent with [`crate::state::Normalizer`]).
        let mut cov = vec![vec![0.0; d]; d];
        for p in points {
            for i in 0..d {
                let di = p[i] - mean[i];
                if di == 0.0 {
                    continue;
                }
                for j in i..d {
                    let dj = p[j] - mean[j];
                    cov[i][j] += di * dj / nf;
                    if i != j {
                        cov[j][i] = cov[i][j];
                    }
                }
            }
        }
        let k = k.min(d).min(n - 1);
        let mut components: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut eigenvalues: Vec<f64> = Vec::with_capacity(k);
        for _comp_idx in 0..k {
            // Deterministic non-degenerate start vector (generic smooth
            // pattern; not aligned with typical principal axes by
            // construction).
            let mut v: Vec<f64> = (0..d)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    // i indexes embedding dims, far below 2^52.
                    let x = (i + 1) as f64;
                    (0.9 * x).sin() + 0.1 * (0.4 * x).cos()
                })
                .collect();
            orthogonalize(&mut v, &components);
            normalize_in_place(&mut v);
            if l2_norm(&v) < EPSILON {
                // Degenerate (fully deflated) matrix: pad with zero
                // components to keep `k` coordinates.
                components.push(vec![0.0; d]);
                eigenvalues.push(0.0);
                continue;
            }
            let mut lambda = 0.0;
            for _ in 0..500 {
                let w = mat_vec(&cov, &v);
                let new_lambda = dot(&w, &v);
                let mut next = w;
                orthogonalize(&mut next, &components);
                normalize_in_place(&mut next);
                let shift = crate::state::l2(&next, &v);
                v = next;
                lambda = new_lambda;
                if shift < 1e-12 {
                    break;
                }
            }
            // Explicit symmetric deflation.
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] -= lambda * v[i] * v[j];
                }
            }
            components.push(v);
            eigenvalues.push(lambda);
        }
        Some(Self {
            mean,
            components,
            eigenvalues,
        })
    }

    /// Number of retained principal coordinates.
    #[must_use]
    pub fn k(&self) -> usize {
        self.components.len()
    }

    /// Eigenvalues in descending order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// PCA coordinates of `u` in the fitted subspace.
    #[must_use]
    pub fn project(&self, u: &[f64]) -> Vec<f64> {
        let centered: Vec<f64> = u.iter().zip(&self.mean).map(|(x, m)| x - m).collect();
        self.components.iter().map(|c| dot(c, &centered)).collect()
    }

    /// The retained principal components (unit eigenvectors, eigenvalue
    /// order). Exposed for validation and downstream readouts.
    #[must_use]
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// δ-proxy: squared reconstruction residual `‖u − mean‖² − ‖proj‖²`.
    #[must_use]
    pub fn residual(&self, u: &[f64]) -> f64 {
        let centered_sq: f64 = u.iter().zip(&self.mean).map(|(x, m)| (x - m).powi(2)).sum();
        let proj_sq: f64 = self.project(u).iter().map(|x| x.powi(2)).sum();
        (centered_sq - proj_sq).max(0.0)
    }
}

/// Training-pool embeddings used for the H-proxy: mean cosine distance to
/// the `k` nearest pool members (0.0 when the pool is empty).
#[derive(Clone, Debug, Default)]
pub struct NearestPool {
    embeddings: Vec<Vec<f64>>,
    k: usize,
}

impl NearestPool {
    /// Pool over the given training embeddings with `k` neighbors.
    #[must_use]
    pub fn new(embeddings: Vec<Vec<f64>>, k: usize) -> Self {
        Self { embeddings, k }
    }

    /// H-proxy for `u`: mean `1 − cos(u, pool_i)` over the `k` nearest pool
    /// embeddings (clamped at `0.0` for an empty pool).
    #[must_use]
    pub fn uncertainty(&self, u: &[f64]) -> f64 {
        if self.embeddings.is_empty() || self.k == 0 {
            return 0.0;
        }
        let mut dists: Vec<f64> = self.embeddings.iter().map(|p| 1.0 - cosine(u, p)).collect();
        dists.sort_by(f64::total_cmp);
        let take = self.k.min(dists.len());
        #[allow(clippy::cast_precision_loss)] // take ≤ pool size, far below 2^52.
        let mean: f64 = dists[..take].iter().sum::<f64>() / take as f64;
        mean
    }
}

fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

fn orthogonalize(v: &mut [f64], against: &[Vec<f64>]) {
    for q in against {
        let proj = dot(v, q);
        for (vi, qi) in v.iter_mut().zip(q) {
            *vi -= proj * qi;
        }
    }
}

fn normalize_in_place(v: &mut [f64]) {
    let n = l2_norm(v);
    if n > EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG noise so the PCA tests need no external crates.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)] // test noise: 53-bit mantissa shift only.
            let v = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
            v
        }
    }

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {a} ≈ {b}");
    }

    #[test]
    fn pca_recovers_principal_direction() {
        // Points along the (1, 2) direction plus small orthogonal noise.
        let mut rng = Lcg(42);
        let mut points = Vec::new();
        for _ in 0..200 {
            let t = rng.next_f64() * 10.0 - 5.0;
            let noise = (rng.next_f64() - 0.5) * 0.01;
            points.push(vec![t + noise, 2.0 * t - noise]);
        }
        let pca = Pca::fit(&points, 1).unwrap();
        // First component ~ ±(1, 2)/√5.
        let n = 5.0f64.sqrt();
        approx(pca.components()[0][0].abs(), 1.0 / n, 0.02);
        approx(pca.components()[0][1].abs(), 2.0 / n, 0.02);
        // On-manifold point projects far from the mean; residual ~ 0.
        approx(pca.residual(&[1.0, 2.0]), 0.0, 1e-6);
    }

    #[test]
    fn pca_residual_small_on_manifold() {
        let mut rng = Lcg(7);
        let mut points = Vec::new();
        for _ in 0..100 {
            let t = rng.next_f64() * 4.0 - 2.0;
            points.push(vec![t, 3.0 * t]);
        }
        let pca = Pca::fit(&points, 1).unwrap();
        let on_manifold = vec![1.0, 3.0];
        let off_manifold = vec![3.0, -1.0];
        assert!(pca.residual(&on_manifold) < pca.residual(&off_manifold));
        approx(pca.residual(&on_manifold), 0.0, 1e-6);
    }

    #[test]
    fn pca_fit_rejects_underdetermined() {
        assert!(Pca::fit(&[], 2).is_none());
        assert!(Pca::fit(&[vec![1.0, 2.0]], 2).is_none());
        assert!(Pca::fit(&[vec![1.0, 2.0], vec![3.0, 4.0]], 0).is_none());
        assert!(Pca::fit(&[vec![1.0], vec![2.0], vec![3.0]], 2).is_some());
    }

    #[test]
    fn nearest_pool_uncertainty() {
        let pool = NearestPool::new(vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]], 2);
        // Close to the first two members → small mean distance.
        let near = pool.uncertainty(&[1.0, 0.05]);
        // Orthogonal to everything → larger.
        let far = pool.uncertainty(&[-1.0, -0.05]);
        assert!(near < far, "near={near} far={far}");
        assert!(NearestPool::default().uncertainty(&[1.0]) < f64::EPSILON);
    }
}
