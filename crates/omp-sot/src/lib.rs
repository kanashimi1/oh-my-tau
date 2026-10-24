//! State of Thought (`SoT`) limited-access controllers, per arXiv 2609.16055
//! ("State of Thought Enables Endogenous Reasoning"), §4.1 / App. D.
//!
//! The paper's full `SoT` reads a learned controller off privileged hidden
//! states. This crate implements the two limited-access variants that remove
//! one of those privileges each:
//!
//! * **Training-free `SoT`** (App. D.1) keeps the endogenous 4-D state
//!   `m_t = [δ, v, c, H]` (App. B.1, Eq. 13-19) but replaces the learned
//!   evidence and stop heads with fixed rules: state-compatibility scoring of
//!   historical steps (Eq. 25) and a deterministic trailing-window stop
//!   (Eq. 27). No controller parameters are fitted.
//! * **SoT-Embed** (App. D.2, Eq. 28) removes hidden-state access: sentence
//!   units are mapped by a fixed [`embed::SentenceEncoder`] to `u_t`, and the
//!   control state is read from the embedding trajectory — dynamics
//!   coordinates [`state`](Eq. 15-16) computed exactly as in the paper, plus
//!   documented proxies for the two snapshot coordinates the paper leaves
//!   open ("compact trajectory summaries derived from `U`, such as a
//!   low-dimensional subspace fit on the training distribution"): PCA
//!   residual stands in for within-sentence dispersion δ, and mean distance
//!   to the k nearest training-pool embeddings stands in for predictive
//!   entropy H.
//!
//! The closed loop is [`sot::run`] (Algorithm 1): generate a sentence-level
//! unit → read the state → select active evidence under budget `B` → decide
//! stop → emit the final answer.
//!
//! Both trackers and both controller operators are pure data structures with
//! deterministic outputs; the only I/O is the injected [`sot::Backend`] that
//! generates text.

pub mod controller;
pub mod embed;
pub mod segment;
pub mod sot;
pub mod state;

pub use controller::{TrainingFreeEvidence, TrainingFreeStop};
pub use sot::{Backend, Outcome, StateTracker};
pub use state::{Normalizer, Observation, RawState, State, Trajectory, C, DELTA, H, V};
