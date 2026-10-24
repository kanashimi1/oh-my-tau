//! Laya turn judge for omp-rs.
//!
//! * [`episode`] — session entries -> episodes (port of `build_traj_outcome.py`: one real
//!   user message -> steps until the next real user message) and the flat `state` / objective.
//! * [`view`] — the judge input (port of `judge_view.view_sel`, byte-identical).
//! * [`policy`] — context-policy head (`noop | clm_edit | rlm_offload`) over the same ONNX runtime.
//! * [`scorer`] — ONNX inference through `laya-ort`.
//! * [`shadow`] — env-gated background worker for the turn-end shadow hook.
//! * [`offline`] — score historical session files with the same code.

pub mod episode;
pub mod offline;
pub mod policy;
pub mod pyfmt;
pub mod scorer;
pub mod shadow;
pub mod view;
