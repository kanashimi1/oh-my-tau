//! Port of omp's context compaction (pi-agent-core `compaction/*`, coding-agent
//! `session/session-context.ts`) for the local, summary-based method (`soft` / `context-full`).
//!
//! Flow: [`plan_compaction`] picks the cut point (`firstKeptEntryId`) and what to summarize →
//! [`summary::compact`] builds the summary requests ([`build_summary_request`] & co., as
//! `pa_types::ai::Context`) and runs them through an injected [`Completer`] → [`apply`] turns the
//! result into the `compaction` entry → [`build_session_context`] + [`convert_to_llm`] give what
//! the model sees afterwards.
//!
//! Session entries and agent messages are handled as `serde_json::Value`s (omp's roles and entry
//! kinds outrun `pa_types`). Provider-native compaction (`OpenAI` remote, Anthropic beta),
//! snapcompact, handoff and shake are out of scope: sessions whose latest compaction depends on
//! such a payload yield an `Unsupported` error instead of a wrong context.

// Token arithmetic mirrors JS number math (f64 <-> u64); `format!` appends keep prompt assembly
// readable; the `expect`s are on static regexes.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::format_push_string,
    clippy::missing_panics_doc
)]

pub mod apply;
pub mod branch;
pub mod context;
pub mod convert;
pub mod cut;
pub mod dialect;
pub mod files;
pub mod handoff;
pub mod json;
pub mod messages;
pub mod native;
pub mod prompts;
pub mod serialize;
pub mod settings;
pub mod summary;
pub mod tokens;

pub use apply::{
    append_compaction, compaction_entry, project_tokens_after, synthetic_compaction_entry,
    ApplyOptions,
};
pub use context::{
    build_session_context, build_session_context_owned, build_session_context_path,
    latest_compaction, ContextError, Leaf, SessionContext,
};
pub use convert::{convert_message, convert_to_llm};
pub use cut::{
    find_cut_point, plan_boundary, plan_compaction, plan_region, BoundaryEntry, CompactionBoundary,
    CompactionPreparation, CutPoint, PlanError,
};
pub use handoff::{
    append_handoff, generate_handoff_from_context, handoff_compaction_entry,
    handoff_summary_from_document, render_handoff_prompt, HandoffCommit,
};
pub use json::parse_iso_ms;
pub use native::{
    anthropic_compaction_payload, archive_source_text, get_preserved_archive, history_blocks,
    history_blocks_for_context, openai_remote_payload, Archive, Frame, FRAME_DATA_BYTES_BUDGET,
};
pub use settings::{should_compact, CompactionSettings};
pub use summary::{
    build_short_summary_request, build_summary_request, build_turn_prefix_request, compact,
    generate_summary, CompactionResult, Completer, SummaryKind, SummaryParams, SummaryRequest,
};
pub use tokens::{ByteEstimate, Tokenizer};
