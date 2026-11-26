//! Deterministic Sans-IO replay of a compaction pass plus the bounded flight
//! log that makes it possible.
//!
//! The flight log ([`FlightLog`]) records the state transitions of every
//! compaction pass that runs with `CompactParams::flight` set: pass start
//! (session + leaf turn identifiers), the plan, each summarizer call and its
//! reply, and exactly one terminal transition (`committed`, `failed`,
//! `hook_cancelled`, `commit_refused`, `persist_failed`). It is bounded: when
//! full, the oldest records are evicted and a `log_truncated` marker records
//! how many were dropped, so a consumer can tell a complete log from a
//! truncated one instead of silently replaying half a pass.
//!
//! Replay ([`replay_pass`]) re-runs the *same* production pipeline
//! ([`compact_now`]) against a copy of the session journal, replacing the only
//! IO edge — the summarizer — with a [`ScriptedSummarizer`] driven by the
//! recorded replies/failures. There is no wall clock and no provider: injected
//! timeouts surface as the same deterministic error a real timeout produces at
//! the completer boundary. The replayed transition sequence must equal the
//! recorded one event-for-event; any drift is a [`ReplayError::Diverged`].
//!
//! Explicit non-goals (reported, never guessed at):
//! - a pass cancelled by the extension hook or refused by the commit guard
//!   depends on live external state and replays as [`ReplayError::NotReplayable`];
//! - a truncated log or a size-capped recorded reply replays as
//!   [`ReplayError::Truncated`];
//! - a journal append failure (`persist_failed`) needs fault injection into the
//!   session writer and replays as [`ReplayError::NotReplayable`].

use std::collections::VecDeque;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use anyhow::{anyhow, Result};
use omp_compact::summary::{CompactionResult, SummaryKind, SummaryRequest};
use omp_compact::{CompactionSettings, Completer};
use pa_types::ai::{AssistantContentBlock, AssistantMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{compact_now, CompactParams, SummarizeFn, Summarizer};
use crate::persist::Persistence;

/// Default bound on the in-memory flight log (a pass emits ~4 + 2×summarizer
/// calls records, so this keeps dozens of passes).
pub const DEFAULT_CAPACITY: usize = 256;
/// Cap on one recorded summarizer reply. Longer replies are stored truncated
/// and flagged; replay refuses them ([`ReplayError::Truncated`]) rather than
/// reconstructing a summary the live pass never produced.
pub const MAX_RECORDED_REPLY_BYTES: usize = 64 * 1024;
/// Default bound of the on-disk flight log (`OMP_COMPACTION_FLIGHT_LOG`):
/// the file is truncated on open once it exceeds this.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 1024 * 1024;
/// The deterministic error an injected [`Injected::Timeout`] produces at the
/// completer boundary — the same surface a real provider timeout has, with no
/// wall-clock wait (Sans-IO).
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
pub const TIMEOUT_ERROR: &str = "summary request timed out";

/// One flight-log transition. No timestamps: replay equality is byte-for-byte.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum FlightEvent {
    /// A pass began. `leaf_id` is the session-turn identifier the plan was cut
    /// from (the branch tip at plan time).
    PassStart {
        session_id: String,
        leaf_id: String,
        model_id: String,
        custom_instructions: Option<String>,
        non_message_tokens: u64,
    },
    /// The boundary/cut was computed.
    Planned {
        boundary_start: u64,
        first_kept_entry_id: String,
        tokens_before: u64,
        messages_to_summarize: u64,
        is_split_turn: bool,
    },
    /// One summarizer (completer) call went out; `call` is 1-based per pass.
    SummarizeCall {
        call: u64,
        kind: String,
        max_tokens: u64,
    },
    /// The summarizer replied; `text` is the reply text the summary pipeline
    /// consumes (size-capped, see [`MAX_RECORDED_REPLY_BYTES`]).
    SummarizeReply {
        call: u64,
        text: String,
        truncated: bool,
    },
    /// The summarizer failed (provider error, or an injected error/timeout).
    SummarizeFailed { call: u64, error: String },
    /// The extension `session_before_compact` hook cancelled the pass.
    HookCancelled,
    /// The commit guard refused the commit (session switched, `/tree` move,
    /// concurrent append, Esc).
    CommitRefused { reason: String },
    /// The summarization itself failed outside a completer call (e.g. the kept
    /// entry had no id).
    Failed { error: String },
    /// The journal append of the `compaction` entry failed.
    PersistFailed { error: String },
    /// Terminal success: the `compaction` entry is in the journal.
    Committed {
        method: String,
        tokens_after: Option<u64>,
        summary: String,
        short_summary: String,
        first_kept_entry_id: String,
        tokens_before: u64,
    },
    /// Bounded-log marker: `dropped` records were evicted before this point.
    LogTruncated { dropped: u64 },
}

impl FlightEvent {
    /// Stable discriminant name, for transition-sequence comparisons.
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::PassStart { .. } => "pass_start",
            Self::Planned { .. } => "planned",
            Self::SummarizeCall { .. } => "summarize_call",
            Self::SummarizeReply { .. } => "summarize_reply",
            Self::SummarizeFailed { .. } => "summarize_failed",
            Self::HookCancelled => "hook_cancelled",
            Self::CommitRefused { .. } => "commit_refused",
            Self::Failed { .. } => "failed",
            Self::PersistFailed { .. } => "persist_failed",
            Self::Committed { .. } => "committed",
            Self::LogTruncated { .. } => "log_truncated",
        }
    }
}

/// One line of the flight log: a monotonically increasing `seq` (per log, the
/// record identifier) and the `pass` (turn) identifier it belongs to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightRecord {
    pub seq: u64,
    pub pass: String,
    #[serde(flatten)]
    pub event: FlightEvent,
}

struct Inner {
    records: VecDeque<FlightRecord>,
    seq: u64,
    pass_counter: u64,
    current_pass: String,
    dropped: u64,
    file: Option<std::fs::File>,
    write_failures: u64,
}

/// Bounded in-memory flight log with an optional JSONL file sink. Recording is
/// best-effort by contract: a sink failure increments
/// [`Self::write_failures`] and never affects the compaction being observed.
pub struct FlightLog {
    capacity: usize,
    sink: Option<PathBuf>,
    #[allow(dead_code)] // replay/test surface (bounded-sink observability)
    max_file_bytes: u64,
    inner: Mutex<Inner>,
}

impl FlightLog {
    /// An in-memory log holding at most `capacity` records.
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::with_sink(capacity, None, DEFAULT_MAX_FILE_BYTES)
    }

    /// A log that also appends every record to `sink` as JSONL. An existing
    /// sink larger than `max_file_bytes` is truncated on construction, so the
    /// on-disk log stays bounded too.
    #[must_use]
    pub fn with_sink(capacity: usize, sink: Option<PathBuf>, max_file_bytes: u64) -> Self {
        if let Some(path) = &sink {
            if std::fs::metadata(path).is_ok_and(|m| m.len() > max_file_bytes) {
                // Explicit bounded-file behavior: start over rather than grow
                // without limit. A failure here is counted on the first write.
                let _ = std::fs::File::create(path);
            }
        }
        Self {
            capacity: capacity.max(2),
            sink,
            max_file_bytes,
            inner: Mutex::new(Inner {
                records: VecDeque::new(),
                seq: 0,
                pass_counter: 0,
                current_pass: String::new(),
                dropped: 0,
                file: None,
                write_failures: 0,
            }),
        }
    }

    /// Begin a new pass; returns its turn identifier
    /// (`{session_id}:{leaf_id}#{n}`) and records [`FlightEvent::PassStart`].
    pub fn start_pass(
        &self,
        session_id: &str,
        leaf_id: &str,
        model_id: &str,
        custom_instructions: Option<&str>,
        non_message_tokens: u64,
    ) -> String {
        let pass = {
            let mut inner = self.inner.lock();
            inner.pass_counter += 1;
            inner.current_pass = format!("{session_id}:{leaf_id}#{}", inner.pass_counter);
            inner.current_pass.clone()
        };
        self.record(FlightEvent::PassStart {
            session_id: session_id.to_owned(),
            leaf_id: leaf_id.to_owned(),
            model_id: model_id.to_owned(),
            custom_instructions: custom_instructions.map(str::to_owned),
            non_message_tokens,
        });
        pass
    }

    /// Append one transition, evicting the oldest records past the bound and
    /// maintaining the `log_truncated` marker, then mirror the new line(s) to
    /// the sink. Sink failures are counted ([`Self::write_failures`]), never
    /// propagated into the compaction being observed.
    pub fn record(&self, event: FlightEvent) {
        let added = {
            let mut inner = self.inner.lock();
            push_record(&mut inner, self.capacity, event)
        };
        let Some(path) = &self.sink else {
            return;
        };
        let mut inner = self.inner.lock();
        for record in &added {
            let Ok(mut line) = serde_json::to_string(record) else {
                inner.write_failures += 1;
                continue;
            };
            line.push('\n');
            if inner.file.is_none() {
                inner.file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok();
            }
            let failed = match inner.file.as_mut() {
                Some(f) => f.write_all(line.as_bytes()).is_err(),
                None => true,
            };
            if failed {
                inner.file = None; // reopen (and re-bound) on the next record
                inner.write_failures += 1;
                break;
            }
        }
    }

    /// A snapshot of the retained records, oldest first.
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn records(&self) -> Vec<FlightRecord> {
        self.inner.lock().records.iter().cloned().collect()
    }

    /// The retained records as JSONL (the sink's format).
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn encode(&self) -> String {
        let inner = self.inner.lock();
        let mut out = String::new();
        for r in &inner.records {
            if let Ok(line) = serde_json::to_string(r) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        out
    }

    /// Records evicted past the bound so far.
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.inner.lock().dropped
    }

    /// Sink write/open failures so far (recording never fails the pass).
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn write_failures(&self) -> u64 {
        self.inner.lock().write_failures
    }

    /// The bound of the on-disk sink in bytes.
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }
}

/// Append `event` under the bound, keeping exactly one trailing
/// `log_truncated` marker whose count is current (a stale marker is replaced
/// by a fresh record with a fresh `seq`, so the append-only sink never sees a
/// seq collision). Returns the records a sink should mirror, in order.
fn push_record(inner: &mut Inner, capacity: usize, event: FlightEvent) -> Vec<FlightRecord> {
    let mut added = Vec::with_capacity(2);
    inner.seq += 1;
    let record = FlightRecord {
        seq: inner.seq,
        pass: inner.current_pass.clone(),
        event,
    };
    inner.records.push_back(record.clone());
    added.push(record);
    while inner.records.len() > capacity {
        inner.records.pop_front();
        inner.dropped += 1;
    }
    if inner.dropped == 0 {
        return added;
    }
    if matches!(
        inner.records.back().map(|r| &r.event),
        Some(FlightEvent::LogTruncated { .. })
    ) {
        inner.records.pop_back();
    }
    // Evict first so the marker reports every drop, including its own room.
    while inner.records.len() >= capacity {
        inner.records.pop_front();
        inner.dropped += 1;
    }
    inner.seq += 1;
    let marker = FlightRecord {
        seq: inner.seq,
        pass: inner.current_pass.clone(),
        event: FlightEvent::LogTruncated {
            dropped: inner.dropped,
        },
    };
    inner.records.push_back(marker.clone());
    added.push(marker);
    added
}

/// The flight log a production pass should record into, from the environment:
/// `OMP_COMPACTION_FLIGHT_LOG=<path>` enables it (JSONL sink),
/// `OMP_COMPACTION_FLIGHT_MAX_BYTES` overrides the on-disk bound. `None`
/// (unset) keeps compaction allocation- and IO-free.
#[must_use]
pub fn flight_log_from_env() -> Option<FlightLog> {
    let path = std::env::var_os("OMP_COMPACTION_FLIGHT_LOG")?;
    if path.is_empty() {
        return None;
    }
    let max = std::env::var("OMP_COMPACTION_FLIGHT_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MAX_FILE_BYTES);
    Some(FlightLog::with_sink(
        DEFAULT_CAPACITY,
        Some(PathBuf::from(path)),
        max,
    ))
}

/// Stable name of one summarizer call kind (matches `omp.gen_ai.oneshot.kind`).
fn kind_name(kind: SummaryKind) -> String {
    match kind {
        SummaryKind::CompactionSummary => "compaction_summary",
        SummaryKind::CompactionTurnPrefix => "compaction_turn_prefix",
        SummaryKind::CompactionShortSummary => "compaction_short_summary",
        SummaryKind::BranchSummary => "branch_summary",
        SummaryKind::Handoff => "handoff",
    }
    .to_owned()
}

/// The text of a summarizer reply (what `completion_text` extracts).
fn reply_text(reply: &AssistantMessage) -> String {
    reply
        .content
        .iter()
        .filter_map(|b| match b {
            AssistantContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// A minimal assistant reply carrying `text` — the shape the summary pipeline
/// consumes, reconstructed without any provider.
fn text_reply(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContentBlock::Text(
            pa_types::ai::TextContent {
                text: text.to_owned(),
                text_signature: None,
                rest: serde_json::Map::default(),
            },
        )],
        api: "openai-completions".to_owned(),
        provider: "replay".to_owned(),
        model: "replay".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: pa_types::ai::Usage::default(),
        stop_reason: pa_types::ai::StopReason::Stop,
        stop_reason_raw: None,
        error_message: None,
        error_status: None,
        timestamp: 1,
        rest: serde_json::Map::default(),
    }
}

/// A [`Completer`] decorator that mirrors every summarizer call and its
/// outcome into a [`FlightLog`]. The production pipeline is otherwise
/// untouched — same requests, same replies, same errors.
pub struct RecordingCompleter<'a, C: Completer> {
    inner: &'a C,
    flight: &'a FlightLog,
    calls: AtomicU64,
}

impl<'a, C: Completer> RecordingCompleter<'a, C> {
    pub fn new(inner: &'a C, flight: &'a FlightLog) -> Self {
        Self {
            inner,
            flight,
            calls: AtomicU64::new(0),
        }
    }
}

impl<C: Completer + Sync> Completer for RecordingCompleter<'_, C> {
    fn complete(
        &self,
        request: SummaryRequest,
    ) -> impl std::future::Future<Output = Result<AssistantMessage>> + Send {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        self.flight.record(FlightEvent::SummarizeCall {
            call,
            kind: kind_name(request.kind),
            max_tokens: request.max_tokens,
        });
        let flight = self.flight;
        async move {
            match self.inner.complete(request).await {
                Ok(reply) => {
                    let mut text = reply_text(&reply);
                    let truncated = text.len() > MAX_RECORDED_REPLY_BYTES;
                    if truncated {
                        text.truncate(MAX_RECORDED_REPLY_BYTES);
                    }
                    flight.record(FlightEvent::SummarizeReply {
                        call,
                        text,
                        truncated,
                    });
                    Ok(reply)
                }
                Err(e) => {
                    flight.record(FlightEvent::SummarizeFailed {
                        call,
                        error: format!("{e:#}"),
                    });
                    Err(e)
                }
            }
        }
    }
}

/// One deterministic summarizer outcome at the completer boundary (the
/// injection point): a recorded reply, a recorded/probe error, or a timeout.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Injected {
    Reply(String),
    Fail(String),
    /// Resolves immediately to [`TIMEOUT_ERROR`] — the Sans-IO model of a
    /// provider timeout (the pipeline only ever observes the error surface).
    Timeout,
}

/// A summarizer driven by a fixed script of [`Injected`] outcomes, in call
/// order. Runs out of script → a deterministic `script exhausted` failure, so
/// a diverging replay fails loudly instead of hanging.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
pub struct ScriptedSummarizer {
    script: Mutex<VecDeque<Injected>>,
    calls: AtomicU64,
}

impl ScriptedSummarizer {
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn new(script: Vec<Injected>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into()),
            calls: AtomicU64::new(0),
        })
    }

    /// The [`SummarizeFn`] view of this script (for `Summarizer::Injected`).
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn summarize_fn(self: &Arc<Self>) -> SummarizeFn {
        let this = Arc::clone(self);
        let f: SummarizeFn = Arc::new(move |_request| {
            let this = Arc::clone(&this);
            Box::pin(async move {
                let call = this.calls.fetch_add(1, Ordering::SeqCst) + 1;
                let next = this.script.lock().pop_front();
                match next {
                    Some(Injected::Reply(text)) => Ok(text_reply(&text)),
                    Some(Injected::Fail(error)) => Err(anyhow!(error)),
                    Some(Injected::Timeout) => Err(anyhow!(TIMEOUT_ERROR)),
                    None => Err(anyhow!("script exhausted at call {call}")),
                }
            })
        });
        f
    }

    /// How many calls the script has served.
    #[allow(dead_code)] // replay/test surface
    #[must_use]
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
}

/// Why a log cannot be (fully) replayed. Replay never guesses past these.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// A line of the encoded log did not parse.
    MalformedRecord { line: usize, error: String },
    /// Two records share a `seq` but differ (an identical duplicate is fine —
    /// at-least-once sinks happen — and is deduplicated).
    DuplicateConflict { seq: u64 },
    /// The log was bounded past what the pass needs (eviction marker, or a
    /// size-capped recorded reply).
    Truncated { reason: String },
    /// No pass in the log.
    NoPass,
    /// The pass has no terminal transition (crashed or still running).
    IncompletePass { pass: String },
    /// The pass depends on live state replay cannot reproduce (hook cancel,
    /// commit-guard refusal, journal write failure).
    NotReplayable { reason: String },
    /// The replayed pass did not match the recording.
    Diverged { field: String, recorded: String, replayed: String },
    /// The replayed pass itself could not run (journal IO, plan failure).
    Replay(String),
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedRecord { line, error } => {
                write!(f, "malformed flight-log record at line {line}: {error}")
            }
            Self::DuplicateConflict { seq } => {
                write!(f, "conflicting duplicate flight-log record seq {seq}")
            }
            Self::Truncated { reason } => write!(f, "flight log truncated: {reason}"),
            Self::NoPass => write!(f, "flight log contains no compaction pass"),
            Self::IncompletePass { pass } => {
                write!(f, "pass {pass} has no terminal transition")
            }
            Self::NotReplayable { reason } => write!(f, "pass is not replayable: {reason}"),
            Self::Diverged {
                field,
                recorded,
                replayed,
            } => write!(f, "replay diverged at {field}: recorded {recorded}, replayed {replayed}"),
            Self::Replay(e) => write!(f, "replay failed: {e}"),
        }
    }
}

impl std::error::Error for ReplayError {}

/// Parse an encoded flight log. Identical duplicate records (same `seq`, same
/// payload — e.g. an at-least-once sink rewrite) are deduplicated; a `seq`
/// collision with a different payload is a [`ReplayError::DuplicateConflict`].
///
/// # Errors
/// Malformed line, conflicting duplicate.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
pub fn decode_log(text: &str) -> Result<Vec<FlightRecord>, ReplayError> {
    let mut out: Vec<FlightRecord> = Vec::new();
    let mut by_seq = std::collections::HashMap::<u64, usize>::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: FlightRecord =
            serde_json::from_str(line).map_err(|e| ReplayError::MalformedRecord {
                line: n + 1,
                error: e.to_string(),
            })?;
        if let Some(&at) = by_seq.get(&record.seq) {
            if out[at] == record {
                continue; // identical duplicate: dedupe
            }
            return Err(ReplayError::DuplicateConflict { seq: record.seq });
        }
        by_seq.insert(record.seq, out.len());
        out.push(record);
    }
    Ok(out)
}

/// The production `compact_now` outputs of a committed pass.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
#[derive(Debug)]
pub struct CommittedPass {
    pub result: CompactionResult,
    pub data: Value,
    pub view: Vec<Value>,
}

/// What a replayed pass produced.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
#[derive(Debug)]
pub enum ReplayOutcome {
    /// The pass committed; carries the production `compact_now` outputs.
    Committed(Box<CommittedPass>),
    /// The pass failed, deterministically, with the recorded error.
    Failed { error: String },
}

/// Settings of the pass being replayed; the recorded `planned` transition
/// cross-checks them, so a mismatch surfaces as [`ReplayError::Diverged`]
/// instead of a silently different cut.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
pub struct ReplaySettings<'a> {
    pub settings: &'a CompactionSettings,
    pub context_window: Option<u64>,
}


/// What the pass's terminal transition implies for the replay: reproduce a
/// commit, reproduce a failure, or refuse (live-state-dependent / incomplete).
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
enum Expect {
    Commit,
    Fail(String),
}

#[allow(dead_code)] // replay surface
fn classify_terminal(
    terminal: &FlightEvent,
    pass_id: String,
) -> Result<Expect, ReplayError> {
    match terminal {
        FlightEvent::Committed { .. } => Ok(Expect::Commit),
        FlightEvent::SummarizeFailed { error, .. } | FlightEvent::Failed { error } => {
            Ok(Expect::Fail(error.clone()))
        }
        FlightEvent::HookCancelled => Err(ReplayError::NotReplayable {
            reason: "extension hook cancellation is an external decision".to_owned(),
        }),
        FlightEvent::CommitRefused { reason } => Err(ReplayError::NotReplayable {
            reason: format!("commit-guard refusal depends on a live session race: {reason}"),
        }),
        FlightEvent::PersistFailed { error } => Err(ReplayError::NotReplayable {
            reason: format!("journal write failure needs writer fault injection: {error}"),
        }),
        _ => Err(ReplayError::IncompletePass { pass: pass_id }),
    }
}

/// The summarizer script of a recorded pass: every reply/failure, in call
/// order (call numbers are positional and reproduced by the replay).
#[allow(dead_code)] // replay surface
fn build_script(pass: &[FlightRecord]) -> Vec<Injected> {
    pass.iter()
        .filter_map(|record| match &record.event {
            FlightEvent::SummarizeReply { text, .. } => Some(Injected::Reply(text.clone())),
            FlightEvent::SummarizeFailed { error, .. } => Some(Injected::Fail(error.clone())),
            _ => None,
        })
        .collect()
}

/// Re-run the last pass of `records` through the production [`compact_now`]
/// against the session at `path` (replay against a copy: the pass appends its
/// `compaction` entry on success), with the summarizer replaced by the
/// recorded replies/failures. No provider, no network, no wall clock.
///
/// On success the replayed transition sequence is asserted equal to the
/// recorded one (event-for-event) and the returned outcome carries the
/// replayed result for A/B comparison against the live pass.
///
/// # Errors
/// See [`ReplayError`]; in particular a recorded hook cancel, commit-guard
/// refusal or journal write failure is [`ReplayError::NotReplayable`], and a
/// truncated log or capped reply is [`ReplayError::Truncated`].
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
pub async fn replay_pass(
    persist: &Persistence,
    path: &Path,
    records: &[FlightRecord],
    replay: &ReplaySettings<'_>,
) -> Result<ReplayOutcome, ReplayError> {
    if let Some(dropped) = records.iter().find_map(|r| match &r.event {
        FlightEvent::LogTruncated { dropped } => Some(*dropped),
        _ => None,
    }) {
        return Err(ReplayError::Truncated {
            reason: format!("{dropped} record(s) evicted before this log"),
        });
    }
    let start = records
        .iter()
        .rposition(|r| matches!(r.event, FlightEvent::PassStart { .. }))
        .ok_or(ReplayError::NoPass)?;
    let pass = &records[start..];
    let pass_id = pass[0].pass.clone();
    let FlightEvent::PassStart {
        model_id,
        custom_instructions,
        non_message_tokens,
        ..
    } = pass[0].event.clone()
    else {
        return Err(ReplayError::NoPass);
    };
    if let Some(r) = pass.iter().find(|r| {
        matches!(&r.event, FlightEvent::SummarizeReply { truncated: true, .. })
    }) {
        return Err(ReplayError::Truncated {
            reason: format!("recorded reply at seq {} exceeded the size cap", r.seq),
        });
    }
    let terminal = pass.last().ok_or(ReplayError::NoPass)?;
    let expect = classify_terminal(&terminal.event, pass_id)?;
    let scripted = ScriptedSummarizer::new(build_script(pass));
    let replay_log = FlightLog::new(DEFAULT_CAPACITY);
    let params = CompactParams {
        settings: replay.settings,
        model_id: &model_id,
        context_window: replay.context_window,
        custom_instructions: custom_instructions.as_deref(),
        trajectory_squash: Some(false), // replay the recorded LLM path only
        non_message_tokens,
        hook: None,
        guard: None, // offline replay: no live session to race
        flight: Some(&replay_log),
    };
    let out = compact_now(
        persist,
        path,
        &Summarizer::Injected(scripted.summarize_fn()),
        &params,
    )
    .await;
    let replayed: Vec<FlightEvent> = replay_log.records().into_iter().map(|r| r.event).collect();
    let recorded: Vec<FlightEvent> = pass.iter().map(|r| r.event.clone()).collect();
    match (expect, out) {
        (Expect::Commit, Ok((result, data, view))) => {
            compare_transitions(&recorded, &replayed)?;
            Ok(ReplayOutcome::Committed(Box::new(CommittedPass {
                result,
                data,
                view,
            })))
        }
        (Expect::Fail(recorded_error), Err(e)) => {
            let error = format!("{e:#}");
            // The terminal errors must match; transition prefixes may differ
            // in how far they got only when the failure stage matches too.
            compare_transitions(&recorded, &replayed)?;
            if error != recorded_error {
                return Err(ReplayError::Diverged {
                    field: "error".to_owned(),
                    recorded: recorded_error,
                    replayed: error,
                });
            }
            Ok(ReplayOutcome::Failed { error })
        }
        (Expect::Commit, Err(e)) => Err(ReplayError::Diverged {
            field: "outcome".to_owned(),
            recorded: "committed".to_owned(),
            replayed: format!("failed: {e:#}"),
        }),
        (Expect::Fail(recorded_error), Ok(_)) => Err(ReplayError::Diverged {
            field: "outcome".to_owned(),
            recorded: format!("failed: {recorded_error}"),
            replayed: "committed".to_owned(),
        }),
    }
}

/// Event-for-event equality of the recorded and replayed transition
/// sequences; the first mismatch is reported with both discriminants (and
/// payloads, abbreviated) for diagnosis.
#[allow(dead_code)] // replay surface: driven by the replay tests today, by offline tooling next
fn compare_transitions(
    recorded: &[FlightEvent],
    replayed: &[FlightEvent],
) -> Result<(), ReplayError> {
    let abbrev = |e: &FlightEvent| {
        let s = serde_json::to_string(e).unwrap_or_default();
        s.chars().take(160).collect::<String>()
    };
    for (i, (rec, rep)) in recorded.iter().zip(replayed.iter()).enumerate() {
        if rec != rep {
            return Err(ReplayError::Diverged {
                field: format!("transition {i} ({}/{})", rec.kind(), rep.kind()),
                recorded: abbrev(rec),
                replayed: abbrev(rep),
            });
        }
    }
    if recorded.len() != replayed.len() {
        return Err(ReplayError::Diverged {
            field: "transition count".to_owned(),
            recorded: format!("{} events", recorded.len()),
            replayed: format!("{} events", replayed.len()),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod tests {
    use std::path::Path;
    use std::time::Instant;

    use serde_json::{json, Value};

    use super::*;
    use crate::compact::{CommitGuard, CompactHook, CompactParams, Summarizer};

    fn settings() -> CompactionSettings {
        CompactionSettings {
            keep_recent_tokens: 100,
            ..CompactionSettings::default()
        }
    }

    /// A session file with six long alternating messages (enough to compact).
    fn build_session(dir: &Path) -> Persistence {
        let persist = Persistence::for_test(dir).expect("session");
        for i in 0..6 {
            let (role, text) = if i % 2 == 0 {
                ("user", format!("question {i} {}", "x".repeat(400)))
            } else {
                ("assistant", format!("answer {i} {}", "y".repeat(400)))
            };
            persist
                .with_writer(|w| {
                    w.append_entry(
                        "message",
                        json!({"message": {
                            "role": role,
                            "content": [{"type": "text", "text": text}],
                            "stopReason": "stop", "timestamp": 1,
                        }}),
                        "2026-01-01T00:00:00.000Z",
                    )
                })
                .expect("message entry");
        }
        persist
    }

    fn params<'a>(
        settings: &'a CompactionSettings,
        flight: Option<&'a FlightLog>,
    ) -> CompactParams<'a> {
        CompactParams {
            settings,
            model_id: "test-model",
            context_window: Some(100_000),
            custom_instructions: None,
            trajectory_squash: Some(false),
            non_message_tokens: 0,
            hook: None,
            guard: None,
            flight,
        }
    }

    fn journal_lines(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn kinds(log: &FlightLog) -> Vec<&'static str> {
        log.records().iter().map(|r| r.event.kind()).collect()
    }

    /// A fresh persistence bound to `path` (which the replay pass may append
    /// to), carrying the recorded session id (`for_test` mints its own).
    fn bind(dir: &Path, path: &Path, session_id: &str) -> Persistence {
        let persist = Persistence::for_test(dir).expect("session");
        persist
            .bind_existing(path, Some(session_id), None)
            .expect("bind");
        persist
    }

    /// A/B: a live pass with an injected summarizer records a flight log; the
    /// replay of that log through the production pipeline (scripted completer,
    /// no provider) must produce the identical result, data payload, rebuilt
    /// view and transition sequence.
    #[tokio::test]
    async fn live_pass_replays_with_zero_semantic_drift() {
        let live_dir = tempfile::tempdir().unwrap();
        let persist = build_session(live_dir.path());
        let path = persist.path();
        let copy_dir = tempfile::tempdir().unwrap();
        let copy = copy_dir.path().join("copy.jsonl");
        std::fs::copy(&path, &copy).unwrap();

        let flight = FlightLog::new(DEFAULT_CAPACITY);
        // Enough replies for the whole pipeline (history summary, optional
        // split-turn prefix summary, short summary); leftovers stay unused.
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("SUMMARY A".to_owned()),
            Injected::Reply("PREFIX A".to_owned()),
            Injected::Reply("SHORT A".to_owned()),
            Injected::Reply("EXTRA A".to_owned()),
        ]);
        let settings = settings();
        let t0 = Instant::now();
        let (result, data, view) = compact_now(
            &persist,
            &path,
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&flight)),
        )
        .await
        .expect("live pass");
        let live_elapsed = t0.elapsed();
        let kinds = kinds(&flight);
        assert_eq!(&kinds[..2], ["pass_start", "planned"]);
        assert_eq!(kinds.last(), Some(&"committed"));
        let middle = &kinds[2..kinds.len() - 1];
        assert!(!middle.is_empty());
        for pair in middle.chunks(2) {
            assert_eq!(pair, ["summarize_call", "summarize_reply"]);
        }
        // Sequence/turn identifiers: strictly increasing seq, one pass id of
        // the form `{session}:{leaf}#{n}` on every record.
        let records = flight.records();
        assert!(records.windows(2).all(|w| w[0].seq < w[1].seq));
        let session_id = persist.session_id();
        assert!(records
            .iter()
            .all(|r| r.pass.starts_with(&session_id) && r.pass.ends_with("#1")));

        let records = decode_log(&flight.encode()).expect("decode");
        let replay_persist = bind(copy_dir.path(), &copy, &session_id);
        let t1 = Instant::now();
        let outcome = replay_pass(
            &replay_persist,
            &copy,
            &records,
            &ReplaySettings {
                settings: &settings,
                context_window: Some(100_000),
            },
        )
        .await
        .expect("replay");
        let replay_elapsed = t1.elapsed();
        let ReplayOutcome::Committed(committed) = outcome else {
            panic!("expected a committed replay, got {outcome:?}");
        };
        let CommittedPass {
            result: r2,
            data: d2,
            view: v2,
        } = *committed;
        assert_eq!(result.summary, r2.summary);
        assert_eq!(result.short_summary, r2.short_summary);
        assert_eq!(result.first_kept_entry_id, r2.first_kept_entry_id);
        assert_eq!(result.tokens_before, r2.tokens_before);
        assert_eq!(result.details, r2.details);
        assert_eq!(data, d2, "rpc data payload drifted");
        // The rebuilt view: identical up to the volatile timestamp.
        let strip = |view: &[Value]| -> Vec<Value> {
            view.iter()
                .map(|e| {
                    let mut e = e.clone();
                    e.as_object_mut().unwrap().remove("timestamp");
                    e
                })
                .collect()
        };
        assert_eq!(strip(&view), strip(&v2), "rebuilt view drifted");
        // Journal equivalence: both files end in a compaction entry with the
        // same payload (ids/timestamps aside).
        let a = journal_lines(&path);
        let b = journal_lines(&copy);
        assert_eq!(a.len(), b.len());
        let mut ca = a.last().unwrap().clone();
        let mut cb = b.last().unwrap().clone();
        for e in [&mut ca, &mut cb] {
            let o = e.as_object_mut().unwrap();
            for k in ["id", "parentId", "timestamp"] {
                o.remove(k);
            }
        }
        assert_eq!(ca, cb, "compaction journal entry drifted");
        eprintln!("A/B timing: live pass {live_elapsed:?}, replay {replay_elapsed:?}");
    }

    /// Chaos: an injected timeout fails the pass deterministically (journal
    /// untouched) and the recorded failure replays to the identical error.
    #[tokio::test]
    async fn injected_timeout_fails_and_replays_the_same_failure() {
        let dir = tempfile::tempdir().unwrap();
        let persist = build_session(dir.path());
        let path = persist.path();
        let before = journal_lines(&path);
        let flight = FlightLog::new(DEFAULT_CAPACITY);
        let scripted = ScriptedSummarizer::new(vec![Injected::Timeout]);
        let settings = settings();
        let err = compact_now(
            &persist,
            &path,
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&flight)),
        )
        .await
        .expect_err("timeout must fail the pass");
        assert!(format!("{err:#}").contains(TIMEOUT_ERROR));
        assert_eq!(journal_lines(&path), before, "failed pass wrote nothing");
        assert_eq!(
            kinds(&flight),
            [
                "pass_start",
                "planned",
                "summarize_call",
                "summarize_failed",
                "failed"
            ]
        );

        let copy_dir = tempfile::tempdir().unwrap();
        let copy = copy_dir.path().join("copy.jsonl");
        std::fs::copy(&path, &copy).unwrap();
        let records = decode_log(&flight.encode()).unwrap();
        let session_id = persist.session_id();
        let outcome = replay_pass(
            &bind(copy_dir.path(), &copy, &session_id),
            &copy,
            &records,
            &ReplaySettings {
                settings: &settings,
                context_window: Some(100_000),
            },
        )
        .await
        .expect("failure replays");
        let ReplayOutcome::Failed { error } = outcome else {
            panic!("expected a failed replay, got {outcome:?}");
        };
        assert_eq!(error, format!("{err:#}"));
        assert_eq!(journal_lines(&copy), before, "replay wrote nothing");
    }

    /// Chaos: at-least-once duplicate records are deduplicated by `seq`;
    /// a conflicting duplicate is rejected, never silently picked.
    #[tokio::test]
    async fn duplicate_records_dedupe_and_conflicts_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let persist = build_session(dir.path());
        let flight = FlightLog::new(DEFAULT_CAPACITY);
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("S".to_owned()),
            Injected::Reply("P".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        let settings = settings();
        compact_now(
            &persist,
            &persist.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&flight)),
        )
        .await
        .expect("pass");
        let encoded = flight.encode();
        let doubled: String = encoded
            .lines()
            .flat_map(|l| [l, l])
            .collect::<Vec<_>>()
            .join("\n");
        let records = decode_log(&doubled).expect("duplicates dedupe");
        assert_eq!(records, flight.records());

        let mut conflicted: Vec<FlightRecord> = flight.records();
        conflicted.push(FlightRecord {
            event: FlightEvent::Failed {
                error: "forged".to_owned(),
            },
            ..conflicted[0].clone()
        });
        let text = conflicted
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            decode_log(&text).unwrap_err(),
            ReplayError::DuplicateConflict {
                seq: conflicted[0].seq
            }
        );
    }

    /// Chaos: malformed lines and missing terminals are explicit errors,
    /// never silent guesses.
    #[tokio::test]
    async fn malformed_and_incomplete_logs_are_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let persist = build_session(dir.path());
        let flight = FlightLog::new(DEFAULT_CAPACITY);
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("S".to_owned()),
            Injected::Reply("P".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        let settings = settings();
        compact_now(
            &persist,
            &persist.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&flight)),
        )
        .await
        .expect("pass");
        let encoded = flight.encode();

        // Malformed line: reported with its line number.
        let mut lines: Vec<&str> = encoded.lines().collect();
        lines.insert(1, "{\"seq\": \"nope\"");
        let err = decode_log(&lines.join("\n")).unwrap_err();
        assert!(matches!(err, ReplayError::MalformedRecord { line: 2, .. }));

        // Missing terminal: an incomplete pass is not replayed.
        let mut records = decode_log(&encoded).unwrap();
        records.pop();
        let err = replay_pass(
            &persist,
            &persist.path(),
            &records,
            &ReplaySettings {
                settings: &settings,
                context_window: Some(100_000),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReplayError::IncompletePass { .. }), "{err}");
    }

    /// Chaos: bounded-log eviction and size-capped replies make the log
    /// incomplete by construction; replay refuses both explicitly.
    #[tokio::test]
    async fn bounded_and_capped_logs_refuse_replay() {
        let settings = settings();

        // Bounded log: eviction past the bound leaves a truncation marker and
        // the replay refuses the partial pass.
        let small_dir = tempfile::tempdir().unwrap();
        let small_session = build_session(small_dir.path());
        let small = FlightLog::new(4);
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("S".to_owned()),
            Injected::Reply("P".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        compact_now(
            &small_session,
            &small_session.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&small)),
        )
        .await
        .expect("second pass");
        assert!(small.dropped() > 0);
        assert!(small.records().len() <= 4);
        assert!(small
            .records()
            .iter()
            .any(|r| matches!(r.event, FlightEvent::LogTruncated { .. })));
        let records = decode_log(&small.encode()).unwrap();
        let err = replay_pass(
            &small_session,
            &small_session.path(),
            &records,
            &ReplaySettings {
                settings: &settings,
                context_window: Some(100_000),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReplayError::Truncated { .. }), "{err}");

        // Size-capped recorded reply: the live pass commits with the full
        // reply, but the log cannot reproduce it — explicit Truncated.
        let big_dir = tempfile::tempdir().unwrap();
        let big = build_session(big_dir.path());
        let big_flight = FlightLog::new(DEFAULT_CAPACITY);
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("z".repeat(MAX_RECORDED_REPLY_BYTES + 10)),
            Injected::Reply("p".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        compact_now(
            &big,
            &big.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&big_flight)),
        )
        .await
        .expect("pass with an oversized reply still commits");
        let records = decode_log(&big_flight.encode()).unwrap();
        assert!(records.iter().any(|r| matches!(
            &r.event,
            FlightEvent::SummarizeReply { truncated: true, .. }
        )));
        let err = replay_pass(
            &big,
            &big.path(),
            &records,
            &ReplaySettings {
                settings: &settings,
                context_window: Some(100_000),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ReplayError::Truncated { .. }), "{err}");
    }

    /// Persistence failure of the flight-log sink is explicit and harmless:
    /// the pass commits, the in-memory log is complete, and the failure is
    /// counted (never propagated into the compaction).
    #[tokio::test]
    async fn sink_failure_never_fails_the_pass() {
        let dir = tempfile::tempdir().unwrap();
        let persist = build_session(dir.path());
        let sink = dir.path().join("no-such-dir/flight.jsonl");
        let flight = FlightLog::with_sink(DEFAULT_CAPACITY, Some(sink), DEFAULT_MAX_FILE_BYTES);
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("S".to_owned()),
            Injected::Reply("P".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        let settings = settings();
        compact_now(
            &persist,
            &persist.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&flight)),
        )
        .await
        .expect("pass commits despite the broken sink");
        assert!(flight.write_failures() > 0);
        assert_eq!(kinds(&flight).last(), Some(&"committed"));

        // A working sink receives the same JSONL the in-memory log encodes,
        // and an oversized pre-existing sink is truncated on open.
        let sink_dir = tempfile::tempdir().unwrap();
        let sink_session = build_session(sink_dir.path());
        let sink = sink_dir.path().join("flight.jsonl");
        std::fs::write(&sink, "x".repeat(4096)).unwrap();
        let flight = FlightLog::with_sink(DEFAULT_CAPACITY, Some(sink.clone()), 1024);
        assert_eq!(std::fs::metadata(&sink).unwrap().len(), 0, "bounded file");
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("S".to_owned()),
            Injected::Reply("P".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        compact_now(
            &sink_session,
            &sink_session.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &params(&settings, Some(&flight)),
        )
        .await
        .expect("pass");
        assert_eq!(flight.write_failures(), 0);
        assert_eq!(std::fs::read_to_string(&sink).unwrap(), flight.encode());
    }

    struct CancelHook;
    impl CompactHook for CancelHook {
        fn before<'a>(
            &'a self,
            _preparation: Value,
            _custom_instructions: Option<&'a str>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>> {
            Box::pin(async { true })
        }
    }

    /// Passes that end on live external state (hook cancel, commit-guard
    /// refusal) are recorded but explicitly not replayable.
    #[tokio::test]
    async fn hook_cancel_and_commit_refusal_are_not_replayable() {
        let dir = tempfile::tempdir().unwrap();
        let persist = build_session(dir.path());
        let settings = settings();
        let replay_settings = ReplaySettings {
            settings: &settings,
            context_window: Some(100_000),
        };

        // Hook cancel.
        let flight = FlightLog::new(DEFAULT_CAPACITY);
        let mut p = params(&settings, Some(&flight));
        p.hook = Some(&CancelHook);
        let scripted = ScriptedSummarizer::new(vec![]);
        let err = compact_now(
            &persist,
            &persist.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &p,
        )
        .await
        .expect_err("hook cancels");
        assert_eq!(format!("{err:#}"), "Compaction cancelled");
        assert_eq!(kinds(&flight), ["pass_start", "planned", "hook_cancelled"]);
        let records = decode_log(&flight.encode()).unwrap();
        let err = replay_pass(&persist, &persist.path(), &records, &replay_settings)
            .await
            .unwrap_err();
        assert!(matches!(err, ReplayError::NotReplayable { .. }), "{err}");

        // Commit-guard refusal: the journal tip moves after capture.
        let flight = FlightLog::new(DEFAULT_CAPACITY);
        let guard = CommitGuard::capture(
            &persist,
            std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        )
        .expect("guard");
        persist
            .with_writer(|w| {
                w.append_entry(
                    "message",
                    json!({"message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "concurrent append"}],
                        "stopReason": "stop", "timestamp": 1,
                    }}),
                    "2026-01-01T00:01:00.000Z",
                )
            })
            .expect("concurrent append");
        let mut p = params(&settings, Some(&flight));
        p.guard = Some(&guard);
        let scripted = ScriptedSummarizer::new(vec![
            Injected::Reply("S".to_owned()),
            Injected::Reply("P".to_owned()),
            Injected::Reply("s".to_owned()),
            Injected::Reply("x".to_owned()),
        ]);
        let err = compact_now(
            &persist,
            &persist.path(),
            &Summarizer::Injected(scripted.summarize_fn()),
            &p,
        )
        .await
        .expect_err("guard refuses the commit");
        assert_eq!(format!("{err:#}"), "Compaction cancelled");
        let kinds = kinds(&flight);
        assert_eq!(&kinds[..2], ["pass_start", "planned"]);
        assert_eq!(kinds.last(), Some(&"commit_refused"));
        for pair in kinds[2..kinds.len() - 1].chunks(2) {
            assert_eq!(pair, ["summarize_call", "summarize_reply"]);
        }
        let records = decode_log(&flight.encode()).unwrap();
        let err = replay_pass(&persist, &persist.path(), &records, &replay_settings)
            .await
            .unwrap_err();
        assert!(matches!(err, ReplayError::NotReplayable { .. }), "{err}");
    }
}
