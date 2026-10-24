//! Summary requests and the summarization flow (`generateSummary`, `generateTurnPrefixSummary`,
//! `generateShortSummary`, `compact`).
//!
//! Requests are built as `pa_types::ai::Context` values; the LLM call itself is injected through
//! [`Completer`] so nothing here talks to a provider. Not ported: the remote-compaction endpoint,
//! provider-native compaction (`OpenAI` Responses / Anthropic), per-model dialect transcripts
//! (set [`SummaryParams::dialect`]), the overflow-halving retry and the transient-error
//! retry of the oneshot (wrap the [`Completer`]), and the two summaries of a split turn run
//! sequentially rather than in parallel.

use std::future::Future;

use anyhow::{anyhow, Result};
use pa_types::ai::{
    AssistantContentBlock, AssistantMessage, Context, Message, StopReason, TextContent,
    UserContent, UserContentBlock, UserMessage,
};
use serde_json::{json, Value};

use crate::convert::{convert_message_cow, convert_to_llm};
use crate::cut::CompactionPreparation;
use crate::dialect::{
    serialize_conversation_for_summary, serialize_conversation_for_summary_refs, Dialect,
};
use crate::files::{compute_file_lists, upsert_file_operations};
use crate::prompts;
use crate::serialize::{escape_summary_boundary_tags, utf16_len, utf16_prefix};
use crate::settings::{DEFAULT_RESERVE_TOKENS, MAX_SUMMARY_TOKENS};
use crate::tokens::{ByteEstimate, Tokenizer};

/// Fallback window for a model whose catalog entry has no usable context window.
const DEFAULT_SUMMARY_INPUT_WINDOW: u64 = 200_000;
/// Floor for one summarization window.
const MIN_SUMMARY_INPUT_TOKENS: u64 = 16_384;

/// Which oneshot a [`SummaryRequest`] is (`omp.gen_ai.oneshot.kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryKind {
    CompactionSummary,
    CompactionTurnPrefix,
    CompactionShortSummary,
    BranchSummary,
    Handoff,
}

/// One summarization LLM call: the context to send and its output budget.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryRequest {
    pub kind: SummaryKind,
    pub context: Context,
    pub max_tokens: u64,
}

/// The injected LLM call. Implement it over `pa-ai` (or a fake in tests); any
/// `Fn(SummaryRequest) -> impl Future<Output = Result<AssistantMessage>>` works too.
pub trait Completer {
    fn complete(
        &self,
        request: SummaryRequest,
    ) -> impl Future<Output = Result<AssistantMessage>> + Send;
}

impl<F, Fut> Completer for F
where
    F: Fn(SummaryRequest) -> Fut,
    Fut: Future<Output = Result<AssistantMessage>> + Send,
{
    fn complete(
        &self,
        request: SummaryRequest,
    ) -> impl Future<Output = Result<AssistantMessage>> + Send {
        self(request)
    }
}

/// Knobs shared by every summarization call of one compaction (`SummaryOptions` + model facts).
pub struct SummaryParams<'a> {
    /// `settings.reserveTokens ?? DEFAULT_RESERVE_TOKENS`.
    pub reserve_tokens: u64,
    /// Summarizer model context window; `None` = unknown (200k assumed).
    pub context_window: Option<u64>,
    /// `/compact <instructions>`.
    pub custom_instructions: Option<&'a str>,
    /// `SummaryOptions.extraContext` lines (`<additional-context>`).
    pub extra_context: &'a [String],
    /// `SummaryOptions.promptOverride`.
    pub prompt_override: Option<&'a str>,
    /// `Date.now()` for request message timestamps.
    pub now_ms: u64,
    pub tokenizer: &'a dyn Tokenizer,
    /// Transcript dialect of the summarizer model (`preferredDialect(model.id)`).
    pub dialect: Dialect,
}

impl SummaryParams<'static> {
    /// Pick the transcript dialect from the summarizer model id.
    #[must_use]
    pub fn for_model(mut self, model_id: &str) -> Self {
        self.dialect = crate::dialect::preferred_dialect(model_id);
        self
    }

    /// Defaults: default reserve, unknown window, byte-estimate tokenizer, plain serializer.
    #[must_use]
    pub fn new(now_ms: u64) -> Self {
        static BYTES: ByteEstimate = ByteEstimate;
        Self {
            reserve_tokens: DEFAULT_RESERVE_TOKENS,
            context_window: None,
            custom_instructions: None,
            extra_context: &[],
            prompt_override: None,
            now_ms,
            tokenizer: &BYTES,
            dialect: Dialect::Xml,
        }
    }
}

/// `SUMMARIZATION_SYSTEM_PROMPT`.
#[must_use]
pub fn summarization_system_prompt() -> String {
    prompts::render_static(prompts::SUMMARIZATION_SYSTEM)
}

/// A single-user-message context under the summarization system prompt.
#[must_use]
pub fn build_context(text: &str, now_ms: u64) -> Context {
    Context {
        system_prompt: Some(summarization_system_prompt()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
                rest: serde_json::Map::default(),
            })]),
            timestamp: now_ms,
            rest: serde_json::Map::default(),
        })],
        tools: None,
    }
}

/// `formatAdditionalContext`.
fn format_additional_context(context: &[String]) -> String {
    if context.is_empty() {
        return String::new();
    }
    let lines = context
        .iter()
        .map(|l| format!("- {l}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("<additional-context>\n{lines}\n</additional-context>\n\n")
}

/// Maximum output tokens for the history summary: `min(floor(0.8 * reserve), MAX_SUMMARY_TOKENS)`.
#[must_use]
pub fn summary_max_tokens(reserve_tokens: u64) -> u64 {
    ((reserve_tokens as f64 * 0.8).floor() as u64).min(MAX_SUMMARY_TOKENS)
}

/// `summarizeConversationWindow`'s request: the history summary (or iterative update when
/// `previous_summary` is given) over one serialized conversation window.
#[must_use]
pub fn build_summary_request(
    conversation_text: &str,
    previous_summary: Option<&str>,
    params: &SummaryParams<'_>,
) -> SummaryRequest {
    let previous_summary = previous_summary.filter(|p| !p.is_empty());
    let mut base_prompt = prompts::render_static(if previous_summary.is_some() {
        prompts::UPDATE_SUMMARIZATION
    } else {
        prompts::SUMMARIZATION
    });
    if let Some(over) = params.prompt_override {
        over.clone_into(&mut base_prompt);
    }
    if let Some(custom) = params.custom_instructions.filter(|c| !c.is_empty()) {
        base_prompt = format!("{base_prompt}\n\nAdditional focus: {custom}");
    }
    let mut text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
    if let Some(prev) = previous_summary {
        text.push_str(&format!(
            "<previous-summary>\n{}\n</previous-summary>\n\n",
            escape_summary_boundary_tags(prev)
        ));
    }
    text.push_str(&format_additional_context(params.extra_context));
    text.push_str(&base_prompt);
    SummaryRequest {
        kind: SummaryKind::CompactionSummary,
        context: build_context(&text, params.now_ms),
        max_tokens: summary_max_tokens(params.reserve_tokens),
    }
}

/// `generateTurnPrefixSummary`'s request (budget `min(floor(0.5 * reserve), MAX)`).
#[must_use]
pub fn build_turn_prefix_request(messages: &[Value], params: &SummaryParams<'_>) -> SummaryRequest {
    let conversation =
        serialize_conversation_for_summary(&convert_to_llm(messages), params.dialect);
    let text = format!(
        "<conversation>\n{conversation}\n</conversation>\n\n{}",
        prompts::render_static(prompts::TURN_PREFIX)
    );
    SummaryRequest {
        kind: SummaryKind::CompactionTurnPrefix,
        context: build_context(&text, params.now_ms),
        max_tokens: ((params.reserve_tokens as f64 * 0.5).floor() as u64).min(MAX_SUMMARY_TOKENS),
    }
}

/// `generateShortSummary`'s request: a PR-style blurb of the kept tail given the history summary
/// (budget `min(512, floor(0.2 * reserve))`).
#[must_use]
pub fn build_short_summary_request(
    recent_messages: &[Value],
    history_summary: Option<&str>,
    params: &SummaryParams<'_>,
) -> SummaryRequest {
    let conversation =
        serialize_conversation_for_summary(&convert_to_llm(recent_messages), params.dialect);
    let mut text = format!("<conversation>\n{conversation}\n</conversation>\n\n");
    if let Some(summary) = history_summary.filter(|s| !s.is_empty()) {
        text.push_str(&format!(
            "<previous-summary>\n{}\n</previous-summary>\n\n",
            escape_summary_boundary_tags(summary)
        ));
    }
    text.push_str(&format_additional_context(params.extra_context));
    text.push_str(&prompts::render_static(prompts::SHORT_SUMMARY));
    SummaryRequest {
        kind: SummaryKind::CompactionShortSummary,
        context: build_context(&text, params.now_ms),
        max_tokens: 512.min((params.reserve_tokens as f64 * 0.2).floor() as u64),
    }
}

/// Text of an assistant reply (`text` blocks joined by `\n`); a `stopReason: "error"` reply is an
/// error carrying `prefix` and the provider message.
///
/// # Errors
/// The reply ended with `stopReason: "error"`.
pub fn completion_text(reply: &AssistantMessage, error_prefix: &str) -> Result<String> {
    if reply.stop_reason == StopReason::Error {
        return Err(anyhow!(
            "{error_prefix}: {}",
            reply
                .error_message
                .as_deref()
                .filter(|m| !m.is_empty())
                .unwrap_or("Unknown error")
        ));
    }
    Ok(reply
        .content
        .iter()
        .filter_map(|b| match b {
            AssistantContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn window_or_default(window: Option<u64>) -> u64 {
    window
        .filter(|w| *w > 0)
        .unwrap_or(DEFAULT_SUMMARY_INPUT_WINDOW)
}

/// `minSummaryInputTokens`.
fn min_summary_input_tokens(window: Option<u64>) -> u64 {
    MIN_SUMMARY_INPUT_TOKENS.min(1024.max(window_or_default(window) / 8))
}

/// `summaryInputBudgetTokens`: usable conversation input for one summarization call.
#[must_use]
pub fn summary_input_budget_tokens(window: Option<u64>, max_tokens: u64) -> u64 {
    let w = window_or_default(window);
    let discounted =
        (w as f64 * 0.8).floor() as i64 - max_tokens as i64 - MAX_SUMMARY_TOKENS as i64;
    min_summary_input_tokens(window).max(u64::try_from(discounted).unwrap_or(0))
}

/// `Tokenizer.checkTokenBudget`: byte length is a hard upper bound on tokens, so text whose bytes
/// fit never needs counting.
fn fits_budget(tokenizer: &dyn Tokenizer, text: &str, budget: u64) -> (bool, u64) {
    let bound = text.len() as u64;
    if bound <= budget {
        return (true, bound);
    }
    let tokens = tokenizer.count_text(text);
    (tokens <= budget, tokens)
}

/// `clampConversationToBudget`: truncate one oversized serialized window to its budget.
fn clamp_conversation_to_budget(text: &str, budget: u64, tokens: u64) -> String {
    if tokens <= budget {
        return text.to_owned();
    }
    let len = utf16_len(text);
    let keep = 1024.max(((len as f64 * budget as f64 * 0.95) / tokens as f64).floor() as usize);
    if keep >= len {
        return text.to_owned();
    }
    format!(
        "{}\n\n[... {} more characters truncated]",
        utf16_prefix(text, keep),
        len - keep
    )
}

/// `planSummaryWindows`: partition `messages` into windows that each fit `budget`.
fn plan_summary_windows<'a>(
    messages: &[&'a Value],
    params: &SummaryParams<'_>,
    budget: u64,
) -> Vec<Vec<&'a Value>> {
    let mut windows: Vec<Vec<&Value>> = Vec::new();
    let mut current: Vec<&Value> = Vec::new();
    let mut current_tokens = 0u64;
    for &message in messages {
        let tokens = params
            .tokenizer
            .count_text(&serialize_conversation_for_summary_refs(
                std::slice::from_ref(&message),
                params.dialect,
            ));
        if current_tokens > 0 && current_tokens + tokens > budget {
            windows.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        current.push(message);
        current_tokens += tokens;
    }
    if !current.is_empty() {
        windows.push(current);
    }
    windows
}

/// `generateSummary`: summarize `messages` (agent messages), folding window by window when the
/// conversation outgrows the summarizer's window. Returns the carried summary.
///
/// # Errors
/// The completer failed or replied with `stopReason: "error"`.
pub async fn generate_summary(
    completer: &impl Completer,
    messages: &[Value],
    previous_summary: Option<&str>,
    params: &SummaryParams<'_>,
) -> Result<String> {
    let max_tokens = summary_max_tokens(params.reserve_tokens);
    // Borrowed conversion: messages the converter passes through verbatim are
    // not copied (the history being summarized can be most of the session).
    let llm: Vec<std::borrow::Cow<'_, Value>> =
        messages.iter().filter_map(convert_message_cow).collect();
    let llm: Vec<&Value> = llm.iter().map(AsRef::as_ref).collect();
    let whole = serialize_conversation_for_summary_refs(&llm, params.dialect);
    let budget = summary_input_budget_tokens(params.context_window, max_tokens);
    // When the whole conversation fits, it is the single window and its
    // serialization is that window's text: reuse it instead of rendering the
    // transcript a second time.
    let whole_fits = fits_budget(params.tokenizer, &whole, budget).0;
    let whole = whole_fits.then_some(whole);
    let windows: Vec<Vec<&Value>> = if whole_fits {
        Vec::new()
    } else {
        plan_summary_windows(&llm, params, budget)
    };
    let texts = whole.into_iter().chain(
        windows
            .iter()
            .map(|window| serialize_conversation_for_summary_refs(window, params.dialect)),
    );
    let mut carried: Option<String> = previous_summary.map(str::to_owned);
    for text in texts {
        let (fits, tokens) = fits_budget(params.tokenizer, &text, budget);
        let text = if fits {
            text
        } else {
            clamp_conversation_to_budget(&text, budget, tokens)
        };
        let request = build_summary_request(&text, carried.as_deref(), params);
        let reply = completer.complete(request).await?;
        carried = Some(completion_text(&reply, "Summarization failed")?);
    }
    Ok(carried.unwrap_or_default())
}

/// What [`compact`] produces (`CompactionResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionResult {
    pub summary: String,
    /// PR-style blurb (may be empty if the model returned nothing).
    pub short_summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: u64,
    /// `{ readFiles, modifiedFiles }`.
    pub details: Value,
    /// This compaction's own `preserveData` (only the cumulative snapcompact
    /// archive carries forward; a stale provider-native payload must not —
    /// TS keeps `result.preserveData` here, never `previousPreserveData`).
    pub preserve_data: Option<Value>,
}

/// Port of `compact()` for the local summary path.
///
/// # Errors
/// Any summarization call failed, or the kept entry has no id.
pub async fn compact(
    completer: &impl Completer,
    preparation: &CompactionPreparation,
    params: &SummaryParams<'_>,
) -> Result<CompactionResult> {
    let previous = preparation
        .previous_summary
        .as_deref()
        .filter(|p| !p.is_empty());
    let mut summary = if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
        let history = if !preparation.messages_to_summarize.is_empty() || previous.is_some() {
            generate_summary(
                completer,
                &preparation.messages_to_summarize,
                previous,
                params,
            )
            .await?
        } else {
            "No prior history.".to_owned()
        };
        let reply = completer
            .complete(build_turn_prefix_request(
                &preparation.turn_prefix_messages,
                params,
            ))
            .await?;
        let prefix = completion_text(&reply, "Turn prefix summarization failed")?;
        format!("{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}")
    } else if !preparation.messages_to_summarize.is_empty() {
        generate_summary(
            completer,
            &preparation.messages_to_summarize,
            previous,
            params,
        )
        .await?
    } else if let Some(prev) = previous {
        prev.to_owned()
    } else {
        "No prior history.".to_owned()
    };

    let reply = completer
        .complete(build_short_summary_request(
            &preparation.recent_messages,
            Some(&summary),
            params,
        ))
        .await?;
    let short_summary = completion_text(&reply, "Short summary failed")?;

    let (read_files, modified_files) = compute_file_lists(&preparation.file_ops);
    summary = upsert_file_operations(
        &summary,
        &read_files,
        &modified_files,
        Some(&preparation.file_ops.read),
    );

    if preparation.first_kept_entry_id.is_empty() {
        return Err(anyhow!(
            "First kept entry has no ID - session may need migration"
        ));
    }
    Ok(CompactionResult {
        summary,
        short_summary,
        first_kept_entry_id: preparation.first_kept_entry_id.clone(),
        tokens_before: preparation.tokens_before,
        details: json!({ "readFiles": read_files, "modifiedFiles": modified_files }),
        // TS (`session-maintenance.ts`, local summary path) writes only the
        // new result's own preserveData. Blindly cloning the previous entry's
        // payload re-stamps a stale `openaiRemoteCompaction` onto a locally
        // summarized entry; the context rebuild then trusts that payload for
        // the kept tail and replays nothing, dropping every message after the
        // boundary (observed as "no assistant response" after auto-compaction
        // on a resumed session). Only the cumulative snapcompact archive slot
        // carries forward.
        preserve_data: preparation.previous_preserve_data.as_ref().and_then(|p| {
            p.get("snapcompact")
                .map(|archive| serde_json::json!({ "snapcompact": archive.clone() }))
        }),
    })
}
