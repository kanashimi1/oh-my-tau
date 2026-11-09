//! Prompt entry point: prompt-result reporting and prompt-text classifiers.

use super::recovery;
use anyhow::Result;
use pa_agent::types::{AgentMessage, Message, StopReason, UserContent};
use serde_json::{json, Value};
use std::sync::Arc;

use super::{lock, message_text, CmdError, RunOutcome, Runtime, Ticket};

impl Runtime {
    /// `prompt_result` / `session_settled` land one tick after the terminal `agent_end` (omp's
    /// `setImmediate`): the response of the command that ended the run (e.g. `abort`) goes first and
    /// queue drains are visible when `sessionSettled` is read.
    pub(super) fn report_prompts(self: &Arc<Self>, outcome: &RunOutcome) {
        // Tickets of the runs that just ended are claimed now: a prompt admitted during the delay
        // below belongs to the next run and must not inherit this result.
        let claimed = {
            let mut p = lock(&self.prompts);
            let starts = p.agent_starts;
            let (ready, rest): (Vec<_>, Vec<_>) = p
                .open
                .drain(..)
                .partition(|t| !t.waiting && starts > t.starts_at_begin);
            p.open = rest;
            ready
        };
        let rt = self.clone();
        let outcome = outcome.clone();
        tokio::spawn(async move {
            // `prompt_result` must never precede run cleanup (`is_streaming =
            // false`, run slot taken): a client that fires its next prompt on
            // prompt_result otherwise races the still-registered run and gets
            // spurious "Agent is already processing" rejections — near-certain
            // with extensions, where the AgentEnd listener chain (ext-host RPC)
            // outlasts the 2 ms delay below. Wait for the run to actually
            // settle first, then keep omp's one-tick delay so the command
            // response still goes out before the result frame.
            rt.agent.wait_for_idle().await;
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            rt.report_prompts_now(&outcome, claimed);
        });
    }

    /// Emit `prompt_result` frames then `session_settled` after a terminal end.
    fn report_prompts_now(&self, outcome: &RunOutcome, claimed: Vec<Ticket>) {
        let drained = self.queued_count() == 0;
        let mut p = lock(&self.prompts);
        let mut remaining = Vec::new();
        let mut reports = claimed;
        for t in p.open.drain(..) {
            if t.waiting && drained {
                reports.push(t);
            } else {
                remaining.push(t);
            }
        }
        p.open = remaining;
        let settle = p.active && drained && p.open.is_empty();
        if settle {
            p.active = false;
        }
        drop(p);
        for t in reports {
            let mut frame = json!({"type": "prompt_result", "agentInvoked": true, "status": outcome.status, "sessionSettled": drained});
            if let Some(id) = t.id {
                frame["id"] = id;
            }
            if let Some(e) = &outcome.error {
                frame["error"] = e.clone();
            }
            self.send(frame);
        }
        if settle {
            self.send(json!({"type": "session_settled"}));
        }
        self.settled.notify_waiters();
    }

    // ------------------------------------------------------------- prompts

    /// Admit a prompt (or queue it into a live run) and open its ticket.
    pub async fn prompt(
        self: &Arc<Self>,
        id: Option<Value>,
        message: &str,
        behavior: Option<&str>,
        images: Vec<pa_agent::types::ImageContent>,
    ) -> Result<(), CmdError> {
        // Budget gate: once the session budget is exceeded, refuse to start a
        // new turn (slash commands are dispatched earlier and still work).
        if let Some(msg) = crate::features::budget::prompt_refusal() {
            self.emit_notice("warning", &msg, Some("budget"));
            return Err(CmdError::new(msg));
        }
        // A prompt during in-flight compaction cancels it and waits for the
        // pass to settle first (TS prompt-side `abortCompaction` +
        // `waitForManualMaintenanceCleanup`): the turn must start on the
        // settled context, not race the compaction's `set_messages`. Runs
        // before the auto-title spawn so the title task cannot append its way
        // into a running pass.
        let admit_t0 = std::time::Instant::now();
        turn_trace!("[turn-trace] prompt-admit t=0ms");
        self.cancel_compaction_and_wait().await;
        self.spawn_auto_title(message);
        if self.agent.is_streaming().await {
            let steer = match behavior {
                Some("steer") => true,
                Some("followUp") => false,
                _ => {
                    return Err(CmdError::new(
                        "Agent is already processing a prompt. Specify streamingBehavior ('steer' or 'followUp') to queue the message.",
                    ))
                }
            };
            // Queued prompts carry their notices in the same batch, ahead of the message.
            let mut batch = self.prompt_notices(message, false).await;
            if steer {
                batch.push(crate::notices::interjection(pa_agent::now_ms()));
            }
            let queued = if batch.is_empty() {
                pa_agent::agent::AgentMessageBatch::Single(user_message(message))
            } else {
                batch.push(user_message(message));
                pa_agent::agent::AgentMessageBatch::Batch(batch)
            };
            if steer {
                self.agent.steer(queued);
            } else {
                self.agent.follow_up(queued);
            }
            lock(&self.prompts).open.push(Ticket {
                id,
                starts_at_begin: 0,
                waiting: true,
            });
            self.emit_queue_update();
            return Ok(());
        }
        self.user_interrupt
            .store(false, std::sync::atomic::Ordering::SeqCst);
        {
            // TS `TodoTracker.resetForPrompt`: reminder budget is per prompt.
            let mut p = lock(&self.prompts);
            p.todo_reminders = 0;
            p.todo_awaiting_progress = false;
        }
        // A new prompt supersedes a pending retry (and waits until the old turn is settled).
        self.cancel_retry_and_wait().await;
        self.maybe_restore_primary().await;
        let starts = {
            let mut p = lock(&self.prompts);
            let starts = p.agent_starts;
            p.open.push(Ticket {
                id: id.clone(),
                starts_at_begin: starts,
                waiting: false,
            });
            starts
        };
        let hook = lock(&self.before_start).clone();
        let mut injected = Vec::new();
        if let Some(hook) = hook {
            let images_json = (!images.is_empty())
                .then(|| serde_json::to_value(&images).ok())
                .flatten();
            injected = hook(message.to_string(), images_json).await;
        }
        turn_trace!(
            "[turn-trace] prompt-before-start-done t={}ms",
            admit_t0.elapsed().as_millis()
        );
        let notices = self.prompt_notices(message, true).await;
        let input = prompt_input(message, images, notices, injected);
        let pua = self.agent.prompt_until_accepted(input).await;
        turn_trace!(
            "[turn-trace] prompt-accepted t={}ms",
            admit_t0.elapsed().as_millis()
        );
        if let Err(e) = pua {
            lock(&self.prompts)
                .open
                .retain(|t| !(t.starts_at_begin == starts && t.id == id && !t.waiting));
            return Err(CmdError::new(format!("{e:#}")));
        }
        // A run now exists: from here on the idle-wake delivery leg may start
        // delivery turns (before the first admission it must stay queued so
        // it can never precede the first user prompt's turn).
        self.first_prompt_admitted
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Hidden notices for a prompt (magic keywords, ultra mode). For a prompt
    /// that starts a turn (`starts_turn`), `ultrathink` under `auto` thinking
    /// also lifts the turn to the model's highest level (TS
    /// `applyAutoThinkingLevel`); queued messages leave the level alone.
    async fn prompt_notices(&self, message: &str, starts_turn: bool) -> Vec<AgentMessage> {
        let present = omp_tui::magic::present(message);
        let mode = self.ultra();
        if present.is_empty() && mode.is_none() {
            return Vec::new();
        }
        let settings = omp_settings::Settings::load(&self.cwd);
        if starts_turn
            && self
                .auto_thinking
                .load(std::sync::atomic::Ordering::Relaxed)
            && present.iter().any(|k| k.id == "ultrathink")
            && omp_tui::magic::enabled(&settings, "ultrathink")
        {
            self.ultrathink_auto().await;
        }
        let tools = self
            .agent
            .tools()
            .await
            .iter()
            .map(|t| t.name().to_owned())
            .collect();
        let ctx = crate::notices::Context::new(tools, &settings);
        crate::notices::build(message, &settings, &ctx, mode, pa_agent::now_ms())
    }
}

/// TS `prompts/system/auto-continue.md`: the hidden resume prompt after a
/// compaction cut the live turn.
pub(super) const AUTO_CONTINUE_PROMPT: &str = include_str!("../../prompts/system/auto-continue.md");

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static MARKDOWN_PROMPT_PREFIX_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^(?:>\s*)?(?:(?:[-*+]|\d+[.)])\s+)*").expect("valid regex")
    });

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static PROMPT_LABEL_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^(?i)(?:q(?:uestion)?|ask)\s*\d*\s*[:.)-]\s*").expect("valid regex")
});

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static QUESTION_PROMPT_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)^(what|which|when|where|why|how|who|whom|whose|do|does|did|can|could|would|will|should|is|are|am|may|shall)\b",
    )
    .expect("valid regex")
});

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static USER_DIRECTED_PROMPT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(?:you|your|we|our)\b").expect("valid regex")
    });

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static NON_ASCII_TEXT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"[^\x00-\x7F]").expect("valid regex"));

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static USER_RESPONSE_CUE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)^(?:please\s+)?(?:confirm|reply|choose|pick|decide|advise)\b|^(?:please\s+)?answer\b|^(?:please\s+)?(?:let\s+me\s+know|tell\s+me)\b",
    )
    .expect("valid regex")
});

// invariant: fixed regex literal always compiles.
#[allow(clippy::expect_used, clippy::unwrap_used)]
static TRAILING_PUNCT_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"[.!?。！？]+$").expect("valid regex"));

/// `promptLine` in TS: strip a markdown list/quote prefix, then a `q1:`-style
/// prompt label. Returns `(text, had_prompt_label)`.
fn prompt_line(line: &str) -> (String, bool) {
    let without_markdown_prefix = MARKDOWN_PROMPT_PREFIX_RE
        .replace(line.trim(), "")
        .trim()
        .to_string();
    let without_prompt_label = PROMPT_LABEL_RE
        .replace(&without_markdown_prefix, "")
        .trim()
        .to_string();
    let had_prompt_label = without_prompt_label != without_markdown_prefix;
    (without_prompt_label, had_prompt_label)
}

/// TS `isQuestionPromptLine`.
fn is_question_prompt_line(line: &str) -> bool {
    let (text, had_prompt_label) = prompt_line(line);
    if !text.ends_with('?') && !text.ends_with('\u{FF1F}') {
        return false;
    }
    had_prompt_label
        || QUESTION_PROMPT_RE.is_match(&text)
        || USER_DIRECTED_PROMPT_RE.is_match(&text)
        || NON_ASCII_TEXT_RE.is_match(&text)
}

/// TS `isResponseCueLine`.
fn is_response_cue_line(line: &str) -> bool {
    let (text, _) = prompt_line(line);
    let candidate = TRAILING_PUNCT_RE
        .replace(text.trim(), "")
        .trim()
        .to_string();
    USER_RESPONSE_CUE_RE.is_match(&candidate)
}

/// TS `isAwaitingUserAnswer`: the assistant's last text line is a direct
/// question or an explicit request for user input, so a completion reminder
/// would interrupt a genuine wait.
pub(super) fn is_awaiting_user_answer(text: &str) -> bool {
    let Some(last_line) = text.lines().last().map(str::trim) else {
        return false;
    };
    if last_line.is_empty() {
        return false;
    }
    is_question_prompt_line(last_line) || is_response_cue_line(last_line)
}

/// The prompt as the agent takes it: the user message, then any extension-injected messages.
/// The prompt's agent input: `prepend` (hidden notices, TS `prependMessages`)
/// ahead of the user message, extension-`injected` messages after it.
pub fn prompt_input(
    text: &str,
    images: Vec<pa_agent::types::ImageContent>,
    prepend: Vec<AgentMessage>,
    injected: Vec<AgentMessage>,
) -> pa_agent::agent::AgentPromptInput {
    if prepend.is_empty() && injected.is_empty() {
        return pa_agent::agent::AgentPromptInput::Text {
            text: text.to_string(),
            images,
        };
    }
    let mut content = vec![pa_agent::types::UserPart::Text(
        pa_agent::types::TextContent {
            text: text.to_string(),
            text_signature: None,
        },
    )];
    content.extend(images.into_iter().map(pa_agent::types::UserPart::Image));
    let user = AgentMessage::Standard(Message::User(pa_agent::types::UserMessage {
        content: UserContent::Parts(content),
        timestamp: pa_agent::now_ms(),
    }));
    let mut all = prepend;
    all.push(user);
    all.extend(injected);
    pa_agent::agent::AgentPromptInput::Messages(all)
}

pub(super) fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pa_agent::types::UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: pa_agent::now_ms(),
    }))
}

/// A user-authored message whose text is exactly `text` (queue edits match exactly).
pub(super) fn is_user_text(m: &AgentMessage, text: &str) -> bool {
    matches!(m, AgentMessage::Standard(Message::User(_))) && message_text(m) == text
}

pub(super) fn run_outcome(messages: &[AgentMessage]) -> RunOutcome {
    for m in messages.iter().rev() {
        if let AgentMessage::Standard(Message::Assistant(a)) = m {
            return match a.stop_reason {
                StopReason::Error => {
                    // TS `promptError`: strip the local raw-request dump lines, classify with the
                    // provider's HTTP status, and report that status as `httpStatus`.
                    let raw = a.error_message.as_deref().unwrap_or("");
                    let message = if raw.is_empty() {
                        "Provider request failed".to_string()
                    } else {
                        pa_ai::utils::http_inspector::strip_raw_http_request_diagnostics(raw)
                    };
                    let mut error = json!({
                        "message": message,
                        "provider": a.provider,
                        "model": a.model,
                        "retryable": recovery::is_transient(a.error_status, raw),
                    });
                    if let Some(status) = a.error_status {
                        error["httpStatus"] = json!(status);
                    }
                    RunOutcome {
                        status: "error",
                        error: Some(error),
                    }
                }
                StopReason::Aborted => RunOutcome {
                    status: "aborted",
                    error: None,
                },
                _ => RunOutcome {
                    status: "completed",
                    error: None,
                },
            };
        }
    }
    RunOutcome {
        status: "completed",
        error: None,
    }
}
