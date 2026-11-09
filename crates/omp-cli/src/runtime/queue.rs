//! Steering and follow-up queues: enqueue, remove, promote, and queue-mode settings.

use pa_agent::agent::QueueMode;
use pa_agent::types::AgentMessage;
use serde_json::json;
use std::sync::Arc;

use super::prompt::{is_user_text, user_message};
use super::{lock, Runtime};

impl Runtime {
    pub(super) fn queued(&self) -> (Vec<String>, Vec<String>) {
        (
            self.agent.steering_previews(),
            self.agent.follow_up_previews(),
        )
    }

    /// `(steering, followUp)` previews of the queued messages.
    #[cfg(test)]
    #[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
    pub fn queued_previews(&self) -> (Vec<String>, Vec<String>) {
        self.queued()
    }

    pub fn queued_count(&self) -> usize {
        let (a, b) = self.queued();
        a.len() + b.len()
    }

    pub(super) fn emit_queue_update(&self) {
        let snapshot = self.queued();
        let mut st = lock(&self.settings);
        if st.last_queue == snapshot {
            return;
        }
        st.last_queue = snapshot.clone();
        drop(st);
        self.emit_event(
            json!({"type": "queue_update", "steering": snapshot.0, "followUp": snapshot.1}),
        );
    }

    /// TS `#drainStrandedQueuedMessages` + `#canAutoContinueForFollowUp`: once the
    /// session settles, a steer/follow-up stranded after the loop's final queue poll
    /// (e.g. by an abort) starts a fresh turn via `Agent::continue_run`. A
    /// follow-up-only resume stays suppressed after a user interrupt until the next
    /// explicit prompt; a queued steer always resumes (it injects before the first
    /// provider call, so any tail is valid). The drain turn is a fresh run with its
    /// own abort signal (TS `#scheduleAgentContinue` -> `agent.continue()`), so an
    /// empty Enter that interrupts a turn delivers the queued steer as a real turn.
    pub(super) fn drain_stranded_queue(self: &Arc<Self>) {
        let (steering, _follow) = self.queued();
        let interrupted = self
            .user_interrupt
            .load(std::sync::atomic::Ordering::SeqCst);
        if steering.is_empty() && interrupted {
            return;
        }
        let rt = self.clone();
        tokio::spawn(async move {
            rt.agent.wait_for_idle().await;
            if !rt.agent.has_queued_messages() {
                return;
            }
            let _ = rt.agent.continue_run_until_accepted().await;
        });
    }

    pub fn steer(&self, message: &str) {
        self.agent.steer(vec![
            crate::notices::interjection(pa_agent::now_ms()),
            user_message(message),
        ]);
        self.emit_queue_update();
    }

    pub fn follow_up(&self, message: &str) {
        self.agent.follow_up(user_message(message));
        self.emit_queue_update();
    }

    /// Remove the first queued user message whose text is exactly `message` from `queue`
    /// (`removeQueuedMessage`); `None` when there is no such message.
    fn take_queued_exact(&self, message: &str, queue: &str) -> Option<AgentMessage> {
        let (steering, follow) = self.queued();
        // The agent scans steering before follow-up and drops a whole batch per hit, so the
        // wanted message is the n-th exact match in that combined order.
        let target = if queue == "steering" {
            steering.iter().any(|p| p == message).then_some(0)?
        } else {
            follow.iter().any(|p| p == message).then_some(())?;
            steering.iter().filter(|p| *p == message).count()
        };
        let seen = std::cell::Cell::new(0usize);
        let removed = self.agent.remove_queued_messages(|m| {
            if is_user_text(m, message) {
                let hit = seen.get() == target;
                seen.set(seen.get() + 1);
                hit
            } else {
                false
            }
        });
        removed.into_iter().find(|m| is_user_text(m, message))
    }

    pub fn remove_queued(&self, message: &str, queue: &str) -> bool {
        let ok = self.take_queued_exact(message, queue).is_some();
        if ok {
            self.emit_queue_update();
        }
        ok
    }

    pub fn promote_queued(&self, message: &str) -> bool {
        let Some(first) = self.take_queued_exact(message, "followUp") else {
            return false;
        };
        self.agent.steer(vec![
            crate::notices::interjection(pa_agent::now_ms()),
            first,
        ]);
        self.emit_queue_update();
        true
    }

    pub fn set_queue_modes(&self, steering: Option<QueueMode>, follow_up: Option<QueueMode>) {
        if let Some(m) = steering {
            self.agent.set_steering_mode(m);
        }
        if let Some(m) = follow_up {
            self.agent.set_follow_up_mode(m);
        }
    }

    // --------------------------------------------------------------- state

    /// Whether steering/follow-up messages are queued (TS `ctx.hasPendingMessages()`).
    pub fn has_pending_messages(&self) -> bool {
        let (steer, follow) = self.queued();
        !steer.is_empty() || !follow.is_empty()
    }
}

pub fn queue_mode(m: QueueMode) -> &'static str {
    match m {
        QueueMode::All => "all",
        QueueMode::OneAtATime => "one-at-a-time",
    }
}
