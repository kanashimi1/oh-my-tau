//! Abort and cancellation: stop the live turn or compaction and wait for it to settle.

use super::{lock, Runtime};

impl Runtime {
    /// Cancel in-flight compaction (TS `abortCompaction`): Esc on the
    /// "compacting… (esc to cancel)" loader, `abort`, and every prompt. The
    /// pass discards at its next checkpoint instead of writing the `compaction`
    /// entry or swapping the context.
    pub fn abort_compaction(&self) {
        if let Some(g) = lock(&self.compaction_guard).as_ref() {
            g.cancel();
        }
    }

    /// Cancel in-flight compaction and wait until it fully settled, so a prompt
    /// admitted behind it starts on the settled context instead of racing the
    /// compaction's context swap (TS `abortCompaction` + prompt-side
    /// `waitForManualMaintenanceCleanup`). Bug-hunt #2: without this, the
    /// concurrent turn was lost and the summary landed in the wrong file.
    pub(super) async fn cancel_compaction_and_wait(&self) {
        self.abort_compaction();
        let handle = lock(&self.post_turn_task).take();
        if let Some(h) = handle {
            let _ = h.await;
        }
        // Manual compactions run inline in their rpc handler (no task handle):
        // park on the end notification like TS parks on the cleanup barrier.
        while self.compacting.load(std::sync::atomic::Ordering::SeqCst) {
            let done = self.compaction_done.notified();
            if !self.compacting.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            done.await;
        }
    }

    pub(super) fn bump_generation(&self) {
        self.switch_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Abort the run and wait until the agent is idle.
    pub async fn abort(&self) {
        self.user_interrupt
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // A2A: cancelling the parent kills its running `task` children
        // (their runner drive loop SIGKILLs the child process groups).
        crate::tools::task::kill_async_children_for(&Some(self.a2a_owner.clone()));
        self.abort_retry();
        self.abort_compaction();
        self.agent.abort();
        self.agent.wait_for_idle().await;
    }

    /// Resolve once every admitted prompt reported and no retry saga is pending (print mode:
    /// `prompt`, then wait here, then read the transcript).
    pub async fn wait_settled(&self) {
        loop {
            let notified = self.settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let idle = {
                let p = lock(&self.prompts);
                p.open.is_empty() && !p.active
            } && !lock(&self.recovery).saga_open();
            if idle {
                return;
            }
            notified.await;
        }
    }
}
