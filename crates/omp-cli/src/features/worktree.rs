//! `/wt`, `/worktree`: per-session git worktrees. TS: slash-commands/builtin-lifecycle.ts.

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    rt.send(serde_json::json!({"type": "command_output", "text": "worktree: not implemented yet"}));
    Outcome::Handled { agent_invoked: false }
}
