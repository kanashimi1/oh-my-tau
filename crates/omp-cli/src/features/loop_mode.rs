//! `/loop`: re-run a prompt until a condition holds. TS: slash-commands/builtin-modes.ts.

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    rt.send(serde_json::json!({"type": "command_output", "text": "loop_mode: not implemented yet"}));
    Outcome::Handled { agent_invoked: false }
}
