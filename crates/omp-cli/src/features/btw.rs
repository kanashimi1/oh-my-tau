//! `/btw`: side question that does not enter the main context. TS: session/btw-history.ts, slash-commands/builtin-lifecycle.ts.

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    rt.send(serde_json::json!({"type": "command_output", "text": "btw: not implemented yet"}));
    Outcome::Handled { agent_invoked: false }
}
