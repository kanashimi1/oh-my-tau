//! `/share`: share the session via a secret gist. TS: slash-commands/builtin-collaboration.ts.

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    rt.send(serde_json::json!({"type": "command_output", "text": "share: not implemented yet"}));
    Outcome::Handled { agent_invoked: false }
}
