//! `/browser`, `/computer`: browser automation and computer use.

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    rt.send(serde_json::json!({"type": "command_output", "text": "browser: not implemented yet"}));
    Outcome::Handled { agent_invoked: false }
}

/// Slash-command entry point for `/computer`.
pub async fn computer_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    command(rt, args).await
}
