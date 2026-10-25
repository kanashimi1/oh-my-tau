//! `/skills`, `/marketplace`: skill and plugin catalogs. TS: slash-commands/builtin-skills.ts, builtin-marketplace.ts.

use std::sync::Arc;

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let _ = args;
    rt.send(serde_json::json!({"type": "command_output", "text": "skills: not implemented yet"}));
    Outcome::Handled { agent_invoked: false }
}

/// Slash-command entry point for `/marketplace`.
pub async fn marketplace_command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    command(rt, args).await
}
