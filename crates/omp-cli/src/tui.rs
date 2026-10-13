//! Interactive TUI mode: bridges omp-cli's `Runtime` to the `omp-tui` crate.
//!
//! Installed by `main` when stdin/stdout are TTYs and no `-p`/`--mode` flag
//! is present. Creates the frame channel, starts the [`Runtime`] exactly the
//! way `--mode rpc` does, then hands the loop to [`omp_tui::run`].

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use pa_agent::agent::Agent;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::agent::Target;
use crate::persist::Persistence;
use crate::runtime::Runtime;

/// Run the interactive TUI until the user quits.
///
/// # Errors
///
/// Fails if the TUI terminal loop errors; the runtime keeps running until
/// the process exits (TS tears the session down with the process).
pub async fn run(
    agent: Agent,
    target: Target,
    cwd: PathBuf,
    persist: Persistence,
) -> Result<()> {
    let (tx, rx) = mpsc::unbounded_channel::<Value>();
    let rt = Runtime::start(Arc::new(agent), target, cwd.clone(), persist, tx).await;
    let model = {
        let s = rt.agent.state().await;
        format!("{}/{}", s.model.provider, s.model.id)
    };
    let port = Port { rt };
    omp_tui::run(
        omp_tui::RunConfig {
            frames: rx,
            model,
            cwd,
        },
        port,
    )
    .await
}

/// [`omp_tui::RuntimePort`] backed by the omp-cli [`Runtime`]; the async
/// `Runtime` methods are spawned onto the tokio runtime from these sync hooks.
struct Port {
    rt: Arc<Runtime>,
}

impl omp_tui::RuntimePort for Port {
    fn submit(&self, id: Option<String>, text: String, behavior: Option<omp_tui::SubmitBehavior>) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            let behavior = match behavior {
                Some(omp_tui::SubmitBehavior::Steer) => Some("steer"),
                Some(omp_tui::SubmitBehavior::FollowUp) => Some("followUp"),
                None => None,
            };
            if let Err(e) = rt
                .prompt(id.map(Value::from), &text, behavior, Vec::new())
                .await
            {
                rt.send(serde_json::json!({
                    "type": "prompt_result",
                    "success": false,
                    "error": e.message,
                }));
            }
        });
    }

    fn steer(&self, text: String) {
        self.rt.steer(&text);
    }

    fn follow_up(&self, text: String) {
        self.rt.follow_up(&text);
    }

    fn remove_queued(&self, text: String, queue: omp_tui::action::QueueKind) {
        let queue = match queue {
            omp_tui::action::QueueKind::Steering => "steering",
            omp_tui::action::QueueKind::FollowUp => "followUp",
        };
        let _ = self.rt.remove_queued(&text, queue);
    }

    fn promote_queued(&self, text: String) {
        let _ = self.rt.promote_queued(&text);
    }

    fn abort(&self) {
        let rt = self.rt.clone();
        tokio::spawn(async move { rt.abort().await });
    }

    fn set_model(&self, provider: String, id: String) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            if let Err(e) = rt.set_model(&provider, &id).await {
                rt.send(serde_json::json!({
                    "type": "prompt_result",
                    "success": false,
                    "error": e.message,
                }));
            }
        });
    }

    fn set_thinking_level(&self, level: String) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            if let Some(level) = crate::runtime::parse_level(&level) {
                rt.set_thinking_level(level).await;
            }
        });
    }

    fn run_slash(&self, name: String, args: String) {
        // Slash dispatch is owned by rpc::slash; the interactive controller
        // gains it when the command palette lands (see ARCHITECTURE.md).
        self.rt.send(serde_json::json!({
            "type": "prompt_result",
            "success": false,
            "error": format!("slash command '/{name} {args}' is not available in the TUI yet"),
        }));
    }

    fn extension_ui_response(&self, id: String, value: Value) {
        self.rt.send(serde_json::json!({
            "type": "extension_ui_response",
            "id": id,
            "response": value,
        }));
    }
}
