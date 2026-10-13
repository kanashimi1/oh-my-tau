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
    resume: Option<PathBuf>,
) -> Result<()> {
    let (tx, rx) = mpsc::unbounded_channel::<Value>();
    let rt = Runtime::start(Arc::new(agent), target, cwd.clone(), persist, tx).await;
    if let Some(path) = resume {
        rt.load_session(&path, None).await?;
        rt.publish_session_file();
    }
    // Session renames have no dedicated runtime frame (they reach hosts via
    // the `sessionName` field of `state` dumps and this listener); forward
    // them into the UI stream as `session_name_changed` frames.
    {
        let listener_rt = rt.clone();
        rt.set_name_listener(Some(Arc::new(move |name: &str| {
            listener_rt.send(serde_json::json!({
                "type": "session_name_changed",
                "name": name,
            }));
        })));
    }
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
        // Envelope shape must match the RPC loop's resolution: `answers` /
        // `value` / `cancelled` live at the TOP level of the frame (see
        // `tools/ask.rs::results_from_response`), not nested under a key.
        let mut envelope = serde_json::json!({
            "type": "extension_ui_response",
            "id": id,
        });
        if let serde_json::Value::Object(members) = value {
            if let serde_json::Value::Object(env) = &mut envelope {
                env.extend(members);
            }
        }
        crate::rpc::resolve_pending_ui_request(&id, envelope.clone());
        self.rt.send(envelope);
    }

    fn resume_session(&self, path: PathBuf, fork: bool) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            let result: anyhow::Result<()> = if fork {
                crate::tree::fork(&rt, None)
                    .await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("{e:?}"))
            } else {
                rt.load_session(&path, None).await
            };
            if let Err(e) = result {
                rt.send(serde_json::json!({
                    "type": "prompt_result",
                    "success": false,
                    "error": format!("resume failed: {e:#}"),
                }));
            }
            rt.publish_session_file();
        });
    }

    fn continue_run(&self) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            if let Err(e) = rt
                .prompt(None, omp_tui::MANUAL_CONTINUE_PROMPT, None, Vec::new())
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

    fn abort_retry(&self) {
        self.rt.abort_retry();
    }

    fn rewind_entry(&self, entry_id: String) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            if let Err(e) =
                crate::tree::navigate_tree(&rt, &entry_id, &serde_json::json!({}), None).await
            {
                rt.send(serde_json::json!({
                    "type": "prompt_result",
                    "success": false,
                    "error": format!("rewind failed: {e:?}"),
                }));
            }
        });
    }

    fn abort_compaction(&self) {
        // No compaction-abort RPC in the omp-rs runtime yet; kept as a
        // stable no-op so the app wiring does not change when it lands.
    }
}
