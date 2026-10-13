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
        let rt = self.rt.clone();
        tokio::spawn(async move {
            // Same dispatch chain as RPC `prompt` (builtins → extensions →
            // file templates); slash.rs emits `command_output`/`notice` frames
            // and `set_model` emits `model_changed`, exactly as in RPC mode.
            let text = if args.is_empty() {
                format!("/{name}")
            } else {
                format!("/{name} {args}")
            };
            match crate::rpc::slash::try_dispatch(&rt, &text).await {
                crate::rpc::slash::Outcome::Handled { agent_invoked } => {
                    // The RPC layer answers with an ok response; the TUI has no
                    // response channel, so close the ticket with the same
                    // completed `prompt_result` the ExtensionHandled RPC path
                    // emits (TS interactive controller settles the session the
                    // same way).
                    rt.send(serde_json::json!({
                        "type": "prompt_result",
                        "success": true,
                        "agentInvoked": agent_invoked,
                        "status": "completed",
                        "sessionSettled": true,
                    }));
                }
                crate::rpc::slash::Outcome::ExtensionHandled => {
                    rt.send(serde_json::json!({
                        "type": "prompt_result",
                        "success": true,
                        "agentInvoked": false,
                        "status": "completed",
                        "sessionSettled": true,
                    }));
                }
                crate::rpc::slash::Outcome::Prompt(rewritten) => {
                    if let Err(e) = rt.prompt(None, &rewritten, None, Vec::new()).await {
                        rt.send(serde_json::json!({
                            "type": "prompt_result",
                            "success": false,
                            "error": e.message,
                        }));
                    }
                }
                crate::rpc::slash::Outcome::NotSlash => {
                    // No builtin/extension/file command matched: the palette
                    // offered an unknown name — report it instead of sending
                    // the literal text to the model.
                    rt.send(serde_json::json!({
                        "type": "prompt_result",
                        "success": false,
                        "error": format!("Unknown command: /{name}"),
                    }));
                }
            }
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use omp_tui::RuntimePort as _;
    use pa_agent::scripted::ScriptedProvider;

    use crate::adapter::{json_round_trip, ProviderTarget};
    use crate::runtime::recovery::{RecoveryConfig, Resolver};
    use omp_providers::ResolvedModel;

    fn pa_model(provider: &str, id: &str) -> pa_types::ai::Model {
        serde_json::from_value(serde_json::json!({
            "id": id, "name": id, "api": "openai-completions", "provider": provider,
            "baseUrl": "http://127.0.0.1:9", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    fn resolved(provider: &str, id: &str) -> ResolvedModel {
        ResolvedModel {
            provider: provider.into(),
            id: id.into(),
            api: "openai-completions".into(),
            base_url: "http://127.0.0.1:9".into(),
            api_key: Some("test-key".into()),
            credential_id: None,
            headers: Vec::new(),
            context_window: 100_000,
            max_tokens: 4096,
            reasoning: true,
            thinking: None,
            thinking_auto: false,
            thinking_source: omp_providers::ThinkingSource::None,
            selector: format!("{provider}/{id}"),
            role: None,
            fallbacks: Vec::new(),
            retry: omp_providers::RetryConfig::default(),
            model: pa_model(provider, id),
        }
    }

    struct Harness {
        port: Port,
        rx: mpsc::UnboundedReceiver<Value>,
        _provider: Arc<ScriptedProvider>,
    }

    async fn harness() -> Harness {
        let primary = pa_model("p", "a");
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&primary).expect("agent model"),
        ));
        provider.push_text_turn("ok");
        let agent = Agent::new(pa_agent::agent::AgentOptions {
            initial_state: pa_agent::agent::AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&primary),
                thinking_level: Some(pa_agent::types::ThinkingLevel::Off),
                tools: Some(Vec::new()),
                messages: None,
            },
            stream_fn: Some(provider.stream_fn()),
            ..pa_agent::agent::AgentOptions::default()
        });
        let cwd = std::env::temp_dir().join(format!("omp-tui-slash-{}", std::process::id()));
        std::fs::create_dir_all(&cwd).unwrap();
        let persist = crate::persist::attach_with(&agent, &cwd, true)
            .await
            .expect("session");
        let target = Arc::new(std::sync::RwLock::new(Some(ProviderTarget {
            api_key: Some("test-key".into()),
            model: primary,
            service_tier: None,
            headers: None,
        })));
        let (tx, rx) = mpsc::unbounded_channel();
        let resolver: Resolver = Arc::new(|sel: String| {
            let out: anyhow::Result<ResolvedModel> = (|| {
                let base = sel.split(':').next().unwrap_or(&sel);
                let (p, i) = base
                    .split_once('/')
                    .ok_or_else(|| anyhow::anyhow!("bad selector"))?;
                Ok(resolved(p, i))
            })();
            Box::pin(async move { out })
        });
        let rt = Runtime::start_with(
            Arc::new(agent),
            target,
            cwd,
            persist,
            tx,
            resolver,
            Some(RecoveryConfig::default()),
        )
        .await;
        Harness {
            port: Port { rt },
            rx,
            _provider: provider,
        }
    }

    /// Frames until a `prompt_result` arrives (10 s guard).
    async fn until_result(rx: &mut mpsc::UnboundedReceiver<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let f = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            let done = f["type"] == "prompt_result";
            out.push(f);
            if done {
                return out;
            }
        }
    }

    fn outputs(frames: &[Value]) -> Vec<String> {
        frames
            .iter()
            .filter(|f| f["type"] == "command_output")
            .filter_map(|f| f["text"].as_str().map(str::to_string))
            .collect()
    }

    #[tokio::test]
    async fn slash_model_reports_current_model_and_settles() {
        let mut h = harness().await;
        h.port.run_slash("model".into(), String::new());
        let frames = until_result(&mut h.rx).await;
        let result = frames.last().unwrap();
        assert_eq!(result["success"], true);
        assert_eq!(result["status"], "completed");
        assert!(
            outputs(&frames).iter().any(|t| t.contains("Current model: p/a")),
            "frames: {frames:?}"
        );
    }

    #[tokio::test]
    async fn slash_model_switch_emits_model_changed() {
        let mut h = harness().await;
        h.port.run_slash("model".into(), "q/b".into());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["success"], true);
        assert!(
            frames.iter().any(|f| f["type"] == "model_changed"),
            "frames: {frames:?}"
        );
        assert!(
            outputs(&frames)
                .iter()
                .any(|t| t.contains("Model set to q/b.")),
            "frames: {frames:?}"
        );
    }

    #[tokio::test]
    async fn slash_effort_lists_and_sets_levels() {
        let mut h = harness().await;
        h.port.run_slash("effort".into(), String::new());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["success"], true);
        assert!(
            outputs(&frames).iter().any(|t| t.contains("Thinking:")),
            "frames: {frames:?}"
        );

        let mut h = harness().await;
        h.port.run_slash("effort".into(), "high".into());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["success"], true);
        assert!(
            outputs(&frames)
                .iter()
                .any(|t| t.contains("Thinking set to high.")),
            "frames: {frames:?}"
        );
    }

    #[tokio::test]
    async fn slash_compact_reports_and_settles() {
        let mut h = harness().await;
        h.port.run_slash("compact".into(), String::new());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["success"], true);
        assert!(
            outputs(&frames).iter().any(|t| t.contains("Compaction")),
            "frames: {frames:?}"
        );
    }

    #[tokio::test]
    async fn slash_clear_starts_a_new_session() {
        let mut h = harness().await;
        let before = h.port.rt.session_id();
        h.port.run_slash("clear".into(), String::new());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["success"], true);
        assert!(
            outputs(&frames).iter().any(|t| t.contains("New session started.")),
            "frames: {frames:?}"
        );
        assert_ne!(before, h.port.rt.session_id());
    }

    #[tokio::test]
    async fn slash_export_writes_transcript_and_prints_path() {
        let mut h = harness().await;
        let dir = h.port.rt.cwd.clone();
        let target = dir.join("transcript.md");
        h.port
            .run_slash("export".into(), target.display().to_string());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["success"], true);
        let text = std::fs::read_to_string(&target).expect("export file");
        assert_eq!(text, "(empty session)");
        assert!(
            outputs(&frames)
                .iter()
                .any(|t| t.contains(&format!("Session exported to: {}", target.display()))),
            "frames: {frames:?}"
        );
        std::fs::remove_file(&target).ok();
    }

    #[tokio::test]
    async fn slash_unknown_command_is_a_clear_error() {
        let mut h = harness().await;
        h.port.run_slash("nosuchcmd".into(), String::new());
        let frames = until_result(&mut h.rx).await;
        let result = frames.last().unwrap();
        assert_eq!(result["success"], false);
        assert_eq!(result["error"], "Unknown command: /nosuchcmd");
    }
}
