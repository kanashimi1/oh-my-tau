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
    extensions: Option<Arc<crate::ext::Extensions>>,
) -> Result<()> {
    // OSC 11 background probe: must run before the TUI takes over the
    // terminal (the response bytes would otherwise land in the crossterm
    // event stream). Drives theme.dark/theme.light selection in the app.
    let osc11_bg = omp_tui::chrome::probe_osc11_background(std::time::Duration::from_millis(200));
    let session_sources = session_sources(&cwd, &persist);
    let (tx, rx) = mpsc::unbounded_channel::<Value>();
    // Model catalogue for the `/model` picker: providers with credentials
    // (same source the RPC `get_available_models` and `/model` builtin use).
    let models = crate::subcmd::models::browser_models(&omp_providers::AgentDir::default())
        .into_iter()
        .map(|b| {
            let m = b.model;
            let mut e = omp_tui::overlays::models::ModelEntry::new(
                &m.provider,
                &m.id,
                Some(m.context_window),
            );
            e.name = if m.name.is_empty() {
                m.id.clone()
            } else {
                m.name.clone()
            };
            e.max_tokens = Some(m.max_tokens);
            e.cost_input = m.cost.input.as_f64();
            e.cost_output = m.cost.output.as_f64();
            e.cost_cache_read = m.cost.cache_read.as_f64();
            e.reasoning = m.reasoning;
            e.vision = m.input.contains(&pa_types::ai::ModelInput::Image);
            e.available = b.available;
            e.kind = b.kind;
            e.priority = b.priority;
            e
        })
        .collect();
    let rt = Runtime::start(Arc::new(agent), target, cwd.clone(), persist, tx).await;
    if let Some(gate) = crate::approval::Gate::get() {
        gate.bind(&rt, true);
    }
    if let Some(path) = resume {
        rt.load_session(&path, None).await?;
        rt.publish_session_file();
        announce_loaded_session(&rt).await;
    }
    // Extension wiring (after the startup resume so `session_start` sees the
    // resumed file/title; the resume itself emits no `session_switch`, as in TS).
    if let Some(e) = &extensions {
        e.attach_session(rt.clone(), crate::ext::ExtMode::Tui).await;
    }
    // Slash catalog for the editor's `/` popup: builtins + extension/file
    // commands, same start-up announce sequence as RPC mode.
    rt.set_available_commands(crate::rpc::slash::dynamic_commands(&rt).await);
    rt.announce_commands();
    // Session renames have no dedicated runtime frame (they reach hosts via
    // the `sessionName` field of `state` dumps and this listener); forward
    // them into the UI stream as `session_name_changed` frames.
    {
        let listener_rt = rt.clone();
        let ext_listener = extensions.as_ref().map(|e| e.name_listener());
        rt.set_name_listener(Some(Arc::new(move |name: &str| {
            if let Some(f) = &ext_listener {
                f(name);
            }
            listener_rt.send(serde_json::json!({
                "type": "session_name_changed",
                "name": name,
            }));
        })));
    }
    let (model, model_name) = {
        let m = rt.agent.model().await;
        (format!("{}/{}", m.provider, m.id), m.name.clone())
    };
    rt.enable_context_usage_frames();
    crate::slash_extra::mark_interactive_ui();
    rt.publish_context_usage();
    let port = Port {
        rt,
        ext: extensions,
    };
    omp_tui::run(
        omp_tui::RunConfig {
            frames: rx,
            model,
            model_name: Some(model_name).filter(|n| !n.is_empty()),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            cwd,
            models,
            session_sources,
            osc11_bg,
        },
        port,
    )
    .await
}

/// Session stores for the `/resume` picker: the live session file's dir
/// (omp-rs store, or `--session-dir`), plus TS omp's dir for the same folder
/// (tagged `ts`) when the live dir is the default omp-rs one. Resuming a TS
/// session reads it; the continuation is written into the omp-rs store.
fn session_sources(
    cwd: &std::path::Path,
    persist: &Persistence,
) -> Vec<omp_tui::overlays::sessions::SessionSource> {
    use omp_tui::overlays::sessions::SessionSource;
    let Some(native) = persist.path().parent().map(std::path::Path::to_path_buf) else {
        return Vec::new();
    };
    if native.as_os_str().is_empty() {
        return Vec::new();
    }
    let is_default = crate::persist::sessions_dir(cwd).is_some_and(|d| d == native);
    let mut sources = vec![SessionSource::new(native)];
    if is_default {
        if let Some(ts) = crate::persist::ts_sessions_dir(cwd) {
            sources.push(SessionSource::labeled(ts, "ts"));
        }
    }
    sources
}

/// Hand the resumed history to the TUI (`session_loaded`): the transcript is
/// rebuilt from the loaded messages (TS `renderInitialMessages`).
async fn announce_loaded_session(rt: &Runtime) {
    let name = rt.session_name();
    rt.send(serde_json::json!({
        "type": "session_loaded",
        "messages": rt.messages().await,
        "name": name,
    }));
}

/// [`omp_tui::RuntimePort`] backed by the omp-cli [`Runtime`]; the async
/// `Runtime` methods are spawned onto the tokio runtime from these sync hooks.
struct Port {
    rt: Arc<Runtime>,
    ext: Option<Arc<crate::ext::Extensions>>,
}

/// Extension `input` gate (TS input-controller `#runInputHandlers`, source
/// `interactive`): `Some(text)` to run (possibly rewritten), `None` when an extension
/// handled the input.
async fn gate_prompt(ext: Option<&Arc<crate::ext::Extensions>>, text: &str) -> Option<String> {
    let Some(ext) = ext else {
        return Some(text.to_string());
    };
    match ext
        .input_gate(text, omp_ext::lifecycle::InputSource::Interactive)
        .await
    {
        Ok(crate::ext::PromptGate::Run(t)) => Some(t),
        Ok(crate::ext::PromptGate::Handled) => None,
        Err(e) => {
            eprintln!("omt: extension input: {e:#}");
            Some(text.to_string())
        }
    }
}

/// `models_changed` frame: the re-read `/model` picker catalogue after an
/// F5 refresh (fields mirror the startup mapping in `tui_main`).
fn models_changed_frame() -> Value {
    let models: Vec<Value> =
        crate::subcmd::models::browser_models(&omp_providers::AgentDir::default())
            .into_iter()
            .map(|b| {
                let m = &b.model;
                serde_json::json!({
                    "provider": m.provider,
                    "id": m.id,
                    "name": if m.name.is_empty() { m.id.clone() } else { m.name.clone() },
                    "contextWindow": m.context_window,
                    "maxTokens": m.max_tokens,
                    "costInput": m.cost.input.as_f64(),
                    "costOutput": m.cost.output.as_f64(),
                    "costCacheRead": m.cost.cache_read.as_f64(),
                    "reasoning": m.reasoning,
                    "vision": m.input.contains(&pa_types::ai::ModelInput::Image),
                    "available": b.available,
                    "kind": b.kind,
                    "priority": b.priority,
                })
            })
            .collect();
    serde_json::json!({"type": "models_changed", "models": models})
}

/// TS `RpcPromptResultFrame` for a prompt that failed before or outside the
/// agent run (the TUI-local paths below; the runtime emits its own).
fn prompt_failed(message: impl Into<String>) -> Value {
    serde_json::json!({
        "type": "prompt_result",
        "agentInvoked": false,
        "status": "error",
        "error": {"message": message.into(), "retryable": false},
        "sessionSettled": true,
    })
}

/// TS `RpcPromptResultFrame` for a prompt completed locally (slash command).
fn prompt_completed(agent_invoked: bool) -> Value {
    serde_json::json!({
        "type": "prompt_result",
        "agentInvoked": agent_invoked,
        "status": "completed",
        "sessionSettled": true,
    })
}

/// en-US integer grouping (TS `Number.prototype.toLocaleString()`).
fn group_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// TS `formatEndpoint` (provider-details.ts): origin + path without a
/// trailing slash (`/` for an empty path); unparseable urls verbatim.
fn format_endpoint(base_url: &str) -> String {
    match reqwest::Url::parse(base_url) {
        Ok(url) => {
            let path = url.path().strip_suffix('/').unwrap_or(url.path());
            let path = if path.is_empty() { "/" } else { path };
            format!("{}{path}", url.origin().ascii_serialization())
        }
        Err(_) => base_url.to_string(),
    }
}

/// TS `hasLocalLoopbackBaseUrl` (append-only-context-mode.ts).
fn has_local_loopback_base_url(base_url: &str) -> bool {
    let Some(host) = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
    else {
        return false;
    };
    if matches!(
        host.as_str(),
        "localhost" | "127.0.0.1" | "0.0.0.0" | "::1" | "[::1]"
    ) {
        return true;
    }
    if host.starts_with("10.") || host.starts_with("192.168.") || host.ends_with(".local") {
        return true;
    }
    host.strip_prefix("172.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|octet| octet.parse::<u8>().ok())
        .is_some_and(|octet| (16..=31).contains(&octet))
}

/// TS `shouldEnableAppendOnlyContext(setting, model)`.
fn append_only_context_enabled(setting: &str, model: &Value) -> bool {
    match setting {
        "on" => true,
        "off" => false,
        _ => {
            let provider = model["provider"].as_str().unwrap_or("");
            let base_url = model["baseUrl"].as_str().unwrap_or("");
            !model.is_null()
                && (provider == "deepseek"
                    || matches!(
                        provider,
                        "ollama" | "ollama-cloud" | "lm-studio" | "llama.cpp"
                    )
                    || base_url.to_ascii_lowercase().contains("xiaomimimo.com")
                    || has_local_loopback_base_url(base_url)
                    || model["compat"]["supportsStore"].as_bool() == Some(true))
        }
    }
}

/// The `/session` info report (TS `CommandController.handleSessionCommand`)
/// as a `session_info` frame: one entry per report line — `{"k","v"}`
/// (dim key + value; `vs` = styled value spans), `{"head"}` (bold heading),
/// `{"text","dim"?}` (free line; `""` = blank).
async fn session_info_frame(rt: &Runtime) -> Value {
    use serde_json::json;
    let kv = |k: &str, v: String| json!({"k": k, "v": v});
    let blank = || json!({"text": ""});
    let head = |h: &str| json!({"head": h});
    let stats = rt.session_stats().await;
    let count = |key: &str| stats[key].as_u64().unwrap_or(0).to_string();
    let session_id = stats["sessionId"].as_str().unwrap_or_default().to_string();
    let mut lines = vec![
        kv(
            "File",
            stats["sessionFile"]
                .as_str()
                .unwrap_or("In-memory")
                .to_string(),
        ),
        kv("ID", session_id),
        blank(),
        head("Provider"),
    ];
    let model = rt.model_info();
    let provider = model["provider"].as_str().map(str::to_string);
    match &provider {
        None => lines.push(json!({"text": "No model selected", "dim": true})),
        Some(provider) => {
            let auth = {
                let provider = provider.clone();
                tokio::task::spawn_blocking(move || {
                    omp_providers::auth_details(&omp_providers::AgentDir::default(), &provider)
                })
                .await
                .ok()
            };
            let field = |key: &str| model[key].as_str().unwrap_or_default().to_string();
            lines.push(kv("Name", provider.clone()));
            lines.push(kv("Model", field("id")));
            lines.push(kv("API", field("api")));
            lines.push(kv(
                "Auth",
                auth.as_ref().map_or("auto", |a| a.mode).to_string(),
            ));
            lines.push(kv("Endpoint", format_endpoint(&field("baseUrl"))));
            if let Some(source) = auth.and_then(|a| a.source) {
                lines.push(kv("Source", source));
            }
        }
    }
    lines.push(blank());
    lines.push(head("Messages"));
    lines.push(kv("User", count("userMessages")));
    lines.push(kv("Assistant", count("assistantMessages")));
    lines.push(kv("Tool Calls", count("toolCalls")));
    lines.push(kv("Tool Results", count("toolResults")));
    lines.push(kv("Total", count("totalMessages")));
    lines.push(blank());
    let setting = omp_settings::Settings::load(&rt.cwd)
        .get("provider.appendOnlyContext")
        .as_str()
        .unwrap_or("auto")
        .to_string();
    let active = append_only_context_enabled(&setting, &model);
    let setting_label = if setting == "auto" {
        format!("auto ({})", provider.as_deref().unwrap_or("?"))
    } else {
        setting
    };
    lines.push(json!({"k": "Append-Only", "vs": [
        if active { json!({"t": "active", "c": "success"}) } else { json!({"t": "inactive", "c": "dim"}) },
        {"t": format!(" (setting: {setting_label})")},
    ]}));
    lines.push(head("Tokens"));
    let tokens = &stats["tokens"];
    let token = |key: &str| tokens[key].as_u64().unwrap_or(0);
    lines.push(kv("Input", group_thousands(token("input"))));
    lines.push(kv("Output", group_thousands(token("output"))));
    if token("cacheRead") > 0 {
        lines.push(kv("Cache Read", group_thousands(token("cacheRead"))));
    }
    if token("cacheWrite") > 0 {
        lines.push(kv("Cache Write", group_thousands(token("cacheWrite"))));
    }
    lines.push(kv("Total", group_thousands(token("total"))));
    let cost = stats["cost"].as_f64().unwrap_or(0.0);
    if cost > 0.0 {
        lines.push(blank());
        lines.push(head("Cost"));
        lines.push(kv("Total", format!("{cost:.4}")));
    }
    let lsp = omp_lsp::servers::get_lsp_status();
    if !lsp.is_empty() {
        lines.push(blank());
        lines.push(head("LSP Servers"));
        for server in lsp {
            let (status, color) = match server.status {
                omp_lsp::client::ClientStatus::Ready => ("ready", "success"),
                omp_lsp::client::ClientStatus::Connecting => ("connecting", "warning"),
                omp_lsp::client::ClientStatus::Error => ("error", "error"),
            };
            let status = match &server.error {
                Some(e) if status == "error" => format!("{status}: {e}"),
                _ => status.to_string(),
            };
            lines.push(json!({"k": server.name, "vs": [
                {"t": status, "c": color},
                {"t": " "},
                {"t": format!("({})", server.file_types.join(", ")), "c": "dim"},
            ]}));
        }
    }
    lines.push(blank());
    lines.push(head("MCP Servers"));
    let mcp = crate::tools::mcp::connected_servers();
    if mcp.is_empty() {
        lines.push(json!({"text": "None connected", "dim": true}));
    }
    for (name, tools) in mcp {
        lines.push(json!({"k": name, "vs": [
            {"t": "connected", "c": "success"},
            {"t": " "},
            {"t": format!("({tools} tools)"), "c": "dim"},
        ]}));
    }
    // The TS report ends with a newline, which renders one trailing blank row.
    lines.push(blank());
    json!({"type": "session_info", "lines": lines})
}

impl omp_tui::RuntimePort for Port {
    fn submit(&self, id: Option<String>, text: String, behavior: Option<omp_tui::SubmitBehavior>) {
        let rt = self.rt.clone();
        let ext = self.ext.clone();
        tokio::spawn(async move {
            let Some(text) = gate_prompt(ext.as_ref(), &text).await else {
                rt.send(prompt_completed(false));
                return;
            };
            let behavior = match behavior {
                Some(omp_tui::SubmitBehavior::Steer) => Some("steer"),
                Some(omp_tui::SubmitBehavior::FollowUp) => Some("followUp"),
                None => None,
            };
            if let Err(e) = rt
                .prompt(id.map(Value::from), &text, behavior, Vec::new())
                .await
            {
                rt.send(prompt_failed(e.message));
            }
        });
    }

    fn steer(&self, text: String) {
        let (rt, ext) = (self.rt.clone(), self.ext.clone());
        tokio::spawn(async move {
            if let Some(text) = gate_prompt(ext.as_ref(), &text).await {
                rt.steer(&text);
            }
        });
    }

    fn follow_up(&self, text: String) {
        let (rt, ext) = (self.rt.clone(), self.ext.clone());
        tokio::spawn(async move {
            if let Some(text) = gate_prompt(ext.as_ref(), &text).await {
                rt.follow_up(&text);
            }
        });
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
                rt.send(prompt_failed(e.message));
            }
        });
    }

    fn set_thinking_level(&self, level: String) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            if level == "auto" {
                rt.set_auto_thinking().await;
            } else if let Some(mode) = crate::runtime::parse_ultra_name(&level) {
                // Shift+Tab past `max` (or the selector) picks an ultra mode.
                rt.set_ultra(Some(mode)).await;
            } else if let Some(level) = crate::runtime::parse_level(&level) {
                rt.set_thinking_level(level).await;
            }
        });
    }

    fn available_thinking_levels(&self) -> Vec<String> {
        self.rt
            .supported_efforts()
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn run_slash(&self, name: String, args: String) {
        let rt = self.rt.clone();
        let ext = self.ext.clone();
        tokio::spawn(async move {
            // Same dispatch chain as RPC `prompt` (builtins → extensions →
            // file templates); slash.rs emits `command_output`/`notice` frames
            // and `set_model` emits `model_changed`, exactly as in RPC mode.
            let text = if args.is_empty() {
                format!("/{name}")
            } else {
                format!("/{name} {args}")
            };
            // TS input-controller runs the `input` handlers before slash handling.
            let Some(text) = gate_prompt(ext.as_ref(), &text).await else {
                rt.send(prompt_completed(false));
                return;
            };
            match crate::rpc::slash::try_dispatch(&rt, &text).await {
                crate::rpc::slash::Outcome::Handled { agent_invoked } => {
                    // The RPC layer answers with an ok response; the TUI has no
                    // response channel, so close the ticket with the same
                    // completed `prompt_result` the ExtensionHandled RPC path
                    // emits (TS interactive controller settles the session the
                    // same way).
                    rt.send(prompt_completed(agent_invoked));
                }
                crate::rpc::slash::Outcome::ExtensionHandled => {
                    rt.send(prompt_completed(false));
                }
                crate::rpc::slash::Outcome::Prompt(rewritten) => {
                    if let Err(e) = rt.prompt(None, &rewritten, None, Vec::new()).await {
                        rt.send(prompt_failed(e.message));
                    }
                }
                crate::rpc::slash::Outcome::NotSlash if !text.starts_with('/') => {
                    // An `input` handler rewrote the command into plain text.
                    if let Err(e) = rt.prompt(None, &text, None, Vec::new()).await {
                        rt.send(prompt_failed(e.message));
                    }
                }
                crate::rpc::slash::Outcome::NotSlash => {
                    // No builtin/extension/file command matched: the palette
                    // offered an unknown name — report it instead of sending
                    // the literal text to the model.
                    rt.send(prompt_failed(
                        if rt.available_commands().iter().any(|c| {
                            c.get("name").and_then(serde_json::Value::as_str) == Some(name.as_str())
                        }) {
                            format!("/{name} is not ported to omt yet")
                        } else {
                            format!("Unknown command: /{name}")
                        },
                    ));
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
            match result {
                // Fork targets the current session: the transcript is already right.
                Ok(()) if fork => {}
                Ok(()) => announce_loaded_session(&rt).await,
                Err(e) => {
                    rt.send(prompt_failed(format!("resume failed: {e:#}")));
                    // A load that failed after the agent reset leaves it with
                    // no history: clear the transcript to match. An early
                    // failure (unreadable file) keeps the current session, so
                    // re-announcing would only re-dump it into scrollback.
                    if !fork && rt.messages().await.is_empty() {
                        announce_loaded_session(&rt).await;
                    }
                }
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
                rt.send(prompt_failed(e.message));
            }
        });
    }

    fn settings_get(&self, ids: &[String]) -> Vec<(String, Value)> {
        // Synchronous read: the layered merge is pure fs + env work (TS
        // `cfg://` get runs inline in the UI host too).
        let settings = omp_settings::Settings::load(&self.rt.cwd);
        ids.iter()
            .map(|id| (id.clone(), settings.get(id)))
            .collect()
    }

    fn settings_set(&self, id: String, value: Value) -> Result<(), String> {
        omp_settings::write_global(&id, &value)
            .map(|_| ())
            .map_err(|e| e.to_string())
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
                rt.send(prompt_failed(format!("rewind failed: {e:?}")));
            }
        });
    }

    fn abort_compaction(&self) {
        // No compaction-abort RPC in the omp-rs runtime yet; kept as a
        // stable no-op so the app wiring does not change when it lands.
    }

    fn refresh_models(&self) {
        let rt = self.rt.clone();
        tokio::task::spawn_blocking(move || {
            // TS `ModelRegistry.refresh`: per-provider /models discovery →
            // `models.db`.
            match omp_providers::refresh(&omp_providers::AgentDir::default(), None) {
                Ok(reports) => {
                    for r in reports {
                        let msg = match &r.note {
                            Some(note) => format!("model refresh {}: {note}", r.provider),
                            None if r.discovered.is_empty() => {
                                format!(
                                    "model refresh {}: no new models ({} cached)",
                                    r.provider, r.total
                                )
                            }
                            None => format!(
                                "model refresh {}: +{} ({})",
                                r.provider,
                                r.discovered.len(),
                                r.discovered.join(", ")
                            ),
                        };
                        // `session_info` is the one free-text frame the TUI
                        // renders verbatim.
                        rt.send(
                            serde_json::json!({"type": "session_info", "lines": [{"text": msg}]}),
                        );
                    }
                    // Re-read the merged catalog so the open /model browser
                    // shows new models immediately (TS model hub behavior).
                    rt.send(models_changed_frame());
                }
                Err(e) => rt.send(prompt_failed(format!("model refresh failed: {e:#}"))),
            }
        });
    }

    fn request_session_info(&self) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            let frame = session_info_frame(&rt).await;
            rt.send(frame);
        });
    }

    fn request_usage(&self) {
        let rt = self.rt.clone();
        tokio::spawn(async move {
            let frame = crate::rpc::slash::usage_panel_frame(&rt).await;
            rt.send(frame);
        });
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
            credential_id: None,
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
            port: Port { rt, ext: None },
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
        assert_eq!(result["status"], "completed");
        assert_eq!(result["status"], "completed");
        assert!(
            outputs(&frames)
                .iter()
                .any(|t| t.contains("Current model: p/a")),
            "frames: {frames:?}"
        );
    }

    #[tokio::test]
    async fn slash_model_switch_emits_model_changed() {
        let mut h = harness().await;
        h.port.run_slash("model".into(), "q/b".into());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["status"], "completed");
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
        assert_eq!(frames.last().unwrap()["status"], "completed");
        assert!(
            outputs(&frames).iter().any(|t| t.contains("Thinking:")),
            "frames: {frames:?}"
        );

        let mut h = harness().await;
        h.port.run_slash("effort".into(), "high".into());
        let frames = until_result(&mut h.rx).await;
        assert_eq!(frames.last().unwrap()["status"], "completed");
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
        assert_eq!(frames.last().unwrap()["status"], "completed");
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
        assert_eq!(frames.last().unwrap()["status"], "completed");
        assert!(
            outputs(&frames)
                .iter()
                .any(|t| t.contains("New session started.")),
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
        assert_eq!(frames.last().unwrap()["status"], "completed");
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
        assert_eq!(result["status"], "error");
        assert_eq!(result["error"]["message"], "Unknown command: /nosuchcmd");
    }
}
