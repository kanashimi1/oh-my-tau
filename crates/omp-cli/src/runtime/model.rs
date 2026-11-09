//! Model, thinking-level, ultra, and service-tier state for the live agent.

use super::recovery::RecoveryConfig;
use crate::adapter::{json_round_trip, ProviderTarget};
use crate::tools::edit_mode_for;
use anyhow::Result;
use omp_tui::magic::Ultra;
use pa_agent::types::ThinkingLevel;
use serde_json::{json, Value};

use super::{level_name, lock, parse_level, CmdError, Runtime};

impl Runtime {
    /// Session preamble omp writes: `model_change` then `thinking_level_change`.
    pub(super) async fn record_model_state(&self) {
        let m = self.agent.model().await;
        let model = format!("{}/{}", m.provider, m.id);
        let _ = self.persist.append_model_change(&model, None, false);
        let ultra = self.ultra().map(Ultra::name);
        let _ = self.persist.append_thinking_level_change(
            Some(level_name(self.agent.thinking_level().await)),
            None,
            ultra,
        );
    }

    pub(super) fn live_provider_id(&self) -> Option<(String, String)> {
        let t = self.target.read();
        t.as_ref()
            .map(|t| (String::from(&*t.model.provider), t.model.id.clone()))
    }

    /// `provider/id[:level]` of the live model (`formatRetryFallbackSelector`).
    pub(super) fn live_selector_with_level(&self, level: Option<ThinkingLevel>) -> String {
        let (p, i) = self.live_provider_id().unwrap_or_default();
        match level {
            Some(l) => format!("{p}/{i}:{}", level_name(l)),
            None => format!("{p}/{i}"),
        }
    }

    /// `ultrathink` under `auto`: resolve this turn to the highest supported
    /// effort, journaled as an auto resolution.
    pub(super) async fn ultrathink_auto(&self) {
        let Some(model) = self.live_model().filter(|m| m.reasoning) else {
            return;
        };
        let effective = crate::model::clamp_level(&model, ThinkingLevel::Max);
        let name = level_name(effective);
        if self.agent.thinking_level().await != effective {
            self.agent.set_thinking_level(effective).await;
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some("auto"), None);
        }
        self.emit_event(json!({
            "type": "thinking_level_changed",
            "thinkingLevel": name,
            "configured": "auto",
            "resolved": name,
            "ultra": null,
        }));
    }

    /// Base URL of the live model's provider endpoint (`None` before a model is set).
    pub fn model_base_url(&self) -> Option<String> {
        self.target
            .read()
            .as_ref()
            .map(|t| t.model.base_url.clone())
            .filter(|u| !u.is_empty())
    }

    pub fn model_info(&self) -> Value {
        let target = self.target.read().clone();
        target.map_or(Value::Null, |t| model_info(&t.model))
    }

    /// Shared provider-target slot (`set_model` swaps it): callers resolve the
    /// live model/credentials per request (`LiveCompleter::from_target`).
    pub(crate) fn provider_target(&self) -> crate::agent::Target {
        self.target.clone()
    }

    /// `set_model`: `model_change` entry with role `default`.
    pub async fn set_model(&self, provider: &str, model_id: &str) -> Result<Value, CmdError> {
        self.select_model(provider, model_id, Some("default")).await
    }

    /// `cycle_model`: same switch, `model_change` without a role (`appendModelChange(model)`).
    pub async fn cycle_to(&self, provider: &str, model_id: &str) -> Result<Value, CmdError> {
        self.select_model(provider, model_id, None).await
    }

    async fn select_model(
        &self,
        provider: &str,
        model_id: &str,
        role: Option<&str>,
    ) -> Result<Value, CmdError> {
        let reference = format!("{provider}/{model_id}");
        let resolved = (self.resolver)(reference.clone())
            .await
            .map_err(|_| CmdError::new(format!("Model not found: {provider}/{model_id}")))?;
        if resolved.provider != provider || resolved.id != model_id {
            return Err(CmdError::new(format!(
                "Model not found: {provider}/{model_id}"
            )));
        }
        if resolved.api_key.is_none() {
            return Err(CmdError::new(format!(
                "No API key for {provider}/{model_id}"
            )));
        }
        let info = model_info(&resolved.model);
        // A user selection ends fallback ownership and clears the selector's cooldown.
        {
            let mut rec = lock(&self.recovery);
            rec.reconfigure(RecoveryConfig::from_resolved(&resolved));
            rec.clear_cooldown(&reference);
        }
        self.swap_model(&resolved, role, false, true).await;
        // Re-apply thinking for the new model: the catalog's `thinking.defaultLevel`, else the current level.
        let catalog_default = crate::model::catalog_extras(provider, model_id)
            .into_iter()
            .flatten()
            .find(|(k, _)| *k == "thinking")
            .and_then(|(_, v)| {
                v.get("defaultLevel")
                    .and_then(Value::as_str)
                    .and_then(parse_level)
            });
        // An ultra mode keeps thinking at the highest level the new model supports.
        let preferred = match catalog_default {
            _ if self.ultra().is_some() => ThinkingLevel::Max,
            Some(l) => l,
            None => self.agent.thinking_level().await,
        };
        self.apply_thinking(preferred).await;
        Ok(info)
    }

    /// Swap the live model: provider target, agent model, edit-mode tools, `model_change` entry,
    /// `model_changed` event (only when the model actually changed). Thinking is left to the caller.
    pub(super) async fn swap_model(
        &self,
        resolved: &omp_providers::ResolvedModel,
        role: Option<&str>,
        is_fallback: bool,
        journal: bool,
    ) -> bool {
        let agent_model: Option<pa_agent::types::Model> = json_round_trip(&resolved.model);
        let Some(agent_model) = agent_model else {
            return false;
        };
        let changing = self
            .live_provider_id()
            .is_none_or(|(p, i)| p != resolved.provider || i != resolved.id);
        *self.target.write() = Some(ProviderTarget {
            api_key: resolved.api_key.clone(),
            model: resolved.model.clone(),
            service_tier: None,
            headers: (!resolved.headers.is_empty())
                .then(|| resolved.headers.iter().cloned().collect()),
            credential_id: resolved.credential_id,
        });
        self.agent.set_model(agent_model).await;
        self.agent.set_credential_id(resolved.credential_id);
        let mode = edit_mode_for(&resolved.id);
        let rebuild = {
            let mut st = lock(&self.settings);
            let changed = st.edit_mode != mode;
            st.edit_mode = mode;
            changed
        };
        if rebuild {
            {
                let settings = omp_settings::Settings::load(&self.cwd);
                let mut tools = crate::tools::session_builtin_tools(&self.cwd, mode, &settings);
                // Fresh Envs: restamp this session's owner token (the tools
                // stamp/consume children and deliveries under it).
                for tool in &tools {
                    tool.set_a2a_owner(&self.a2a_owner);
                }
                tools.extend(crate::tools::mcp::mcp_tools(&self.cwd).await);
                tools.extend(crate::tools::wasm::wasm_tools(&self.cwd));
                // TS `reconcileThinkTool`: model swaps re-evaluate the hidden
                // think tool's eligibility.
                let hidden =
                    crate::tools::hidden_tools(&settings, Some(&resolved.model), false, false);
                tools.extend(hidden);
                // The live context strategy keeps its `context_edit` tool.
                if let Some(shared) = crate::context::live() {
                    let tool = crate::context::tool(shared);
                    tools.extend(match crate::approval::Gate::get() {
                        Some(gate) => gate.wrap_tools(vec![tool]),
                        None => vec![tool],
                    });
                }
                self.agent.set_tools(tools).await;
            }
        }
        if changing {
            self.emit_event(json!({"type": "model_changed"}));
            if self
                .ui_context_frames
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                let usage = self.context_usage_frame().await;
                self.send(usage);
            }
        }
        if journal {
            let selector = format!("{}/{}", resolved.provider, resolved.id);
            let _ = self
                .persist
                .append_model_change(&selector, role, is_fallback);
        }
        true
    }

    /// `ModelControls.setThinkingLevel`: clamp to the live model; journal and announce only a change.
    /// Returns the effective level.
    pub(super) async fn apply_thinking(&self, requested: ThinkingLevel) -> ThinkingLevel {
        let model = self.target.read().as_ref().map(|t| t.model.clone());
        let effective = model.map_or(requested, |m| crate::model::clamp_level(&m, requested));
        let current = self.agent.thinking_level().await;
        if current != effective {
            self.agent.set_thinking_level(effective).await;
            let name = level_name(effective);
            let ultra = self.ultra().map(Ultra::name);
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some(name), ultra);
            self.emit_event(
                json!({"type": "thinking_level_changed", "thinkingLevel": name, "ultra": ultra}),
            );
        }
        effective
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        // Pinning a concrete level leaves `auto` (TS `setThinkingLevel`) and any ultra mode.
        self.auto_thinking
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let left_ultra = self.leave_ultra().await;
        let before = self.agent.thinking_level().await;
        let effective = self.apply_thinking(level).await;
        if left_ultra && effective == before {
            // Same level, but the mode is gone: journal and announce that.
            let name = level_name(effective);
            let _ = self
                .persist
                .append_thinking_level_change(Some(name), Some(name), None);
            self.emit_event(
                json!({"type": "thinking_level_changed", "thinkingLevel": name, "ultra": null}),
            );
        }
    }

    /// The active ultra mode.
    #[must_use]
    pub fn ultra(&self) -> Option<Ultra> {
        *lock(&self.ultra)
    }

    /// Enter (`Some`) or leave (`None`, `/ultra off`) an ultra mode.
    ///
    /// Entering pins thinking to the highest level the live model supports
    /// (leaving `auto`); leaving keeps the current level. Either way one
    /// `thinking_level_change` is journaled (`ultra` only while a mode is on)
    /// and `thinking_level_changed` announced; `ultralong`/`hypercode` switch
    /// the `clm-host` context strategy on, other modes / off switch it back.
    pub async fn set_ultra(&self, mode: Option<Ultra>) {
        let prev = std::mem::replace(&mut *lock(&self.ultra), mode);
        if prev.is_none() && mode.is_none() {
            return;
        }
        let name = match mode {
            Some(_) => {
                self.auto_thinking
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                let effective = self.live_model().map_or(ThinkingLevel::Max, |m| {
                    crate::model::clamp_level(&m, ThinkingLevel::Max)
                });
                if self.agent.thinking_level().await != effective {
                    self.agent.set_thinking_level(effective).await;
                }
                level_name(effective)
            }
            None => level_name(self.agent.thinking_level().await),
        };
        let ultra = mode.map(Ultra::name);
        let _ = self
            .persist
            .append_thinking_level_change(Some(name), Some(name), ultra);
        self.emit_event(json!({
            "type": "thinking_level_changed",
            "thinkingLevel": name,
            "configured": name,
            "ultra": ultra,
        }));
        self.sync_clm(prev, mode).await;
    }

    /// Drop the ultra mode without journaling (the caller records the new
    /// level); `true` when one was active.
    pub(super) async fn leave_ultra(&self) -> bool {
        let prev = lock(&self.ultra).take();
        self.sync_clm(prev, None).await;
        prev.is_some()
    }

    pub(super) fn live_model(&self) -> Option<pa_types::ai::Model> {
        self.target.read().as_ref().map(|t| t.model.clone())
    }

    /// Switch the `clm-host` strategy (and its `context_edit` tool) on or off
    /// when the ultra mode's CLM leg changes. A session started with
    /// `clm`/`clm-host` keeps its strategy: ultra never touches it.
    pub(super) async fn sync_clm(&self, prev: Option<Ultra>, next: Option<Ultra>) {
        let (had, want) = (prev.is_some_and(Ultra::clm), next.is_some_and(Ultra::clm));
        if had == want || crate::context::active() != crate::context::Kind::Append {
            return;
        }
        let mut tools = self.agent.tools().await;
        tools.retain(|t| t.name() != crate::context::TOOL_NAME);
        if want {
            let settings = omp_settings::Settings::load(&self.cwd);
            let shared = crate::context::ultra_shared(&settings);
            let tool = crate::context::tool(shared.clone());
            tools.extend(match crate::approval::Gate::get() {
                Some(gate) => gate.wrap_tools(vec![tool]),
                None => vec![tool],
            });
            crate::context::set_live(Some(shared), true);
        } else {
            crate::context::set_live(None, false);
        }
        self.agent.set_tools(tools).await;
        let message = if want {
            format!(
                "Context strategy changed to clm-host ({}); the prompt cache resets once.",
                next.map_or("ultra", Ultra::name)
            )
        } else {
            "Context strategy changed back to append; the prompt cache resets once.".to_owned()
        };
        self.emit_notice("warning", &message, None);
    }

    /// `setThinkingLevel("auto")`: run a provisional concrete level and flag
    /// the session as auto (the status line shows `auto` until classified).
    pub async fn set_auto_thinking(&self) {
        self.leave_ultra().await;
        let provisional = self
            .target
            .read()
            .as_ref()
            .map_or(ThinkingLevel::Medium, |t| {
                crate::model::clamp_level(&t.model, ThinkingLevel::Medium)
            });
        self.apply_thinking(provisional).await;
        let name = level_name(provisional);
        let _ = self
            .persist
            .append_thinking_level_change(Some(name), Some("auto"), None);
        self.auto_thinking
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.emit_event(json!({"type": "thinking_level_changed", "thinkingLevel": name, "configured": "auto", "ultra": null}));
    }

    /// Efforts the live model supports, least to most intensive, without
    /// `off` (TS `getSupportedEfforts`: catalog `thinking.efforts`; models
    /// without catalog metadata offer the standard ladder). Empty when the
    /// model cannot think.
    #[must_use]
    pub fn supported_efforts(&self) -> Vec<&'static str> {
        let target = self.target.read().clone();
        let Some(t) = target.filter(|t| t.model.reasoning) else {
            return Vec::new();
        };
        match crate::model::catalog_efforts(&t.model.provider, &t.model.id) {
            Some(efforts) => efforts
                .into_iter()
                .map(level_name)
                .filter(|l| *l != "off")
                .collect(),
            // No catalog entry: offer the ladder levels the runtime would not
            // clamp away, so every selector value actually takes effect.
            None => ["minimal", "low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .filter(|name| {
                    parse_level(name).is_some_and(|l| crate::model::clamp_level(&t.model, l) == l)
                })
                .collect(),
        }
    }

    /// Current service-tier override on the live provider target
    /// (TS `session.serviceTier`, driven by `/fast` and `/slow`).
    #[must_use]
    pub fn service_tier(&self) -> Option<pa_types::ai::ServiceTier> {
        self.target.read().as_ref().and_then(|t| t.service_tier)
    }

    /// Set the service-tier override for subsequent requests (TS
    /// `session.setFastMode`/`setSlowMode`: `None` restores the default tier).
    pub fn set_service_tier(&self, tier: Option<pa_types::ai::ServiceTier>) {
        if let Some(t) = self.target.write().as_mut() {
            t.service_tier = tier;
        }
    }

    /// Selectable levels for the live model, `off` first.
    pub fn available_thinking_levels(&self) -> Vec<&'static str> {
        let target = self.target.read().clone();
        let mut levels = vec!["off"];
        if let Some(t) = target {
            if t.model.reasoning {
                levels.extend(["minimal", "low", "medium", "high", "xhigh"]);
            }
        }
        levels
    }
}

/// omp `ModelInfo` from a pa-ai model, plus the catalog metadata (`int`, `tps`, `thinking`,
/// `identity`, `tokenizer`, ...) pa-ai's descriptor does not carry.
pub fn model_info(model: &pa_types::ai::Model) -> Value {
    let mut v = serde_json::to_value(model).unwrap_or(Value::Null);
    if let Some(o) = v.as_object_mut() {
        o.remove("featured");
        o.remove("maxTokensExplicit");
        o.remove("thinking_level_map");
        o.remove("thinkingLevelMap");
        if let Some(extras) = crate::model::catalog_extras(&model.provider, &model.id) {
            let compat = o.remove("compat");
            for (k, val) in extras {
                o.insert(k.to_string(), val);
            }
            if let Some(c) = compat {
                o.insert("compat".into(), c);
            }
        }
    }
    v
}

/// Head-only view of a `message` entry: enough for the context rebuild's
/// default-model inference (`role`/`provider`/`model`) without the payload.
/// Set by `main` when `--model`/`--provider` (resp. `--thinking`) was given:
/// the startup resume keeps the explicit choice instead of the session's
/// (TS `hasExplicitModel`); cleared by the first `load_session`.
pub static MODEL_PINNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub static THINKING_PINNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set by `main` for `-p` runs that persist no session: automatic titling
/// is skipped (see `spawn_auto_title`).
pub static NO_SESSION_PRINT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set by `main` when `--model` was an ordered chain: `(provider/id of the
/// selected entry, the entries after it)`. The startup recovery config of that
/// model walks this chain instead of its configured `retry.fallbackChains`
/// (TS `installSubagentRetryFallbackChain`: a non-empty selector chain wins).
pub static LAUNCH_FALLBACKS: std::sync::OnceLock<(String, Vec<String>)> =
    std::sync::OnceLock::new();

/// [`LAUNCH_FALLBACKS`] applied to the startup resolution of the live model.
pub(super) fn apply_launch_fallbacks(resolved: &mut omp_providers::ResolvedModel) {
    if let Some((primary, chain)) = LAUNCH_FALLBACKS.get() {
        if *primary == format!("{}/{}", resolved.provider, resolved.id) && !chain.is_empty() {
            resolved.fallbacks.clone_from(chain);
        }
    }
}

/// Models whose provider has credentials (plus the live model), for
/// `get_available_models` / `cycle_model`.
pub async fn available_models(current: Option<&pa_types::ai::Model>) -> Vec<pa_types::ai::Model> {
    let mut out: Vec<pa_types::ai::Model> = Vec::new();
    for provider in pa_ai::models_generated::get_providers() {
        let models = pa_ai::models_generated::get_models(provider);
        let Some(first) = models.first() else {
            continue;
        };
        let has_key = omp_providers::resolve_async(format!("{provider}/{}", first.id))
            .await
            .is_ok_and(|r| r.api_key.is_some());
        if has_key {
            out.extend(models.into_iter().cloned());
        }
    }
    if let Some(c) = current {
        if !out.iter().any(|m| m.provider == c.provider && m.id == c.id) {
            out.insert(0, c.clone());
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
mod ultra_tests {
    use crate::runtime::parse_ultra_name;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use tokio::sync::mpsc::UnboundedReceiver;

    use super::*;
    use crate::adapter::ProviderTarget;

    /// Tests that switch the process-wide live context strategy run one at a time.
    static CLM_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    struct Named(&'static str);

    impl pa_agent::types::AgentTool for Named {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "test tool"
        }
        fn parameters(&self) -> &Value {
            static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| json!({"type": "object", "properties": {}}))
        }
        fn execute(
            self: Arc<Self>,
            _id: String,
            _params: Value,
            _signal: pa_agent::abort::AbortSignal,
            _on_update: pa_agent::types::AgentToolUpdateCallback,
        ) -> pa_agent::BoxFut<'static, anyhow::Result<pa_agent::types::AgentToolResult>> {
            Box::pin(async { Ok(pa_agent::types::AgentToolResult::text("ok")) })
        }
    }

    fn model() -> pa_types::ai::Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": true, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
        provider: Arc<ScriptedProvider>,
        _cwd: tempfile::TempDir,
    }

    /// A runtime over a scripted provider, the CLI's `convert_to_llm` (custom
    /// notices reach the wire as developer messages) and tools named `tools`.
    /// `config` is the project `.omp/config.yml`; `dir` a real session directory.
    async fn harness(tools: &[&'static str], config: &str, dir: Option<&Path>) -> Harness {
        let model = model();
        let provider = Arc::new(ScriptedProvider::new(
            json_round_trip(&model).expect("agent model"),
        ));
        for i in 0..4 {
            provider.push_text_turn(&format!("answer {i}"));
        }
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&model),
                thinking_level: Some(ThinkingLevel::Medium),
                tools: Some(
                    tools
                        .iter()
                        .map(|n| Arc::new(Named(n)) as Arc<dyn pa_agent::types::AgentTool>)
                        .collect(),
                ),
                messages: None,
            },
            convert_to_llm: Some(crate::agent::convert_to_llm()),
            stream_fn: Some(provider.stream_fn()),
            ..AgentOptions::default()
        });
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(cwd.path().join(".omp")).unwrap();
        std::fs::write(cwd.path().join(".omp/config.yml"), config).unwrap();
        let persist = match dir {
            Some(d) => {
                crate::persist::attach_in(&agent, cwd.path(), d, None, crate::persist::uuid_v7())
                    .await
            }
            None => crate::persist::attach_with(&agent, cwd.path(), true).await,
        }
        .expect("session");
        let target = Arc::new(parking_lot::RwLock::new(Some(ProviderTarget {
            api_key: Some("k".into()),
            model,
            service_tier: None,
            headers: None,
            credential_id: None,
        })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let rt = Runtime::start(
            Arc::new(agent),
            target,
            cwd.path().to_path_buf(),
            persist,
            tx,
        )
        .await;
        Harness {
            rt,
            rx,
            provider,
            _cwd: cwd,
        }
    }

    async fn until_settled(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv())
                .await
                .expect("frame in time")
                .expect("open");
            let done = f["type"] == "session_settled";
            out.push(f);
            if done {
                return out;
            }
        }
    }

    fn drain(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    fn of_type<'a>(frames: &'a [Value], kind: &str) -> Vec<&'a Value> {
        frames.iter().filter(|f| f["type"] == kind).collect()
    }

    fn thinking_entries(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|v| v["type"] == "thinking_level_change")
            .collect()
    }

    /// `(role, customType or text)` of the last request's messages.
    fn last_wire(provider: &ScriptedProvider) -> Vec<(String, String)> {
        let calls = provider.calls();
        let ctx = calls.last().expect("a provider call");
        ctx.messages
            .iter()
            .map(|m| {
                let v = serde_json::to_value(m).unwrap();
                let text = match &v["content"] {
                    Value::String(s) => s.clone(),
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                (v["role"].as_str().unwrap_or_default().to_owned(), text)
            })
            .collect()
    }

    async fn prompt(h: &mut Harness, text: &str) -> Vec<Value> {
        h.rt.prompt(None, text, None, Vec::new())
            .await
            .expect("prompt");
        until_settled(&mut h.rx).await
    }

    /// Journal + restore round trip: the session line carries `ultra` next to a
    /// TS-readable `thinkingLevel`, and loading the session brings the mode back.
    #[tokio::test]
    async fn set_ultra_journals_and_restores() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = harness(&["task", "eval"], "", Some(dir.path())).await;
        prompt(&mut h, "first").await;
        h.rt.set_ultra(Some(Ultra::Code)).await;
        let frames = drain(&mut h.rx);
        let changed = of_type(&frames, "thinking_level_changed");
        assert_eq!(changed.len(), 1, "{frames:?}");
        assert_eq!(changed[0]["ultra"], "ultracode");
        let level = h.rt.agent.thinking_level().await;
        assert_eq!(
            level,
            crate::model::clamp_level(&model(), ThinkingLevel::Max)
        );
        assert_eq!(changed[0]["thinkingLevel"], level_name(level));
        assert_eq!(h.rt.state().await["ultra"], "ultracode");
        assert_eq!(h.rt.context_usage_frame().await["ultra"], "ultracode");
        prompt(&mut h, "second").await;

        let file = h.rt.session_file();
        let entries = thinking_entries(&file);
        let last = entries.last().expect("thinking entry");
        assert_eq!(last["ultra"], "ultracode", "{last}");
        assert!(
            parse_level(last["thinkingLevel"].as_str().unwrap()).is_some(),
            "TS-readable level: {last}"
        );
        assert_eq!(last["configured"], last["thinkingLevel"]);
        // Entries without a mode stay TS-shaped (no `ultra` key at all).
        assert!(entries[0].get("ultra").is_none(), "{}", entries[0]);

        let mut second = harness(&["task", "eval"], "", Some(dir.path())).await;
        assert_eq!(second.rt.ultra(), None);
        second.rt.load_session(&file, None).await.expect("resume");
        assert_eq!(second.rt.ultra(), Some(Ultra::Code));
        assert_eq!(second.rt.state().await["ultra"], "ultracode");
        let frames = drain(&mut second.rx);
        assert!(
            of_type(&frames, "thinking_level_changed")
                .iter()
                .any(|f| f["ultra"] == "ultracode"),
            "{frames:?}"
        );
    }

    #[tokio::test]
    async fn set_thinking_level_clears_ultra() {
        let mut h = harness(&[], "", None).await;
        h.rt.set_ultra(Some(Ultra::Deep)).await;
        let level = h.rt.agent.thinking_level().await;
        drain(&mut h.rx);
        // Same level as ultra pinned: the mode still ends, announced and journaled.
        h.rt.set_thinking_level(level).await;
        assert_eq!(h.rt.ultra(), None);
        let frames = drain(&mut h.rx);
        let changed = of_type(&frames, "thinking_level_changed");
        assert_eq!(changed.len(), 1, "{frames:?}");
        assert_eq!(changed[0]["ultra"], Value::Null);
        assert_eq!(h.rt.state().await["ultra"], Value::Null);
        // `auto` clears it too; `/ultra off` keeps the level.
        h.rt.set_ultra(Some(Ultra::Code)).await;
        h.rt.set_auto_thinking().await;
        assert_eq!(h.rt.ultra(), None);
        assert!(of_type(&drain(&mut h.rx), "thinking_level_changed")
            .iter()
            .all(|f| f.get("ultra").is_some()));
        h.rt.set_ultra(Some(Ultra::Code)).await;
        let level = h.rt.agent.thinking_level().await;
        h.rt.set_ultra(None).await;
        assert_eq!(h.rt.ultra(), None);
        assert_eq!(h.rt.agent.thinking_level().await, level);
        let frames = drain(&mut h.rx);
        assert_eq!(
            of_type(&frames, "thinking_level_changed").last().unwrap()["ultra"],
            Value::Null
        );
    }

    /// Bare `/modelpreset` used to index `tokens[1..]` of an empty arg list and kill the process.
    #[tokio::test]
    async fn bare_modelpreset_reports_instead_of_panicking() {
        let mut h = harness(&[], "", None).await;
        drain(&mut h.rx);
        let outcome = crate::rpc::slash::try_dispatch(&h.rt, "/modelpreset").await;
        assert!(matches!(
            outcome,
            crate::rpc::slash::Outcome::Handled {
                agent_invoked: false
            }
        ));
        assert_eq!(of_type(&drain(&mut h.rx), "command_output").len(), 1);
    }

    #[tokio::test]
    async fn ultra_notice_prepended_on_every_prompt() {
        let mut h = harness(&["task", "eval"], "", None).await;
        prompt(&mut h, "plain question").await;
        assert!(
            last_wire(&h.provider)
                .iter()
                .all(|(role, _)| role != "developer"),
            "no notice without a mode"
        );
        h.rt.set_ultra(Some(Ultra::Code)).await;
        for text in ["fix the parser", "ultracode it"] {
            prompt(&mut h, text).await;
            let wire = last_wire(&h.provider);
            let n = wire.len();
            assert_eq!(wire[n - 1].0, "user", "{wire:?}");
            assert_eq!(
                wire[n - 2].0,
                "developer",
                "notice right before the user message: {wire:?}"
            );
            assert!(
                wire[n - 2].1.contains("Ultra mode **ultracode** is active"),
                "{}",
                wire[n - 2].1
            );
            assert!(
                !wire[n - 3].1.contains("ultracode"),
                "exactly one notice: {wire:?}"
            );
        }
    }

    #[tokio::test]
    async fn keyword_notices_prepended_skipped_when_disabled_or_tool_missing() {
        let mut h = harness(&["task", "eval"], "", None).await;
        prompt(&mut h, "please jevify these files").await;
        let wire = last_wire(&h.provider);
        let n = wire.len();
        assert_eq!(wire[n - 2].0, "developer", "{wire:?}");
        assert!(wire[n - 2].1.contains("**jevify**"), "{}", wire[n - 2].1);
        assert_eq!(
            wire[n - 1],
            ("user".to_owned(), "please jevify these files".to_owned())
        );
        // Persisted as a hidden custom message.
        let notice =
            h.rt.messages()
                .await
                .into_iter()
                .find(|m| m["customType"] == "jevify-notice")
                .expect("journaled");
        assert_eq!(notice["display"], false);
        assert_eq!(notice["attribution"], "user");

        // Required tool missing.
        let mut h = harness(&["task"], "", None).await;
        prompt(&mut h, "please jevify these files").await;
        assert!(last_wire(&h.provider)
            .iter()
            .all(|(role, _)| role != "developer"));

        // Disabled by setting.
        let mut h = harness(&["task", "eval"], "magicKeywords:\n  jevify: false\n", None).await;
        prompt(&mut h, "please jevify these files").await;
        assert!(last_wire(&h.provider)
            .iter()
            .all(|(role, _)| role != "developer"));
    }

    #[tokio::test]
    async fn clm_switch_adds_and_removes_context_edit() {
        let _g = CLM_LOCK.lock().await;
        let mut h = harness(&["task", "eval"], "", None).await;
        let names = |rt: &Arc<Runtime>| {
            let rt = rt.clone();
            async move {
                rt.agent
                    .tools()
                    .await
                    .iter()
                    .map(|t| t.name().to_owned())
                    .collect::<Vec<_>>()
            }
        };
        h.rt.set_ultra(Some(Ultra::Code)).await;
        assert!(!names(&h.rt)
            .await
            .iter()
            .any(|n| n == crate::context::TOOL_NAME));
        assert!(crate::context::live().is_none());
        drain(&mut h.rx);

        h.rt.set_ultra(Some(Ultra::Long)).await;
        assert!(names(&h.rt)
            .await
            .iter()
            .any(|n| n == crate::context::TOOL_NAME));
        let live = crate::context::live().expect("clm live");
        assert_eq!(live.kind, crate::context::Kind::ClmHost);
        assert_eq!(crate::context::skip_extensions(), ["context-slim"]);
        let frames = drain(&mut h.rx);
        let notices = of_type(&frames, "notice");
        assert_eq!(notices.len(), 1, "{frames:?}");
        assert_eq!(notices[0]["level"], "warning");
        assert!(notices[0]["message"]
            .as_str()
            .unwrap()
            .contains("prompt cache"));

        // Long -> Hyper keeps CLM: no second switch.
        h.rt.set_ultra(Some(Ultra::Hyper)).await;
        assert!(of_type(&drain(&mut h.rx), "notice").is_empty());
        assert_eq!(
            names(&h.rt)
                .await
                .iter()
                .filter(|n| *n == crate::context::TOOL_NAME)
                .count(),
            1
        );

        // Leaving CLM (here via a plain thinking level) removes the tool again.
        h.rt.set_thinking_level(ThinkingLevel::High).await;
        assert!(!names(&h.rt)
            .await
            .iter()
            .any(|n| n == crate::context::TOOL_NAME));
        assert!(crate::context::live().is_none());
        assert!(crate::context::skip_extensions().is_empty());
        assert_eq!(of_type(&drain(&mut h.rx), "notice").len(), 1);
        assert_eq!(names(&h.rt).await, ["task", "eval"]);
    }

    /// Ultra is a mode, not a thinking level: no default / CLI flag / model
    /// suffix carries it (subagents never inherit it).
    #[test]
    fn ultra_is_not_a_thinking_level() {
        for u in Ultra::ALL {
            assert_eq!(parse_level(u.name()), None);
            assert_eq!(parse_ultra_name(u.name()), Some(u));
        }
        assert_eq!(parse_ultra_name("code"), None);
        assert_eq!(parse_ultra_name("max"), None);
    }
}
