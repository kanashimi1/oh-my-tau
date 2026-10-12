//! Context compaction glue around `omp-compact`: settings from `config.yml`, the model-visible
//! context rebuilt from a session file (`rebuild_context`), the summarizer call over `pa-ai`
//! (`LiveCompleter`), the manual/auto compaction commit (`compact_now`) and the post-turn
//! threshold decision (`auto_check`).
//!
//! Only the local summary path (`soft` / `context-full`) is implemented; `snapcompact`, `handoff`,
//! `shake`, remote and speculative/idle compaction are not (a `methodOrder` without `soft` therefore
//! disables automatic compaction).

use std::future::Future;
use std::path::Path;

use anyhow::{anyhow, Context as _, Result};
use omp_compact::context::branch_path;
use omp_compact::settings::{calculate_context_tokens, compaction_context_tokens, last_assistant_usage};
use omp_compact::tokens::{count_messages, CountOptions};
use omp_compact::{
    build_session_context, convert_to_llm, plan_compaction, project_tokens_after, should_compact, ApplyOptions, ByteEstimate, CompactionResult, CompactionSettings, Completer, Leaf,
    SummaryParams, SummaryRequest,
};
use pa_agent::types::AgentMessage;
use pa_types::ai::{AssistantMessage, Model};
use serde_json::{json, Value};

use crate::persist::{iso_from_ms, Persistence};

/// `compaction.*` from `config.yml` (flat dotted keys are honoured too), TS defaults otherwise.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn settings_from_config(config: &serde_yaml_ng::Value) -> CompactionSettings {
    let get = |key: &str| {
        config
            .get("compaction")
            .and_then(|c| c.get(key))
            .or_else(|| config.get(format!("compaction.{key}")))
    };
    let num = |key: &str| get(key).and_then(serde_yaml_ng::Value::as_f64);
    let mut s = CompactionSettings::default();
    if let Some(b) = get("enabled").and_then(serde_yaml_ng::Value::as_bool) {
        s.enabled = b;
    }
    if let Some(n) = num("thresholdPercent") {
        s.threshold_percent = Some(n);
    }
    if let Some(n) = num("thresholdTokens") {
        s.threshold_tokens = Some(n);
    }
    if let Some(n) = num("reserveTokens").filter(|n| *n >= 0.0) {
        s.reserve_tokens = Some(n as u64);
    }
    if let Some(n) = num("keepRecentTokens").filter(|n| *n >= 0.0) {
        s.keep_recent_tokens = n as u64;
    }
    // Only the `soft` method is available here: an explicit order without it leaves nothing to run.
    if let Some(order) = get("methodOrder").and_then(serde_yaml_ng::Value::as_sequence) {
        let has_soft = order.iter().filter_map(serde_yaml_ng::Value::as_str).any(|m| matches!(m, "soft" | "context-full"));
        if !has_soft {
            s.strategy = Some("off".to_owned());
        }
    }
    s
}

/// Settings of the user's omp install (`$OMP_AGENT_DIR` or `~/.omp/agent`, `config.yml`).
#[must_use]
pub fn load_settings() -> CompactionSettings {
    let dir = omp_providers::AgentDir::default().0;
    load_settings_from(&dir)
}

/// [`load_settings`] for an explicit agent directory.
#[must_use]
pub fn load_settings_from(dir: &Path) -> CompactionSettings {
    std::fs::read_to_string(dir.join("config.yml"))
        .ok()
        .and_then(|text| serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text).ok())
        .map(|config| settings_from_config(&config))
        .unwrap_or_default()
}

/// The messages the model sees for `entries` (session entries without the header): the latest
/// compaction summary first, then the kept tail and everything after it.
///
/// # Errors
/// The latest compaction carries a payload this port cannot replay (remote/snapcompact/...).
pub fn rebuild_context(entries: &[Value]) -> Result<Vec<AgentMessage>> {
    let ctx = build_session_context(entries, Leaf::Last).map_err(|e| anyhow!("{e}"))?;
    Ok(convert_to_llm(&ctx.messages)
        .into_iter()
        .filter_map(|m| serde_json::from_value::<AgentMessage>(m).ok())
        .collect())
}

/// Summarizer calls over `pa-ai` against the live model.
pub struct LiveCompleter {
    pub model: Model,
    pub api_key: Option<String>,
    pub headers: Option<std::collections::BTreeMap<String, String>>,
}

impl Completer for LiveCompleter {
    fn complete(&self, request: SummaryRequest) -> impl Future<Output = Result<AssistantMessage>> + Send {
        let model = self.model.clone();
        let api_key = self.api_key.clone();
        let headers = self.headers.clone();
        async move {
            let options = pa_ai::types::SimpleStreamOptions {
                base: pa_ai::types::StreamOptions {
                    max_tokens: Some(request.max_tokens),
                    api_key,
                    headers: headers.map(|h| h.into_iter().collect()),
                    ..pa_ai::types::StreamOptions::default()
                },
                // `resolveCompactionEffort`: High by default, only for reasoning models.
                reasoning: model.reasoning.then_some(pa_types::ai::ModelThinkingLevel::High),
                thinking_budgets: None,
            };
            pa_ai::complete_simple(&model, &request.context, Some(options)).await.map_err(|e| anyhow!("{e:?}"))
        }
    }
}

impl LiveCompleter {
    /// The live provider target (`set_model` swaps it, so resolve per compaction).
    ///
    /// # Errors
    /// No provider target is set.
    pub fn from_target(target: &crate::agent::Target) -> Result<Self> {
        let t = target.read().map_err(|_| anyhow!("provider target lock poisoned"))?.clone().ok_or_else(|| anyhow!("no provider target"))?;
        Ok(Self { model: t.model, api_key: t.api_key, headers: t.headers })
    }
}

/// An injected summarizer (tests, hosts that route summaries elsewhere).
pub type SummarizeFn = std::sync::Arc<dyn Fn(SummaryRequest) -> std::pin::Pin<Box<dyn Future<Output = Result<AssistantMessage>> + Send>> + Send + Sync>;

/// The summarizer a compaction runs with: the live model, or an injected function.
#[derive(Clone)]
pub enum Summarizer {
    Live(std::sync::Arc<LiveCompleter>),
    Injected(SummarizeFn),
}

impl Completer for Summarizer {
    fn complete(&self, request: SummaryRequest) -> impl Future<Output = Result<AssistantMessage>> + Send {
        let this = self.clone();
        async move {
            match this {
                Self::Live(live) => live.complete(request).await,
                Self::Injected(f) => f(request).await,
            }
        }
    }
}

/// What a compaction pass needs besides the session file.
pub struct CompactParams<'a> {
    pub settings: &'a CompactionSettings,
    /// Summarizer model id (`preferredDialect`) and window.
    pub model_id: &'a str,
    pub context_window: Option<u64>,
    /// `/compact <instructions>`.
    pub custom_instructions: Option<&'a str>,
    /// System prompt + tool schema tokens, added to the `tokensAfter` projection.
    pub non_message_tokens: u64,
}

fn now_ms() -> u64 {
    u64::try_from(pa_agent::now_ms()).unwrap_or(0)
}

/// Plan, summarize and commit one compaction of the live session; `Ok` carries the rpc `compact`
/// result (`summary`, `shortSummary`, `firstKeptEntryId`, `tokensBefore`, `details`, `preserveData`).
/// The caller re-reads the session file and swaps the agent's messages via [`rebuild_context`].
///
/// # Errors
/// Nothing to compact / already compacted (TS messages), summarization failure, unsupported
/// previous compaction payload, or a write failure.
pub async fn compact_now(persist: &Persistence, entries: &[Value], completer: &impl Completer, params: &CompactParams<'_>) -> Result<(CompactionResult, Value)> {
    let branch: Vec<Value> = branch_path(entries, Leaf::Last).into_iter().cloned().collect();
    let tokenizer = ByteEstimate;
    let prep = plan_compaction(&branch, params.settings, &tokenizer).map_err(|e| anyhow!("{e}"))?;
    let Some(prep) = prep else {
        if branch.last().is_some_and(|e| e.get("type").and_then(Value::as_str) == Some("compaction")) {
            return Err(anyhow!("Already compacted"));
        }
        return Err(anyhow!("Nothing to compact (session too small)"));
    };
    let mut summary_params = SummaryParams::new(now_ms()).for_model(params.model_id);
    summary_params.reserve_tokens = params.settings.reserve_tokens.unwrap_or(omp_compact::settings::DEFAULT_RESERVE_TOKENS);
    summary_params.context_window = params.context_window;
    summary_params.custom_instructions = params.custom_instructions;
    let result = omp_compact::compact(completer, &prep, &summary_params).await?;
    let ts = iso_from_ms(i64::try_from(now_ms()).unwrap_or(0));
    let tokens_after = project_tokens_after(&branch, &result, Some("soft"), &ts, params.non_message_tokens, &tokenizer).ok();
    let options = ApplyOptions { method: Some("soft".to_owned()), tokens_after, from_extension: Some(false) };
    persist
        .with_writer(|w| omp_compact::append_compaction(w, &result, &ts, &options))
        .context("writing the compaction entry")?;
    let mut data = json!({
        "summary": result.summary,
        "shortSummary": result.short_summary,
        "firstKeptEntryId": result.first_kept_entry_id,
        "tokensBefore": result.tokens_before,
        "details": result.details,
    });
    if let Some(p) = &result.preserve_data {
        data["preserveData"] = p.clone();
    }
    Ok((result, data))
}

/// The post-turn threshold decision (`checkCompaction`, case "threshold"): the provider-billed
/// context of the newest successful assistant turn after the latest compaction, floored by the
/// local estimate of the rebuilt context, against `resolveThresholdTokens`.
#[must_use]
pub fn auto_check(entries: &[Value], context_window: u64, settings: &CompactionSettings) -> bool {
    if !settings.enabled || context_window == 0 {
        return false;
    }
    let branch: Vec<Value> = branch_path(entries, Leaf::Last).into_iter().cloned().collect();
    let after = branch.iter().rposition(|e| e.get("type").and_then(Value::as_str) == Some("compaction")).map_or(0, |i| i + 1);
    // An assistant turn that predates the latest compaction carries stale usage.
    let usage = last_assistant_usage(&branch[after..]).map_or(0, calculate_context_tokens);
    let stored = rebuild_estimate(&branch);
    should_compact(compaction_context_tokens(usage, stored), context_window, settings)
}

fn rebuild_estimate(branch: &[Value]) -> u64 {
    build_session_context(branch, Leaf::Last).map_or(0, |ctx| count_messages(&ByteEstimate, &convert_to_llm(&ctx.messages), CountOptions::default()))
}

/// Copy the active branch of a loaded session into the (fresh) live session file so its
/// `compaction` entries keep working: ids are regenerated by the writer, `firstKeptEntryId` is
/// remapped to the copied entry. Model/thinking changes and other bookkeeping entries come along.
///
/// # Errors
/// An entry does not fit its schema or the write failed.
pub fn seed_entries(persist: &Persistence, entries: &[Value]) -> Result<()> {
    let mut ids = std::collections::HashMap::<String, String>::new();
    for entry in branch_path(entries, Leaf::Last) {
        let Some(kind) = entry.get("type").and_then(Value::as_str) else { continue };
        let Some(Value::Object(mut fields)) = Some(entry.clone()) else { continue };
        let old_id = fields.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
        let ts = fields.get("timestamp").and_then(Value::as_str).unwrap_or_default().to_owned();
        for k in ["type", "id", "parentId", "timestamp"] {
            fields.remove(k);
        }
        for key in ["firstKeptEntryId", "providerReplayThroughEntryId"] {
            if let Some(mapped) = fields.get(key).and_then(Value::as_str).and_then(|id| ids.get(id)) {
                fields.insert(key.to_owned(), Value::String(mapped.clone()));
            }
        }
        let new_id = persist.with_writer(|w| w.append_entry(kind, Value::Object(fields), &ts))?;
        ids.insert(old_id, new_id);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use tokio::sync::mpsc::UnboundedReceiver;

    use super::*;
    use crate::adapter::{json_round_trip, ProviderTarget};
    use crate::runtime::Runtime;

    fn pa_model() -> Model {
        serde_json::from_value(json!({
            "id": "a", "name": "a", "api": "openai-completions", "provider": "p",
            "baseUrl": "http://127.0.0.1:9", "reasoning": false, "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000, "maxTokens": 4096
        }))
        .expect("model json")
    }

    fn reply(text: &str) -> AssistantMessage {
        serde_json::from_value(json!({
            "content": [{"type": "text", "text": text}], "api": "openai-completions", "provider": "p", "model": "a",
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "stop", "timestamp": 1
        }))
        .expect("reply json")
    }

    /// An injected summarizer answering every request with `text`, counting the calls.
    fn fake_summarizer(text: &'static str) -> (SummarizeFn, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let f: SummarizeFn = Arc::new(move |_req| {
            seen.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(reply(text)) })
        });
        (f, calls)
    }

    struct Harness {
        rt: Arc<Runtime>,
        rx: UnboundedReceiver<Value>,
    }

    async fn harness(turns: usize) -> Harness {
        let model = pa_model();
        let provider = Arc::new(ScriptedProvider::new(json_round_trip(&model).expect("agent model")));
        for i in 0..turns {
            provider.push_text_turn(&format!("answer {i} {}", "y".repeat(400)));
        }
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("test".into()),
                model: json_round_trip(&model),
                thinking_level: Some(pa_agent::types::ThinkingLevel::Off),
                tools: Some(Vec::new()),
                messages: None,
            },
            stream_fn: Some(provider.stream_fn()),
            ..AgentOptions::default()
        });
        let cwd = std::env::temp_dir();
        let persist = crate::persist::attach_with(&agent, &cwd, true).await.expect("session");
        let target = Arc::new(std::sync::RwLock::new(Some(ProviderTarget { api_key: Some("k".into()), model, service_tier: None, headers: None })));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let rt = Runtime::start(Arc::new(agent), target, cwd, persist, tx).await;
        // keep the provider alive for the runtime's lifetime
        std::mem::forget(provider);
        Harness { rt, rx }
    }

    async fn until_settled(rx: &mut UnboundedReceiver<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let f = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("frame in time").expect("open");
            let done = f["type"] == "session_settled";
            out.push(f);
            if done {
                return out;
            }
        }
    }

    fn small_keep() -> CompactionSettings {
        CompactionSettings { keep_recent_tokens: 100, ..CompactionSettings::default() }
    }

    async fn talk(h: &mut Harness, prompts: usize) {
        for i in 0..prompts {
            h.rt.prompt(None, &format!("question {i} {}", "x".repeat(400)), None, Vec::new()).await.expect("prompt");
            until_settled(&mut h.rx).await;
        }
    }

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../omp-compact/tests/fixtures/compacted-session.jsonl")
    }

    fn user_text(m: &AgentMessage) -> String {
        serde_json::to_value(m).unwrap()["content"].to_string()
    }

    #[test]
    fn config_yaml_maps_to_settings() {
        let cfg: serde_yaml_ng::Value = serde_yaml_ng::from_str("compaction:\n  enabled: false\n  thresholdPercent: 70\n  reserveTokens: 5000\n  keepRecentTokens: 123\n  methodOrder: [snapcompact, shake]\n").unwrap();
        let s = settings_from_config(&cfg);
        assert!(!s.enabled);
        assert_eq!((s.threshold_percent, s.reserve_tokens, s.keep_recent_tokens), (Some(70.0), Some(5000), 123));
        assert_eq!(s.strategy.as_deref(), Some("off"), "no soft method configured");
        let defaults = settings_from_config(&serde_yaml_ng::from_str("compaction:\n  methodOrder: [snapcompact, soft]\n").unwrap());
        assert!(defaults.enabled && defaults.strategy.as_deref() == Some("context-full"));
    }

    #[tokio::test]
    async fn resume_of_compacted_session_rebuilds_summary_plus_tail() {
        let h = harness(0).await;
        let raw = crate::runtime::read_entries(&fixture()).expect("fixture");
        let expected = rebuild_context(&raw).expect("rebuild");
        assert!(user_text(&expected[0]).contains("<summary>"), "summary comes first: {}", user_text(&expected[0]));
        assert!(expected.len() >= 2);

        h.rt.load_session(&fixture(), None).await.expect("load");
        let live = h.rt.messages().await;
        assert_eq!(live.len(), expected.len(), "model sees the rebuilt context, not the raw log");
        assert!(live[0].to_string().contains("<summary>"));

        // The seeded file keeps the compaction usable: same context when rebuilt from it, and
        // `firstKeptEntryId` points at a copied entry.
        let seeded = h.rt.entries().expect("entries");
        let compaction = seeded.iter().find(|e| e["type"] == "compaction").expect("compaction copied");
        let kept = compaction["firstKeptEntryId"].as_str().unwrap();
        assert!(seeded.iter().any(|e| e["id"] == kept), "firstKeptEntryId remapped to the new file");
        assert_eq!(rebuild_context(&seeded).unwrap().len(), expected.len());
    }

    #[tokio::test]
    async fn forced_compaction_with_fake_summarizer_commits_entry_and_swaps_context() {
        let mut h = harness(3).await;
        h.rt.set_compaction_settings(CompactionSettings { enabled: false, ..small_keep() });
        let (fake, calls) = fake_summarizer("FAKE SUMMARY");
        h.rt.set_summarizer(Some(fake));
        talk(&mut h, 3).await;
        let before = h.rt.messages().await.len();
        assert_eq!(before, 6);

        let result = h.rt.compact(None).await.expect("compact");
        assert!(result["summary"].as_str().unwrap().contains("FAKE SUMMARY"), "{result}");
        assert!(calls.load(Ordering::SeqCst) >= 1);
        let entries = h.rt.entries().unwrap();
        let comp = entries.iter().find(|e| e["type"] == "compaction").expect("compaction entry");
        assert_eq!(comp["method"], "soft");
        assert_eq!(comp["firstKeptEntryId"], result["firstKeptEntryId"]);
        assert!(entries.iter().any(|e| e["id"] == comp["firstKeptEntryId"]), "kept entry exists");
        assert!(comp["tokensAfter"].as_u64().is_some());

        let after = h.rt.messages().await;
        assert!(after.len() < before + 1 && after[0].to_string().contains("FAKE SUMMARY"), "{after:?}");
        // a second manual compaction right away is a no-op
        let again = h.rt.compact(None).await.unwrap_err();
        assert!(again.message.contains("Already compacted") || again.message.contains("Nothing to compact"), "{}", again.message);
    }

    #[tokio::test]
    async fn auto_compaction_fires_after_a_turn_over_threshold_and_emits_events() {
        let mut h = harness(4).await;
        let (fake, _) = fake_summarizer("AUTO SUMMARY");
        h.rt.set_summarizer(Some(fake));
        h.rt.set_compaction_settings(CompactionSettings { enabled: false, ..small_keep() });
        talk(&mut h, 3).await;
        assert!(h.rt.entries().unwrap().iter().all(|e| e["type"] != "compaction"), "disabled: no compaction yet");

        h.rt.set_compaction_settings(CompactionSettings { threshold_tokens: Some(200.0), ..small_keep() });
        h.rt.prompt(Some(json!("p4")), &format!("question 4 {}", "x".repeat(400)), None, Vec::new()).await.expect("prompt");
        let frames = until_settled(&mut h.rx).await;
        let kinds: Vec<&str> = frames.iter().map(|f| f["type"].as_str().unwrap_or("?")).filter(|k| k.starts_with("auto_") || matches!(*k, "agent_end" | "prompt_result" | "session_settled")).collect();
        assert_eq!(kinds, ["agent_end", "auto_compaction_start", "auto_compaction_end", "prompt_result", "session_settled"], "{kinds:?}");
        let start = frames.iter().find(|f| f["type"] == "auto_compaction_start").unwrap();
        assert_eq!((start["reason"].clone(), start["action"].clone()), (json!("threshold"), json!("context-full")));
        let end = frames.iter().find(|f| f["type"] == "auto_compaction_end").unwrap();
        assert_eq!((end["aborted"].clone(), end["willRetry"].clone()), (json!(false), json!(false)));
        assert!(end["result"]["summary"].as_str().unwrap().contains("AUTO SUMMARY"), "{end}");
        assert!(h.rt.entries().unwrap().iter().any(|e| e["type"] == "compaction"));
        assert!(h.rt.messages().await[0].to_string().contains("AUTO SUMMARY"));
    }
}
