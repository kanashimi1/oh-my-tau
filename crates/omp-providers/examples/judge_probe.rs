//! `cargo run -p omp-providers --example judge_probe -- ["<state text>"]`
//! One live System One judgment through the resolved `judge` chain (keys are never printed).
use omp_providers::{AgentDir, JudgeRequest};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The build failed with a linker error.".into());
    let chain = omp_providers::resolve_judge_chain(&AgentDir::default())?;
    for c in &chain {
        eprintln!(
            "[chain] {:?} {}/{} url={:?}",
            c.kind, c.model.provider, c.model.id, c.url
        );
    }
    let questions = json!({
        "kind": {"type": "choice", "instructions": "What kind of message is this?",
                 "criteria": {"error": "reports a failure", "success": "reports success", "other": null}},
        "urgent": {"type": "noul", "instructions": "Does this need immediate attention?"}
    });
    let request = JudgeRequest {
        state: json!(state),
        questions: questions.as_object().cloned().unwrap_or_default(),
    };
    // Opt-in: the default cache is omp's own database.
    let cache = std::env::var_os("JUDGE_PROBE_CACHE").and_then(|_| {
        omp_providers::JudgmentCache::open(
            &omp_providers::judgment_cache_path(&AgentDir::default()),
        )
        .ok()
    });
    let r = omp_providers::judge(&chain, &request, cache.as_ref()).await?;
    println!("api={} provider={} model={}", r.api, r.provider, r.model);
    println!("answers={}", serde_json::to_string(&r.answers)?);
    println!("usage={:?}", r.usage);
    Ok(())
}
