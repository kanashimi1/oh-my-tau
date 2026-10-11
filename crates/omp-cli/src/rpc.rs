//! `--mode rpc`: JSON lines on stdin/stdout, a subset of omp's RPC protocol
//! (commands: prompt, abort, get_state; frames: response + agent events).

use std::sync::Arc;

use anyhow::Result;
use pa_agent::agent::Agent;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::events;

fn response(id: &Option<Value>, command: &str, success: bool, extra: Value) -> Value {
    let mut v = json!({"type":"response","command":command,"success":success});
    if let Some(id) = id {
        v["id"] = id.clone();
    }
    if let (Some(obj), Some(more)) = (v.as_object_mut(), extra.as_object()) {
        for (k, val) in more {
            obj.insert(k.clone(), val.clone());
        }
    }
    v
}

async fn state(agent: &Agent, session_id: &str) -> Value {
    let s = agent.state().await;
    json!({
        "model": serde_json::to_value(&s.model).unwrap_or(Value::Null),
        "thinkingLevel": serde_json::to_value(s.thinking_level).unwrap_or(Value::Null),
        "isStreaming": s.is_streaming,
        "isCompacting": false,
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "interruptMode": "immediate",
        "sessionId": session_id,
        "autoCompactionEnabled": false,
        "messageCount": s.messages.len(),
        "queuedMessageCount": 0,
        "isSettled": !s.is_streaming,
        "systemPrompt": [s.system_prompt],
    })
}

/// Run the RPC loop until stdin closes.
///
/// # Errors
/// Fails on stdout write errors.
pub async fn run(agent: Agent) -> Result<()> {
    let agent = Arc::new(agent);
    let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
    let ev_tx = tx.clone();
    let _sub = agent
        .subscribe(move |event, _signal| {
            let _ = ev_tx.send(events::frame(&event));
            Box::pin(async { Ok(()) })
        })
        .await;

    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(v) = rx.recv().await {
            let mut line = v.to_string();
            line.push('\n');
            if out.write_all(line.as_bytes()).await.is_err() || out.flush().await.is_err() {
                break;
            }
        }
    });

    let session_id = format!("omprs-{}", pa_agent::now_ms());
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(cmd) = serde_json::from_str::<Value>(&line) else {
            let _ = tx.send(json!({"type":"response","command":"parse","success":false,"error":"invalid JSON"}));
            continue;
        };
        let id = cmd.get("id").cloned();
        let kind = cmd.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        let reply = match kind.as_str() {
            "prompt" => match cmd.get("message").and_then(Value::as_str) {
                None => response(&id, &kind, false, json!({"error":"missing message"})),
                Some(message) => match agent.prompt_until_accepted(message).await {
                    Ok(()) => response(&id, &kind, true, json!({"data":{"agentInvoked":true}})),
                    Err(e) => response(&id, &kind, false, json!({"error":format!("{e:#}")})),
                },
            },
            "abort" => {
                agent.abort();
                response(&id, &kind, true, json!({}))
            }
            "get_state" => {
                let data = state(&agent, &session_id).await;
                response(&id, &kind, true, json!({"data":data}))
            }
            other => response(
                &id,
                other,
                false,
                json!({"error":format!("unsupported command: {other}")}),
            ),
        };
        let _ = tx.send(reply);
    }
    // stdin closed: let an in-flight run finish and flush its events.
    agent.wait_for_idle().await;
    drop(tx);
    drop(_sub);
    // Subscription holds a sender clone inside the agent; abort the writer
    // after giving it a moment to drain.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    writer.abort();
    Ok(())
}
