//! `omt hub` — the a2a hub CLI (`SPEC.md` item 6): serve, send, inbox,
//! status, ack, answer, claim, release, claims, check, subscribe. JSON
//! output with `--json`; short human output otherwise.

use std::future::Future;
use std::path::PathBuf;

use anyhow::{Context, Result};
use omp_hub::{Envelope, HubClient, HubDir, Priority, State};
use serde_json::{json, Value};

/// Bodies larger than this never go inline (`SPEC.md`).
const INLINE_LIMIT: usize = 8 * 1024;

/// Dispatch `omt hub <command> [flags]`; returns the process exit code.
///
/// # Errors
/// Parse, connection, and hub-side failures (printed by `main`).
pub fn run(args: &[String]) -> Result<i32> {
    let (flags, rest) = parse_flags(args);
    let flags = std::sync::Arc::new(flags);
    let Some(cmd) = rest.first().map(String::as_str) else {
        usage();
        return Ok(2);
    };
    match cmd {
        "serve" => cmd_serve(&flags),
        "send" => block_on(cmd_send(flags)),
        "inbox" => block_on(cmd_inbox(flags)),
        "status" => block_on(cmd_status(flags)),
        "ack" => block_on(cmd_ack(flags)),
        "answer" => block_on(cmd_answer(flags)),
        "claim" => block_on(cmd_claim(flags)),
        "release" => block_on(cmd_release(flags)),
        "claims" => block_on(cmd_claims(flags)),
        "check" => block_on(cmd_check(flags)),
        "subscribe" => block_on(cmd_subscribe(flags)),
        "mcp" => super::hub_mcp::run(args),
        _ => {
            eprintln!("unknown hub command: {cmd}\n");
            usage();
            Ok(2)
        }
    }
}

fn usage() {
    eprintln!(
        "omt hub — a2a hub (SPEC.md)

Usage: omt hub <command> [flags]

Commands:
  serve [--dir DIR] [--sock PATH]     run the hub in the foreground; exits when
                                      unreachable (dir/socket gone) or idle with
                                      no client for OMP_HUB_IDLE_EXIT_SECS
                                      (default 900, 0 = never)
  send --from A --to B [flags]        send an envelope (idempotent on --id)
  inbox --recipient R                 list mail, urgent first
  status [--recipient R|--message-id M]  unread / reply-expected with age / claims
  ack --message-id M --state S        acknowledge delivered|read
  answer --causation-id M [flags]     answer a question (validated vs reply_schema)
  claim --resource R --owner O --ttl-s N [--intent I]
  release --resource R --owner O --fencing-token N
  claims [--prefix P]
  check --resource R
  subscribe --agent A [--last-seq N]  stream deliver pushes
  mcp [--agent NAME] [--dir DIR] [--sock PATH]
                                      stdio MCP server (protocol 2025-06-18)

Common flags: --json   machine-readable output
              --dir/--sock override OMP_HUB_DIR/OMP_HUB_SOCK for serve"
    );
}

/// Simple `--key value` / `--key=value` / boolean-flag parser.
pub(crate) fn parse_flags(
    args: &[String],
) -> (std::collections::HashMap<String, String>, Vec<String>) {
    let mut flags = std::collections::HashMap::new();
    let mut rest = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if let Some(raw) = a.strip_prefix("--") {
            if let Some((k, v)) = raw.split_once('=') {
                flags.insert(k.to_string(), v.to_string());
            } else if let Some(v) = it.peek().filter(|v| !v.starts_with("--")) {
                flags.insert(raw.to_string(), (*v).clone());
                it.next();
            } else {
                flags.insert(raw.to_string(), String::new());
            }
        } else {
            rest.push(a.clone());
        }
    }
    (flags, rest)
}

pub(crate) fn flag<'m>(
    flags: &'m std::sync::Arc<std::collections::HashMap<String, String>>,
    k: &str,
) -> Option<&'m str> {
    flags.get(k).map(String::as_str).filter(|v| !v.is_empty())
}

fn need_flag<'m>(
    flags: &'m std::sync::Arc<std::collections::HashMap<String, String>>,
    k: &str,
) -> Result<&'m str> {
    flag(flags, k).with_context(|| format!("missing --{k}"))
}

fn json_mode(flags: &std::sync::Arc<std::collections::HashMap<String, String>>) -> bool {
    flags.contains_key("json")
}

fn print_json(v: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
    );
}

fn hub_dir(flags: &std::sync::Arc<std::collections::HashMap<String, String>>) -> HubDir {
    match flag(flags, "dir") {
        Some(dir) => HubDir::explicit(PathBuf::from(dir), flag(flags, "sock").map(PathBuf::from)),
        None => HubDir::resolve(),
    }
}

/// Dedicated-thread current-thread runtime (the `subcmd/stats.rs` pattern;
/// the CLI's own runtime must not be `block_on`ed).
#[allow(clippy::disallowed_methods)] // entry point: dedicated runtime on its own thread
fn block_on<T>(fut: impl Future<Output = Result<T>> + Send + 'static) -> Result<T>
where
    T: Send + 'static,
{
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("create tokio runtime")
            .and_then(|rt| rt.block_on(fut))
    })
    .join()
    .unwrap_or_else(|e| anyhow::bail!("hub worker thread panicked: {e:?}"))
}

#[allow(clippy::disallowed_methods)] // entry point: dedicated runtime on its own thread
fn connect(
    flags: &std::sync::Arc<std::collections::HashMap<String, String>>,
) -> impl Future<Output = Result<HubClient>> + Send + 'static {
    let hub = hub_dir(flags);
    let auto_start = std::env::var_os("OMP_HUB_NO_AUTOSTART").is_none();
    async move { HubClient::connect_with(hub, auto_start).await }
}

#[allow(clippy::disallowed_methods)] // entry point: dedicated runtime on its own thread
fn cmd_serve(flags: &std::sync::Arc<std::collections::HashMap<String, String>>) -> Result<i32> {
    let hub = hub_dir(flags);
    eprintln!(
        "omt hub serve: dir={} sock={}",
        hub.dir.display(),
        hub.sock.display()
    );
    // `subcmd::run` executes on the CLI's own runtime thread, so serve()
    // must drive its fresh runtime on a dedicated thread (same reason as
    // `block_on` above); this thread parks until the server dies.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("create tokio runtime")
            .and_then(|rt| rt.block_on(omp_hub::serve(&hub)));
        let _ = tx.send(result);
    });
    match rx.recv() {
        Ok(result) => {
            let exit = result?;
            eprintln!("omt hub serve: exiting ({exit:?})");
            Ok(0)
        }
        Err(e) => anyhow::bail!("hub serve thread died: {e}"),
    }
}

/// Build the envelope from flags; bodies > 8 KiB go by ref (`SPEC.md`).
fn envelope_from_flags(
    flags: &std::sync::Arc<std::collections::HashMap<String, String>>,
    allow_payload_ref: bool,
) -> Result<Envelope> {
    let mut env = Envelope {
        message_id: flag(flags, "id").unwrap_or_default().to_string(),
        run_id: flag(flags, "run-id").map(ToOwned::to_owned),
        parent_id: flag(flags, "parent-id").map(ToOwned::to_owned),
        correlation_id: flag(flags, "correlation-id").map(ToOwned::to_owned),
        causation_id: flag(flags, "causation-id").map(ToOwned::to_owned),
        sender: need_flag(flags, "from")?.to_string(),
        // Optional on `answer`: the hub routes the reply to the original
        // sender; required for `send` (checked there).
        recipient: flag(flags, "to").unwrap_or_default().to_string(),
        kind: flag(flags, "kind")
            .unwrap_or(omp_hub::Kind::AGENT_MESSAGE)
            .to_string(),
        priority: Priority(
            flag(flags, "priority")
                .and_then(|p| p.parse().ok())
                .unwrap_or(1),
        ),
        expected: flag(flags, "expected").map(ToOwned::to_owned),
        reply_schema: reply_schema_from_flags(flags)?,
        deadline: flag(flags, "deadline-ms").and_then(|d| d.parse().ok()),
        ..Envelope::default()
    };
    let payload = match flag(flags, "payload") {
        Some(p) => {
            if p.len() > INLINE_LIMIT {
                anyhow::bail!("inline --payload exceeds {INLINE_LIMIT} bytes; use --payload-file");
            }
            Some(p.to_string())
        }
        None => match flag(flags, "payload-file") {
            Some(path) => {
                let path = PathBuf::from(path);
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("read {}", path.display()))?;
                if text.len() > INLINE_LIMIT && allow_payload_ref {
                    env.payload_ref = Some(omp_hub::PayloadRef {
                        path: std::fs::canonicalize(&path)
                            .unwrap_or(path)
                            .display()
                            .to_string(),
                        sha256: sha256_hex(text.as_bytes()),
                        bytes: text.len() as u64,
                    });
                    None
                } else {
                    Some(text)
                }
            }
            None => None,
        },
    };
    env.payload = payload;
    Ok(env)
}

fn reply_schema_from_flags(
    flags: &std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<Option<Value>> {
    let Some(raw) = flag(flags, "reply-schema") else {
        return Ok(None);
    };
    let text = match std::fs::read_to_string(raw) {
        Ok(t) => t,
        Err(_) => raw.to_string(),
    };
    let v: Value = serde_json::from_str(&text).context("--reply-schema: invalid JSON")?;
    Ok(Some(v))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

async fn cmd_send(flags: std::sync::Arc<std::collections::HashMap<String, String>>) -> Result<i32> {
    let env = envelope_from_flags(&flags, true)?;
    let c = connect(&flags).await?;
    let receipt = c.send(env).await?;
    if json_mode(&flags) {
        print_json(&json!(receipt));
    } else {
        let line = format!(
            "sent {} seq={} state={}{}{}",
            receipt.message_id,
            receipt.seq,
            receipt.state,
            receipt
                .error
                .as_ref()
                .map_or(String::new(), |e| format!(" error={e}")),
            if receipt.dedup { " (dedup)" } else { "" },
        );
        println!("{line}");
    }
    Ok(i32::from(receipt.state == State::FAILED))
}

async fn cmd_inbox(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let recipient = need_flag(&flags, "recipient")?.to_string();
    let since = flag(&flags, "since-seq").and_then(|v| v.parse().ok());
    let limit = flag(&flags, "limit").and_then(|v| v.parse().ok());
    let c = connect(&flags).await?;
    let envelopes = c.inbox(&recipient, since, limit).await?;
    if json_mode(&flags) {
        print_json(&json!({ "recipient": recipient, "envelopes": envelopes }));
    } else {
        if envelopes.is_empty() {
            println!("(inbox {recipient}: empty)");
        }
        for env in &envelopes {
            println!(
                "#{} [{}] {} -> {} state={} kind={}{}",
                env.seq.unwrap_or(0),
                env.priority.0,
                env.sender,
                env.recipient,
                env.state.as_deref().unwrap_or("?"),
                env.kind,
                env.expected
                    .as_deref()
                    .filter(|e| !e.is_empty() && *e != "none")
                    .map_or(String::new(), |e| format!(" expected={e}")),
            );
            if let Some(p) = &env.payload {
                let preview: String = p.chars().take(120).collect();
                println!("  {preview}");
            }
            if let Some(r) = &env.payload_ref {
                println!("  path={} sha256={} bytes={}", r.path, r.sha256, r.bytes);
            }
        }
    }
    Ok(0)
}

async fn cmd_status(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let c = connect(&flags).await?;
    if let Some(id) = flag(&flags, "message-id") {
        let v = c.status_message(id).await?;
        print_or(&flags, &v, print_message_human);
        return Ok(0);
    }
    let recipient = flag(&flags, "recipient").map(ToOwned::to_owned);
    let v = c.status(recipient.as_deref()).await?;
    if json_mode(&flags) {
        print_json(&v);
    } else {
        println!("unread by recipient:");
        for pair in v["unread_by_recipient"].as_array().into_iter().flatten() {
            let p = pair.as_array().cloned().unwrap_or_default();
            if p.len() == 2 {
                println!("  {}: {}", p[0], p[1]);
            }
        }
        if let Some(rec) = &recipient {
            println!("unread for {rec}:");
            for e in v["unread"].as_array().into_iter().flatten() {
                print_message_human(e);
            }
        }
        println!("pending replies (reply-expected, age):");
        for p in v["pending_replies"].as_array().into_iter().flatten() {
            println!(
                "  {} -> {} kind={} age_ms={} deadline_ms={}",
                p["message_id"].as_str().unwrap_or("?"),
                p["recipient"].as_str().unwrap_or("?"),
                p["kind"].as_str().unwrap_or("?"),
                p["age_ms"].as_u64().unwrap_or(0),
                p["deadline_ms"]
                    .as_u64()
                    .map_or_else(|| "none".into(), |d| d.to_string()),
            );
        }
        println!("active claims:");
        for cl in v["active_claims"].as_array().into_iter().flatten() {
            println!(
                "  {} owner={} token={} expires_ms={}",
                cl["resource"].as_str().unwrap_or("?"),
                cl["owner"].as_str().unwrap_or("?"),
                cl["fencing_token"].as_u64().unwrap_or(0),
                cl["expires_ms"].as_u64().unwrap_or(0),
            );
        }
    }
    Ok(0)
}

fn print_message_human(v: &Value) {
    println!(
        "  {} seq={} state={} {} -> {} kind={}",
        v["message_id"].as_str().unwrap_or("?"),
        v["seq"].as_u64().unwrap_or(0),
        v["state"].as_str().unwrap_or("?"),
        v["sender"].as_str().unwrap_or("?"),
        v["recipient"].as_str().unwrap_or("?"),
        v["kind"].as_str().unwrap_or("?"),
    );
}

fn print_or(
    flags: &std::sync::Arc<std::collections::HashMap<String, String>>,
    v: &Value,
    human: impl Fn(&Value),
) {
    if json_mode(flags) {
        print_json(v);
    } else {
        human(v);
    }
}

async fn cmd_ack(flags: std::sync::Arc<std::collections::HashMap<String, String>>) -> Result<i32> {
    let id = need_flag(&flags, "message-id")?.to_string();
    let state = need_flag(&flags, "state")?.to_string();
    let c = connect(&flags).await?;
    c.ack(&id, &state).await?;
    if json_mode(&flags) {
        print_json(&json!({"message_id": id, "state": state, "ok": true}));
    } else {
        println!("ack {id} -> {state}");
    }
    Ok(0)
}

async fn cmd_answer(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let causation_id = need_flag(&flags, "causation-id")?.to_string();
    let mut env = envelope_from_flags(&flags, false)?;
    env.causation_id = Some(causation_id.clone());
    env.kind = flag(&flags, "kind")
        .unwrap_or(omp_hub::KIND_ANSWER)
        .to_string();
    let c = connect(&flags).await?;
    let v = c.answer(&causation_id, env).await?;
    if json_mode(&flags) {
        print_json(&v);
    } else {
        println!(
            "answered {} (question state={})",
            v["answer"]["message_id"].as_str().unwrap_or("?"),
            v["question"]["state"].as_str().unwrap_or("?"),
        );
    }
    Ok(0)
}

async fn cmd_claim(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let resource = need_flag(&flags, "resource")?.to_string();
    let owner = need_flag(&flags, "owner")?.to_string();
    let ttl = flag(&flags, "ttl-s")
        .and_then(|v| v.parse().ok())
        .context("missing --ttl-s")?;
    let intent = flag(&flags, "intent").map(ToOwned::to_owned);
    let c = connect(&flags).await?;
    let outcome = c.claim(&resource, &owner, ttl, intent.as_deref()).await?;
    if json_mode(&flags) {
        print_json(&serde_json::to_value(&outcome)?);
    } else if outcome.granted {
        println!(
            "claimed {resource} token={} expires_ms={}",
            outcome.fencing_token.unwrap_or(0),
            outcome.expires_ms.unwrap_or(0),
        );
    } else {
        println!(
            "conflict: {resource} held by {} until expires_ms={}",
            outcome.holder.as_deref().unwrap_or("?"),
            outcome.expires_ms.unwrap_or(0),
        );
    }
    Ok(i32::from(!outcome.granted))
}

async fn cmd_release(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let resource = need_flag(&flags, "resource")?.to_string();
    let owner = need_flag(&flags, "owner")?.to_string();
    let token = flag(&flags, "fencing-token")
        .and_then(|v| v.parse().ok())
        .context("missing --fencing-token")?;
    let c = connect(&flags).await?;
    c.release(&resource, &owner, token).await?;
    if json_mode(&flags) {
        print_json(&json!({"resource": resource, "released": true}));
    } else {
        println!("released {resource}");
    }
    Ok(0)
}

async fn cmd_claims(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let prefix = flag(&flags, "prefix").map(ToOwned::to_owned);
    let c = connect(&flags).await?;
    let v = c.claims(prefix.as_deref(), None).await?;
    if json_mode(&flags) {
        print_json(&v);
    } else {
        for cl in v["claims"].as_array().into_iter().flatten() {
            println!(
                "{} owner={} token={} expires_ms={}",
                cl["resource"].as_str().unwrap_or("?"),
                cl["owner"].as_str().unwrap_or("?"),
                cl["fencing_token"].as_u64().unwrap_or(0),
                cl["expires_ms"].as_u64().unwrap_or(0),
            );
        }
    }
    Ok(0)
}

async fn cmd_check(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let resource = need_flag(&flags, "resource")?.to_string();
    let c = connect(&flags).await?;
    let v = c.check(&resource).await?;
    if json_mode(&flags) {
        print_json(&v);
    } else if v["held"].as_bool().unwrap_or(false) {
        let h = &v["holder"];
        println!(
            "{resource} held by {} token={} expires_ms={}",
            h["owner"].as_str().unwrap_or("?"),
            h["fencing_token"].as_u64().unwrap_or(0),
            h["expires_ms"].as_u64().unwrap_or(0),
        );
    } else {
        println!("{resource} free");
    }
    Ok(0)
}

async fn cmd_subscribe(
    flags: std::sync::Arc<std::collections::HashMap<String, String>>,
) -> Result<i32> {
    let agent = need_flag(&flags, "agent")?.to_string();
    let last_seq = flag(&flags, "last-seq").and_then(|v| v.parse().ok());
    let c = connect(&flags).await?;
    let mut c = c;
    let mut stream = c.take_deliver_stream().context("no deliver stream")?;
    let hello = c.hello(&agent, last_seq.unwrap_or(0)).await?;
    if json_mode(&flags) {
        print_json(&json!({"hello": hello}));
    } else {
        println!(
            "subscribed agent={} last_seq={} replayed={} (Ctrl-C to exit)",
            hello["agent"].as_str().unwrap_or("?"),
            hello["last_seq"].as_u64().unwrap_or(0),
            hello["replayed"].as_u64().unwrap_or(0),
        );
    }
    while let Some(env) = stream.recv().await {
        if json_mode(&flags) {
            println!("{}", serde_json::to_string(&env)?);
        } else {
            println!(
                "[deliver] {} <- {} kind={} seq={} state={}",
                env.recipient,
                env.sender,
                env.kind,
                env.seq.unwrap_or(0),
                env.state.as_deref().unwrap_or("?"),
            );
            if let Some(p) = &env.payload {
                let preview: String = p.chars().take(200).collect();
                println!("  {preview}");
            }
        }
    }
    Ok(0)
}
