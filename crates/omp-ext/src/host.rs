use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot};

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// How to start the host process.
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// `bun` executable.
    pub bun: String,
    /// Path to `ext-host/host.ts`.
    pub host_script: PathBuf,
    /// Append one JSON line per emitted event (per-handler action/ms) to this file.
    pub trace: Option<PathBuf>,
    /// Handler budget in ms (host default 30000).
    pub handler_timeout_ms: Option<u64>,
    /// Extra environment for the host. `HOOK_TRACE=0` is set by default so the user's own
    /// `lib/hook-trace.ts` wrapper does not write into the real `~/.omp/state/hook_trace.jsonl`.
    pub env: Vec<(String, String)>,
}

impl HostConfig {
    #[must_use]
    pub fn new(host_script: impl Into<PathBuf>) -> Self {
        Self {
            bun: "bun".into(),
            host_script: host_script.into(),
            trace: None,
            handler_timeout_ms: None,
            env: Vec::new(),
        }
    }
}

/// Message initiated by the host (not a response).
#[derive(Debug, Clone)]
pub enum HostMessage {
    /// Fire-and-forget: `append_entry`, `send_message`, `send_user_message`, `log`, `ui_notify`, ...
    Notification { method: String, params: Value },
    /// Needs an answer via [`ExtHost::respond`]: `compact`, `navigate_tree`, `ui_confirm`, ...
    Request {
        id: Value,
        method: String,
        params: Value,
    },
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadedExtension {
    pub path: String,
    pub name: String,
    #[serde(default)]
    pub handlers: HashMap<String, u64>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadResult {
    pub loaded: Vec<LoadedExtension>,
    pub errors: Vec<LoadError>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadError {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct HandlerRecord {
    pub ext: String,
    pub event: String,
    #[serde(default)]
    pub tool: Option<String>,
    pub ms: f64,
    pub action: String,
    #[serde(default)]
    pub error: Option<String>,
}

/// Aggregated outcome of one `emit`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EmitResult {
    /// Event-specific aggregate (see `ext-host/host.ts`); `Null` when no handler produced anything.
    #[serde(default)]
    pub result: Value,
    #[serde(default)]
    pub handlers: Vec<HandlerRecord>,
    /// Handlers that threw or timed out; the host keeps running (omp reports these per handler).
    #[serde(default)]
    pub errors: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    pub extension: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CommandInfo {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub extension: String,
}

/// Connection to a running extension host.
pub struct ExtHost {
    stdin: tokio::sync::Mutex<ChildStdin>,
    child: tokio::sync::Mutex<Child>,
    pending: Pending,
    next_id: AtomicU64,
    incoming: Mutex<Option<mpsc::UnboundedReceiver<HostMessage>>>,
}

impl ExtHost {
    /// Spawn `bun <host_script>` and start the reader tasks.
    ///
    /// # Errors
    /// Fails when the process cannot be spawned.
    pub fn spawn(cfg: &HostConfig) -> Result<Arc<Self>> {
        let mut cmd = Command::new(&cfg.bun);
        cmd.arg(&cfg.host_script)
            .env("HOOK_TRACE", "0")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(t) = &cfg.trace {
            cmd.env("OMP_EXT_TRACE", t);
        }
        if let Some(ms) = cfg.handler_timeout_ms {
            cmd.env("OMP_EXT_TIMEOUT_MS", ms.to_string());
        }
        cmd.envs(cfg.env.iter().map(|(k, v)| (k, v)));
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {} {}", cfg.bun, cfg.host_script.display()))?;
        let stdin = child.stdin.take().context("host stdin")?;
        let stdout = child.stdout.take().context("host stdout")?;
        let stderr = child.stderr.take().context("host stderr")?;

        let pending: Pending = Arc::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let host = Arc::new(Self {
            stdin: tokio::sync::Mutex::new(stdin),
            child: tokio::sync::Mutex::new(child),
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            incoming: Mutex::new(Some(rx)),
        });

        let weak = Arc::downgrade(&host);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                route(&msg, &pending, &tx, weak.upgrade());
            }
            // Host gone: fail everything still waiting.
            for (_, w) in pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain()
            {
                let _ = w.send(Err("extension host exited".into()));
            }
        });
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[ext-host] {line}");
            }
        });
        Ok(host)
    }

    /// Receiver of host-initiated messages; available once. Without a taker, host requests are
    /// answered with a "not supported" error so the extension's promise settles.
    pub fn take_incoming(&self) -> Option<mpsc::UnboundedReceiver<HostMessage>> {
        self.incoming
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Answer a [`HostMessage::Request`].
    ///
    /// # Errors
    /// Fails when the host's stdin is closed.
    pub async fn respond(&self, id: Value, result: Result<Value, String>) -> Result<()> {
        let msg = match result {
            Ok(v) => json!({"jsonrpc":"2.0","id":id,"result":v}),
            Err(m) => json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":m}}),
        };
        self.write(&msg).await
    }

    async fn write(&self, msg: &Value) -> Result<()> {
        let mut line = serde_json::to_vec(msg)?;
        line.push(b'\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(&line).await?;
        stdin.flush().await?;
        Ok(())
    }

    /// Raw JSON-RPC request.
    ///
    /// # Errors
    /// Transport failure, host exit, or a JSON-RPC error response.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, tx);
        if let Err(e) = self
            .write(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .await
        {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            return Err(e);
        }
        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(m)) => Err(anyhow!("{method}: {m}")),
            Err(_) => bail!("{method}: extension host dropped the request"),
        }
    }

    /// Load extensions: explicit `paths`, or every extension in `dir` (default
    /// `~/.omp/agent/extensions`) when `paths` is `None`.
    ///
    /// # Errors
    /// Transport failure. Per-extension failures are returned in [`LoadResult::errors`].
    pub async fn load(
        &self,
        paths: Option<&[PathBuf]>,
        dir: Option<&str>,
        cwd: &str,
        ctx: Option<Value>,
    ) -> Result<LoadResult> {
        let mut p = json!({ "cwd": cwd });
        if let Some(paths) = paths {
            p["paths"] = json!(paths);
        }
        if let Some(dir) = dir {
            p["dir"] = json!(dir);
        }
        if let Some(ctx) = ctx {
            p["ctx"] = ctx;
        }
        Ok(serde_json::from_value(self.request("load", p).await?)?)
    }

    /// Emit an omp event (`{"type": ..., ...}`); `ctx` merges into the host's session state
    /// (`sessionId`, `sessionFile`, `branch`, `leafId`, `model`, `contextUsage`, `hasUI`, ...).
    ///
    /// # Errors
    /// Transport failure. Handler failures are reported in [`EmitResult::handlers`]/`errors`.
    pub async fn emit(&self, event: Value, ctx: Option<Value>) -> Result<EmitResult> {
        let mut p = json!({ "event": event });
        if let Some(ctx) = ctx {
            p["ctx"] = ctx;
        }
        Ok(serde_json::from_value(self.request("emit", p).await?)?)
    }

    /// # Errors
    /// Transport failure.
    pub async fn list_tools(&self) -> Result<Vec<ToolInfo>> {
        Ok(serde_json::from_value(
            self.request("list_tools", json!({})).await?,
        )?)
    }

    /// Execute an extension tool; returns its `{content, details, isError?}` result.
    ///
    /// # Errors
    /// Unknown tool, tool failure, or transport failure.
    pub async fn call_tool(
        &self,
        name: &str,
        tool_call_id: &str,
        params: Value,
        ctx: Option<Value>,
    ) -> Result<Value> {
        self.request(
            "call_tool",
            json!({"name": name, "toolCallId": tool_call_id, "params": params, "ctx": ctx}),
        )
        .await
    }

    /// # Errors
    /// Transport failure.
    pub async fn list_commands(&self) -> Result<Vec<CommandInfo>> {
        Ok(serde_json::from_value(
            self.request("list_commands", json!({})).await?,
        )?)
    }

    /// # Errors
    /// Unknown command, command failure, or transport failure.
    pub async fn call_command(&self, name: &str, args: &str) -> Result<Value> {
        self.request("call_command", json!({"name": name, "args": args}))
            .await
    }

    /// Merge fields into the host's session state without emitting an event.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn set_ctx(&self, ctx: Value) -> Result<()> {
        self.request("set_ctx", json!({ "ctx": ctx })).await?;
        Ok(())
    }

    /// Graceful exit: ask the host to exit, wait up to 2 s, then kill.
    ///
    /// # Errors
    /// Only on kill failure.
    pub async fn shutdown(&self) -> Result<()> {
        let _ =
            tokio::time::timeout(Duration::from_secs(2), self.request("shutdown", json!({}))).await;
        let mut child = self.child.lock().await;
        if tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .is_err()
        {
            child.kill().await?;
        }
        Ok(())
    }
}

fn route(
    msg: &Value,
    pending: &Pending,
    tx: &mpsc::UnboundedSender<HostMessage>,
    host: Option<Arc<ExtHost>>,
) {
    let method = msg.get("method").and_then(Value::as_str);
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    match (method, msg.get("id")) {
        (None, Some(id)) => {
            let Some(id) = id.as_u64() else { return };
            let Some(w) = pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id)
            else {
                return;
            };
            let _ = w.send(match msg.get("error") {
                Some(e) => Err(e
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
                    .to_owned()),
                None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
            });
        }
        (Some(m), None) => {
            let _ = tx.send(HostMessage::Notification {
                method: m.to_owned(),
                params,
            });
        }
        (Some(m), Some(id)) => {
            let req = HostMessage::Request {
                id: id.clone(),
                method: m.to_owned(),
                params,
            };
            let untaken = host.as_ref().is_some_and(|h| {
                h.incoming
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .is_some()
            });
            if untaken || tx.send(req).is_err() {
                // Answer on its own task: the reply needs the stdin lock, and
                // awaiting it here would stall the stdout reader (a large write
                // holding stdin while the host's stdout pipe is full wedges both).
                if let Some(host) = host {
                    let m = m.to_owned();
                    let id = id.clone();
                    tokio::spawn(async move {
                        let _ = host
                            .respond(id, Err(format!("{m} is not supported by the core")))
                            .await;
                    });
                }
            }
        }
        (None, None) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{serve, HeadlessResponder};

    fn pid_alive(pid: u32) -> bool {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }

    fn fake_host(dir: &std::path::Path, body: &str) -> HostConfig {
        let script = dir.join("fake-host.sh");
        std::fs::write(&script, body).unwrap();
        HostConfig {
            bun: "/bin/sh".into(),
            host_script: script,
            trace: None,
            handler_timeout_ms: None,
            env: Vec::new(),
        }
    }

    /// Once `serve()` is running, the loop must not keep the host alive:
    /// dropping the last `Arc<ExtHost>` handle kills the bun child
    /// (`kill_on_drop`). Previously `serve` held a strong reference and the
    /// process outlived every handle until the explicit close call.
    #[tokio::test]
    async fn host_process_exits_when_last_handle_drops() {
        let dir = tempfile::tempdir().unwrap();
        // `sleep 60` keeps the child alive unless it is killed on drop.
        let cfg = fake_host(dir.path(), "sleep 60\n");
        let host = ExtHost::spawn(&cfg).unwrap();
        let pid = host.child.lock().await.id().expect("child pid");
        assert!(pid_alive(pid));
        let incoming = host.take_incoming().expect("incoming is available once");
        let _notes = serve(&host, incoming, Arc::new(HeadlessResponder));
        drop(host);
        // Tokio reaps the killed child asynchronously; allow generous time,
        // but far less than the 60 s the fake host would otherwise sleep.
        let mut exited = false;
        for _ in 0..200 {
            if !pid_alive(pid) {
                exited = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(exited, "bun host (pid {pid}) outlived its last handle");
    }

    /// An unclaimed host request (no `take_incoming` taker) is answered on its
    /// own task, not inside the stdout reader loop: the "not supported" error
    /// still reaches the child's stdin. The fake host records the reply by
    /// touching a marker file.
    #[tokio::test]
    async fn unclaimed_request_is_answered_off_the_reader_loop() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("replied");
        let script = dir.path().join("fake-host.sh");
        std::fs::write(
            &script,
            concat!(
                "printf '%s\n' '{\"jsonrpc\":\"2.0\",\"id\":101,\"method\":\"compact\",\"params\":{}}'\n",
                "while IFS= read -r line; do\n",
                "  case \"$line\" in\n",
                "    *'not supported'*) touch \"$MARKER\"; break;;\n",
                "  esac\n",
                "done\n",
                "sleep 60\n",
            ),
        )
        .unwrap();
        let mut cfg = HostConfig {
            bun: "/bin/sh".into(),
            host_script: script,
            trace: None,
            handler_timeout_ms: None,
            env: vec![("MARKER".into(), marker.display().to_string())],
        };
        cfg.handler_timeout_ms = None;
        let host = ExtHost::spawn(&cfg).unwrap();
        // Deliberately no `take_incoming`: the request is unclaimed.
        let answered = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if marker.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await;
        drop(host);
        assert!(answered.is_ok(), "unclaimed request was never answered");
    }
}
