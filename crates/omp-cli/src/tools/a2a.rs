//! Agent-to-agent bus between a parent session and its `task` children.
//!
//! The parent owns a unix-socket server for as long as at least one
//! [`BusLease`] is alive (a `task` call holds one). Every child it starts
//! gets [`SOCK_ENV`] (the socket) and [`SELF_ENV`] (its own agent id). A child's
//! `write agent://<id>` that does not name one of its own local handles is sent
//! over the socket and routed by the parent:
//!
//! - `agent://parent` -> the parent's inbox ([`take_parent_messages`]); the
//!   `task` tool drains it into its result, so a message sent while the parent
//!   is blocked in `task` is never lost and lands in the parent's next turn.
//! - `agent://<sibling-id>` / `agent://all` -> a steer on the sibling run(s)
//!   registered in the shared agent registry, delivered at the sibling's next
//!   turn boundary, prefixed with the sender's id.
//!
//! Wire format: one JSON request line `{"from","to","message"}`, one JSON
//! reply line `{"ok":true,"text"}` / `{"ok":false,"error"}`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::task::JoinHandle;

pub(crate) use omp_task::runner::{A2A_SELF_ENV as SELF_ENV, A2A_SOCK_ENV as SOCK_ENV};

/// Bound for one request/reply exchange (either side).
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest accepted request line.
const MAX_LINE: usize = 1 << 20;

/// A message a child addressed to `agent://parent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParentMessage {
    pub from: String,
    pub message: String,
}

static PARENT_INBOX: Mutex<Vec<ParentMessage>> = Mutex::new(Vec::new());

fn push_parent_message(from: &str, message: &str) {
    PARENT_INBOX
        .lock()
        .expect("a2a inbox lock")
        .push(ParentMessage {
            from: from.to_string(),
            message: message.to_string(),
        });
}

/// Drain the messages children sent to the parent since the last call.
pub(crate) fn take_parent_messages() -> Vec<ParentMessage> {
    std::mem::take(&mut *PARENT_INBOX.lock().expect("a2a inbox lock"))
}

/// Render drained parent messages as the block appended to a `task` result.
pub(crate) fn render_parent_messages(messages: &[ParentMessage]) -> String {
    let mut out = String::from("<agent-messages>");
    for m in messages {
        out.push_str(&format!("\nfrom agent://{}: {}", m.from, m.message));
    }
    out.push_str("\n</agent-messages>");
    out
}

struct BusState {
    leases: usize,
    sock: Option<PathBuf>,
    dir: Option<PathBuf>,
    server: Option<JoinHandle<()>>,
}

static BUS: Mutex<BusState> = Mutex::new(BusState {
    leases: 0,
    sock: None,
    dir: None,
    server: None,
});

/// Keeps the bus server alive; the last drop stops it and removes the socket.
pub(crate) struct BusLease {
    sock: PathBuf,
}

impl BusLease {
    /// Socket path to hand to children as [`SOCK_ENV`].
    pub(crate) fn sock(&self) -> &Path {
        &self.sock
    }

    /// Start (or join) this process's bus. Must run inside a tokio runtime.
    pub(crate) fn acquire() -> std::io::Result<Self> {
        let mut bus = BUS.lock().expect("a2a bus lock");
        if bus.leases == 0 {
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "omt-a2a-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            {
                use std::os::unix::fs::DirBuilderExt;
                std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
            }
            let sock = dir.join("bus.sock");
            let listener = match UnixListener::bind(&sock) {
                Ok(listener) => listener,
                Err(e) => {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err(e);
                }
            };
            bus.server = Some(tokio::spawn(serve(listener)));
            bus.sock = Some(sock);
            bus.dir = Some(dir);
        }
        bus.leases += 1;
        let sock = bus.sock.clone().expect("bus socket set while leased");
        Ok(Self { sock })
    }
}

impl Drop for BusLease {
    fn drop(&mut self) {
        let mut bus = BUS.lock().expect("a2a bus lock");
        bus.leases = bus.leases.saturating_sub(1);
        if bus.leases == 0 {
            if let Some(server) = bus.server.take() {
                server.abort();
            }
            bus.sock = None;
            if let Some(dir) = bus.dir.take() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }
}

async fn serve(listener: UnixListener) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        tokio::spawn(async move {
            let _ = tokio::time::timeout(IO_TIMEOUT, handle_conn(stream)).await;
        });
    }
}

async fn handle_conn(stream: UnixStream) {
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    let mut reader = BufReader::new(read).take(MAX_LINE as u64);
    let reply = match reader.read_line(&mut line).await {
        Ok(n) if n > 0 => match serde_json::from_str::<Value>(&line) {
            Ok(req) => route(&req).await,
            Err(e) => Err(format!("bad a2a request: {e}")),
        },
        _ => Err("empty a2a request".to_string()),
    };
    let frame = match reply {
        Ok(text) => json!({"ok": true, "text": text}),
        Err(error) => json!({"ok": false, "error": error}),
    };
    let mut out = frame.to_string();
    out.push('\n');
    let _ = write.write_all(out.as_bytes()).await;
    let _ = write.shutdown().await;
}

/// Route one child request inside the parent process.
async fn route(req: &Value) -> Result<String, String> {
    let field = |name: &str| req.get(name).and_then(Value::as_str).unwrap_or_default();
    let (from, to, message) = (field("from"), field("to"), field("message"));
    if from.is_empty() || to.is_empty() || message.trim().is_empty() {
        return Err("a2a request needs from, to and a non-empty message".to_string());
    }
    if to == "parent" {
        push_parent_message(from, message);
        return Ok("Delivered to parent.".to_string());
    }
    if to == from {
        return Err("an agent cannot message itself".to_string());
    }
    let text = format!("[message from agent://{from}]\n{message}");
    super::eval_agents::deliver_from_peer(to, &text, from).await
}

/// Child-side handle on the parent's bus (from the inherited environment).
pub(crate) struct Client {
    sock: PathBuf,
    from: String,
}

impl Client {
    /// `Some` iff this process was started by a parent that runs a bus.
    pub(crate) fn from_env() -> Option<Self> {
        let sock = std::env::var_os(SOCK_ENV).filter(|v| !v.is_empty())?;
        let from = std::env::var(SELF_ENV).ok().filter(|v| !v.is_empty())?;
        Some(Self {
            sock: PathBuf::from(sock),
            from,
        })
    }

    /// A client for an explicit socket/sender (tests drive the bus directly).
    #[cfg(test)]
    pub(crate) fn for_test(sock: &Path, from: &str) -> Self {
        Self {
            sock: sock.to_path_buf(),
            from: from.to_string(),
        }
    }

    /// Send `message` to `to` (`parent`, a sibling id, or `all`) via the parent.
    pub(crate) async fn send(&self, to: &str, message: &str) -> Result<String, String> {
        tokio::time::timeout(IO_TIMEOUT, self.exchange(to, message))
            .await
            .map_err(|_| "parent bus timed out".to_string())?
    }

    async fn exchange(&self, to: &str, message: &str) -> Result<String, String> {
        let stream = UnixStream::connect(&self.sock)
            .await
            .map_err(|e| format!("parent bus unreachable ({e}); the parent may have finished"))?;
        let (read, mut write) = stream.into_split();
        let mut req = json!({"from": self.from, "to": to, "message": message}).to_string();
        req.push('\n');
        write
            .write_all(req.as_bytes())
            .await
            .map_err(|e| format!("parent bus write failed: {e}"))?;
        let mut line = String::new();
        BufReader::new(read)
            .read_line(&mut line)
            .await
            .map_err(|e| format!("parent bus read failed: {e}"))?;
        let reply: Value =
            serde_json::from_str(&line).map_err(|e| format!("bad parent bus reply: {e}"))?;
        if reply.get("ok") == Some(&Value::Bool(true)) {
            Ok(reply
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("Delivered.")
                .to_string())
        } else {
            Err(reply
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("delivery failed")
                .to_string())
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The bus is process-global: serialize the tests that lease it.
    pub(crate) static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn parent_messages_queue_and_drain_once() {
        let _serial = LOCK.lock().await;
        let lease = BusLease::acquire().unwrap();
        let client = Client {
            sock: lease.sock().to_path_buf(),
            from: "kid-a2a-test".into(),
        };
        let reply = client.send("parent", "need the schema").await.unwrap();
        assert_eq!(reply, "Delivered to parent.");
        let drained: Vec<_> = take_parent_messages()
            .into_iter()
            .filter(|m| m.from == "kid-a2a-test")
            .collect();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].message, "need the schema");
        assert!(take_parent_messages()
            .iter()
            .all(|m| m.from != "kid-a2a-test"));
        assert!(render_parent_messages(&drained).contains("from agent://kid-a2a-test: need"));
    }

    #[tokio::test]
    async fn unknown_peer_and_self_are_refused() {
        let _serial = LOCK.lock().await;
        let lease = BusLease::acquire().unwrap();
        let client = Client {
            sock: lease.sock().to_path_buf(),
            from: "kid-x".into(),
        };
        let err = client.send("no-such-peer", "hi").await.unwrap_err();
        assert!(err.contains("no-such-peer"), "{err}");
        let err = client.send("kid-x", "hi").await.unwrap_err();
        assert!(err.contains("itself"), "{err}");
    }

    #[tokio::test]
    async fn socket_is_removed_after_last_lease() {
        let _serial = LOCK.lock().await;
        let lease = BusLease::acquire().unwrap();
        let second = BusLease::acquire().unwrap();
        let sock = lease.sock().to_path_buf();
        assert!(sock.exists());
        drop(lease);
        assert!(sock.exists(), "still leased");
        drop(second);
        assert!(!sock.exists());
    }
}
