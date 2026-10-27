//! Dictation for the web composer: `POST /api/transcribe` (raw audio body)
//! → local speech-to-text by a persistent faster-whisper worker
//! (`stt_worker.py`, embedded and written to `~/.omp-rs/stt/`).
//!
//! The worker keeps the model loaded between requests and exits on its own
//! after `OMT_STT_IDLE` seconds idle; the next request respawns it. Requests
//! are serialized (one model, one GPU). `OMT_STT_PYTHON` (default `python3`)
//! and `OMT_STT_MODEL` (default `large-v3-turbo`) pick the runtime.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

/// Largest accepted recording (≈ 25 min of opus voice).
pub const MAX_AUDIO: usize = 25 << 20;
const WORKER_PY: &str = include_str!("stt_worker.py");

struct Worker {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

static WORKER: Mutex<Option<Worker>> = Mutex::const_new(None);
static SEQ: AtomicU64 = AtomicU64::new(1);

fn stt_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join(".omp-rs/stt")
}

fn spawn() -> Result<Worker> {
    let dir = stt_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let script = dir.join("worker.py");
    if std::fs::read_to_string(&script).ok().as_deref() != Some(WORKER_PY) {
        std::fs::write(&script, WORKER_PY)
            .with_context(|| format!("write {}", script.display()))?;
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("worker.log"))
        .context("open stt worker log")?;
    let python = std::env::var("OMT_STT_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let mut child = tokio::process::Command::new(&python)
        .arg("-u")
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(log)
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("start {python} (faster-whisper)"))?;
    let stdin = child.stdin.take().context("worker stdin")?;
    let stdout = child.stdout.take().context("worker stdout")?;
    Ok(Worker {
        child,
        stdin,
        lines: BufReader::new(stdout).lines(),
    })
}

async fn ask(worker: &mut Worker, req: &Value) -> Result<Value> {
    let mut line = req.to_string();
    line.push('\n');
    worker.stdin.write_all(line.as_bytes()).await?;
    worker.stdin.flush().await?;
    let id = &req["id"];
    while let Some(out) = worker.lines.next_line().await? {
        if let Ok(v) = serde_json::from_str::<Value>(&out) {
            if &v["id"] == id {
                return Ok(v);
            }
        }
    }
    bail!("speech worker exited (see ~/.omp-rs/stt/worker.log)")
}

/// Transcribe one recording. `ext` names the container (`webm`, `ogg`,
/// `wav`, …); `language` is an ISO code hint (`None` = auto-detect).
///
/// # Errors
/// The worker cannot start, or the model failed on this audio.
pub async fn transcribe(audio: &[u8], ext: &str, language: Option<&str>) -> Result<Value> {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = stt_dir();
    std::fs::create_dir_all(&dir)?;
    let ext: String = ext
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(8)
        .collect();
    let path = dir.join(format!(
        "rec-{}-{n}.{}",
        std::process::id(),
        if ext.is_empty() { "webm" } else { &ext }
    ));
    std::fs::write(&path, audio).context("save recording")?;
    let req = json!({"id": n, "path": path.display().to_string(), "language": language});
    let mut slot = WORKER.lock().await;
    let mut result = Err(anyhow!("speech worker unavailable"));
    // One retry: the worker may have exited on its idle timer since the
    // last request.
    for _ in 0..2 {
        let alive = match slot.as_mut() {
            Some(w) => w.child.try_wait().ok().flatten().is_none(),
            None => false,
        };
        if !alive {
            *slot = Some(match spawn() {
                Ok(w) => w,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            });
        }
        let Some(worker) = slot.as_mut() else { break };
        match ask(worker, &req).await {
            Ok(v) => {
                result = Ok(v);
                break;
            }
            Err(e) => {
                *slot = None;
                result = Err(e);
            }
        }
    }
    drop(slot);
    let _ = std::fs::remove_file(&path);
    let v = result?;
    if let Some(e) = v["error"].as_str() {
        bail!("speech-to-text: {e}");
    }
    Ok(json!({"text": v["text"], "language": v["language"]}))
}
