//! Session persistence (omp session JSONL v3) via `omp-session`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use omp_session::SessionWriter;
use pa_agent::agent::Agent;
use pa_agent::types::AgentEvent;

use crate::agent::civil;

/// ISO-8601 UTC with milliseconds, like omp ("2026-10-11T14:08:39.790Z").
#[must_use]
pub fn iso_from_ms(ms: i64) -> String {
    let (y, mo, d) = civil(ms.div_euclid(86_400_000));
    let rem = ms.rem_euclid(86_400_000);
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3_600_000,
        rem / 60_000 % 60,
        rem / 1000 % 60,
        rem % 1000
    )
}

#[must_use]
pub fn now_iso() -> String {
    iso_from_ms(pa_agent::now_ms())
}

/// UUID v7 string (48-bit ms timestamp + random).
#[must_use]
pub fn uuid_v7() -> String {
    use rand::RngCore;
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    let ms = u64::try_from(pa_agent::now_ms()).unwrap_or(0);
    b[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    b[6] = 0x70 | (b[6] & 0x0f);
    b[8] = 0x80 | (b[8] & 0x3f);
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..])
}

fn sessions_dir(cwd: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let slug = cwd.display().to_string().replace('/', "-");
    Some(PathBuf::from(home).join(".omp-rs/sessions").join(slug))
}

/// Live session file: the agent's finished messages are appended to
/// `writer`, which `start_new` swaps for a fresh file (`new_session`).
pub struct Persistence {
    writer: Arc<Mutex<SessionWriter>>,
    session_id: Mutex<String>,
}

impl Persistence {
    #[must_use]
    pub fn session_id(&self) -> String {
        self.session_id.lock().unwrap().clone()
    }

    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.writer.lock().unwrap().path().to_path_buf()
    }

    /// Append a non-message entry (`model_change`, `title_change`, ...).
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_entry(&self, kind: &str, payload: serde_json::Value) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_entry(kind, payload, &now_iso())
            .map(|_| ())
    }

    /// Append a finished conversation message.
    ///
    /// # Errors
    /// Fails on serialization or I/O errors.
    pub fn append_message(&self, message: &pa_types::session::AgentMessage, timestamp_ms: i64) -> Result<()> {
        self.writer
            .lock()
            .unwrap()
            .append_message(message, &iso_from_ms(timestamp_ms))
            .map(|_| ())
    }

    /// Switch to a brand-new session file in `dir` (default: the cwd slot).
    ///
    /// # Errors
    /// Fails when the file cannot be created.
    pub fn start_new(&self, cwd: &Path, dir: Option<&Path>) -> Result<()> {
        let dir = match dir {
            Some(d) => d.to_path_buf(),
            None => sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))?,
        };
        let id = uuid_v7();
        let fresh = SessionWriter::create(&dir, &cwd.display().to_string(), &id, &now_iso())?;
        *self.writer.lock().unwrap() = fresh;
        *self.session_id.lock().unwrap() = id;
        Ok(())
    }
}

/// Create a session file for this run and append every completed message.
///
/// # Errors
/// Fails when the session file cannot be created.
pub async fn attach(agent: &Agent, cwd: &Path) -> Result<Persistence> {
    let dir = sessions_dir(cwd).ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    let session_id = uuid_v7();
    let writer = SessionWriter::create(&dir, &cwd.display().to_string(), &session_id, &now_iso())?;
    let writer = Arc::new(Mutex::new(writer));
    let sub = agent
        .subscribe({
            let writer = writer.clone();
            move |event, _| {
                if let AgentEvent::MessageEnd { message } = &event {
                    let converted = serde_json::to_value(message)
                        .map_err(anyhow::Error::from)
                        .and_then(|v| {
                            serde_json::from_value::<pa_types::session::AgentMessage>(v)
                                .map_err(anyhow::Error::from)
                        });
                    match converted {
                        Ok(m) => {
                            let ts = iso_from_ms(message.timestamp());
                            if let Err(e) = writer.lock().unwrap().append_message(&m, &ts) {
                                eprintln!("omp-rs: session write failed: {e:#}");
                            }
                        }
                        Err(e) => eprintln!("omp-rs: session message conversion failed: {e:#}"),
                    }
                }
                Box::pin(async { Ok(()) })
            }
        })
        .await;
    // Dropping a Subscription does not unsubscribe; the listener lives with the agent.
    drop(sub);
    Ok(Persistence { writer, session_id: Mutex::new(session_id) })
}
