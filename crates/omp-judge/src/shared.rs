//! Shared laya judge: one model per machine instead of one per `omt` process.
//!
//! Every long-running `omt` with `context.policy=laya` (and the shadow / offline scorers)
//! used to load its own copy of the judge (`laya_judge_tjsel.onnx`, ~1.3 GB, plus the
//! tokenizer): ten agents held ten identical ~1.7 GB heaps. [`Client`] instead talks to one
//! `omt laya-serve` daemon per `(ckpt, onnx)` pair over a Unix socket and starts it on
//! first use. The daemon holds `<sock>.lock` for its lifetime (one daemon per model),
//! answers requests one at a time (the ONNX session is not `Sync`) and exits after
//! [`IDLE_EXIT`] without requests.
//!
//! Wire format, one request per connection: the client writes one JSON line
//! `{"state": str, "questions": object}` and shuts down its write half; the daemon answers
//! one JSON line `{"ok": [{"qid", "options", "probs"}]}` or `{"err": str}`.

use std::fs::{self, File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use crate::onnx::QuestionProbs;
use serde_json::{json, Value};

use crate::scorer::ModelPaths;

/// The daemon exits after this long without a request (the next client restarts it).
pub const IDLE_EXIT: Duration = Duration::from_mins(30);
/// How long a client waits for a starting daemon to load the model and bind.
const START_WAIT: Duration = Duration::from_mins(5);
/// Upper bound for one judge call (queueing behind other clients included).
const REQUEST_TIMEOUT: Duration = Duration::from_mins(5);

/// Socket of the daemon serving `paths`: `$HOME/.omp-rs/judge/run/laya-<hash>.sock`,
/// the hash over the canonical ckpt and onnx paths.
#[must_use]
pub fn socket_path(paths: &ModelPaths) -> PathBuf {
    let mut h = DefaultHasher::new();
    for p in [&paths.ckpt, &paths.onnx] {
        fs::canonicalize(p).unwrap_or_else(|_| p.clone()).hash(&mut h);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".omp-rs/judge/run")
        .join(format!("laya-{:016x}.sock", h.finish()))
}

/// Client of the shared judge daemon (drop-in for `onnx::Judge::judge`).
#[derive(Debug)]
pub struct Client {
    paths: ModelPaths,
    sock: PathBuf,
}

impl Client {
    /// Connects to the daemon for `paths`, starting it (and waiting for the model load) when
    /// none is running.
    ///
    /// # Errors
    /// Model files missing, or the daemon could not be started or failed to load the model
    /// (its log is quoted).
    pub fn connect(paths: &ModelPaths) -> Result<Self> {
        // Fail here, not after a daemon start, when the model is not on disk.
        for f in [paths.ckpt.join("rl_agent_config.json"), paths.onnx.clone()] {
            if !f.exists() {
                bail!("{} not found", f.display());
            }
        }
        let c = Self {
            paths: paths.clone(),
            sock: socket_path(paths),
        };
        c.ensure_daemon()?;
        Ok(c)
    }

    /// Probabilities for every question of `questions` given `state`.
    ///
    /// # Errors
    /// Daemon unreachable after a restart attempt, or the daemon's inference error.
    pub fn judge(&mut self, state: &str, questions: &Value) -> Result<Vec<QuestionProbs>> {
        let req = json!({ "state": state, "questions": questions }).to_string();
        let resp = match request(&self.sock, &req) {
            Ok(r) => r,
            Err(first) => {
                // Daemon gone (idle exit, crash): restart it once.
                self.ensure_daemon()
                    .with_context(|| format!("laya daemon restart after: {first:#}"))?;
                request(&self.sock, &req)?
            }
        };
        decode_response(&resp)
    }

    fn ensure_daemon(&self) -> Result<()> {
        if UnixStream::connect(&self.sock).is_ok() {
            return Ok(());
        }
        let dir = self.sock.parent().context("socket path without parent")?;
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let log = log_path(&self.sock);
        let mut child = spawn_daemon(&self.paths, &self.sock, &log)?;
        let t0 = Instant::now();
        loop {
            if UnixStream::connect(&self.sock).is_ok() {
                // Reap the daemon when it eventually exits (it outlives most requests).
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return Ok(());
            }
            // Exit 0 = another daemon already holds the lock (still loading): keep waiting.
            if let Some(st) = child.try_wait().context("wait laya daemon")? {
                if !st.success() {
                    bail!(
                        "laya daemon exited with {st}; log {}:\n{}",
                        log.display(),
                        tail(&log, 2000)
                    );
                }
            }
            if t0.elapsed() > START_WAIT {
                bail!(
                    "laya daemon did not come up within {}s; log {}:\n{}",
                    START_WAIT.as_secs(),
                    log.display(),
                    tail(&log, 2000)
                );
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

fn log_path(sock: &Path) -> PathBuf {
    sock.with_extension("log")
}

fn spawn_daemon(paths: &ModelPaths, sock: &Path, log: &Path) -> Result<Child> {
    let exe = std::env::current_exe().context("current_exe for laya-serve")?;
    // A rebuilt `omt` replaces the file under running processes; /proc/self/exe then reads
    // "<path> (deleted)". The daemon is started from the binary now at that path.
    let exe = match exe.to_str().and_then(|s| s.strip_suffix(" (deleted)")) {
        Some(live) => PathBuf::from(live),
        None => exe,
    };
    let err = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .with_context(|| format!("open {}", log.display()))?;
    Command::new(&exe)
        .arg("laya-serve")
        .arg("--ckpt")
        .arg(&paths.ckpt)
        .arg("--onnx")
        .arg(&paths.onnx)
        .arg("--threads")
        .arg(paths.threads.max(1).to_string())
        .arg("--sock")
        .arg(sock)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(err)
        // Own process group: a Ctrl-C in the agent's terminal must not take the shared
        // daemon (and every other agent's judge) down with it.
        .process_group(0)
        .spawn()
        .with_context(|| format!("spawn {} laya-serve", exe.display()))
}

fn tail(path: &Path, max: usize) -> String {
    let s = fs::read_to_string(path).unwrap_or_default();
    let cut = s.len().saturating_sub(max);
    let start = (cut..=s.len()).find(|&i| s.is_char_boundary(i)).unwrap_or(s.len());
    s[start..].to_owned()
}

fn request(sock: &Path, req: &str) -> Result<String> {
    let mut s =
        UnixStream::connect(sock).with_context(|| format!("connect {}", sock.display()))?;
    s.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    s.set_write_timeout(Some(REQUEST_TIMEOUT))?;
    s.write_all(req.as_bytes())?;
    s.write_all(b"\n")?;
    s.shutdown(std::net::Shutdown::Write)?;
    let mut resp = String::new();
    s.read_to_string(&mut resp).context("read laya daemon reply")?;
    if resp.trim().is_empty() {
        bail!("laya daemon closed the connection without a reply");
    }
    Ok(resp)
}

fn encode_ok(qs: &[QuestionProbs]) -> String {
    let v: Vec<Value> = qs
        .iter()
        .map(|q| json!({ "qid": q.qid, "options": q.options, "probs": q.probs }))
        .collect();
    json!({ "ok": v }).to_string()
}

fn decode_response(resp: &str) -> Result<Vec<QuestionProbs>> {
    let v: Value = serde_json::from_str(resp.trim()).context("parse laya daemon reply")?;
    if let Some(e) = v.get("err") {
        bail!("laya daemon: {}", e.as_str().unwrap_or("unknown error"));
    }
    let arr = v
        .get("ok")
        .and_then(Value::as_array)
        .context("laya daemon reply without `ok`")?;
    arr.iter()
        .map(|q| {
            let strs = |k: &str| -> Result<Vec<String>> {
                q[k].as_array()
                    .with_context(|| format!("`{k}` not an array"))?
                    .iter()
                    .map(|x| x.as_str().map(str::to_owned).context("option not a string"))
                    .collect()
            };
            #[allow(clippy::cast_possible_truncation)] // probabilities were f32 on the daemon
            let probs = q["probs"]
                .as_array()
                .context("`probs` not an array")?
                .iter()
                .map(|x| x.as_f64().map(|f| f as f32).context("prob not a number"))
                .collect::<Result<Vec<f32>>>()?;
            Ok(QuestionProbs {
                qid: q["qid"].as_str().context("`qid` not a string")?.to_owned(),
                options: strs("options")?,
                probs,
            })
        })
        .collect()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Judge function the daemon serves (the real ONNX judge, or a stub in tests).
pub type JudgeFn = dyn FnMut(&str, &Value) -> Result<Vec<QuestionProbs>> + Send;

/// Accept loop over `listener`: one request per connection, answered under one lock.
/// Returns when no request arrived for `idle` (the caller removes the socket).
///
/// # Errors
/// The listener failed.
pub fn serve_listener(listener: &UnixListener, judge: Box<JudgeFn>, idle: Duration) -> Result<()> {
    let judge = Arc::new(Mutex::new(judge));
    let last = Arc::new(AtomicU64::new(now_secs()));
    listener.set_nonblocking(true)?;
    loop {
        match listener.accept() {
            Ok((conn, _)) => {
                last.store(now_secs(), Ordering::Relaxed);
                let judge = Arc::clone(&judge);
                let last = Arc::clone(&last);
                std::thread::spawn(move || {
                    if let Err(e) = handle(&conn, &judge) {
                        eprintln!("laya-serve: connection: {e:#}");
                    }
                    last.store(now_secs(), Ordering::Relaxed);
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if now_secs().saturating_sub(last.load(Ordering::Relaxed)) >= idle.as_secs() {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e).context("accept"),
        }
    }
}

fn handle(conn: &UnixStream, judge: &Mutex<Box<JudgeFn>>) -> Result<()> {
    conn.set_nonblocking(false)?;
    conn.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    let mut line = String::new();
    BufReader::new(conn).read_line(&mut line)?;
    if line.trim().is_empty() {
        return Ok(()); // liveness probe (connect + close) from a starting client
    }
    let reply = match serde_json::from_str::<Value>(&line) {
        Ok(req) => {
            let state = req["state"].as_str().unwrap_or("");
            let mut j = judge.lock().map_err(|_| anyhow!("judge lock poisoned"))?;
            match (*j)(state, &req["questions"]) {
                Ok(qs) => encode_ok(&qs),
                Err(e) => json!({ "err": format!("{e:#}") }).to_string(),
            }
        }
        Err(e) => json!({ "err": format!("bad request: {e}") }).to_string(),
    };
    let mut w = conn;
    w.write_all(reply.as_bytes())?;
    w.write_all(b"\n")?;
    Ok(())
}

/// `omt laya-serve`: hold `<sock>.lock`, load the judge, serve until idle.
/// Exits 0 without loading when another daemon already holds the lock.
///
/// # Errors
/// Lock/socket I/O or model load failure.
pub fn serve(paths: &ModelPaths, sock: &Path) -> Result<()> {
    let lock_path = sock.with_extension("lock");
    let lock = File::create(&lock_path).with_context(|| format!("create {}", lock_path.display()))?;
    if lock.try_lock().is_err() {
        return Ok(()); // a daemon for this model is running (or loading)
    }
    let t0 = Instant::now();
    let mut judge = crate::onnx::Judge::load(&paths.ckpt, &paths.onnx, paths.threads.max(1))
        .with_context(|| {
            format!(
                "load laya judge (ckpt {}, onnx {})",
                paths.ckpt.display(),
                paths.onnx.display()
            )
        })?;
    let _ = fs::remove_file(sock); // stale socket of a dead daemon (we hold the lock)
    let listener =
        UnixListener::bind(sock).with_context(|| format!("bind {}", sock.display()))?;
    eprintln!(
        "laya-serve pid {} ready in {:.1}s on {}",
        std::process::id(),
        t0.elapsed().as_secs_f64(),
        sock.display()
    );
    let r = serve_listener(
        &listener,
        Box::new(move |state, qs| judge.judge(state, qs)),
        IDLE_EXIT,
    );
    let _ = fs::remove_file(sock);
    eprintln!("laya-serve pid {} exiting ({r:?})", std::process::id());
    drop(lock);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub(state: &str, qs: &Value) -> Result<Vec<QuestionProbs>> {
        if state == "fail" {
            bail!("stub failure");
        }
        Ok(qs
            .as_object()
            .unwrap()
            .keys()
            .map(|k| QuestionProbs {
                qid: k.clone(),
                options: vec!["true".into(), "false".into()],
                probs: vec![0.25, 0.75],
            })
            .collect())
    }

    #[test]
    fn roundtrip_errors_and_idle_exit() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("j.sock");
        let listener = UnixListener::bind(&sock).unwrap();
        let server = std::thread::spawn(move || {
            serve_listener(&listener, Box::new(stub), Duration::from_secs(2)).unwrap();
        });
        let mut c = Client {
            paths: ModelPaths::default(),
            sock: sock.clone(),
        };
        let qs = json!({ "success": {}, "rework": {} });
        let r = c.judge("s", &qs).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].qid, "success");
        assert_eq!(r[0].options, ["true", "false"]);
        assert!((r[0].probs[1] - 0.75).abs() < 1e-6);
        // concurrent clients are all answered
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let sock = sock.clone();
                std::thread::spawn(move || {
                    let req = json!({ "state": "s", "questions": { "q": {} } }).to_string();
                    decode_response(&request(&sock, &req).unwrap()).unwrap()
                })
            })
            .collect();
        for h in hs {
            assert_eq!(h.join().unwrap()[0].qid, "q");
        }
        // inference errors travel back as errors
        let req = json!({ "state": "fail", "questions": {} }).to_string();
        let e = decode_response(&request(&sock, &req).unwrap()).unwrap_err();
        assert!(format!("{e:#}").contains("stub failure"));
        server.join().unwrap(); // returns after 2 s idle
    }

    #[test]
    fn socket_path_depends_on_model() {
        let a = ModelPaths::default();
        let mut b = a.clone();
        b.onnx = "/other/model.onnx".into();
        assert_ne!(socket_path(&a), socket_path(&b));
        assert_eq!(socket_path(&a), socket_path(&a.clone()));
    }
}
