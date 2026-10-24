//! Shared-mux integration test: N sessions over one server process through
//! the `omp-lsp-mux` daemon, with document-sync isolation, server exit after
//! the last session plus the linger timeout, and RSS/process-count
//! measurement for N = 1/4/8 against private spawns.

use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use omp_lsp::mux::protocol::lsp_mux_endpoint;
use omp_lsp::mux::{connect_shared_lsp_transport, MuxTransport};

fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "omp-lsp-shared-mux-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp project");
    dir
}

fn frame(message: &Value) -> Vec<u8> {
    let body = serde_json::to_string(message).expect("serialize frame");
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

/// Minimal framed-message channel over a mux link for the test.
struct Link {
    transport: MuxTransport,
    read_buf: Vec<u8>,
}

impl Link {
    async fn send(&mut self, message: &Value) {
        self.transport
            .writer
            .write_all(&frame(message))
            .await
            .expect("write mux link");
    }

    /// Read the next JSON-RPC message on the link.
    async fn recv(&mut self) -> Value {
        loop {
            if let Some(msg) = pop_frame(&mut self.read_buf) {
                return msg;
            }
            let mut chunk = [0u8; 8192];
            let n = self
                .transport
                .reader
                .read(&mut chunk)
                .await
                .expect("read mux link");
            assert!(n > 0, "mux link closed");
            self.read_buf.extend_from_slice(&chunk[..n]);
        }
    }

    /// Read until a message with `method` arrives; returns its params.
    async fn recv_notification(&mut self, method: &str) -> Value {
        loop {
            let msg = self.recv().await;
            if msg.get("method").and_then(Value::as_str) == Some(method) {
                return msg.get("params").cloned().unwrap_or(Value::Null);
            }
        }
    }

    async fn request(&mut self, method: &str, params: Value, id: u64) -> Value {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await;
        loop {
            let msg = self.recv().await;
            if msg.get("id").and_then(Value::as_u64) == Some(id) {
                return msg;
            }
        }
    }
}

fn pop_frame(buf: &mut Vec<u8>) -> Option<Value> {
    let header_end = find_subslice(buf, b"\r\n\r\n")?;
    let header = &buf[..header_end];
    let mut len = None;
    for line in header.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(colon) = line.iter().position(|&b| b == b':') {
            let (name, value) = (&line[..colon], &line[colon + 1..]);
            if name.eq_ignore_ascii_case(b"content-length") {
                len = std::str::from_utf8(value)
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok());
            }
        }
    }
    let len = len?;
    if buf.len() < header_end + 4 + len {
        return None;
    }
    let body: Vec<u8> = buf
        .drain(..header_end + 4 + len)
        .skip(header_end + 4)
        .collect();
    serde_json::from_slice(&body).ok()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Processes whose executable basename is `exe` (other agents share this
/// machine and run their own mux fixtures), with their RSS in kB.
fn matching_processes(exe: &str) -> Vec<(u32, u64)> {
    let mut out = Vec::new();
    let proc_dir = std::fs::read_dir("/proc").expect("read /proc");
    for entry in proc_dir.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        let argv0_end = cmdline
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(cmdline.len());
        let argv0 = Path::new(std::str::from_utf8(&cmdline[..argv0_end]).unwrap_or(""));
        if argv0.file_name().and_then(|n| n.to_str()) != Some(exe) {
            continue;
        }
        let Ok(status) = std::fs::read_to_string(entry.path().join("status")) else {
            continue;
        };
        let rss = status
            .lines()
            .find_map(|l| l.strip_prefix("VmRSS:"))
            .and_then(|l| l.trim().strip_suffix(" kB"))
            .and_then(|l| l.trim().parse::<u64>().ok());
        if let Some(rss) = rss {
            out.push((pid, rss));
        }
    }
    out
}

/// Locate the `omp-lsp-mux` binary built alongside the `fake_lsp` fixture
/// (integration tests do not get a `CARGO_BIN_EXE_` for hyphenated names).
fn mux_bin() -> std::path::PathBuf {
    let fake = std::path::PathBuf::from(env!("CARGO_BIN_EXE_fake_lsp"));
    let candidate = fake.with_file_name(format!("omp-lsp-mux{}", std::env::consts::EXE_SUFFIX));
    assert!(
        candidate.is_file(),
        "omp-lsp-mux binary not found at {}",
        candidate.display()
    );
    candidate
}

/// Whether the process's argv0 path contains `needle`.
fn process_argv0_contains(pid: u32, needle: &str) -> bool {
    let Ok(cmdline) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
        return false;
    };
    let argv0_end = cmdline
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(cmdline.len());
    String::from_utf8_lossy(&cmdline[..argv0_end]).contains(needle)
}

/// Directory (with trailing slash) holding this test build's binaries, so
/// the process filters work for whichever profile cargo built (debug or
/// release) while ignoring other profiles' fixtures on this machine.
fn build_bin_dir() -> String {
    let fake = std::path::PathBuf::from(env!("CARGO_BIN_EXE_fake_lsp"));
    format!("{}/", fake.parent().expect("bin dir").display())
}

/// `fake_lsp` processes belonging to this test's build profile (sibling
/// measurement harnesses run fixtures of other profiles on this machine).
fn fake_lsp_processes() -> Vec<(u32, u64)> {
    matching_processes("fake_lsp")
        .into_iter()
        .filter(|(pid, _)| process_argv0_contains(*pid, &build_bin_dir()))
        .collect()
}

/// This build's `omp-lsp-mux` daemon processes.
fn mux_daemon_processes() -> Vec<(u32, u64)> {
    matching_processes("omp-lsp-mux")
        .into_iter()
        .filter(|(pid, _)| process_argv0_contains(*pid, &build_bin_dir()))
        .collect()
}

fn process_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

async fn wait_for<F: FnMut() -> bool>(mut cond: F, timeout: Duration, what: &str) {
    let start = std::time::Instant::now();
    while !cond() {
        assert!(start.elapsed() < timeout, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Initialize a fresh link through the mux (handshake + initialize +
/// initialized). Returns the link and the shared server pid.
async fn open_initialized_link(cwd: &Path, server_bin: &str) -> (Link, u32) {
    let transport = connect_shared_lsp_transport(server_bin, &[], cwd, None)
        .await
        .expect("shared transport");
    let pid = transport.pid.expect("connect result pid");
    let mut link = Link {
        transport,
        read_buf: Vec::new(),
    };
    let root = format!("file://{}", cwd.display());
    let init = link
        .request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": root,
                "workspaceFolders": [{"uri": root, "name": "ra-smoke"}],
                "capabilities": {},
            }),
            1,
        )
        .await;
    assert!(
        init.pointer("/result/capabilities").is_some(),
        "initialize result: {init}"
    );
    link.send(&json!({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
        .await;
    (link, pid)
}

/// Kill leftover same-profile `fake_lsp`/`omp-lsp-mux` processes from
/// earlier runs of this test (a failed run leaves the detached daemon and
/// its server behind).
fn kill_stale_mux_processes() {
    for (pid, _) in fake_lsp_processes()
        .into_iter()
        .chain(mux_daemon_processes())
    {
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status();
    }
    std::thread::sleep(Duration::from_millis(100));
}

#[allow(clippy::too_many_lines)] // one long end-to-end scenario with measurement
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_mux_sessions_isolation_and_exit() {
    kill_stale_mux_processes();
    // Deterministic short timers for the daemon spawned by this test.
    std::env::set_var("OMP_LSP_MUX_BIN", mux_bin());
    std::env::set_var("OMP_LSP_MUX_SERVER_LINGER_MS", "400");
    std::env::set_var("OMP_LSP_MUX_IDLE_MS", "600");
    let server_bin = env!("CARGO_BIN_EXE_fake_lsp");
    let cwd = unique_temp_dir("project");
    let file_a = cwd.join("a.fake");
    let file_b = cwd.join("b.fake");
    std::fs::write(&file_a, "a\n").unwrap();
    std::fs::write(&file_b, "b\n").unwrap();
    let uri_a = format!("file://{}", file_a.display());
    let uri_b = format!("file://{}", file_b.display());

    // --- N=4 links over one server -------------------------------------
    let mut links: Vec<Link> = Vec::new();
    let mut pids = Vec::new();
    for _ in 0..4 {
        let (link, pid) = open_initialized_link(&cwd, server_bin).await;
        links.push(link);
        pids.push(pid);
    }
    assert_eq!(
        pids[1..],
        pids[..3],
        "all four sessions must share one server"
    );
    let server_pid = pids[0];
    assert!(process_alive(server_pid));
    assert_eq!(
        fake_lsp_processes().len(),
        1,
        "exactly one fake_lsp process for N=4 sessions"
    );

    // --- Diagnostics: didOpen on session 0 is broadcast (cached on the mux)
    // and echoes the document version (TS mux behavior).
    links[0]
        .send(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri_a, "languageId": "fake", "version": 7, "text": "a\n"}},
        }))
        .await;
    let diag0 = links[0]
        .recv_notification("textDocument/publishDiagnostics")
        .await;
    let diag1 = links[1]
        .recv_notification("textDocument/publishDiagnostics")
        .await;
    assert_eq!(diag0["uri"], json!(uri_a));
    assert_eq!(diag0["version"], json!(7));
    assert_eq!(diag1["uri"], json!(uri_a));
    assert_eq!(diag1["version"], json!(7));

    // --- Isolation: dropping session 0 closes its documents on the shared
    // server; the other sessions keep working.
    let dropped_transport = links.remove(0);
    drop(dropped_transport);
    // A didOpen from session 0 must not have severed the server: session 1
    // still gets answers.
    links[0]
        .send(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri_b, "languageId": "fake", "version": 3, "text": "b\n"}},
        }))
        .await;
    let _ = links[0]
        .recv_notification("textDocument/publishDiagnostics")
        .await;
    let hover = links[0]
        .request(
            "textDocument/hover",
            json!({"textDocument": {"uri": uri_b}, "position": {"line": 0, "character": 0}}),
            50,
        )
        .await;
    assert_eq!(hover["result"]["contents"]["value"], json!("fake hover"));
    assert!(
        process_alive(server_pid),
        "server survives one session dropping"
    );

    // --- RSS/process measurement for N = 1/4/8 ---------------------------
    // Drop the remaining session links so the server lingers out first.
    drop(links);
    wait_for(
        || !process_alive(server_pid),
        Duration::from_secs(10),
        "server exit after last session + linger",
    )
    .await;

    let mut shared_results = Vec::new();
    for n in [1usize, 4, 8] {
        let mut mlinks = Vec::new();
        let mut mpid = None;
        for _ in 0..n {
            let (link, pid) = open_initialized_link(&cwd, server_bin).await;
            mlinks.push(link);
            mpid = Some(pid);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let procs = fake_lsp_processes();
        let rss: u64 = procs.iter().map(|(_, r)| *r).sum::<u64>() + mux_daemon_rss();
        assert_eq!(procs.len(), 1, "shared mode: one server process for N={n}");
        shared_results.push((n, procs.len(), rss));
        let pid = mpid.expect("pid");
        drop(mlinks);
        wait_for(
            || !process_alive(pid),
            Duration::from_secs(10),
            "shared server linger exit",
        )
        .await;
    }

    // Private baseline: N directly spawned fake_lsp processes.
    let mut private_results = Vec::new();
    for n in [1usize, 4, 8] {
        let mut children = Vec::new();
        for _ in 0..n {
            let child = std::process::Command::new(server_bin)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn private fake_lsp");
            children.push(child);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let procs = fake_lsp_processes();
        let rss: u64 = procs.iter().map(|(_, r)| *r).sum();
        private_results.push((n, procs.len(), rss));
        for mut child in children {
            let _ = child.kill();
            let _ = child.wait();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // --- Client-level integration: get_or_create_client over the mux ------
    omp_lsp::mux::set_shared_lsp_enabled(true);
    invalidate_and_configure(&cwd, server_bin);
    let config = omp_lsp::config::get_config(&cwd);
    let file_path = cwd.join("main.fake");
    std::fs::write(&file_path, "fake body\n").unwrap();
    let (_name, server) =
        omp_lsp::config::get_server_for_file(&config, &file_path).expect("fakeserver match");
    let client =
        omp_lsp::client::get_or_create_client(&server, &cwd, Some(Duration::from_secs(10)), None)
            .await
            .expect("shared client initialize");
    assert!(
        client.is_shared_mux(),
        "client must use the shared mux link"
    );
    let client_pid = client.mux_server_pid().expect("shared pid");
    client.refresh_file(&file_path).await;
    let uri = omp_lsp::utils::file_to_uri(&file_path);
    let published =
        omp_lsp::diagnostics::wait_for_diagnostics(&client, &uri, 5000, None, None, Some(1))
            .await
            .expect("diagnostics through mux");
    assert_eq!(published.len(), 1);
    // A second, independent transport to the same server must reuse it.
    let extra = connect_shared_lsp_transport(
        server
            .resolved_command
            .as_deref()
            .unwrap_or(&server.command),
        &server.args.clone().unwrap_or_default(),
        &cwd,
        None,
    )
    .await
    .expect("second link");
    assert_eq!(
        extra.pid,
        Some(client_pid),
        "same shared server behind the client link"
    );
    drop(extra);
    let _ = client.shutdown_instance().await;
    wait_for(
        || !process_alive(client_pid),
        Duration::from_secs(10),
        "shared server exit after client shutdown + linger",
    )
    .await;

    // --- Daemon idle exit -------------------------------------------------
    let endpoint = lsp_mux_endpoint(&cwd).expect("endpoint");
    wait_for(
        || !endpoint.exists(),
        Duration::from_secs(15),
        "mux daemon idle exit (socket removed)",
    )
    .await;

    // --- Report the measurements on stdout for the bench log -------------
    let mut report = String::new();
    report.push_str("shared (1 server + mux daemon):\n");
    for (n, count, rss) in &shared_results {
        report.push_str("  N=");
        report.push_str(&n.to_string());
        report.push_str(": server_processes=");
        report.push_str(&count.to_string());
        report.push_str(" total_rss_kib=");
        report.push_str(&rss.to_string());
        report.push('\n');
    }
    report.push_str("private baseline:\n");
    for (n, count, rss) in &private_results {
        report.push_str("  N=");
        report.push_str(&n.to_string());
        report.push_str(": server_processes=");
        report.push_str(&count.to_string());
        report.push_str(" total_rss_kib=");
        report.push_str(&rss.to_string());
        report.push('\n');
    }
    println!("MUX_MEASUREMENT\n{report}MUX_MEASUREMENT_END");
    std::fs::write(cwd.join("mux-measurement.txt"), &report).ok();
    std::env::remove_var("OMP_LSP_MUX_BIN");
    std::env::remove_var("OMP_LSP_MUX_SERVER_LINGER_MS");
    std::env::remove_var("OMP_LSP_MUX_IDLE_MS");
}

fn invalidate_and_configure(cwd: &Path, server_bin: &str) {
    let lsp_json = json!({
        "servers": {
            "fakeserver": {
                "command": server_bin,
                "fileTypes": [".fake"],
                "rootMarkers": ["."],
                "languageId": "fake"
            }
        }
    });
    std::fs::write(
        cwd.join("lsp.json"),
        serde_json::to_string_pretty(&lsp_json).unwrap(),
    )
    .unwrap();
    omp_lsp::config::invalidate_config(cwd);
}

/// RSS of the `omp-lsp-mux` daemon process(es) in kB.
fn mux_daemon_rss() -> u64 {
    mux_daemon_processes().iter().map(|(_, r)| *r).sum()
}

/// RSS of one pid in kB (`None` when the process is gone).
fn rss_of_pid(pid: u32) -> u64 {
    let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) else {
        return 0;
    };
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|l| l.trim().strip_suffix(" kB"))
        .and_then(|l| l.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Manual measurement against a real language server
/// (`cargo test -p omp-lsp --test shared_mux -- --ignored`): N mux links over
/// one rust-analyzer on a tiny crate, versus N private spawns. Skipped in
/// normal runs (rust-analyzer may be absent; init takes seconds).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "live rust-analyzer smoke test; run explicitly"]
#[allow(clippy::too_many_lines)] // one long end-to-end measurement scenario
async fn shared_mux_rust_analyzer_measurement() {
    use std::io::Read as _;

    kill_stale_mux_processes();
    let ra = std::process::Command::new("rust-analyzer")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ra {
        eprintln!("rust-analyzer not installed; skipping");
        return;
    }
    std::env::set_var("OMP_LSP_MUX_BIN", mux_bin());
    std::env::set_var("OMP_LSP_MUX_SERVER_LINGER_MS", "1000");
    std::env::set_var("OMP_LSP_MUX_IDLE_MS", "1500");

    // A tiny cargo project for rust-analyzer to load.
    let cwd = unique_temp_dir("ra-crate");
    std::fs::write(
        cwd.join("Cargo.toml"),
        "[package]\nname = \"ra-smoke\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(cwd.join("src")).unwrap();
    std::fs::write(
        cwd.join("src").join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();

    // Warm the daemon + shared server with N=4 initialized links.
    let mut links = Vec::new();
    let mut pid = None;
    for _ in 0..4 {
        let (mut link, p) = open_initialized_link(&cwd, "rust-analyzer").await;
        pid = Some(p);
        // A cheap request so the server is genuinely serving every link;
        // retry while rust-analyzer's workspace scan catches up.
        let uri = format!("file://{}/src/lib.rs", cwd.display());
        let mut hover = Value::Null;
        for _ in 0..20 {
            hover = link
                .request(
                    "textDocument/hover",
                    json!({"textDocument": {"uri": uri}, "position": {"line": 0, "character": 20}}),
                    99,
                )
                .await;
            if hover.pointer("/result/contents").is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        assert!(
            hover.pointer("/result/contents").is_some(),
            "hover: {hover}"
        );
        links.push(link);
    }
    let server_pid = pid.expect("server pid");
    tokio::time::sleep(Duration::from_millis(500)).await;
    // Measure by exact pid: sibling measurement harnesses on this machine
    // also run rust-analyzer processes that name-matching would count.
    let shared_rss = rss_of_pid(server_pid) + mux_daemon_rss();
    println!("RA_SHARED N=4 server_processes=1 server_pid={server_pid} total_rss_kib={shared_rss}");

    drop(links);
    wait_for(
        || !process_alive(server_pid),
        Duration::from_secs(20),
        "ra linger exit",
    )
    .await;
    wait_for(
        || !lsp_mux_endpoint(&cwd).expect("endpoint").exists(),
        Duration::from_secs(20),
        "mux daemon idle exit",
    )
    .await;

    // Private baseline: one direct spawn, warmed with the same initialize +
    // hover handshake before measuring.
    let mut child = std::process::Command::new("rust-analyzer")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn private rust-analyzer");
    {
        let stdin = child.stdin.as_mut().expect("private stdin");
        let root = format!("file://{}", cwd.display());
        let init = json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "processId": std::process::id(),
                "rootUri": root,
                "workspaceFolders": [{"uri": root, "name": "ra-smoke"}],
                "capabilities": {},
            },
        });
        let body = serde_json::to_string(&init).unwrap();
        std::io::Write::write_all(
            stdin,
            format!("Content-Length: {}\r\n\r\n{}", body.len(), body).as_bytes(),
        )
        .expect("write init");
        std::io::Write::flush(stdin).unwrap();
        let initialized = serde_json::to_string(&json!({
            "jsonrpc": "2.0", "method": "initialized", "params": {},
        }))
        .unwrap();
        std::io::Write::write_all(
            stdin,
            format!(
                "Content-Length: {}\r\n\r\n{}",
                initialized.len(),
                initialized
            )
            .as_bytes(),
        )
        .expect("write initialized");
        std::io::Write::flush(stdin).unwrap();
        let init_req = serde_json::to_string(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/hover",
            "params": {"textDocument": {"uri": format!("file://{}/src/lib.rs", cwd.display())},
                       "position": {"line": 0, "character": 20}},
        }))
        .unwrap();
        std::io::Write::write_all(
            stdin,
            format!("Content-Length: {}\r\n\r\n{}", init_req.len(), init_req).as_bytes(),
        )
        .expect("write hover");
        std::io::Write::flush(stdin).unwrap();
        // Retry the hover until it succeeds (the workspace scan takes a
        // while); the server is warm once it answers. A reader thread
        // streams stdout bytes back so the retry loop can tick.
        let mut framed = Vec::new();
        let mut got_hover = false;
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let mut stdout_pipe = child.stdout.take().expect("private stdout");
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match stdout_pipe.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(90);
        let mut next_hover = std::time::Instant::now() + Duration::from_secs(2);
        let mut hover_id = 2u64;
        while !got_hover && std::time::Instant::now() < deadline {
            if std::time::Instant::now() >= next_hover {
                let req = serde_json::to_string(&json!({
                    "jsonrpc": "2.0", "id": hover_id, "method": "textDocument/hover",
                    "params": {"textDocument": {"uri": format!("file://{}/src/lib.rs", cwd.display())},
                               "position": {"line": 0, "character": 20}},
                }))
                .unwrap();
                std::io::Write::write_all(
                    stdin,
                    format!("Content-Length: {}\r\n\r\n{}", req.len(), req).as_bytes(),
                )
                .expect("write hover");
                std::io::Write::flush(stdin).unwrap();
                hover_id += 1;
                next_hover = std::time::Instant::now() + Duration::from_secs(2);
            }
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(bytes) => framed.extend_from_slice(&bytes),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            let text = String::from_utf8_lossy(&framed);
            got_hover = (2..hover_id).any(|id| text.contains(&format!("\"id\":{id},\"result\"")));
        }
        assert!(got_hover, "private rust-analyzer did not answer hover");
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    let private_pid = child.id();
    let private_rss = rss_of_pid(private_pid);
    println!(
        "RA_PRIVATE N=1 server_processes=1 server_pid={private_pid} total_rss_kib={private_rss}"
    );
    let _ = child.kill();
    let _ = child.wait();
    std::env::remove_var("OMP_LSP_MUX_BIN");
    std::env::remove_var("OMP_LSP_MUX_SERVER_LINGER_MS");
    std::env::remove_var("OMP_LSP_MUX_IDLE_MS");
}
