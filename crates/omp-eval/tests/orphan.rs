//! Orphan-kernel regression: a runner (python `while True: pass`, bun
//! `for(;;){}`) must die within ~2 s of its host being `SIGKILLed`, even while
//! a cell is busy. The helper is the test binary itself, re-exec'd with
//! `OMP_ORPHAN_KIND` set; it starts a kernel, records the runner pid, and
//! parks in a busy cell until the parent test SIGKILLs it.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::PythonKernel;
mod common;

/// Process alive check for a pid we do not own (orphaned runners are
/// reparented to init): `kill(pid, 0)` fails with ESRCH once it is gone.
fn pid_alive(pid: u32) -> bool {
    // Orphaned runners are reparented away from this process, so a
    // wait()-based check is impossible; /proc is the portable-liveness probe
    // here (Linux CI), and signal 0 needs `unsafe` the workspace forbids.
    std::path::Path::new("/proc").join(pid.to_string()).exists()
}

#[test]
fn orphan_helper_entry() {
    let Ok(kind) = std::env::var("OMP_ORPHAN_KIND") else {
        return; // normal test run: entry point is inert
    };
    let dir = std::path::PathBuf::from(
        std::env::var("OMP_ORPHAN_DIR").expect("OMP_ORPHAN_DIR set by the parent test"),
    );
    let pid_file = dir.join("kernel.pid");
    let worker = std::thread::spawn(move || match kind.as_str() {
        "py" => {
            let k = PythonKernel::start("python3", &dir).expect("kernel starts");
            write!(
                std::fs::File::create(&pid_file).expect("pid file"),
                "{}",
                k.pid().expect("pid")
            )
            .expect("write pid");
            // Let the busy cell actually start spinning before the parent kills us.
            std::thread::sleep(Duration::from_millis(500));
            let _ = k.execute("while True: pass", None);
        }
        "js" => {
            let k = JsKernel::start(default_runtime(), &dir).expect("kernel starts");
            write!(
                std::fs::File::create(&pid_file).expect("pid file"),
                "{}",
                k.pid().expect("pid")
            )
            .expect("write pid");
            std::thread::sleep(Duration::from_millis(500));
            let _ = k.execute("for(;;){}", None);
        }
        other => panic!("unknown kind {other}"),
    });
    // Park until the parent SIGKILLs this helper.
    std::thread::sleep(Duration::from_mins(2));
    let _ = worker.join();
}

fn run_orphan_case(kind: &'static str) {
    let dir = std::env::temp_dir().join(format!("omp-eval-orphan-{}-{}", kind, std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    let mut helper = Command::new(std::env::current_exe().expect("test exe"))
        .args(["orphan_helper_entry", "--exact", "--nocapture"])
        .env("OMP_ORPHAN_KIND", kind)
        .env("OMP_ORPHAN_DIR", &dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn helper");

    // Wait for the helper to report the runner pid (kernel start ~1 s).
    let pid_file = dir.join("kernel.pid");
    let deadline = Instant::now() + Duration::from_secs(30);
    let kernel_pid: u32 = loop {
        assert!(
            helper.try_wait().expect("helper poll").is_none(),
            "helper died early"
        );
        if let Ok(text) = std::fs::read_to_string(&pid_file) {
            break text.trim().parse().expect("kernel pid");
        }
        assert!(
            Instant::now() < deadline,
            "helper never reported the kernel pid"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    // Make sure the busy cell is actually running.
    std::thread::sleep(Duration::from_millis(700));
    assert!(pid_alive(kernel_pid), "kernel should be running");

    // SIGKILL the host mid-cell; the kernel must be gone within 3 s.
    let host_pid = i32::try_from(helper.id()).expect("host pid");
    assert_eq!(
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(host_pid),
            nix::sys::signal::Signal::SIGKILL,
        ),
        Ok(()),
        "SIGKILL helper"
    );
    helper.wait().expect("reap helper");

    let started = Instant::now();
    let gone_deadline = started + Duration::from_secs(3);
    let gone = loop {
        if !pid_alive(kernel_pid) {
            break true;
        }
        if Instant::now() >= gone_deadline {
            break false;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    // Best-effort cleanup if the assertion below would fail.
    if !gone {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(i32::try_from(kernel_pid).expect("pid")),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(gone, "{kind} kernel survived a SIGKILLed host for >3 s");
}

#[test]
fn python_kernel_exits_when_host_is_sigkilled() {
    run_orphan_case("py");
}

#[test]
fn js_kernel_exits_when_host_is_sigkilled() {
    if !common::require_bun("js_kernel_exits_when_host_is_sigkilled") {
        return;
    }
    run_orphan_case("js");
}
