//! Regression for the 2026-10-31 desktop incident: an agent process whose
//! live `PATH` lost `~/.bun/bin` hours after startup could no longer respawn
//! the JS kernel. The kernel now pins an absolute runtime path at start.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::CellStatus;

#[test]
fn respawn_survives_path_rotation() {
    let Ok(orig_path) = std::env::var("PATH") else {
        eprintln!("skipping: no PATH in environment");
        return;
    };
    temp_env::with_var("PATH", Some(orig_path), || {
        let dir = std::env::temp_dir().join(format!("omp-eval-js-rot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let k = match JsKernel::start(default_runtime(), &dir) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: bun unavailable ({e})");
                return;
            }
        };
        let first = k.execute("1 + 1", Some(Duration::from_secs(30))).unwrap();
        assert_eq!(first.status, CellStatus::Ok);

        temp_env::with_var("PATH", Some("/usr/bin:/bin"), || {
            let pid = k.pid().expect("runner pid");
            std::process::Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while k.is_alive() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(!k.is_alive(), "runner should be gone after SIGKILL");

            let second = k
                .execute("2 + 2", Some(Duration::from_secs(30)))
                .expect("transparent respawn must survive a rotated PATH");
            assert_eq!(second.status, CellStatus::Ok);
        });
    });
}
