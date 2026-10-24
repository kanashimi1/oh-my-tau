//! Print mode answers handled slash commands with their output (temp HOME,
//! no provider call): `omp-rs -p "/rlm stats"` used to print
//! "Error: no assistant response" because command output frames were dropped.

use std::process::Command;

#[test]
fn print_mode_prints_handled_command_output() {
    let home = tempfile::tempdir().unwrap();
    let agent = home.path().join(".omp/agent");
    std::fs::create_dir_all(&agent).unwrap();
    std::fs::write(
        agent.join("models.yml"),
        "providers:\n  mock:\n    baseUrl: http://127.0.0.1:9/v1\n    apiKey: unused\n    api: openai-completions\n    models:\n      - id: m\n        name: M\n        contextWindow: 8000\n        maxTokens: 100\n",
    )
    .unwrap();
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_omt"))
        .env("HOME", home.path())
        .env_remove("OMP_AGENT_DIR")
        .env_remove("OMP_AUTH_BROKER_URL")
        .env_remove("OMP_AUTH_BROKER_TOKEN")
        .env("NO_COLOR", "1")
        .current_dir(&work)
        .args([
            "-p",
            "--no-session",
            "--no-extensions",
            "--model",
            "mock/m",
            "/rlm stats",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("\"calls\": 0"),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(!stderr.contains("no assistant response"), "{stderr}");
}
