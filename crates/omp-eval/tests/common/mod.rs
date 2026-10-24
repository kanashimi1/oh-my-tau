//! Shared prerequisite detection for the integration tests: a test whose
//! external prerequisite is missing prints exactly one `skipping ...` line
//! and returns early, so a clean machine (fresh clone, no `bun`) still gets
//! a green `cargo test`, while machines that have the prerequisite always
//! run the real test.

/// Is `bun` resolvable on PATH? Probes each PATH entry for an executable
/// file named `bun` (symlinks are followed); nothing is spawned.
pub fn bun_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| is_executable(dir.join("bun")))
}

/// Prerequisite gate for tests that need the Bun JS runtime: returns `true`
/// when `bun` is on PATH, otherwise prints the single skip line for `test`
/// and returns `false` so the caller can return early.
pub fn require_bun(test: &str) -> bool {
    if bun_on_path() {
        return true;
    }
    eprintln!(
        "skipping {test}: bun is not on PATH (install bun from https://bun.sh and make the `bun` executable resolve on PATH)"
    );
    false
}

#[cfg(unix)]
fn is_executable(path: std::path::PathBuf) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::metadata(&path)
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: std::path::PathBuf) -> bool {
    std::fs::metadata(&path).is_ok_and(|m| m.is_file())
}
