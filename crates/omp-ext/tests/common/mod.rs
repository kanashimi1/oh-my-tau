//! Prerequisite checks for the omp-ext integration tests.
//!
//! Every test in this crate drives the real extension host (`bun
//! ext-host/host.ts`), and the `api.typebox`/`api.zod` surface additionally
//! needs the installed omp TS package. When a prerequisite is missing a test
//! prints ONE `skipping <test>: ...` line to stderr and returns early, so a
//! fresh clone without `bun` still gets a green `cargo test -p omp-ext`;
//! where the prerequisites exist (the owner's machine) the full suite always
//! runs — nothing is skipped silently.

use std::path::{Path, PathBuf};

/// `bun` is resolvable on PATH: some PATH entry holds an executable `bun`
/// file. Nothing is spawned to probe.
fn bun_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| is_executable(&dir.join("bun")))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.is_file()
        && std::fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// The omp TS package root, resolved exactly like `ext-host/host.ts` resolves
/// it for `api.typebox`/`api.zod`: `OMP_PKG_ROOT`, else
/// `$HOME/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent`.
// The omp-pkg helpers are only exercised by the host tests; the lifecycle
// test binaries compile this module without using them.
#[allow(dead_code)]
fn omp_pkg_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("OMP_PKG_ROOT") {
        return Some(PathBuf::from(root));
    }
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".bun/install/global/node_modules/@oh-my-pi/pi-coding-agent")
    })
}

/// Print the skip line for `test` and report whether to return early:
/// `if bun_missing("<test>") { return; }`.
pub fn bun_missing(test: &str) -> bool {
    if bun_on_path() {
        return false;
    }
    eprintln!("skipping {test}: bun not found on PATH (install it from https://bun.sh)");
    true
}

/// Same contract as [`bun_missing`] for the installed omp TS package backing
/// `api.typebox`/`api.zod`: `if !omp_pkg_missing("<test>") { ... }` guards just
/// the assertions that need it, so the rest of the test still runs.
#[allow(dead_code)]
pub fn omp_pkg_missing(test: &str) -> bool {
    match omp_pkg_root() {
        Some(root) if root.is_dir() => false,
        Some(root) => {
            eprintln!(
                "skipping {test}: omp TS package @oh-my-pi/pi-coding-agent not found at {} (install with `bun i -g @oh-my-pi/pi-coding-agent` or point OMP_PKG_ROOT at it)",
                root.display()
            );
            true
        }
        None => {
            eprintln!(
                "skipping {test}: omp TS package @oh-my-pi/pi-coding-agent not found (set OMP_PKG_ROOT to its install dir)"
            );
            true
        }
    }
}
