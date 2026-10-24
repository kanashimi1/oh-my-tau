//! `omp-lsp-mux`: the broker-shared LSP mux daemon binary (omp
//! `src/lsp/mux/server.ts` `startLspMuxFromEnvironment`). Started detached by
//! `omp-lsp`'s mux client side on first shared use; exits after an idle
//! grace with no connected sessions.

use std::path::PathBuf;

use omp_lsp::mux::protocol::{LSP_MUX_PROJECT_DIR_ENV, LSP_MUX_SOCKET_ENV};

fn main() {
    let endpoint = std::env::var_os(LSP_MUX_SOCKET_ENV).map(PathBuf::from);
    let project_dir = std::env::var_os(LSP_MUX_PROJECT_DIR_ENV).map(PathBuf::from);
    let (Some(endpoint), Some(project_dir)) = (endpoint, project_dir) else {
        eprintln!("LSP mux environment is incomplete: set {LSP_MUX_SOCKET_ENV} and {LSP_MUX_PROJECT_DIR_ENV}");
        std::process::exit(2);
    };
    // The mux child servers pin `processId` to us; do not leak the env on.
    std::env::remove_var(LSP_MUX_SOCKET_ENV);
    std::env::remove_var(LSP_MUX_PROJECT_DIR_ENV);
    // One thread is enough: the daemon is a socket accept loop plus async
    // pipe pump tasks, and a lean runtime keeps the per-project daemon's
    // own footprint small (it is charged to every shared session).
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .thread_name("omp-lsp-mux")
        .build()
        .expect("build mux runtime");
    if let Err(error) = runtime.block_on(omp_lsp::mux::server::run(&endpoint, &project_dir)) {
        eprintln!("omp lsp mux failed on {}: {error}", endpoint.display());
        std::process::exit(1);
    }
}
