//! `omp-rs <subcommand>` — standalone subcommands operating on data omp-rs
//! already has (model catalog, session usage stats, settings, on-disk session
//! maintenance, session HTML export), mirroring the TS `omp
//! models|stats|config|gc` CLI surface and the `omp --export <file>` launch
//! flag.

pub mod config;
pub mod export;
pub mod gc;
pub mod models;
pub mod stats;

/// Subcommands this module implements.
#[must_use]
pub fn known(name: &str) -> bool {
    matches!(name, "models" | "stats" | "config" | "gc" | "export")
}

/// Dispatch `omp-rs <name> <args>`; `args` excludes the subcommand token.
/// Returns the process exit code.
///
/// # Errors
/// Whatever the subcommand handler reports (parse or data errors).
pub fn run(name: &str, args: &[String]) -> anyhow::Result<i32> {
    match name {
        "models" => models::run(args),
        "stats" => stats::run(args),
        "config" => config::run(args),
        "gc" => gc::run(args),
        "export" => export::run(args),
        _ => anyhow::bail!("unknown subcommand: {name}"),
    }
}
