//! `omp-rs <subcommand>` — standalone subcommands operating on data omp-rs
//! already has (model catalog, session usage stats, settings, on-disk session
//! maintenance, session HTML export, provider login/logout), mirroring the TS `omp
//! models|stats|config|gc` CLI surface and the `omp --export <file>` launch
//! flag.

pub mod collab;
pub mod config;
pub mod export;
pub mod gc;
pub mod judge_sessions;
pub mod login;
pub mod models;
pub mod serve_collab;
pub mod stats;
pub mod stats_judge;
pub mod usage_broker;
pub mod web;

/// Subcommands this module implements.
#[must_use]
pub fn known(name: &str) -> bool {
    matches!(
        name,
        "models"
            | "stats"
            | "config"
            | "gc"
            | "export"
            | "judge-sessions"
            | "login"
            | "logout"
            | "collab"
            | "serve-collab"
            | "web"
    )
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
        "judge-sessions" => judge_sessions::run(args),
        "login" => login::run_login(args),
        "logout" => login::run_logout(args),
        "collab" => collab::run(args),
        "serve-collab" => serve_collab::run(args),
        "web" => web::run(args),
        _ => anyhow::bail!("unknown subcommand: {name}"),
    }
}
