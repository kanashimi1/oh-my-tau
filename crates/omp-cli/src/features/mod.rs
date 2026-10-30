//! Runtime-backed feature commands (`/btw`, …): behaviors that need the live
//! agent/model beyond plain text handling. Each feature owns one module with a
//! `command(rt, args) -> Outcome` entry, wired from `slash_extra::dispatch_extra`.

pub mod budget;
pub mod btw;
pub mod refine;
pub mod loop_mode;
pub mod plan;
pub mod restart;
pub mod skills;
pub mod browser;
pub mod sandbox;
pub mod share;
pub mod shake;
pub mod ssh;
pub mod worktree;
