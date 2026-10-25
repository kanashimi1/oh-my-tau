//! omt feature commands. Each module exposes `command(rt, args)`, wired from
//! `rpc/slash.rs` (`dispatch_ported`) and `slash_extra.rs` (`dispatch_extra`).

pub mod plan;
pub mod worktree;
pub mod btw;
pub mod loop_mode;
pub mod shake;
pub mod restart;
pub mod skills;
pub mod share;
pub mod ssh;
pub mod browser;
pub mod budget;
pub mod sandbox;
