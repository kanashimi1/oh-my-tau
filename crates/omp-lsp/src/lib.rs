//! LSP support for omp-rs: a port of omp's `src/lsp` (v18.6.1).
//!
//! - [`config`]: server discovery/merging from `defaults.json` + user/project
//!   config files (`lsp.json`/`lsp.yaml` + hidden variants), root-marker gating
//!   and binary resolution.
//! - [`client`]: JSON-RPC 2.0 over stdio language-server client (initialize
//!   handshake, didOpen/didChange/didSave, publishDiagnostics tracking,
//!   pending-request routing, graceful shutdown, idle reaping).
//! - [`servers`]: server selection helpers, read-only action set, reload.
//! - [`diagnostics`]: publish/pull diagnostic waiting, glob target resolution,
//!   symbol-column resolution.
//! - [`edits`]: workspace-edit application (text edits, resource operations,
//!   rename-with-rollback).
//! - [`mux`]: broker-shared language servers — one mux daemon per project
//!   multiplexes every session over one server process per identity key.
//! - [`utils`]: URI conversion, output formatting, language-id detection.
//!
//! Cancellation is injected through [`Cancel`] so the crate stays independent
//! of the agent runtime; `omp-cli` implements it for `pa_agent`'s `AbortSignal`.

use std::future::Future;
use std::pin::Pin;

pub mod client;
pub mod config;
pub mod diagnostics;
pub mod edits;
pub mod mux;
pub mod servers;
pub mod types;
pub mod utils;

/// Cancellation handle threaded through every potentially blocking LSP call.
///
/// Mirrors the TS `AbortSignal` usage in `src/lsp`: an aborted signal rejects
/// the in-flight request (sending `$/cancelRequest`) instead of leaving a
/// pending entry behind.
pub trait Cancel: Send + Sync {
    fn is_cancelled(&self) -> bool;
    /// Resolves when the handle is cancelled.
    fn cancelled(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

/// Never-cancelled handle (TS `undefined` signal).
#[derive(Debug, Clone, Copy, Default)]
pub struct NeverCancel;

impl Cancel for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
    fn cancelled(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(std::future::pending())
    }
}

/// Wall-clock deadline plus optional caller cancellation.
#[derive(Clone, Copy, Default)]
pub struct Deadline<'a> {
    pub cancel: Option<&'a dyn Cancel>,
    pub timeout: Option<std::time::Duration>,
}

impl std::fmt::Debug for Deadline<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Deadline")
            .field("cancel", &self.cancel.map(|_| ".."))
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl<'a> Deadline<'a> {
    #[must_use]
    pub fn new(cancel: Option<&'a dyn Cancel>, timeout: Option<std::time::Duration>) -> Self {
        Self { cancel, timeout }
    }
}
