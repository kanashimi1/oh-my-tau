//! Internal URL layer: parsed URLs, scheme specs, the provider trait,
//! and a process-global registry the read/write tools route through.
//!
//! Port of `src/internal-urls` (router, parse, types) from the TS agent.

use std::path::{Path, PathBuf};
use std::sync::RwLock;

pub mod parse;
pub mod registry;
pub mod schemes;
pub mod selectors;
pub mod spec;

pub use parse::{extract_uri_scheme, parse_internal_url, InternalUrl};
pub use registry::{Registry, UrlTarget};
pub use selectors::{split_selectors, split_url_scheme};
pub use spec::{
    InternalResource, SchemeBacking, SchemeSelectors, SchemeSpec, UrlCompletion, WritePolicy,
    WriteScope, WriteVia,
};

/// Caller-supplied context threaded into providers (TS `ResolveContext`/`WriteContext`).
#[derive(Debug, Clone, Default)]
pub struct UrlCtx {
    /// Current session file (`<name>.jsonl`); session-scoped schemes
    /// (local/artifact/agent/history) derive their stores from it.
    pub session_file: Option<PathBuf>,
    /// Working directory of the calling tool (for relative-path fallbacks).
    pub cwd: Option<PathBuf>,
    /// Return directory shape without materializing an expensive listing.
    pub skip_directory_listing: bool,
}

impl UrlCtx {
    /// The session's artifacts directory: the session file path minus its
    /// `.jsonl` suffix (TS `artifactsDirectoryFor`).
    pub fn artifacts_dir(&self) -> Option<PathBuf> {
        let file = self.session_file.as_ref()?;
        let s = file.to_str()?;
        s.strip_suffix(".jsonl").map(PathBuf::from)
    }

    /// The session id: the session file's file stem (TS `getSessionId`).
    pub fn session_id(&self) -> Option<String> {
        self.session_file
            .as_ref()
            .and_then(|f| f.file_stem())
            .and_then(|s| s.to_str())
            .map(ToOwned::to_owned)
    }
}

static SESSION_FILE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Install the process-global session file so tools can default to it
/// (TS `AgentRegistry` first-main-session fallback).
pub fn set_global_session_file(path: Option<PathBuf>) {
    *SESSION_FILE.write().expect("session file lock") = path;
}

/// The installed global session file, if any.
pub fn global_session_file() -> Option<PathBuf> {
    SESSION_FILE.read().expect("session file lock").clone()
}

/// A provider for one internal URL scheme (TS `ProtocolHandler`).
pub trait UrlProvider: Send + Sync {
    /// Scheme this provider owns, e.g. `"local"` (lowercase, no `://`).
    fn scheme(&self) -> &'static str;

    /// Declared facts about the scheme.
    fn spec(&self) -> &SchemeSpec;

    /// Whether `write` is implemented (Rust cannot probe method presence;
    /// providers with `write` return true — checked by [`Registry::register`]).
    fn impl_write(&self) -> bool {
        false
    }

    /// Whether `locate` is implemented.
    fn impl_locate(&self) -> bool {
        false
    }

    /// Resolve the URL to a resource.
    ///
    /// # Errors
    /// Resolution failure with the scheme's canonical error message.
    fn read(&self, url: &InternalUrl, ctx: &UrlCtx) -> anyhow::Result<InternalResource>;

    /// Write `content` to the URL. Default: unsupported.
    ///
    /// # Errors
    /// Read-only scheme or write failure.
    fn write(&self, url: &InternalUrl, content: &str, ctx: &UrlCtx) -> anyhow::Result<()> {
        let _ = (url, content, ctx);
        anyhow::bail!(
            "{}:// URLs are read-only for write; use the protocol-specific tool for mutations.",
            self.scheme()
        )
    }

    /// Locate the URL on the local filesystem, when it is file-backed.
    fn locate(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        let _ = (url, ctx);
        None
    }

    /// Locate for writes: returns the lexical path a write should create,
    /// after proving creation stays inside the scheme root (TS `locate` with
    /// `options.create`). Defaults to [`Self::locate`].
    fn locate_create(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        self.locate(url, ctx)
    }

    /// Synchronous locate for renderers; only linkable schemes.
    fn locate_sync(&self, url: &InternalUrl, ctx: &UrlCtx) -> Option<PathBuf> {
        let _ = (url, ctx);
        None
    }

    /// Autocomplete candidates for the host/path portion of `scheme://<query>`.
    fn complete(&self, query: &str, ctx: &UrlCtx) -> Vec<UrlCompletion> {
        let _ = (query, ctx);
        Vec::new()
    }

    /// Prompt documentation line for this scheme (system prompt listing).
    fn prompt_doc(&self) -> Option<String> {
        None
    }
}

/// The artifacts directory for a session file (`<file>.jsonl` → `<file>/`),
/// or `None` when the file has no `.jsonl` suffix.
pub fn artifacts_directory_for(session_file: &Path) -> Option<PathBuf> {
    let s = session_file.to_str()?;
    s.strip_suffix(".jsonl").map(PathBuf::from)
}
