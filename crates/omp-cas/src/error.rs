//! Error type shared by the store and the pack codec.

use std::path::{Path, PathBuf};

/// Result alias for the whole crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong in a CAS operation. Library code returns
/// these instead of panicking; callers wrap in `anyhow` at the edges.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A referenced object (pack root, dependency or explicit lookup) is in
    /// neither tier of the store, and pack import could not satisfy it.
    #[error("missing object {digest}")]
    MissingObject {
        /// Lowercase-hex digest that could not be resolved.
        digest: String,
    },
    /// Stored bytes (or a pack payload) do not hash to their declared
    /// digest. Reads verify every object, so this is the corruption signal.
    #[error("corrupt object {digest}: {reason}")]
    Corrupt {
        /// Lowercase-hex digest whose bytes failed verification.
        digest: String,
        /// What mismatched.
        reason: String,
    },
    /// A digest string was not exactly 64 lowercase hex chars. Strict
    /// validation is what makes digest-derived paths traversal-safe.
    #[error("invalid digest {input:?}: not 64 lowercase hex chars")]
    InvalidDigest {
        /// The rejected string, truncated for display.
        input: String,
    },
    /// A configured resource bound (budget, entry count, byte cap) was hit.
    #[error("resource limit exceeded: {0}")]
    Limit(&'static str),
    /// The pack container is malformed (bad magic, truncated, codec
    /// mismatch, manifest inconsistent with its payload).
    #[error("pack format: {0}")]
    Format(&'static str),
    /// Filesystem failure, with the path that failed.
    #[error("io error at {}: {source}", path.display())]
    Io {
        /// Path the failed syscall touched.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
}

pub(crate) fn io_err(path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.to_path_buf(),
        source,
    }
}
