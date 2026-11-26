//! SHA-256 content digests.
//!
//! Same encoding as the existing digest surfaces this store is meant to
//! interoperate with: lowercase hex in [`omp-hub`](https://docs.rs) `PayloadRef.sha256`
//! and the omp-collab replication hashes. Strict lowercase-only parsing is a
//! security property here: every on-disk path in the store is derived from a
//! parsed digest, so non-hex input (including `..`/absolute paths) can never
//! reach the filesystem.

use std::fmt;

use sha2::{Digest as _, Sha256};

use crate::error::Error;

/// Hash `bytes` with SHA-256.
#[must_use]
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// A validated SHA-256 digest. Ordered by raw bytes; `hex()` is canonical.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    /// Digest of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(sha256(bytes))
    }

    /// Wrap an already-hashed value.
    #[must_use]
    pub const fn from_raw(raw: [u8; 32]) -> Self {
        Self(raw)
    }

    /// Strict parse: exactly 64 lowercase hex characters. Uppercase, short,
    /// long and non-hex input (including anything resembling a path) are
    /// rejected, so parsed digests are safe to embed in file names.
    ///
    /// # Errors
    /// [`Error::InvalidDigest`] when the string is not canonical.
    pub fn parse(input: &str) -> crate::Result<Self> {
        let bytes = input.as_bytes();
        if bytes.len() != 64
            || !bytes
                .iter()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(Error::InvalidDigest {
                input: truncate_for_error(input),
            });
        }
        let mut raw = [0_u8; 32];
        // Length and alphabet checked above; hex can only fail on those.
        hex::decode_to_slice(input, &mut raw)
            .map_err(|_| Error::InvalidDigest { input: truncate_for_error(input) })?;
        Ok(Self(raw))
    }

    /// Canonical lowercase-hex form (64 chars) — the on-disk and wire form,
    /// matching `PayloadRef.sha256`.
    #[must_use]
    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Raw 32 bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({})", self.hex())
    }
}

/// Keep hostile input out of error messages/paths.
fn truncate_for_error(input: &str) -> String {
    const MAX: usize = 72;
    if input.len() <= MAX {
        input.to_owned()
    } else {
        let mut cut = MAX;
        while !input.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}…", &input[..cut])
    }
}
