//! Agent loop for Prime Agent: the loop works with [`types::AgentMessage`]
//! throughout and converts to LLM-bound [`types::Message`] values only at
//! the model call boundary.
#![allow(clippy::disallowed_methods)] // vendored third-party code: the sync-bridge rule applies to omp-rs first-party crates only

pub mod abort;
pub mod agent;
pub mod agent_loop;
pub mod proxy;
pub mod scripted;
pub mod stream;
pub mod types;
pub mod validation;

use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

/// Boxed, sendable future used across the crate's hook and stream traits.
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Unix timestamp in milliseconds, mirroring `Date.now()` in the TS reference.
///
/// # Panics
///
/// Panics if the epoch timestamp exceeds i64 milliseconds — beyond year 292 million.
#[must_use]
// invariant: epoch millis fit in i64 until ~year 292 million; clamping the
// map_or arm already handles pre-epoch clocks.
#[allow(clippy::expect_used)]
pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        i64::try_from(d.as_millis()).expect("millis since epoch fit in i64")
    })
}
