//! The single permitted sync→async bridge in library code.
//!
//! [`block_on`] exists for the few genuinely sync call sites that cannot be
//! made async — traits whose implementors are invoked from non-runtime
//! threads (e.g. `omp_eval::HostBridge::call`, driven by the eval kernel's
//! `spawn_blocking` cell loop). New call sites must not be added: prefer
//! making the caller async, or gate sync tools through `spawn_blocking` and
//! `.await` them.

/// Drive `fut` to completion from a thread that is *not* a Tokio runtime
/// worker thread.
///
/// This is only valid when the calling thread cannot be a worker of the
/// runtime that drives `fut` — blocking a runtime worker on its own reactor
/// deadlocks. The eval kernel invokes [`omp_eval::HostBridge::call`] from its
/// `spawn_blocking` cell loop, which satisfies that contract. A process
/// lifetime fallback runtime covers callers with no ambient runtime (tests).
// invariant: the fallback runtime only fails to build on resource exhaustion.
#[allow(clippy::expect_used)]
pub(crate) fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        // `Handle::block_on` on a `spawn_blocking` thread parks only that
        // thread while the runtime keeps driving `fut`.
        Ok(handle) =>
        {
            #[allow(clippy::disallowed_methods)]
            handle.block_on(fut)
        }
        Err(_no_runtime) => {
            static RUNTIME: std::sync::LazyLock<tokio::runtime::Runtime> =
                std::sync::LazyLock::new(|| {
                    tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("sync bridge runtime")
                });
            #[allow(clippy::disallowed_methods)]
            RUNTIME.block_on(fut)
        }
    }
}
