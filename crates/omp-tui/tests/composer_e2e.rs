#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! First headless e2e over the [`omp_tui::controller::run_headless`] seam:
//! startup renders welcome + composer, typed keys land in the composer
//! draft, the frame settles, and the `app.exit` chord ends `run()` with
//! `Ok`. No tty, no provider, no network — CI-safe.
//!
//! Exit is ctrl+d, not Esc: Esc is `app.interrupt`, and while idle its
//! ladder step is a no-op that keeps the draft (`app.rs` `esc_ladder`), so
//! it can never end the loop.

mod support;

use crossterm::event::{KeyCode, KeyModifiers};
use support::TuiHarness;

/// `provider/id` for the status line; spelled to collide with nothing the
/// startup frame draws (welcome wordmark, tips, placeholder).
const MODEL: &str = "bench-mock/model-x";

#[tokio::test]
async fn composer_typing_lands_in_the_frame_and_ctrl_d_exits() {
    let mut harness = TuiHarness::new(MODEL, 80, 24);
    harness.start();

    // Startup settles on the initial frame: the welcome is committed above
    // the live region and the empty composer + status band render in it.
    let frame = harness.wait_rendered(2).await;
    let text = support::buffer_text(&frame);
    assert!(text.contains(MODEL), "startup status line:\n{text}");
    assert!(!text.contains("hello"), "composer starts empty:\n{text}");

    harness.type_text("hello");
    harness.wait_settled().await;
    let text = harness.frame_text();
    assert!(
        text.contains("hello"),
        "composer draft after typing:\n{text}"
    );
    assert!(text.contains(MODEL), "status line still present:\n{text}");
    let calls = harness.port_calls();
    assert!(
        calls.is_empty(),
        "typing must not dispatch to the runtime: {calls:?}"
    );

    // The runtime double's inbound direction: one frame through
    // `RunConfig::frames` must surface as a system row in the transcript.
    harness.send_frame(serde_json::json!({
        "type": "notice",
        "level": "info",
        "message": "runtime-double check"
    }));
    harness.wait_settled().await;
    let text = harness.frame_text();
    assert!(
        text.contains("runtime-double check"),
        "runtime frame must surface in the transcript:\n{text}"
    );

    // Esc while idle is a no-op (the interrupt ladder keeps the draft); the
    // loop must survive it.
    harness.push_key(KeyCode::Esc, KeyModifiers::NONE);
    harness.wait_settled().await;

    let result = harness.quit().await;
    assert!(result.is_ok(), "run() must exit cleanly: {result:?}");
}
