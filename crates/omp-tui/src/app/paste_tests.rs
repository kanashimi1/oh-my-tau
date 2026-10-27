use super::*;
use crate::clipboard::ClipboardOutcome;
use crossterm::event::{
    Event, KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods,
};
use std::path::Path;
use std::time::Instant;

/// An app pinned to built-in default settings: independent of the
/// process-global `HOME`, which the temp-HOME tests mutate.
fn app() -> App {
    let mut app = App::new("zai/glm-5.3-flash".into(), Path::new("/repo"));
    app.set_settings_global(None);
    // Capture deliveries instead of ringing the real terminal / desktop.
    app.set_notify_hook(Arc::new(|_| {}));
    app
}

fn port() -> crate::port::FakePort {
    crate::port::FakePort::default()
}

fn key(code: CtKeyCode) -> KeyEvent {
    CtKeyEvent::new(code, CtMods::NONE)
}

fn log(port: &crate::port::FakePort) -> Vec<String> {
    port.log.lock().clone()
}

fn shift_ctrl(code: char) -> KeyEvent {
    CtKeyEvent::new(CtKeyCode::Char(code), CtMods::CONTROL | CtMods::SHIFT)
}

#[test]
fn paste_chords_arm_clipboard_read() {
    let mut app = app();
    let port = port();
    let ctrl_v = CtKeyEvent::new(CtKeyCode::Char('v'), CtMods::CONTROL);
    assert!(app.handle_key(ctrl_v, &port, Instant::now()).is_some());
    assert!(app.clipboard_rx.is_some(), "ctrl+v arms the clipboard read");
    app.clipboard_rx = None;
    let csi_u = CtKeyEvent::new(CtKeyCode::Char('v'), CtMods::CONTROL | CtMods::SHIFT);
    assert!(app.handle_key(csi_u, &port, Instant::now()).is_some());
    assert!(
        app.clipboard_rx.is_some(),
        "ctrl+shift+v arms the clipboard read"
    );
}

#[test]
fn bracketed_paste_small_inserts_verbatim() {
    let mut app = app();
    let port = port();
    assert!(app.handle_terminal_event(&Event::Paste("hello\nworld".into()), &port));
    assert_eq!(app.editor.text(), "hello\nworld");
}

#[test]
fn bracketed_paste_large_collapses_to_marker() {
    let mut app = app();
    let port = port();
    let payload = "x".repeat(1200);
    assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
    // TS input-controller stages a marker-sized paste as a text chip
    // (`insertTextAttachment`); the raw-text marker stays the editor
    // default for hosts without the large-paste gate.
    assert_eq!(app.editor.text(), "[Text #1, paste.md]");
}

#[test]
fn bracketed_paste_keeps_payload_verbatim_including_newlines() {
    let mut app = app();
    let port = port();
    // A trailing newline inside the markers is paste content; a real
    // Enter sharing the burst arrives as its own Event::Key (TS
    // `leadingEnter` splits bytes after `\x1b[201~`).
    assert!(app.handle_terminal_event(&Event::Paste("hi there\n".into()), &port));
    assert_eq!(app.editor.text(), "hi there\n");
    assert!(
        !log(&port).iter().any(|l| l.contains("hi there")),
        "no auto-submit"
    );
}

#[test]
fn paste_text_raw_inserts_verbatim_without_collapse() {
    let mut app = app();
    let port = port();
    assert!(app.handle_terminal_event(&Event::Key(shift_ctrl('v')), &port));
    // The read is in flight; nothing inserted yet.
    assert_eq!(app.editor.text(), "");
    app.apply_clipboard_outcome(ClipboardOutcome::TextRaw("a\nb\nc".into()));
    assert_eq!(app.editor.text(), "a\nb\nc");
}

#[test]
fn clipboard_smart_text_routes_through_paste_pipeline() {
    let mut app = app();
    app.apply_clipboard_outcome(ClipboardOutcome::Text("x".repeat(1200)));
    assert_eq!(app.editor.text(), "[Text #1, paste.md]");
    // Discard the pasted chip so the next outcome starts from an empty editor.
    let _ = app.editor.take();
    app.apply_clipboard_outcome(ClipboardOutcome::Empty);
    assert_eq!(app.editor.text(), "", "empty clipboard inserts nothing");
}

#[test]
fn clipboard_image_attaches_chip() {
    let mut app = app();
    app.apply_clipboard_outcome(ClipboardOutcome::Image(b"png".to_vec()));
    assert!(
        app.editor.text().starts_with("[Image #1]"),
        "{}",
        app.editor.text()
    );
    assert_eq!(app.editor.attachments().len(), 1);
}

#[test]
fn large_paste_menu_opens_at_threshold_and_inline_collapses() {
    let mut app = app();
    let port = port();
    let mut cfg_home = std::env::temp_dir();
    cfg_home.push(format!("omp-paste-menu-test-{}", std::process::id()));
    std::fs::create_dir_all(&cfg_home).unwrap();
    // Default settings file location derives from cwd/env; keep the
    // default threshold (100 lines) and use a 120-line paste.
    let payload = "line\n".repeat(119) + "line";
    assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
    assert_eq!(
        app.overlays.open(),
        Some(OpenOverlay::LargePaste),
        "menu gate"
    );
    assert_eq!(
        app.editor.text(),
        "",
        "insert deferred while the menu is open"
    );
    // Choose "Paste inline" (third row) → collapse to the inline marker.
    app.overlays.handle_key(key(CtKeyCode::Down));
    app.overlays.handle_key(key(CtKeyCode::Down));
    app.overlays.handle_key(key(CtKeyCode::Enter));
    app.poll_deadlines(Instant::now(), &port);
    assert_eq!(app.overlays.open(), None);
    assert!(
        app.editor.text().contains("[Paste #1,") || app.editor.text().contains("[Text #1,"),
        "{}",
        app.editor.text()
    );
    let _ = std::fs::remove_dir_all(cfg_home);
}

#[test]
fn large_paste_menu_cancel_stages_text_chip() {
    let mut app = app();
    let port = port();
    let payload = "line\n".repeat(119) + "line";
    assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
    assert_eq!(app.overlays.open(), Some(OpenOverlay::LargePaste));
    // Esc bubbles out of the panel; the host then applies the fallback.
    assert!(app.handle_terminal_event(&Event::Key(key(CtKeyCode::Esc)), &port));
    app.poll_deadlines(Instant::now(), &port);
    assert_eq!(app.overlays.open(), None);
    assert!(
        app.editor.text().starts_with("[Text #1, paste.md]"),
        "{}",
        app.editor.text()
    );
}

#[test]
fn large_paste_below_threshold_stages_chip_without_menu() {
    let mut app = app();
    let port = port();
    // Marker-sized (>10 lines) but under the 100-line menu gate.
    let payload = "line\n".repeat(20);
    assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
    assert_eq!(app.overlays.open(), None, "below gate: no menu");
    assert!(
        app.editor.text().starts_with("[Text #1, paste.md]"),
        "{}",
        app.editor.text()
    );
}
