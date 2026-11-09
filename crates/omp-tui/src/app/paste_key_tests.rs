use super::*;
use crossterm::event::{KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods};
use std::path::Path;

#[test]
fn ctrl_shift_v_csi_u_matches_paste_text_raw() {
    let app = App::new("m".into(), Path::new("/repo"));
    let key = CtKeyEvent::new(CtKeyCode::Char('v'), CtMods::CONTROL | CtMods::SHIFT);
    assert_eq!(
        crate::keybindings::key_id_from_event(&key).as_deref(),
        Some("ctrl+shift+v"),
        "CSI-u ctrl+shift+v canonicalizes to the binding id"
    );
    assert!(app.keys.matches_event(&key, "app.clipboard.pasteTextRaw"));
}
