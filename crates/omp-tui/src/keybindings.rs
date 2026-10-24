//! Keybinding registry — ports of `pi-tui/src/keybindings.ts`,
//! `app-keybindings.ts` (defaults + `keybindings.yml` loader) and
//! `key-hint-format.ts` (UI hint formatting).
//!
//! Defaults are identical to omp; user overrides come from
//! `~/.omp/agent/keybindings.yml` (flat map `"binding-id": key | [keys]`,
//! `[]` disables). A `KeybindingsManager` is built once at startup and
//! queried on the hot input path via `matches_event` / `matches_canonical`
//! (precomputed alias sets, no per-keystroke table scan).
//!
//! Key ids are strings like `ctrl+alt+shift+super+<key>`, canonicalized the
//! same way TS `canonicalKeyId` does: modifiers sorted `ctrl,shift,alt,super`,
//! base lowercased, `esc`→`escape`, `return`→`enter`, a bare uppercase letter
//! becomes `shift+<lower>`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// One registered binding: default chords plus a human description.
#[derive(Debug, Clone, Copy)]
pub struct KeybindingDef {
    /// Default key ids (empty = unbound until the user binds it).
    pub default_keys: &'static [&'static str],
    /// Description shown in help / hint UIs.
    pub description: &'static str,
}

const fn def(default_keys: &'static [&'static str], description: &'static str) -> KeybindingDef {
    KeybindingDef {
        default_keys,
        description,
    }
}

/// Base TUI editor/input bindings (TS `TUI_KEYBINDINGS`).
#[rustfmt::skip]
pub static TUI_KEYBINDINGS: &[(&str, KeybindingDef)] = &[
    ("tui.editor.cursorUp", def(&["up"], "Move cursor up")),
    ("tui.editor.cursorDown", def(&["down"], "Move cursor down")),
    ("tui.editor.cursorLeft", def(&["left", "ctrl+b"], "Move cursor left")),
    ("tui.editor.cursorRight", def(&["right", "ctrl+f"], "Move cursor right")),
    ("tui.editor.cursorWordLeft", def(&["alt+left", "ctrl+left", "alt+b"], "Move cursor word left")),
    ("tui.editor.cursorWordRight", def(&["alt+right", "ctrl+right", "alt+f"], "Move cursor word right")),
    ("tui.editor.cursorLineStart", def(&["home", "ctrl+a"], "Move to line start")),
    ("tui.editor.cursorLineEnd", def(&["end", "ctrl+e"], "Move to line end")),
    ("tui.editor.jumpForward", def(&["ctrl+]"], "Jump forward to character")),
    ("tui.editor.jumpBackward", def(&["ctrl+alt+]"], "Jump backward to character")),
    ("tui.editor.pageUp", def(&["pageUp"], "Page up")),
    ("tui.editor.pageDown", def(&["pageDown"], "Page down")),
    ("tui.editor.deleteCharBackward", def(&["backspace"], "Delete character backward")),
    ("tui.editor.deleteCharForward", def(&["delete", "ctrl+d"], "Delete character forward")),
    ("tui.editor.deleteWordBackward", def(&["ctrl+w", "alt+backspace", "ctrl+backspace", "super+alt+backspace"], "Delete word backward")),
    ("tui.editor.deleteWordForward", def(&["alt+delete", "alt+d", "super+alt+delete", "super+alt+d"], "Delete word forward")),
    ("tui.editor.deleteToLineStart", def(&["ctrl+u"], "Delete to line start")),
    ("tui.editor.deleteToLineEnd", def(&["ctrl+k"], "Delete to line end")),
    ("tui.editor.yank", def(&["ctrl+y"], "Yank")),
    ("tui.editor.yankPop", def(&["alt+y"], "Yank pop")),
    ("tui.editor.undo", def(&["ctrl+-", "ctrl+_"], "Undo")),
    ("tui.editor.spellingSuggestions", def(&["ctrl+."], "Show spelling replacements")),
    ("tui.input.newLine", def(&["shift+enter", "ctrl+j"], "Insert newline")),
    ("tui.input.submit", def(&["enter"], "Submit input")),
    ("tui.input.tab", def(&["tab"], "Tab / autocomplete")),
    ("tui.input.copy", def(&["ctrl+c"], "Copy selection")),
    ("tui.select.up", def(&["up"], "Move selection up")),
    ("tui.select.down", def(&["down"], "Move selection down")),
    ("tui.select.pageUp", def(&["pageUp"], "Selection page up")),
    ("tui.select.pageDown", def(&["pageDown"], "Selection page down")),
    ("tui.select.confirm", def(&["enter"], "Confirm selection")),
    ("tui.select.cancel", def(&["escape", "ctrl+c"], "Cancel selection")),
];

/// App-level bindings (TS `app-keybindings.ts` `KEYBINDINGS`, non-TUI half).
#[rustfmt::skip]
pub static APP_KEYBINDINGS: &[(&str, KeybindingDef)] = &[
    ("app.interrupt", def(&["escape"], "Interrupt current operation")),
    ("app.clear", def(&["ctrl+c"], "Clear screen or cancel")),
    ("app.exit", def(&["ctrl+d"], "Exit application")),
    ("app.suspend", def(&["ctrl+z"], "Suspend application")),
    ("app.display.reset", def(&["alt+l"], "Reset terminal display")),
    ("app.thinking.cycle", def(&["shift+tab"], "Cycle thinking level")),
    ("app.thinking.toggle", def(&["ctrl+t"], "Toggle thinking mode")),
    ("app.model.cycleForward", def(&["ctrl+p"], "Cycle to next model")),
    ("app.model.cycleBackward", def(&["shift+ctrl+p"], "Cycle to previous model")),
    ("app.model.select", def(&["alt+m"], "Select model")),
    ("app.model.selectTemporary", def(&["alt+p"], "Select temporary model for current session")),
    ("app.tools.expand", def(&["ctrl+o"], "Expand tools")),
    ("app.tools.toggleVisibility", def(&["ctrl+shift+o"], "Show or hide tool activity")),
    ("app.editor.external", def(&["ctrl+g"], "Open external editor")),
    ("app.message.followUp", def(&["ctrl+q", "ctrl+enter"], "Send follow-up message")),
    ("app.retry", def(&["f5", "alt+r"], "Retry last failed assistant turn")),
    ("app.message.dequeue", def(&["alt+up", "shift+up"], "Dequeue message")),
    ("app.clipboard.pasteImage", def(&["ctrl+v"], "Paste image or text from clipboard")),
    ("app.clipboard.pasteTextRaw", def(&["ctrl+shift+v", "alt+shift+v"], "Paste text from clipboard as raw text (no collapse)")),
    ("app.clipboard.copyLine", def(&["alt+shift+l"], "Copy current line")),
    ("app.clipboard.copyPrompt", def(&["alt+shift+c"], "Copy prompt")),
    ("app.session.new", def(&[], "Create new session")),
    ("app.session.tree", def(&[], "Show session tree")),
    ("app.session.fork", def(&[], "Fork session")),
    ("app.session.resume", def(&[], "Resume session")),
    ("app.agents.hub", def(&["alt+a"], "Open the agent hub")),
    ("app.session.observe", def(&["ctrl+s"], "Open the agent hub")),
    ("app.session.togglePath", def(&["ctrl+p"], "Toggle session path display")),
    ("app.session.toggleSort", def(&["ctrl+s"], "Toggle session sort order")),
    ("app.session.rename", def(&["ctrl+r"], "Rename session")),
    ("app.session.delete", def(&["ctrl+d"], "Delete session")),
    ("app.session.deleteNoninvasive", def(&["ctrl+backspace"], "Delete session (non-invasive)")),
    ("app.tree.foldOrUp", def(&["ctrl+left", "alt+left"], "Fold or move up")),
    ("app.tree.unfoldOrDown", def(&["ctrl+right", "alt+right"], "Unfold or move down")),
    ("app.plan.toggle", def(&["alt+shift+p"], "Toggle plan mode")),
    ("app.history.search", def(&["ctrl+r"], "Search history")),
    ("app.stt.toggle", def(&[], "Toggle speech-to-text (default gesture: hold Space)")),
    ("app.live.toggle", def(&["ctrl+l"], "Start or stop live voice mode (/live)")),
];

/// All definitions (TS `KEYBINDINGS` = `TUI_KEYBINDINGS` + app half).
#[must_use]
pub fn keybindings() -> Vec<(&'static str, KeybindingDef)> {
    let mut all = TUI_KEYBINDINGS.to_vec();
    all.extend_from_slice(APP_KEYBINDINGS);
    all
}

/// Symbols whose `shift+<sym>` alias is registered so a `!` binding also
/// matches when the terminal reports the shifted key with a SHIFT modifier
/// (TS `SHIFTED_SYMBOL_KEYS`).
const SHIFTED_SYMBOL_KEYS: &[&str] = &[
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "{", "}", "|", ":", "<", ">", "?",
    "~",
];

const MODIFIER_ORDER: [&str; 4] = ["ctrl", "shift", "alt", "super"];

fn starts_with_modifier(key: &str, offset: usize, modifier: &str) -> bool {
    let rest = &key.as_bytes()[offset.min(key.len())..];
    if rest.len() <= modifier.len() {
        return false;
    }
    if rest[modifier.len()] != b'+' {
        return false;
    }
    rest[..modifier.len()]
        .iter()
        .zip(modifier.as_bytes())
        .all(|(a, b)| *a == *b || *a == b - 32)
}

fn is_ascii_uppercase_letter(key: &str) -> bool {
    key.len() == 1 && key.as_bytes()[0].is_ascii_uppercase()
}

/// Canonicalize a key id: order modifiers `ctrl,shift,alt,super`, lowercase
/// the base, `esc`→`escape`, `return`→`enter`, bare uppercase letter →
/// `shift+<lower>` (TS `canonicalKeyId`).
#[must_use]
pub fn canonical_key_id(key: &str) -> String {
    let mut offset = 0;
    let mut modifiers: Vec<&str> = Vec::new();
    let mut found = true;
    while found {
        found = false;
        for modifier in MODIFIER_ORDER {
            if starts_with_modifier(key, offset, modifier) {
                modifiers.push(modifier);
                offset += modifier.len() + 1;
                found = true;
                break;
            }
        }
    }
    let raw_base = &key[offset.min(key.len())..];
    let base = match raw_base.to_lowercase().as_str() {
        "esc" => "escape".to_string(),
        "return" => "enter".to_string(),
        lower => lower.to_string(),
    };
    if is_ascii_uppercase_letter(raw_base) && !modifiers.contains(&"shift") {
        modifiers.push("shift");
    }
    if modifiers.is_empty() {
        return base;
    }
    modifiers.sort_by_key(|m| {
        MODIFIER_ORDER
            .iter()
            .position(|o| o == m)
            .unwrap_or(usize::MAX)
    });
    format!("{}+{}", modifiers.join("+"), base)
}

/// Register `key` plus its shifted-symbol aliases into `keys` (TS
/// `addKeyAliases`).
pub fn add_key_aliases(keys: &mut BTreeSet<String>, key: &str) {
    let canonical = canonical_key_id(key);
    if SHIFTED_SYMBOL_KEYS.contains(&canonical.as_str()) {
        keys.insert(format!("shift+{canonical}"));
    }
    keys.insert(canonical);
}

fn normalize_keys(keys: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    keys.iter()
        .map(|k| k.to_lowercase())
        .filter(|k| seen.insert(k.clone()))
        .collect()
}

/// User overrides: binding id → replacement keys (empty = disabled).
pub type UserBindings = BTreeMap<String, Vec<String>>;

/// A chord/id conflict: the same key claimed by more than one binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeybindingConflict {
    /// The canonical key claimed twice.
    pub key: String,
    /// Binding ids claiming it.
    pub bindings: Vec<String>,
}

/// Resolved bindings: id → chords, user overrides replacing (not merging)
/// defaults; per-key conflicts among user overrides detected (TS
/// `KeybindingsManager`).
#[derive(Debug, Clone)]
pub struct KeybindingsManager {
    keys_by_id: BTreeMap<&'static str, Vec<String>>,
    match_keys_by_id: BTreeMap<&'static str, BTreeSet<String>>,
    conflicts: Vec<KeybindingConflict>,
    user: UserBindings,
}

impl KeybindingsManager {
    /// Build from the full definition table and (optional) user overrides.
    #[must_use]
    pub fn new(user: UserBindings) -> Self {
        let definitions = keybindings();

        let mut user_claims: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
        for (id, _) in &definitions {
            if let Some(keys) = user.get(*id) {
                for key in normalize_keys(keys) {
                    user_claims.entry(key).or_default().insert(id);
                }
            }
        }
        let conflicts = user_claims
            .into_iter()
            .filter(|(_, ids)| ids.len() > 1)
            .map(|(key, ids)| KeybindingConflict {
                key,
                bindings: ids.into_iter().map(String::from).collect(),
            })
            .collect();

        let mut keys_by_id = BTreeMap::new();
        let mut match_keys_by_id = BTreeMap::new();
        for (id, definition) in &definitions {
            let keys = match user.get(*id) {
                Some(keys) => normalize_keys(keys),
                None => normalize_keys(
                    &definition
                        .default_keys
                        .iter()
                        .map(|s| (*s).to_string())
                        .collect::<Vec<_>>(),
                ),
            };
            let mut aliases = BTreeSet::new();
            for key in &keys {
                add_key_aliases(&mut aliases, key);
            }
            keys_by_id.insert(*id, keys);
            match_keys_by_id.insert(*id, aliases);
        }

        Self {
            keys_by_id,
            match_keys_by_id,
            conflicts,
            user,
        }
    }

    /// Build with default (no-user-override) bindings.
    #[must_use]
    pub fn defaults() -> Self {
        Self::new(UserBindings::new())
    }

    /// Hot-path probe: `canonical` was computed once for the current
    /// keystroke (TS `matchesCanonical`).
    #[must_use]
    pub fn matches_canonical(&self, canonical: &str, id: &str) -> bool {
        self.match_keys_by_id
            .get(id)
            .is_some_and(|set| set.contains(canonical))
    }

    /// Match a crossterm event against a binding id.
    #[must_use]
    pub fn matches_event(&self, event: &KeyEvent, id: &str) -> bool {
        match key_id_from_event(event) {
            Some(canonical) => self.matches_canonical(&canonical, id),
            None => false,
        }
    }

    /// The resolved keys for `id` (empty when unbound).
    #[must_use]
    pub fn get_keys(&self, id: &str) -> Vec<String> {
        self.keys_by_id.get(id).cloned().unwrap_or_default()
    }

    /// First key for `id`, if any (for hint rendering).
    #[must_use]
    pub fn first_key(&self, id: &str) -> Option<String> {
        self.keys_by_id
            .get(id)
            .and_then(|keys| keys.first().cloned())
    }

    /// Definition metadata for `id`.
    #[must_use]
    pub fn get_definition(&self, id: &str) -> Option<KeybindingDef> {
        keybindings()
            .into_iter()
            .find(|(key, _)| *key == id)
            .map(|(_, def)| def)
    }

    /// Detected user-override conflicts.
    #[must_use]
    pub fn get_conflicts(&self) -> &[KeybindingConflict] {
        &self.conflicts
    }

    /// The user overrides in effect.
    #[must_use]
    pub fn get_user_bindings(&self) -> &UserBindings {
        &self.user
    }

    /// id → resolved keys (single key as a one-element list), definition
    /// order (TS `getResolvedBindings`).
    #[must_use]
    pub fn get_resolved_bindings(&self) -> Vec<(&'static str, Vec<String>)> {
        keybindings()
            .into_iter()
            .map(|(id, _)| {
                let keys = self.keys_by_id.get(id).cloned().unwrap_or_default();
                (id, keys)
            })
            .collect()
    }
}

/// Convert a crossterm event into a canonical key id (`parseKey(data)` +
/// `canonicalKeyId` in one step -- crossterm has already parsed the bytes).
#[must_use]
pub fn key_id_from_event(event: &KeyEvent) -> Option<String> {
    let base = match event.code {
        KeyCode::Char(' ') => "space".to_string(),
        KeyCode::Char(c) => c.to_string(),
        KeyCode::Enter => "enter".to_string(),
        KeyCode::Esc => "escape".to_string(),
        KeyCode::Tab => "tab".to_string(),
        KeyCode::BackTab => "tab".to_string(),
        KeyCode::Backspace => "backspace".to_string(),
        KeyCode::Delete => "delete".to_string(),
        KeyCode::Insert => "insert".to_string(),
        KeyCode::Home => "home".to_string(),
        KeyCode::End => "end".to_string(),
        KeyCode::PageUp => "pageUp".to_string(),
        KeyCode::PageDown => "pageDown".to_string(),
        KeyCode::Up => "up".to_string(),
        KeyCode::Down => "down".to_string(),
        KeyCode::Left => "left".to_string(),
        KeyCode::Right => "right".to_string(),
        KeyCode::F(n) => format!("f{n}"),
        KeyCode::KeypadBegin => "begin".to_string(),
        _ => return None,
    };
    let mut prefixed = String::new();
    let mods = event.modifiers;
    if mods.contains(KeyModifiers::CONTROL) {
        prefixed.push_str("ctrl+");
    }
    // Shift is already expressed by an uppercase char base for plain typed
    // bytes; an explicit SHIFT flag (CSI-u / modifyOtherKeys encodings, e.g.
    // `ctrl+shift+v` = `\x1b[118;6u`) must still surface — TS sees the same
    // press as an uppercase base (`ctrl+V` → `ctrl+shift+v`).
    if mods.contains(KeyModifiers::SHIFT) {
        prefixed.push_str("shift+");
    }
    if mods.contains(KeyModifiers::ALT) {
        prefixed.push_str("alt+");
    }
    if mods.contains(KeyModifiers::SUPER) {
        prefixed.push_str("super+");
    }
    prefixed.push_str(&base);
    Some(canonical_key_id(&prefixed))
}

/// The user overrides file (TS `app-keybindings.ts` loader): the flat map
/// `~/.omp/agent/keybindings.yml` (`OMP_AGENT_DIR` when set), where each
/// value is a key id, a list of key ids, or `[]`/`null` to disable the
/// binding.
#[must_use]
pub fn user_keybindings_path() -> PathBuf {
    let mut p = std::env::var_os("OMP_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
            home.push(".omp");
            home.push("agent");
            home
        });
    p.push("keybindings.yml");
    p
}

/// Load user overrides from `path`. Missing file → empty map (no overrides).
/// A malformed file yields an error string (the caller surfaces it as a
/// notice and keeps the defaults, like TS which logs and continues).
pub fn load_user_bindings(path: &Path) -> Result<UserBindings, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(UserBindings::new()),
        Err(e) => return Err(format!("{path:?}: {e}")),
    };
    let raw: serde_yaml_ng::Value =
        serde_yaml_ng::from_str(&text).map_err(|e| format!("{path:?}: {e}"))?;
    let mapping = raw
        .as_mapping()
        .ok_or_else(|| format!("{path:?}: expected a flat key→keys mapping"))?;
    let mut out = UserBindings::new();
    for (k, v) in mapping {
        let Some(id) = k.as_str() else { continue };
        let keys: Vec<String> = match v {
            serde_yaml_ng::Value::Null => Vec::new(),
            serde_yaml_ng::Value::String(s) => vec![s.clone()],
            serde_yaml_ng::Value::Sequence(items) => items
                .iter()
                .filter_map(|i| i.as_str().map(str::to_owned))
                .collect(),
            _ => continue,
        };
        out.insert(id.to_owned(), keys);
    }
    Ok(out)
}

/// The startup manager: user overrides from [`user_keybindings_path`], then
/// defaults. Unreadable/invalid override files are ignored (defaults win) —
/// the app shows the parse error as a system notice at startup.
#[must_use]
pub fn load_default_manager() -> KeybindingsManager {
    KeybindingsManager::new(load_user_bindings(&user_keybindings_path()).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent as CtKeyEvent, KeyModifiers as CtMods};

    fn ev(code: KeyCode, mods: KeyModifiers) -> CtKeyEvent {
        CtKeyEvent::new(code, mods)
    }

    // ---- canonical ids (TS canonicalKeyId) -------------------------------

    #[test]
    fn canonicalizes_modifier_order_and_aliases() {
        assert_eq!(canonical_key_id("esc"), "escape");
        assert_eq!(canonical_key_id("return"), "enter");
        assert_eq!(canonical_key_id("ctrl+shift+p"), "ctrl+shift+p");
        assert_eq!(canonical_key_id("shift+ctrl+p"), "ctrl+shift+p");
        assert_eq!(canonical_key_id("Ctrl+Alt+Right"), "ctrl+alt+right");
        assert_eq!(canonical_key_id("A"), "shift+a");
        assert_eq!(canonical_key_id("pageup"), "pageup");
    }

    #[test]
    fn event_to_canonical_id() {
        assert_eq!(
            key_id_from_event(&ev(KeyCode::Esc, CtMods::NONE)),
            Some("escape".into())
        );
        assert_eq!(
            key_id_from_event(&ev(KeyCode::Char('c'), CtMods::CONTROL)),
            Some("ctrl+c".into())
        );
        assert_eq!(
            key_id_from_event(&ev(KeyCode::Char('D'), CtMods::SHIFT)),
            Some("shift+d".into())
        );
        assert_eq!(
            key_id_from_event(&ev(KeyCode::Up, CtMods::ALT)),
            Some("alt+up".into())
        );
        assert_eq!(
            key_id_from_event(&ev(KeyCode::Enter, CtMods::CONTROL)),
            Some("ctrl+enter".into())
        );
    }

    // ---- defaults reproduce the app's hard-coded input path --------------

    #[test]
    fn defaults_reproduce_app_input_bindings() {
        let km = KeybindingsManager::defaults();
        assert!(km.matches_event(&ev(KeyCode::Esc, CtMods::NONE), "app.interrupt"));
        assert!(km.matches_event(&ev(KeyCode::Char('c'), CtMods::CONTROL), "app.clear"));
        assert!(km.matches_event(&ev(KeyCode::Char('d'), CtMods::CONTROL), "app.exit"));
        assert!(km.matches_event(&ev(KeyCode::Char('o'), CtMods::CONTROL), "app.tools.expand"));
        assert!(km.matches_event(&ev(KeyCode::Up, CtMods::ALT), "app.message.dequeue"));
        assert!(km.matches_event(&ev(KeyCode::Up, CtMods::SHIFT), "app.message.dequeue"));
        assert!(!km.matches_event(&ev(KeyCode::Char('x'), CtMods::NONE), "app.exit"));
        assert!(!km.matches_event(&ev(KeyCode::Esc, CtMods::NONE), "app.exit"));
    }

    #[test]
    fn shifted_symbol_alias_matches_terminal_reports() {
        let mut user = UserBindings::new();
        user.insert("app.tools.expand".into(), vec!["?".into()]);
        let km = KeybindingsManager::new(user);
        assert!(km.matches_event(&ev(KeyCode::Char('?'), CtMods::SHIFT), "app.tools.expand"));
        assert!(km.matches_event(&ev(KeyCode::Char('?'), CtMods::NONE), "app.tools.expand"));
        // Unchanged bindings keep their defaults.
        assert!(km.matches_event(&ev(KeyCode::Esc, CtMods::NONE), "app.interrupt"));
    }

    // ---- user overrides ---------------------------------------------------

    #[test]
    fn user_override_replaces_default_keys() {
        let mut user = UserBindings::new();
        user.insert(
            "app.exit".into(),
            vec!["ctrl+q".into(), "ctrl+alt+q".into()],
        );
        let km = KeybindingsManager::new(user);
        assert!(km.matches_event(&ev(KeyCode::Char('q'), CtMods::CONTROL), "app.exit"));
        assert!(!km.matches_event(&ev(KeyCode::Char('d'), CtMods::CONTROL), "app.exit"));
        assert_eq!(km.get_keys("app.exit"), vec!["ctrl+q", "ctrl+alt+q"]);
    }

    #[test]
    fn empty_override_disables_binding() {
        let mut user = UserBindings::new();
        user.insert("app.exit".into(), Vec::new());
        let km = KeybindingsManager::new(user);
        assert!(!km.matches_event(&ev(KeyCode::Char('d'), CtMods::CONTROL), "app.exit"));
        assert!(km.get_keys("app.exit").is_empty());
    }

    #[test]
    fn detects_user_override_conflicts() {
        let mut user = UserBindings::new();
        user.insert("app.exit".into(), vec!["ctrl+q".into()]);
        user.insert("app.clear".into(), vec!["ctrl+q".into()]);
        let km = KeybindingsManager::new(user);
        assert_eq!(km.get_conflicts().len(), 1);
        assert_eq!(km.get_conflicts()[0].key, "ctrl+q");
        assert!(km.get_conflicts()[0].bindings.len() == 2);
    }

    #[test]
    fn ignores_unknown_binding_ids() {
        let mut user = UserBindings::new();
        user.insert("app.nonexistent".into(), vec!["ctrl+q".into()]);
        let km = KeybindingsManager::new(user);
        assert!(km.get_conflicts().is_empty());
        assert!(km.get_keys("app.nonexistent").is_empty());
    }

    // ---- yml loader -------------------------------------------------------

    fn write_temp_yaml(contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "omp-kb-test-{}-{}.yml",
            std::process::id(),
            contents.len()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn loads_flat_yaml_overrides() {
        let path = write_temp_yaml(
            "app.exit: ctrl+q\napp.clear: [ctrl+c, ctrl+shift+c]\napp.model.select: []\n",
        );
        let user = load_user_bindings(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            user.get("app.exit").map(Vec::as_slice),
            Some(&["ctrl+q".to_owned()][..])
        );
        assert_eq!(user.get("app.clear").map(Vec::len), Some(2));
        assert_eq!(user.get("app.model.select").map(Vec::len), Some(0));
    }

    #[test]
    fn missing_yaml_means_no_overrides() {
        let user = load_user_bindings(&PathBuf::from("/nonexistent/omp-keybindings.yml")).unwrap();
        assert!(user.is_empty());
    }

    #[test]
    fn malformed_yaml_is_an_error_not_a_panic() {
        let path = write_temp_yaml("app.exit: [unclosed\n  - x");
        let err = load_user_bindings(&path);
        std::fs::remove_file(&path).ok();
        assert!(err.is_err());
    }
}
