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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};

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
    KeybindingDef { default_keys, description }
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
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "{", "}", "|", ":", "<", ">",
    "?", "~",
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
        lower => lower,
    };
    if is_ascii_uppercase_letter(raw_base) && !modifiers.contains(&"shift") {
        modifiers.push("shift");
    }
    if modifiers.is_empty() {
        return base;
    }
    modifiers.sort_by_key(|m| MODIFIER_ORDER.iter().position(|o| o == m).unwrap_or(usize::MAX));
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

        Self { keys_by_id, match_keys_by_id, conflicts, user }
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
        self.keys_by_id.get(id).and_then(|keys| keys.first().cloned())
    }

    /// Definition metadata for `id`.
    #[must_use]
    pub fn get_definition(&self, id: &str) -> Option<KeybindingDef> {
        keybindings().into_iter().find(|(key, _)| *key == id).map(|(_, def)| def)
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
    // Shift is already expressed by an uppercase char base; add the explicit
    // prefix only for non-char keys (BackTab handled above as plain `tab`).
    if mods.contains(KeyModifiers::SHIFT) && !event.code.is_char() {
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
