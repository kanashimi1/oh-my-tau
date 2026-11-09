//! Tau settings, `%LOCALAPPDATA%\Tau\settings.json`: the screen-capture
//! shortcut and the optional WebView2 DevTools port. Edited from the web UI
//! (Appearance → Desktop) through the `tau_settings` / `set_*` commands.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_CAPTURE_SHORTCUT: &str = "Ctrl+Shift+S";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DictationSettings {
    pub enabled: bool,
    pub hold_ms: u64,
    pub language: String,
}

impl Default for DictationSettings {
    fn default() -> Self {
        Self { enabled: true, hold_ms: 250, language: "ru".into() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// Global shortcut for "capture screen → pin editor", e.g. `Ctrl+Alt+S`.
    pub capture_shortcut: String,
    /// `--remote-debugging-port` for every Tau webview (127.0.0.1 only).
    /// Off by default: anything local could drive the UI through it.
    pub devtools_port: Option<u16>,
    pub dictation: DictationSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            capture_shortcut: DEFAULT_CAPTURE_SHORTCUT.to_string(),
            devtools_port: None,
            dictation: DictationSettings::default(),
        }
    }
}

fn path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")?;
    Some(PathBuf::from(base).join("Tau").join("settings.json"))
}

pub fn load() -> Settings {
    path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(settings: &Settings) -> Result<(), String> {
    let p = path().ok_or("LOCALAPPDATA is not set")?;
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(&p, json).map_err(|e| format!("{}: {e}", p.display()))
}

/// DevTools port in effect for this process: `TAU_DEBUG_PORT` wins over the
/// settings file (fixed at startup — WebView2 takes browser args once).
pub fn devtools_port(settings: &Settings) -> Option<u16> {
    std::env::var("TAU_DEBUG_PORT")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .or(settings.devtools_port)
}

/// WebView2 browser arguments shared by every window (all webviews of one
/// user-data folder must use identical arguments). Keeps Tauri's defaults.
pub fn browser_args(port: Option<u16>) -> String {
    let mut args = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection".to_string();
    if let Some(port) = port {
        args.push_str(&format!(" --remote-debugging-port={port}"));
    }
    args
}
