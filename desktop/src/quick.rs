//! Quick ask (feature 5): a global Ctrl+Shift+Space shortcut toggles a small
//! always-on-top "quick" window (the omt service with `?quick=1`), plus the
//! `quick_capture` (screen → PNG data URL) and `continue_main` (hand the quick
//! agent to the main window) commands used by assets/quick.js.
//!
//! Parent wiring (lib.rs, not edited here):
//!   * `mod quick;` next to `mod settings;` and a `quick::register(app.handle())`
//!     call in `setup` (after `setup_global_shortcut` — both shortcuts coexist);
//!   * `quick::quick_toggle`, `quick::quick_capture`, `quick::continue_main`
//!     appended to `invoke_handler`.
//! Requires `SERVICE_URL` to be visible to this module (`pub(crate) const`).

use base64::Engine as _;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};

const QUICK_LABEL: &str = "quick";

/// Register the global quick-ask shortcut (Ctrl+Shift+Space). Registers a
/// second shortcut alongside the existing capture hotkey; neither replaces the
/// other. Failures are logged, never fatal.
pub fn register(app: &AppHandle) {
    use tauri_plugin_global_shortcut::{
        Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
    };
    let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
    let result = app
        .global_shortcut()
        .on_shortcut(shortcut, |app, _sc, event| {
            if event.state() == ShortcutState::Pressed {
                if let Err(e) = toggle(app) {
                    super::log_line(&format!("quick toggle: {e}"));
                }
            }
        });
    if let Err(e) = result {
        super::log_line(&format!("quick shortcut Ctrl+Shift+Space: {e}"));
    }
}

/// Show the quick window (creating it on first use), or hide it when visible.
fn toggle(app: &AppHandle) -> Result<(), String> {
    if let Some(win) = app.get_webview_window(QUICK_LABEL) {
        if win.is_visible().unwrap_or(false) {
            win.hide().map_err(|e| e.to_string())?;
            return Ok(());
        }
        let _ = win.show();
        let _ = win.set_focus();
        return Ok(());
    }
    let url = format!("{}/?quick=1", super::SERVICE_URL);
    let parsed = url::Url::parse(&url).map_err(|e| e.to_string())?;
    let port = app
        .try_state::<super::AppState>()
        .map(|s| s.devtools_port)
        .unwrap_or(None);
    let win = WebviewWindowBuilder::new(app, QUICK_LABEL, WebviewUrl::External(parsed))
        .additional_browser_args(&super::settings::browser_args(port))
        .title("Quick ask")
        .inner_size(560.0, 720.0)
        .min_inner_size(360.0, 420.0)
        .always_on_top(true)
        .focused(true)
        .build()
        .map_err(|e| e.to_string())?;
    let _ = win.show();
    let _ = win.set_focus();
    Ok(())
}

/// Global-shortcut/JS toggle entry point (also callable from the page so a
/// main-window affordance can open the quick window).
#[tauri::command]
pub async fn quick_toggle(app: AppHandle) -> Result<(), String> {
    toggle(&app)
}

/// Capture the primary monitor and return it as a PNG data URL for the quick
/// window's screenshot attachment. Hide the ask window first so it does not
/// obscure the content being queried, and always restore it afterwards.
#[tauri::command]
pub fn quick_capture(window: tauri::WebviewWindow) -> Result<String, String> {
    let visible = window.is_visible().unwrap_or(false);
    if visible {
        window.hide().map_err(|e| e.to_string())?;
        std::thread::sleep(std::time::Duration::from_millis(180));
    }
    let result = capture_primary_png();
    if visible {
        let _ = window.show();
        let _ = window.set_focus();
    }
    result
}

fn capture_primary_png() -> Result<String, String> {
    let monitor = xcap::Monitor::all()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .ok_or("no monitor found")?;
    let img = monitor.capture_image().map_err(|e| e.to_string())?;
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(out.into_inner());
    Ok(format!("data:image/png;base64,{b64}"))
}

/// Continue the quick agent in the main window: notify the main page with
/// `omp:quick-continue` {id,cwd} (quick.js there calls app.connect), raise the
/// main window and hide (not close) the quick window.
#[tauri::command]
pub async fn continue_main(app: AppHandle, id: String, cwd: String) -> Result<(), String> {
    let _ = app.emit_to(
        "main",
        "omp:quick-continue",
        serde_json::json!({ "id": id, "cwd": cwd }),
    );
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.unminimize();
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(quick) = app.get_webview_window(QUICK_LABEL) {
        let _ = quick.hide();
    }
    Ok(())
}
