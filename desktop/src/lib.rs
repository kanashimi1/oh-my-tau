// Tau: Tauri 2 desktop app for the omt web UI.
//
// Responsibilities:
//   * ensure the local omt service (http://127.0.0.1:7467) is up, starting it
//     via WSL if needed, then load the UI in the main window
//   * tray icon (Show/Hide, New window, Quit); closing a window hides to tray
//   * global capture shortcut (default Ctrl+Shift+S, settable): captures the monitor under the cursor and
//     lets the user drag a region in the window; the crop lands in that
//     window's composer
//   * commands used by desktop-bridge.js: open_window(hash), notify(title,
//     body, agent, key) — Windows toast, click opens that session —
//     capture_rect(x, y, w, h) — screen pixels of a page region (preview pins),
//     tau_settings / set_capture_shortcut / set_devtools_port (settings.rs)

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};
use url::Url;

use base64::Engine;
use std::os::windows::process::CommandExt;

const SERVICE_URL: &str = "http://127.0.0.1:7467";
const SERVICE_HOST: &str = "127.0.0.1:7467";
const SERVICE_INIT_TIMEOUT: Duration = Duration::from_secs(90);
const WSL_DISTRO: &str = "Ubuntu";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

static WINDOW_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

mod dictation;
mod quick;
mod settings;

struct AppState {
    quitting: AtomicBool,
    /// Capture shortcut currently registered (None: registration failed).
    shortcut: std::sync::Mutex<Option<Shortcut>>,
    /// Why the configured shortcut could not be registered (taken by another app…).
    shortcut_error: std::sync::Mutex<Option<String>>,
    /// DevTools port this process started with (browser args are fixed at startup).
    devtools_port: Option<u16>,
}

/// Current desktop settings plus live state, for the web UI's Desktop section.
#[tauri::command]
async fn tau_settings(app: AppHandle) -> serde_json::Value {
    let s = settings::load();
    let state = app.state::<AppState>();
    let error = state
        .shortcut_error
        .lock()
        .map(|e| e.clone())
        .unwrap_or(None);
    serde_json::json!({
        "captureShortcut": s.capture_shortcut,
        "captureShortcutError": error,
        "devtoolsPort": s.devtools_port,
        "devtoolsActive": state.devtools_port,
    })
}

/// Dictation preferences, including defaults for older settings files.
#[tauri::command]
async fn dictation_settings() -> settings::DictationSettings {
    settings::load().dictation
}

#[tauri::command]
async fn set_dictation(enabled: bool, hold_ms: u64, language: String) -> Result<(), String> {
    let language = language.trim();
    if language.is_empty() || language.len() > 32 || !language.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-') {
        return Err("invalid language".into());
    }
    let mut s = settings::load();
    s.dictation = settings::DictationSettings { enabled, hold_ms, language: language.into() };
    settings::save(&s)
}

/// Re-register the capture shortcut live; on failure the previous one stays.
#[tauri::command]
async fn set_capture_shortcut(app: AppHandle, shortcut: String) -> Result<String, String> {
    let wanted: Shortcut = shortcut
        .trim()
        .parse()
        .map_err(|e| format!("\"{shortcut}\": {e}"))?;
    let state = app.state::<AppState>();
    let old = state.shortcut.lock().map_err(|e| e.to_string())?.take();
    if let Some(old) = old {
        let _ = app.global_shortcut().unregister(old);
    }
    match register_capture(&app, wanted) {
        Ok(()) => {
            *state.shortcut.lock().map_err(|e| e.to_string())? = Some(wanted);
            *state.shortcut_error.lock().map_err(|e| e.to_string())? = None;
            let mut s = settings::load();
            s.capture_shortcut = shortcut.trim().to_string();
            settings::save(&s)?;
            Ok(s.capture_shortcut)
        }
        Err(e) => {
            if let Some(old) = old {
                if register_capture(&app, old).is_ok() {
                    *state.shortcut.lock().map_err(|e| e.to_string())? = Some(old);
                }
            }
            Err(format!("{} is taken or invalid: {e}", shortcut.trim()))
        }
    }
}

/// Store the DevTools port (None turns it off); applies on the next start.
#[tauri::command]
async fn set_devtools_port(port: Option<u16>) -> Result<(), String> {
    if port.is_some_and(|p| p < 1024) {
        return Err("port must be 1024–65535".into());
    }
    let mut s = settings::load();
    s.devtools_port = port;
    settings::save(&s)
}

// Both commands are async: a sync command runs on the main thread, and
// building a webview window there deadlocks on Windows (the whole app froze on
// the first "Open in new window").
#[tauri::command]
async fn open_window(app: AppHandle, hash: String) -> Result<(), String> {
    open_new_window(&app, &hash).map_err(|e| e.to_string())
}

/// Toast source registered under `HKCU\\Software\\Classes\\AppUserModelId`
/// (name + icon in the notification); PowerShell's ID is the fallback.
const TOAST_APP_ID: &str = "com.omt.tau";

/// Recent toast keys (agent + kind): every window polls the roster, so the
/// same transition arrives once per window.
static TOAST_SEEN: std::sync::Mutex<Vec<(String, Instant)>> = std::sync::Mutex::new(Vec::new());

/// Windows toast for an agent event. Skipped (returns `false`) while a Tau
/// window has focus; clicking it shows the window and opens `agent`.
#[tauri::command]
async fn notify(
    app: AppHandle,
    title: String,
    body: String,
    agent: Option<String>,
    key: Option<String>,
) -> Result<bool, String> {
    use tauri_winrt_notification::{Duration as ToastDuration, Toast};
    let focused = app
        .webview_windows()
        .values()
        .any(|w| w.is_focused().unwrap_or(false) && !w.is_minimized().unwrap_or(false));
    if focused {
        return Ok(false);
    }
    if let Some(key) = key {
        let mut seen = TOAST_SEEN.lock().map_err(|e| e.to_string())?;
        seen.retain(|(_, at)| at.elapsed() < Duration::from_secs(30));
        if seen.iter().any(|(k, _)| *k == key) {
            return Ok(false);
        }
        seen.push((key, Instant::now()));
    }
    // Taskbar flash: visible even with Windows notifications turned off.
    flash_taskbar(&app);
    let toast = |id: &str| {
        let (app, agent) = (app.clone(), agent.clone());
        Toast::new(id)
            .duration(ToastDuration::Short)
            .title(&title)
            .text1(&body)
            .on_activated(move |_| {
                show_agent(&app, agent.as_deref());
                Ok(())
            })
            .show()
    };
    match toast(TOAST_APP_ID) {
        Ok(()) => Ok(true),
        Err(e) => {
            log_line(&format!("toast ({TOAST_APP_ID}): {e}"));
            // A portable exe has no installer-registered ID; PowerShell's
            // always is.
            toast(Toast::POWERSHELL_APP_ID)
                .map(|()| true)
                .map_err(|e| e.to_string())
        }
    }
}

/// Flash the main window's taskbar button until the window is focused.
fn flash_taskbar(app: &AppHandle) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        FlashWindowEx, FLASHWINFO, FLASHW_TIMERNOFG, FLASHW_TRAY,
    };
    let windows = app.webview_windows();
    let Some(win) = windows.get("main").or_else(|| windows.values().next()) else {
        return;
    };
    let Ok(hwnd) = win.hwnd() else { return };
    let info = FLASHWINFO {
        cbSize: u32::try_from(std::mem::size_of::<FLASHWINFO>()).unwrap_or(0),
        hwnd: hwnd.0.cast(),
        dwFlags: FLASHW_TRAY | FLASHW_TIMERNOFG,
        uCount: 0,
        dwTimeout: 0,
    };
    // SAFETY: `info` is a fully initialised FLASHWINFO for a live window handle.
    unsafe { FlashWindowEx(&info) };
}

/// Show the main (else any) window and navigate it to `agent`.
fn show_agent(app: &AppHandle, agent: Option<&str>) {
    let windows = app.webview_windows();
    let Some(win) = windows.get("main").or_else(|| windows.values().next()) else {
        return;
    };
    let _ = win.unminimize();
    let _ = win.show();
    let _ = win.set_focus();
    if let Some(id) = agent {
        let _ = win.emit("omp:navigate", id.to_string());
    }
}

/// Register [`TOAST_APP_ID`] (display name + icon) so toasts show as Tau.
fn register_toast_app_id() {
    use windows_sys::Win32::System::Registry::{
        RegCloseKey, RegCreateKeyW, RegSetValueExW, HKEY, HKEY_CURRENT_USER, REG_SZ,
    };
    let icon_dir = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .map(|d| d.join("Tau"));
    let icon = icon_dir.and_then(|d| {
        std::fs::create_dir_all(&d).ok()?;
        let p = d.join("toast-icon.png");
        std::fs::write(&p, include_bytes!("../icons/icon.png")).ok()?;
        Some(p)
    });
    let wide = |s: &str| {
        s.encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<u16>>()
    };
    let subkey = wide(&format!(
        "Software\\Classes\\AppUserModelId\\{TOAST_APP_ID}"
    ));
    let mut key: HKEY = std::ptr::null_mut();
    // SAFETY: valid NUL-terminated wide strings and out-pointer; the key is closed below.
    let rc = unsafe { RegCreateKeyW(HKEY_CURRENT_USER, subkey.as_ptr(), &mut key) };
    if rc != 0 {
        log_line(&format!("register toast app id: {rc}"));
        return;
    }
    let set = |name: &str, value: &str| {
        let (n, v) = (wide(name), wide(value));
        // SAFETY: `key` is open; `v` is a NUL-terminated UTF-16 buffer of the given byte length.
        unsafe {
            RegSetValueExW(
                key,
                n.as_ptr(),
                0,
                REG_SZ,
                v.as_ptr().cast(),
                (v.len() * 2) as u32,
            )
        };
    };
    set("DisplayName", "Tau");
    if let Some(icon) = icon {
        set("IconUri", &icon.display().to_string());
    }
    set("IconBackgroundColor", "FF181818");
    // SAFETY: `key` was opened above.
    unsafe { RegCloseKey(key) };
}

/// Screen pixels of a region of the calling window's page (CSS px relative
/// to the viewport) as a `data:image/png;base64,…` URL. Used to pin pages,
/// PDFs and documents in the preview panel, whose content a page script
/// cannot read.
#[tauri::command]
async fn capture_rect(
    window: tauri::WebviewWindow,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
) -> Result<String, String> {
    // Monitor pixels: the caller is the window the user just clicked, so it
    // is on top.
    let scale = window.scale_factor().map_err(|e| e.to_string())?;
    let origin = window.inner_position().map_err(|e| e.to_string())?;
    #[allow(clippy::cast_possible_truncation)]
    let (px, py) = (
        origin.x + (x * scale).round() as i32,
        origin.y + (y * scale).round() as i32,
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (pw, ph) = (
        (w * scale).round().max(1.0) as u32,
        (h * scale).round().max(1.0) as u32,
    );
    let monitor = xcap::Monitor::from_point(px + 1, py + 1).map_err(|e| e.to_string())?;
    let (mx, my) = (
        monitor.x().map_err(|e| e.to_string())?,
        monitor.y().map_err(|e| e.to_string())?,
    );
    let (mw, mh) = (
        monitor.width().map_err(|e| e.to_string())?,
        monitor.height().map_err(|e| e.to_string())?,
    );
    let rx = u32::try_from((px - mx).max(0))
        .unwrap_or(0)
        .min(mw.saturating_sub(1));
    let ry = u32::try_from((py - my).max(0))
        .unwrap_or(0)
        .min(mh.saturating_sub(1));
    let img = monitor
        .capture_region(rx, ry, pw.min(mw - rx), ph.min(mh - ry))
        .map_err(|e| e.to_string())?;
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(out.into_inner())
    ))
}

pub fn run() {
    tauri::Builder::default()
        // One Tau per user: launching it again (Start menu, the hub's "raise")
        // shows the running window instead of a second tray icon fighting
        // over the Ctrl+Shift+S shortcut.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(main) = app.get_webview_window("main") {
                let _ = main.unminimize();
                let _ = main.show();
                let _ = main.set_focus();
            }
        }))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            open_window,
            notify,
            capture_done,
            capture_rect,
            tau_settings,
            dictation_settings,
            set_dictation,
            set_capture_shortcut,
            quick::quick_toggle,
            quick::quick_capture,
            quick::continue_main,
            set_devtools_port
        ])
        // Composer dictation: windows only ever show the local omt service,
        // so the microphone is granted without a WebView2 prompt per launch.
        .on_permission_request(|_webview, kind| match kind {
            tauri::webview::PermissionKind::Microphone => tauri::webview::PermissionResponse::Allow,
            _ => tauri::webview::PermissionResponse::Default,
        })
        .setup(|app| {
            let prefs = settings::load();
            app.manage(AppState {
                quitting: AtomicBool::new(false),
                shortcut: std::sync::Mutex::new(None),
                shortcut_error: std::sync::Mutex::new(None),
                devtools_port: settings::devtools_port(&prefs),
            });
            register_toast_app_id();
            setup_tray(app)?;
            setup_global_shortcut(app)?;
            quick::register(app.handle());
            setup_main_window(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let quitting = window
                    .app_handle()
                    .state::<AppState>()
                    .quitting
                    .load(Ordering::Relaxed);
                if !quitting {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Tau");
}

fn setup_tray(app: &tauri::App) -> tauri::Result<()> {
    let show_hide = MenuItem::with_id(app, "show_hide", "Show/Hide", true, None::<&str>)?;
    let new_window = MenuItem::with_id(app, "new_window", "New window", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_hide, &new_window, &quit])?;

    let icon = app.default_window_icon().cloned().unwrap_or_else(|| {
        tauri::image::Image::from_bytes(include_bytes!("../icons/tray.png"))
            .expect("embedded tray icon")
    });

    TrayIconBuilder::with_id("main-tray")
        .icon(icon)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("Tau")
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show_hide" => toggle_windows(app),
            "new_window" => {
                let _ = open_new_window(app, "");
            }
            "quit" => {
                app.state::<AppState>()
                    .quitting
                    .store(true, Ordering::Relaxed);
                app.exit(0);
            }
            _ => {}
        })
        .build(app)?;
    Ok(())
}

fn toggle_windows(app: &AppHandle) {
    let windows = app.webview_windows();
    let wins: Vec<_> = windows.values().cloned().collect();
    let any_visible = wins.iter().any(|w| w.is_visible().unwrap_or(false));
    if any_visible {
        for w in wins {
            let _ = w.hide();
        }
    } else if let Some(main) = app.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    } else if let Some(w) = wins.into_iter().next() {
        let _ = w.show();
        let _ = w.set_focus();
    }
}

fn setup_global_shortcut(app: &tauri::App) -> tauri::Result<()> {
    let configured = settings::load().capture_shortcut;
    let shortcut = configured.parse::<Shortcut>().unwrap_or_else(|e| {
        log_line(&format!(
            "capture shortcut {configured:?} invalid ({e}), using default"
        ));
        Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyS)
    });
    let state = app.state::<AppState>();
    match register_capture(app.handle(), shortcut) {
        Ok(()) => {
            if let Ok(mut s) = state.shortcut.lock() {
                *s = Some(shortcut);
            }
        }
        Err(e) => {
            log_line(&format!("register {configured} failed: {e}"));
            if let Ok(mut err) = state.shortcut_error.lock() {
                *err = Some(format!("{configured} is taken by another app: {e}"));
            }
        }
    }
    Ok(())
}

fn register_capture(app: &AppHandle, shortcut: Shortcut) -> Result<(), String> {
    app.global_shortcut()
        .on_shortcut(shortcut, |app, _sc, event| {
            if event.state() == ShortcutState::Pressed {
                trigger_screenshot(app);
            }
        })
        .map_err(|e| e.to_string())
}

fn apply_window_chrome(window: &tauri::WebviewWindow) {
    // Match the native Windows caption to the default graphite harness theme.
    use std::ffi::c_void;
    use windows_sys::Win32::Graphics::Dwm::{
        DwmSetWindowAttribute, DWMWA_CAPTION_COLOR, DWMWA_TEXT_COLOR, DWMWA_USE_IMMERSIVE_DARK_MODE,
    };
    let Ok(hwnd) = window.hwnd() else { return };
    let dark = 1_i32;
    let caption = 0x0018_1818_u32; // COLORREF #181818
    let text = 0x00d6_d6d6_u32; // COLORREF #d6d6d6
    for (attribute, value, size) in [
        (
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark as *const _ as *const c_void,
            std::mem::size_of::<i32>(),
        ),
        (
            DWMWA_CAPTION_COLOR,
            &caption as *const _ as *const c_void,
            std::mem::size_of::<u32>(),
        ),
        (
            DWMWA_TEXT_COLOR,
            &text as *const _ as *const c_void,
            std::mem::size_of::<u32>(),
        ),
    ] {
        // SAFETY: valid window handle and pointers to initialized values of the declared sizes.
        unsafe {
            DwmSetWindowAttribute(hwnd.0 as *mut c_void, attribute as u32, value, size as u32)
        };
    }
}

fn setup_main_window(app: &tauri::App) -> tauri::Result<()> {
    let handle = app.handle().clone();

    let port = app.state::<AppState>().devtools_port;
    if let Some(port) = port {
        log_line(&format!("devtools: remote debugging on 127.0.0.1:{port}"));
    }
    let win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .additional_browser_args(&settings::browser_args(port))
        .title("Tau")
        .inner_size(1280.0, 820.0)
        .min_inner_size(760.0, 540.0)
        .visible(false)
        .build()?;
    apply_window_chrome(&win);

    std::thread::spawn(move || {
        let up = wait_for_service();
        if let Some(win) = handle.get_webview_window("main") {
            if up {
                let _ = win.navigate(Url::parse(SERVICE_URL).expect("service url"));
                let _ = win.show();
                let _ = win.set_focus();
            } else {
                let _ = win.eval(
                    r##"document.getElementById("spinner").style.display="none";
                       document.getElementById("status").textContent =
                         "Could not reach the omt service (127.0.0.1:7467) within 90s.";
                       document.getElementById("status").style.color="#e07878";"##,
                );
                let _ = win.show();
            }
        }
    });

    Ok(())
}

fn open_new_window(app: &AppHandle, hash: &str) -> tauri::Result<()> {
    let seq = WINDOW_SEQ.fetch_add(1, Ordering::Relaxed);
    let label = format!("win-{seq}");
    let mut url = SERVICE_URL.to_string();
    let h = hash.trim().trim_start_matches('/');
    if !h.is_empty() {
        url.push('/');
        url.push_str(h);
    }
    let parsed = Url::parse(&url).map_err(|e| tauri::Error::Anyhow(e.into()))?;
    let port = app.state::<AppState>().devtools_port;
    let win = WebviewWindowBuilder::new(app, &label, WebviewUrl::External(parsed))
        .additional_browser_args(&settings::browser_args(port))
        .title("Tau")
        .inner_size(1280.0, 820.0)
        .min_inner_size(760.0, 540.0)
        .build()?;
    apply_window_chrome(&win);
    let _ = win.show();
    let _ = win.set_focus();
    Ok(())
}

// Service lifecycle

fn probe_service() -> bool {
    let Ok(mut stream) =
        TcpStream::connect_timeout(&SERVICE_HOST.parse().expect("host"), Duration::from_secs(2))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let req =
        format!("GET /api/info HTTP/1.1\r\nHost: {SERVICE_HOST}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 1024];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

fn start_wsl_service() {
    let _ = Command::new("wsl.exe")
        .args([
            "-d",
            WSL_DISTRO,
            "--",
            "systemctl",
            "--user",
            "start",
            "omp-rs-web",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .spawn();
}

fn wait_for_service() -> bool {
    if probe_service() {
        return true;
    }
    start_wsl_service();
    let deadline = Instant::now() + SERVICE_INIT_TIMEOUT;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(700));
        if probe_service() {
            return true;
        }
    }
    false
}

// Screenshot hotkey
//
// Own region capture (no dependency on the Snipping Tool, whose
// `ms-screenclip:` protocol is missing on some Windows installs): hide our
// windows, grab the monitor under the cursor, then show the target window
// fullscreen with the capture; desktop-bridge.js draws the region selector and
// calls `capture_done` to restore the window.

fn trigger_screenshot(app: &AppHandle) {
    log_line("hotkey: capture");
    let app = app.clone();
    std::thread::spawn(move || run_capture_flow(&app));
}

fn run_capture_flow(app: &AppHandle) {
    let windows: Vec<_> = app.webview_windows().into_values().collect();
    let target = windows
        .iter()
        .find(|w| w.is_focused().unwrap_or(false))
        .cloned()
        .or_else(|| app.get_webview_window("main"));
    let Some(target) = target else { return };
    let visible: Vec<_> = windows
        .iter()
        .filter(|w| w.is_visible().unwrap_or(false))
        .cloned()
        .collect();
    for w in &visible {
        let _ = w.hide();
    }
    std::thread::sleep(Duration::from_millis(250)); // let DWM drop our windows from the frame
    let shot = capture_monitor_under_cursor();
    log_line(&format!(
        "capture: {}",
        shot.as_ref()
            .map_or("failed".to_string(), |p| format!("{} bytes png", p.len()))
    ));
    for w in &visible {
        let _ = w.show();
    }
    let Some(png) = shot else {
        let _ = target.show();
        return;
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    let _ = target.show();
    let _ = target.set_fullscreen(true);
    let _ = target.set_focus();
    let _ = target.emit("omp:screen-capture", b64);
}

fn capture_monitor_under_cursor() -> Option<Vec<u8>> {
    let monitors = match xcap::Monitor::all() {
        Ok(m) => m,
        Err(e) => {
            log_line(&format!("xcap monitors: {e}"));
            return None;
        }
    };
    let (cx, cy) = cursor_pos();
    let monitor = monitors
        .iter()
        .find(|m| {
            let (x, y) = (m.x().unwrap_or(0), m.y().unwrap_or(0));
            let (w, h) = (
                m.width().unwrap_or(0) as i32,
                m.height().unwrap_or(0) as i32,
            );
            cx >= x && cx < x + w && cy >= y && cy < y + h
        })
        .or_else(|| monitors.iter().find(|m| m.is_primary().unwrap_or(false)))
        .or_else(|| monitors.first())?;
    let img = match monitor.capture_image() {
        Ok(i) => i,
        Err(e) => {
            log_line(&format!("xcap capture: {e}"));
            return None;
        }
    };
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut out, image::ImageFormat::Png)
        .ok()?;
    Some(out.into_inner())
}

fn cursor_pos() -> (i32, i32) {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;
    let mut p = POINT { x: 0, y: 0 };
    // SAFETY: GetCursorPos writes one POINT through a valid pointer.
    if unsafe { GetCursorPos(&mut p) } == 0 {
        return (0, 0);
    }
    (p.x, p.y)
}

/// Leave the capture view: the region selector finished or was cancelled.
#[tauri::command]
async fn capture_done(window: tauri::WebviewWindow) {
    let _ = window.set_fullscreen(false);
}

/// Append one line to `%TEMP%\\tau.log` (a GUI-subsystem exe has no stderr).
fn log_line(msg: &str) {
    let path = std::env::temp_dir().join("tau.log");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let _ = writeln!(f, "{secs} {msg}");
    }
}
