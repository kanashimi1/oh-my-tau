// omrs: Tauri 2 desktop shell for the omt web UI.
//
// Responsibilities:
//   * ensure the local omt service (http://127.0.0.1:7467) is up, starting it
//     via WSL if needed, then load the UI in the main window
//   * tray icon (Show/Hide, New window, Quit); closing a window hides to tray
//   * global Ctrl+Shift+S shortcut: captures the monitor under the cursor and
//     lets the user drag a region in the window; the crop lands in that
//     window's composer
//   * commands used by desktop-bridge.js: open_window(hash), notify(title, body)

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

struct AppState {
    quitting: AtomicBool,
}

// Both commands are async: a sync command runs on the main thread, and
// building a webview window there deadlocks on Windows (the whole app froze on
// the first "Open in new window").
#[tauri::command]
async fn open_window(app: AppHandle, hash: String) -> Result<(), String> {
    open_new_window(&app, &hash).map_err(|e| e.to_string())
}

#[tauri::command]
async fn notify(title: String, body: String) -> Result<(), String> {
    use tauri_winrt_notification::{Duration as ToastDuration, Toast};
    // A portable exe has no registered AppUserModelID (no Start-menu
    // shortcut), and Windows drops toasts from unregistered IDs; PowerShell's
    // ID is always registered.
    Toast::new(Toast::POWERSHELL_APP_ID)
        .duration(ToastDuration::Short)
        .title(&title)
        .text1(&body)
        .show()
        .map_err(|e| e.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![open_window, notify, capture_done])
        .setup(|app| {
            app.manage(AppState {
                quitting: AtomicBool::new(false),
            });
            setup_tray(app)?;
            setup_global_shortcut(app)?;
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
        .expect("error while running omrs");
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
        .tooltip("omt")
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show_hide" => toggle_windows(app),
            "new_window" => {
                let _ = open_new_window(app, "");
            }
            "quit" => {
                app.state::<AppState>().quitting.store(true, Ordering::Relaxed);
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
    let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyS);
    app.global_shortcut()
        .on_shortcut(shortcut, |app, _sc, event| {
            if event.state() == ShortcutState::Pressed {
                trigger_screenshot(app);
            }
        })
        .unwrap_or_else(|e| log_line(&format!("register Ctrl+Shift+S failed: {e}")));
    Ok(())
}

fn setup_main_window(app: &tauri::App) -> tauri::Result<()> {
    let handle = app.handle().clone();

    let _win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("omt")
        .inner_size(1280.0, 820.0)
        .min_inner_size(760.0, 540.0)
        .visible(false)
        .build()?;

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
    let win = WebviewWindowBuilder::new(app, &label, WebviewUrl::External(parsed))
        .title("omt")
        .inner_size(1280.0, 820.0)
        .min_inner_size(760.0, 540.0)
        .build()?;
    let _ = win.show();
    let _ = win.set_focus();
    Ok(())
}

// Service lifecycle

fn probe_service() -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(
        &SERVICE_HOST.parse().expect("host"),
        Duration::from_secs(2),
    ) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let req = format!("GET /api/info HTTP/1.1\r\nHost: {SERVICE_HOST}\r\nConnection: close\r\n\r\n");
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
    log_line("hotkey: Ctrl+Shift+S");
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
    let visible: Vec<_> = windows.iter().filter(|w| w.is_visible().unwrap_or(false)).cloned().collect();
    for w in &visible {
        let _ = w.hide();
    }
    std::thread::sleep(Duration::from_millis(250)); // let DWM drop our windows from the frame
    let shot = capture_monitor_under_cursor();
    log_line(&format!("capture: {}", shot.as_ref().map_or("failed".to_string(), |p| format!("{} bytes png", p.len()))));
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
            let (w, h) = (m.width().unwrap_or(0) as i32, m.height().unwrap_or(0) as i32);
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
    image::DynamicImage::ImageRgba8(img).write_to(&mut out, image::ImageFormat::Png).ok()?;
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

/// Append one line to `%TEMP%\\omrs.log` (a GUI-subsystem exe has no stderr).
fn log_line(msg: &str) {
    let path = std::env::temp_dir().join("omrs.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let _ = writeln!(f, "{secs} {msg}");
    }
}
