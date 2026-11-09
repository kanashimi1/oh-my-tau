# Tau — omt desktop app (Tauri 2, Windows)

A native Windows shell for the omt web UI. It loads the UI from the local
service (`http://127.0.0.1:7467`), starting that service via WSL when it is
down, and adds a tray icon, native notifications and a global screenshot
hotkey.

## Layout

- `src/lib.rs` — app logic (service autostart, tray, hotkey, commands).
- `src/main.rs` — Windows GUI entry point.
- `tauri.conf.json` — Tauri 2 config (no embedded frontend; remote UI).
- `capabilities/default.json` — ACL for the remote UI origin.
- `ui/index.html` — splash / error page shown while the service boots.
- `icons/` — app + tray icons generated from `assets/favicon.svg` (WebView2).
- `desktop-bridge.js` lives in the web UI assets
  (`crates/omp-cli/src/subcmd/web/assets/desktop-bridge.js`) and is what the
  remote page uses to talk back to this shell.

## Build (from WSL)

Cross-compile for Windows x86_64 with `cargo-xwin`:

```sh
cd desktop
cargo xwin build --target x86_64-pc-windows-msvc --release
```

The portable binary is produced at:

```sh
target/x86_64-pc-windows-msvc/release/tau.exe
```

Copy it to the install location:

```sh
cp target/x86_64-pc-windows-msvc/release/tau.exe \
  /mnt/c/Users/<you>/AppData/Local/Programs/Tau/tau.exe
```

No installer is required; `tau.exe` runs standalone (WebView2 is installed on
the host).

## Run

```sh
/mnt/c/Users/<you>/AppData/Local/Programs/Tau/tau.exe &
```

On startup the app probes `http://127.0.0.1:7467/api/info`. If the service is
down it runs `wsl.exe -d Ubuntu -- systemctl --user start omp-rs-web` (no
browser opens) and polls until it is up (90s timeout, after which an error page
is shown). Closing the app never stops the service.

## Features

- **Tray**: Show/Hide windows, New window, Quit. Closing the main window hides
  it to the tray instead of exiting.
- **Notifications**: in the web UI, `desktop-bridge.js` listens on the
  `omp:frame` bus and notifies when an agent ends (`agent_end`) or waits for
  input (`extension_ui_request` with a dialog method) — only when that window
  is not focused. On the shell they are native Windows toasts; in a plain
  browser they fall back to the Web Notification API when the tab is hidden.
- **Screenshot hotkey**: `Ctrl+Shift+S` (global) opens the Windows snipping
  overlay; the selected region is read from the clipboard and attached to the
  focused window's composer via `OMP.app.composer.addImageData`.
- **Multi-window**: `OMP.desktop.openWindow(hash)` opens a new app window at
  `http://127.0.0.1:7467/<hash>` (the tray and `open_window` command use the
  same path).

## Behaviour notes

- Screenshot hotkey **Ctrl+Shift+S**: captures the monitor under the cursor
  (own capture via `xcap`; the Snipping Tool's `ms-screenclip:` protocol is
  absent on some Windows builds), shows it fullscreen in the focused window;
  drag a region (or Enter = whole screen, Esc = cancel) and the crop lands in
  that window's composer.
- Notifications use PowerShell's registered AppUserModelID (a portable exe has
  none). They only appear if Windows toasts are enabled
  (Settings → System → Notifications).
- The tray icon starts in the taskbar overflow (^); drag it out to pin it.
- Log: `%TEMP%\tau.log` (hotkey, capture, shortcut registration errors).
