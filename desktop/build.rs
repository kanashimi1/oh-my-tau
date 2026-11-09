fn main() {
    // App commands must be declared to get `allow-*` permissions; the remote
    // UI origin (capabilities/default.json) can only invoke what it is granted.
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "open_window",
            "notify",
            "capture_done",
            "capture_rect",
            "tau_settings",
            "dictation_settings",
            "set_dictation",
            "set_capture_shortcut",
            "set_devtools_port",
            "quick_toggle",
            "quick_capture",
            "continue_main",
        ]),
    ))
    .expect("tauri build");
}
