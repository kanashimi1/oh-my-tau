fn main() {
    // App commands must be declared to get `allow-*` permissions; the remote
    // UI origin (capabilities/default.json) can only invoke what it is granted.
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "open_window",
            "notify",
            "capture_done",
            "capture_rect",
        ]),
    ))
    .expect("tauri build");
}
