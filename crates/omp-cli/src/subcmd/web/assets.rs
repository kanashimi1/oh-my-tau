//! The browser app, embedded at build time (plain HTML/CSS/JS, no bundler).
//! The scripts are classic scripts sharing one `OMP` namespace, loaded in
//! the order `index.html` lists them.

use hyper::body::Bytes;

const FILES: &[(&str, &str, &str)] = &[
    (
        "/",
        "text/html; charset=utf-8",
        include_str!("assets/index.html"),
    ),
    (
        "/assets/app.css",
        "text/css; charset=utf-8",
        include_str!("assets/app.css"),
    ),
    (
        "/assets/util.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/util.js"),
    ),
    (
        "/assets/markdown.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/markdown.js"),
    ),
    (
        "/assets/tools.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/tools.js"),
    ),
    (
        "/assets/transcript.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/transcript.js"),
    ),
    (
        "/assets/dialogs.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/dialogs.js"),
    ),
    (
        "/assets/annotate.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/annotate.js"),
    ),
    (
        "/assets/composer.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/composer.js"),
    ),
    (
        "/assets/appearance.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/appearance.js"),
    ),
    (
        "/assets/models.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/models.js"),
    ),
    (
        "/assets/shell.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/shell.js"),
    ),
    (
        "/assets/app.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/app.js"),
    ),
    (
        "/assets/sidebar.css",
        "text/css; charset=utf-8",
        include_str!("assets/sidebar.css"),
    ),
    (
        "/assets/composer.css",
        "text/css; charset=utf-8",
        include_str!("assets/composer.css"),
    ),
    (
        "/assets/artifacts.css",
        "text/css; charset=utf-8",
        include_str!("assets/artifacts.css"),
    ),
    (
        "/assets/stream.css",
        "text/css; charset=utf-8",
        include_str!("assets/stream.css"),
    ),
    (
        "/assets/sidebar.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/sidebar.js"),
    ),
    (
        "/assets/artifacts.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/artifacts.js"),
    ),
    (
        "/assets/desktop-bridge.js",
        "text/javascript; charset=utf-8",
        include_str!("assets/desktop-bridge.js"),
    ),
    (
        "/favicon.svg",
        "image/svg+xml",
        include_str!("assets/favicon.svg"),
    ),
    (
        "/wallpapers/midnight.svg",
        "image/svg+xml",
        include_str!("assets/wallpapers/midnight.svg"),
    ),
    (
        "/wallpapers/anime-sakura.svg",
        "image/svg+xml",
        include_str!("assets/wallpapers/anime-sakura.svg"),
    ),
    (
        "/wallpapers/anime-city.svg",
        "image/svg+xml",
        include_str!("assets/wallpapers/anime-city.svg"),
    ),
    (
        "/wallpapers/anime-sky.svg",
        "image/svg+xml",
        include_str!("assets/wallpapers/anime-sky.svg"),
    ),
    (
        "/wallpapers/vaporwave-sunset.svg",
        "image/svg+xml",
        include_str!("assets/wallpapers/vaporwave-sunset.svg"),
    ),
    (
        "/wallpapers/vaporwave-grid.svg",
        "image/svg+xml",
        include_str!("assets/wallpapers/vaporwave-grid.svg"),
    ),
];

/// `(content type, body)` for an app path; unknown non-API paths fall back
/// to the app shell so deep links (`/s/<id>`) load the SPA.
///
/// With `OMP_WEB_DEV_ASSETS=<dir>` the listed files are read from `<dir>`
/// (the `assets/` source folder) on every request, so UI work needs no rebuild.
pub fn get(path: &str) -> Option<(&'static str, Bytes)> {
    let (rel, mime, body) = FILES
        .iter()
        .find(|(p, _, _)| *p == path)
        .copied()
        .or_else(|| {
            let shell =
                !(path.starts_with("/assets/") || path.starts_with("/api/") || path.contains('.'));
            if shell {
                FILES.first().copied()
            } else {
                None
            }
        })?;
    if let Some(dir) = std::env::var_os("OMP_WEB_DEV_ASSETS") {
        let file = match rel {
            "/" => "index.html".to_owned(),
            p => p
                .trim_start_matches("/assets/")
                .trim_start_matches('/')
                .to_owned(),
        };
        if let Ok(bytes) = std::fs::read(std::path::Path::new(&dir).join(file)) {
            return Some((mime, Bytes::from(bytes)));
        }
    }
    Some((mime, Bytes::from_static(body.as_bytes())))
}
