//! The browser app, embedded at build time (plain HTML/CSS/JS, no bundler).
//! The scripts are classic scripts sharing one `OMP` namespace, loaded in
//! the order `index.html` lists them.

const FILES: &[(&str, &str, &str)] = &[
    ("/", "text/html; charset=utf-8", include_str!("assets/index.html")),
    ("/assets/app.css", "text/css; charset=utf-8", include_str!("assets/app.css")),
    ("/assets/util.js", "text/javascript; charset=utf-8", include_str!("assets/util.js")),
    ("/assets/markdown.js", "text/javascript; charset=utf-8", include_str!("assets/markdown.js")),
    ("/assets/tools.js", "text/javascript; charset=utf-8", include_str!("assets/tools.js")),
    ("/assets/transcript.js", "text/javascript; charset=utf-8", include_str!("assets/transcript.js")),
    ("/assets/dialogs.js", "text/javascript; charset=utf-8", include_str!("assets/dialogs.js")),
    ("/assets/composer.js", "text/javascript; charset=utf-8", include_str!("assets/composer.js")),
    ("/assets/appearance.js", "text/javascript; charset=utf-8", include_str!("assets/appearance.js")),
    ("/assets/models.js", "text/javascript; charset=utf-8", include_str!("assets/models.js")),
    ("/assets/shell.js", "text/javascript; charset=utf-8", include_str!("assets/shell.js")),
    ("/assets/app.js", "text/javascript; charset=utf-8", include_str!("assets/app.js")),
    ("/favicon.svg", "image/svg+xml", include_str!("assets/favicon.svg")),
    ("/wallpapers/midnight.svg", "image/svg+xml", include_str!("assets/wallpapers/midnight.svg")),
    ("/wallpapers/anime-sakura.svg", "image/svg+xml", include_str!("assets/wallpapers/anime-sakura.svg")),
    ("/wallpapers/anime-city.svg", "image/svg+xml", include_str!("assets/wallpapers/anime-city.svg")),
    ("/wallpapers/anime-sky.svg", "image/svg+xml", include_str!("assets/wallpapers/anime-sky.svg")),
    ("/wallpapers/vaporwave-sunset.svg", "image/svg+xml", include_str!("assets/wallpapers/vaporwave-sunset.svg")),
    ("/wallpapers/vaporwave-grid.svg", "image/svg+xml", include_str!("assets/wallpapers/vaporwave-grid.svg")),
];

/// `(content type, body)` for an app path; unknown non-API paths fall back
/// to the app shell so deep links (`/s/<id>`) load the SPA.
pub fn get(path: &str) -> Option<(&'static str, &'static str)> {
    if let Some((_, mime, body)) = FILES.iter().find(|(p, _, _)| *p == path) {
        return Some((mime, body));
    }
    if path.starts_with("/assets/") || path.starts_with("/api/") || path.contains('.') {
        return None;
    }
    FILES.first().map(|(_, mime, body)| (*mime, *body))
}
