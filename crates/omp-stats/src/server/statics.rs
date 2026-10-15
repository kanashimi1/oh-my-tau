//! Static dashboard client: the bun-built bundle in `client/`, embedded.

use super::api::ApiResponse;

const INDEX_HTML: &[u8] = include_bytes!("../../client/index.html");
const INDEX_JS: &[u8] = include_bytes!("../../client/index.js");
const INDEX_CSS: &[u8] = include_bytes!("../../client/index.css");
const FAVICON_ICO: &[u8] = include_bytes!("../../client/favicon.ico");

fn lookup(name: &str) -> Option<(&'static [u8], &'static str)> {
    match name {
        "index.html" => Some((INDEX_HTML, "text/html;charset=utf-8")),
        "index.js" => Some((INDEX_JS, "text/javascript;charset=utf-8")),
        "index.css" => Some((INDEX_CSS, "text/css;charset=utf-8")),
        "favicon.ico" => Some((FAVICON_ICO, "image/x-icon")),
        _ => None,
    }
}

/// Handle static file requests: the exact bundle file, else the SPA `index.html`.
#[must_use]
pub fn handle_static(request_path: &str) -> ApiResponse {
    let name = request_path.trim_start_matches('/');
    let name = if name.is_empty() { "index.html" } else { name };
    match lookup(name).or_else(|| lookup("index.html")) {
        Some((body, content_type)) => ApiResponse {
            status: 200,
            content_type: content_type.to_string(),
            headers: Vec::new(),
            body: body.to_vec(),
        },
        None => ApiResponse::text(404, "Not Found"),
    }
}
