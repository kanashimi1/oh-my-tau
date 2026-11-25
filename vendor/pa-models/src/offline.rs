//! Offline mode (`PI_OFFLINE`): no network at all, serve cache/bundled/compiled.

/// Whether the catalog subsystem must avoid every network request.
#[must_use]
pub fn is_catalog_offline() -> bool {
    offline_flag().is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
}

fn offline_flag() -> Option<String> {
    std::env::var("PI_OFFLINE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: process-global env mutation in tests is unsafe in Rust 2024 and
    // racy across the test binary; the parsing rule is covered through the
    // private helper instead.
    #[test]
    fn parses_offline_values() {
        // temp_env serializes and restores the mutation (no direct set_var, which the
        // Blaze checker refuses even though this crate is edition 2021).
        let yes = |raw: &str| temp_env::with_var("PI_OFFLINE", Some(raw), is_catalog_offline);
        assert!(yes("1"));
        assert!(yes("true"));
        assert!(yes("YES"));
        assert!(yes(" Yes "));
        assert!(!yes("0"));
        assert!(!yes("no"));
        assert!(!yes("maybe"));
        assert!(!temp_env::with_var("PI_OFFLINE", None::<&str>, is_catalog_offline));
    }
}
