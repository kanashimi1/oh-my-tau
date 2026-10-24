//! Request timeout resolution: a port of `mcp/timeout.ts`.
//!
//! `OMP_MCP_TIMEOUT_MS` overrides every config; per-server `timeout` (ms)
//! overrides the 30000 ms default; `0` disables client-side timeouts.

/// Default MCP request/connect timeout.
pub const DEFAULT_TIMEOUT_MS: u64 = 30_000;

const TIMEOUT_ENV: &str = "OMP_MCP_TIMEOUT_MS";

/// Resolve the effective timeout in milliseconds.
#[must_use]
pub fn resolve_timeout_ms(config_timeout: Option<u64>) -> u64 {
    if let Some(raw) = std::env::var(TIMEOUT_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        if let Ok(value) = raw.trim().parse::<u64>() {
            return value;
        }
        tracing::warn!(
            "ignoring invalid {TIMEOUT_ENV} env value; expected a non-negative number: {raw}"
        );
    }
    config_timeout.unwrap_or(DEFAULT_TIMEOUT_MS)
}

/// Timeouts are enabled when strictly positive.
#[must_use]
pub const fn is_timeout_enabled(timeout_ms: u64) -> bool {
    timeout_ms > 0
}

/// Human-readable timeout for diagnostics ("30000ms" / "disabled").
#[must_use]
pub fn describe_timeout(timeout_ms: u64) -> String {
    if is_timeout_enabled(timeout_ms) {
        format!("{timeout_ms}ms")
    } else {
        "disabled".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_config() {
        std::env::remove_var(TIMEOUT_ENV);
        assert_eq!(resolve_timeout_ms(None), DEFAULT_TIMEOUT_MS);
        assert_eq!(resolve_timeout_ms(Some(5)), 5);
        assert!(is_timeout_enabled(DEFAULT_TIMEOUT_MS));
        assert!(!is_timeout_enabled(0));
        assert_eq!(describe_timeout(0), "disabled");
        assert_eq!(describe_timeout(12), "12ms");
    }
}
