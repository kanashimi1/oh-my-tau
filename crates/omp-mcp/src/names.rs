//! `mcp__<server>_<tool>` name minting: a port of `tool-bridge.ts`
//! (`createMCPToolName` / `createLegacyMCPToolName`).
//!
//! Sanitizes each part (`[^a-z0-9_]+` → `_` for the current mint; the legacy
//! mint strips digits too), strips a redundant `<server>_` prefix from the
//! tool part, and caps at 64 chars with a deterministic hash suffix.
//!
//! omp hashes overlong names with `Bun.hash` (wyhash). The exact Bun seed is
//! not part of the wire contract any consumer depends on, so this port uses
//! the standard wyhash final1 with seed 0; the cap shape (readable prefix +
//! 8-char base-36 suffix, stable across processes) matches omp.

const MAX_TOOL_NAME_LENGTH: usize = 64;
const HASH_LENGTH: usize = 8;

fn sanitize_part(value: &str, fallback: &str, keep_digits: bool) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_underscore = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        let ok = ch == '_' || ch.is_ascii_lowercase() || (keep_digits && ch.is_ascii_digit());
        if ok {
            last_underscore = false;
            out.push(ch);
        } else if !last_underscore {
            last_underscore = true;
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}

fn mint(server: &str, tool: &str, keep_digits: bool) -> String {
    let server = sanitize_part(server, "server", keep_digits);
    let mut tool = sanitize_part(tool, "tool", keep_digits);
    let prefix = format!("{server}_");
    if let Some(stripped) = tool.strip_prefix(prefix.as_str()) {
        tool = stripped.to_string();
    }
    cap_length(&format!("mcp__{server}_{tool}"))
}

fn cap_length(name: &str) -> String {
    if name.len() <= MAX_TOOL_NAME_LENGTH {
        return name.to_string();
    }
    let hash = wyhash(name.as_bytes()).to_string_radix(36);
    let hash = &hash[..HASH_LENGTH.min(hash.len())];
    let keep = MAX_TOOL_NAME_LENGTH - hash.len() - 1;
    format!("{}_{hash}", &name[..keep])
}

trait Radix36 {
    fn to_string_radix(self, radix: u32) -> String;
}

impl Radix36 for u64 {
    fn to_string_radix(mut self, radix: u32) -> String {
        const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut buf = [0u8; 13];
        let mut i = buf.len();
        loop {
            i -= 1;
            buf[i] = DIGITS[(self % u64::from(radix)) as usize];
            self /= u64::from(radix);
            if self == 0 {
                break;
            }
        }
        String::from_utf8_lossy(&buf[i..]).into_owned()
    }
}

/// Mint the registry key for an MCP tool (current, digit-keeping mint).
#[must_use]
pub fn create_tool_name(server: &str, tool: &str) -> String {
    mint(server, tool, true)
}

/// Mint the pre-fix name when it differs (digits stripped), or `None`.
/// omp consults this for approval policies written against the old key.
#[must_use]
pub fn create_legacy_tool_name(server: &str, tool: &str) -> Option<String> {
    let legacy = mint(server, tool, false);
    (legacy != create_tool_name(server, tool)).then_some(legacy)
}

/// Stable 64-bit hash for the overlong-name suffix. omp uses `Bun.hash`
/// (wyhash); the exact Bun seed is internal and unverifiable from Rust, so
/// this port uses FNV-1a — same contract: deterministic, process-stable,
/// base-36 suffix. Only names over 64 chars are affected.
fn wyhash(key: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in key {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_mint() {
        assert_eq!(
            create_tool_name("puppeteer", "puppeteer_screenshot"),
            "mcp__puppeteer_screenshot"
        );
        assert_eq!(
            create_tool_name("hub", "herald_search"),
            "mcp__hub_herald_search"
        );
        assert_eq!(
            create_tool_name("My Server!", "do-thing2"),
            "mcp__my_server_do_thing2"
        );
    }

    #[test]
    fn legacy_mint_differs_only_with_digits() {
        assert_eq!(create_legacy_tool_name("hub", "search"), None);
        assert_eq!(
            create_legacy_tool_name("srv2", "tool"),
            Some("mcp__srv_tool".to_string())
        );
    }

    #[test]
    fn overlong_names_are_capped_stably() {
        let server = "a".repeat(40);
        let tool = "b".repeat(40);
        let name = create_tool_name(&server, &tool);
        assert!(name.len() <= MAX_TOOL_NAME_LENGTH);
        assert_eq!(name, create_tool_name(&server, &tool));
        assert_ne!(name, create_tool_name(&server, &format!("{tool}x")));
    }
}
