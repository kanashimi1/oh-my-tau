//! Formatting helpers for status segments — the port of
//! `status-line/metrics.ts`, `chrome/format.ts` and `chrome/shared.ts`.

use std::fmt::Write as _;

/// Display cell width of a string: wide ranges count 2, combining 0.
#[must_use]
pub fn str_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Width of one char (a small `wcwidth`: good enough for status text).
#[must_use]
pub fn char_width(c: char) -> usize {
    let cp = u32::from(c);
    if cp == 0 {
        return 0;
    }
    // Zero-width joiners / combining marks / variation selectors.
    if (0x0300..=0x036F).contains(&cp)
        || (0xFE00..=0xFE0F).contains(&cp)
        || (0x200B..=0x200F).contains(&cp)
    {
        return 0;
    }
    // Wide: CJK, fullwidth forms, most emoji blocks.
    if (0x1100..=0x115F).contains(&cp)
        || (0x2E80..=0xA4CF).contains(&cp)
        || (0xAC00..=0xD7A3).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0xFE30..=0xFE4F).contains(&cp)
        || (0xFF00..=0xFF60).contains(&cp)
        || (0xFFE0..=0xFFE6).contains(&cp)
        || (0x1F300..=0x1F64F).contains(&cp)
        || (0x1F900..=0x1F9FF).contains(&cp)
        || (0x20000..=0x2FFFD).contains(&cp)
        || (0x30000..=0x3FFFD).contains(&cp)
    {
        return 2;
    }
    1
}

/// `formatNumber`: compact `K`/`M` with one decimal (`12.3K`, `4.5M`);
/// values below 1000 render as-is (`metrics.ts`).
#[allow(clippy::cast_precision_loss)] // display math; one-decimal rounding hides the loss
#[must_use]
pub fn format_number(v: u64) -> String {
    if v < 1000 {
        return v.to_string();
    }
    let s = if v < 1_000_000 {
        format!("{:.1}K", v as f64 / 1_000.0)
    } else {
        format!("{:.1}M", v as f64 / 1_000_000.0)
    };
    // TS `formatNumber`: whole values drop the decimal (`200K`, not
    // `200.0K`).
    s.replace(".0K", "K").replace(".0M", "M")
}

/// `formatDuration` (`chrome/format.ts`): `45s`, `1h 23m`, `2d 3h`.
#[must_use]
pub fn format_duration(ms: u64) -> String {
    let secs = ms / 1000;
    let days = secs / 86_400;
    let hours = (secs % 86_400) / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        if mins > 0 {
            format!("{hours}h {mins}m")
        } else {
            format!("{hours}h")
        }
    } else if mins > 0 {
        format!("{mins}m")
    } else {
        format!("{s}s")
    }
}

/// `brandTimer` (`segments.ts:241`): whole-unit timer `12s`, `3m`, `7h`,
/// capped at 99h.
#[must_use]
pub fn format_brand_timer(elapsed_ms: u64) -> String {
    let secs = elapsed_ms / 1000;
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h", (secs / 3600).min(99))
    }
}

/// Collapse whitespace, trim, strip control chars (`chrome/shared.ts:12`).
#[must_use]
pub fn sanitize_status_text(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Left-truncate with a `…` prefix (`clampPathLength`, `segments.ts:39`).
#[must_use]
pub fn clamp_path_length(path: &str, max: usize) -> String {
    if str_width(path) <= max {
        return path.to_string();
    }
    // Left-truncate: keep the tail, mark with a leading ellipsis
    // (`/v/l/project/path` → `…ct/path`).
    let floor = max.max(1);
    let mut kept: Vec<char> = Vec::new();
    let mut w = 1usize; // the ellipsis
    for c in path.chars().rev() {
        let cw = char_width(c);
        if w + cw > floor {
            break;
        }
        kept.push(c);
        w += cw;
    }
    let mut out = String::from("…");
    out.extend(kept.into_iter().rev());
    out
}

/// `shortenPath`: parent dirs become initials, `~/` kept (`path.ts`).
#[must_use]
pub fn shorten_path(path: &str) -> String {
    let home_prefix = path
        .strip_prefix(home_dir().as_str())
        .map(|rest| format!("~/{}", rest.trim_start_matches('/')));
    let p = home_prefix.as_deref().unwrap_or(path);
    let parts: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 2 {
        return p.to_string();
    }
    let mut out = String::new();
    if p.starts_with('/') {
        out.push('/');
    }
    // Keep the final two components whole; initial the parents
    // (`/very/long/project/path` → `/v/l/project/path`).
    let keep_from = parts.len().saturating_sub(2);
    for (i, part) in parts.iter().enumerate() {
        if i >= keep_from {
            out.push_str(part);
            if i == parts.len() - 1 {
                break;
            }
            out.push('/');
        } else if *part == "~" {
            out.push_str(part);
            out.push('/');
        } else {
            let initial = part.chars().next().unwrap_or('?');
            out.push(initial);
            out.push('/');
        }
    }
    out
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_default()
}

/// One billed quantity in the billing summary (§3.4).
#[derive(Debug, Clone, Copy)]
pub struct BillingPart {
    /// Total spend in the session's billing unit.
    pub cost: f64,
    /// Subagent tree spend (appended as `(+x.xx)`).
    pub subagent_cost: f64,
    /// Premium requests ( appended as `★ n` when > 0).
    pub premium_requests: u64,
    /// True when billing is subscription-based (`S<amount>`).
    pub subscription: bool,
}

/// `formatBillingSummary` (`metrics.ts:59-140`): parts joined by single
/// spaces; `None` when there is nothing to bill.
#[must_use]
pub fn format_billing_summary(part: &BillingPart) -> Option<String> {
    if part.cost <= 0.0 && part.subagent_cost <= 0.0 && part.premium_requests == 0 {
        return None;
    }
    let mut out = String::new();
    if part.cost > 0.0 {
        if part.subscription {
            let _ = write!(out, "S{:.2}", part.cost);
        } else {
            let _ = write!(out, "${:.2}", part.cost);
        }
    }
    if part.subagent_cost > 0.0 {
        if !out.is_empty() {
            out.push(' ');
        }
        let _ = write!(out, "(+{:.2})", part.subagent_cost);
    }
    if part.premium_requests > 0 {
        if !out.is_empty() {
            out.push(' ');
        }
        let _ = write!(out, "★ {}", part.premium_requests);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

use super::*;

    #[test]
    fn format_number_compacts() {
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(12_345), "12.3K");
        assert_eq!(format_number(4_567_890), "4.6M");
    }

    #[test]
    fn durations_use_largest_units() {
        assert_eq!(format_duration(45_000), "45s");
        assert_eq!(format_duration(83 * 60_000), "1h 23m");
        assert_eq!(format_duration(900), "0s");
    }

    #[test]
    fn brand_timer_caps_at_99h() {
        assert_eq!(format_brand_timer(12_000), "12s");
        assert_eq!(format_brand_timer(3 * 60_000), "3m");
        assert_eq!(format_brand_timer(10 * 86_400_000), "99h");
    }

    #[test]
    fn sanitize_strips_and_collapses() {
        assert_eq!(sanitize_status_text("  a\t b\u{7} c  "), "a b c");
    }

    #[test]
    fn clamp_left_truncates_with_ellipsis() {
        assert_eq!(clamp_path_length("short", 10), "short");
        let long = clamp_path_length("a/very/long/path/that/keeps/going", 8);
        assert!(str_width(&long) <= 8);
        assert!(long.starts_with('…'));
    }

    #[test]
    fn shorten_path_uses_initials() {
        assert_eq!(shorten_path("/work/repo/src/main"), "/w/r/src/main");
        assert_eq!(shorten_path("/repo"), "/repo");
    }

    #[test]
    fn billing_summary_joins_parts() {
        let p = BillingPart {
            cost: 64.49,
            subagent_cost: 49.07,
            premium_requests: 0,
            subscription: false,
        };
        assert_eq!(
            format_billing_summary(&p).as_deref(),
            Some("$64.49 (+49.07)")
        );
        let empty = BillingPart {
            cost: 0.0,
            subagent_cost: 0.0,
            premium_requests: 0,
            subscription: false,
        };
        assert_eq!(format_billing_summary(&empty), None);
        let sub = BillingPart {
            subscription: true,
            premium_requests: 3,
            ..empty
        };
        assert_eq!(format_billing_summary(&sub).as_deref(), Some("★ 3"));
    }
}
