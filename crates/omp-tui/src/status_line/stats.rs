//! Token/cost/context/quota segments — the rest of
//! `status-line/segments.ts` (`singleStatSegment` family, `context_pct`,
//! `usage`).

use super::metrics as fmt;
use super::presets::SegmentId;
use super::segments::{Ctx, Piece};
use super::style;
use crate::theme::Theme;

/// Render a stats segment; `None` means hidden.
#[must_use]
pub fn render(id: SegmentId, ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    match id {
        SegmentId::TokenIn => single(
            ctx,
            theme,
            "icon.input",
            "⤵",
            ctx.st.tokens.input,
            "statusLineSpend",
        ),
        SegmentId::TokenOut => single(
            ctx,
            theme,
            "icon.output",
            "⤴",
            ctx.st.tokens.output,
            "statusLineOutput",
        ),
        SegmentId::CacheRead => single(
            ctx,
            theme,
            "icon.cache",
            "💾",
            ctx.st.tokens.cache_read,
            "statusLineSpend",
        ),
        SegmentId::CacheWrite => single(
            ctx,
            theme,
            "icon.cache",
            "💾",
            ctx.st.tokens.cache_write,
            "statusLineOutput",
        ),
        SegmentId::TokenRate => token_rate(ctx, theme),
        SegmentId::TokenTotal => token_total(ctx, theme),
        SegmentId::CacheHit => cache_hit(ctx, theme),
        SegmentId::Cost => cost(ctx),
        SegmentId::ContextPct => context_pct(ctx, theme),
        SegmentId::ContextTotal => context_total(ctx, theme),
        SegmentId::Usage => usage(ctx, theme),
        _ => None,
    }
}

/// `singleStatSegment` (`segments.ts:110`): icon + compact number, hidden
/// at 0.
fn single(
    _ctx: &Ctx<'_>,
    theme: &Theme,
    icon_name: &'static str,
    default_icon: &'static str,
    value: u64,
    token: &'static str,
) -> Option<Vec<Piece>> {
    if value == 0 {
        return None;
    }
    Some(vec![Piece::new(
        token,
        format!(
            "{} {}",
            style::icon(theme, icon_name, default_icon),
            fmt::format_number(value)
        ),
    )])
}

/// `token_total` (`segments.ts:733-756`): input+output+cacheWrite+
/// orchestration, cacheRead excluded.
fn token_total(ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    let t = &ctx.st.tokens;
    let total = t.input + t.output + t.cache_write + t.orchestration_input + t.orchestration_output;
    if total == 0 {
        return None;
    }
    Some(vec![Piece::new(
        "statusLineSpend",
        format!(
            "{} {}",
            style::icon(theme, "icon.tokens", "🪙"),
            fmt::format_number(total)
        ),
    )])
}

/// `token_rate` (`segments.ts:758-781`): `⚡ 23.4 tok/s`.
fn token_rate(ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    let rate = ctx.st.tokens_per_second?;
    if rate <= 0.0 {
        return None;
    }
    Some(vec![Piece::new(
        "statusLineOutput",
        format!(
            "{} {:.1} tok/s",
            style::icon(theme, "icon.throughput", "⚡"),
            rate
        ),
    )])
}

/// `cache_hit` (`segments.ts:993-1019`): `💾 97.42%`.
#[allow(clippy::cast_precision_loss)]
fn cache_hit(ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    let t = &ctx.st.tokens;
    if t.cache_read == 0 {
        return None;
    }
    let denom = t.cache_read + t.cache_write + t.input;
    if denom == 0 {
        return None;
    }
    let rate = t.cache_read as f64 / denom as f64 * 100.0;
    Some(vec![Piece::new(
        "statusLineSpend",
        format!("{} {rate:.2}%", style::icon(theme, "icon.cache", "💾")),
    )])
}

/// `cost` — billing summary (`segments.ts:783-830`).
fn cost(ctx: &Ctx<'_>) -> Option<Vec<Piece>> {
    let part = fmt::BillingPart {
        cost: ctx.st.tokens.cost,
        subagent_cost: ctx.st.tokens.subagent_cost.max(ctx.st.subagent_tree_cost),
        premium_requests: ctx.st.tokens.premium_requests,
        subscription: ctx.st.using_subscription,
    };
    let text = fmt::format_billing_summary(&part)?;
    Some(vec![Piece::new("statusLineCost", text)])
}

/// Context usage level (`chrome/context-thresholds.ts:6-54`): fires at the
/// percent **or** token threshold, whichever comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextLevel {
    /// Below all thresholds.
    Normal,
    /// ≥50% or ≥150K.
    Warning,
    /// ≥70% or ≥270K.
    Purple,
    /// ≥90% or ≥500K.
    Error,
}

/// Classify `(percent, tokens)` into a level.
#[must_use]
pub fn context_level(percent: Option<f64>, tokens: Option<u64>) -> ContextLevel {
    let by_pct = percent.is_some_and(|p| p >= 90.0);
    let by_tok = tokens.is_some_and(|t| t >= 500_000);
    if by_pct || by_tok {
        return ContextLevel::Error;
    }
    if percent.is_some_and(|p| p >= 70.0) || tokens.is_some_and(|t| t >= 270_000) {
        return ContextLevel::Purple;
    }
    if percent.is_some_and(|p| p >= 50.0) || tokens.is_some_and(|t| t >= 150_000) {
        return ContextLevel::Warning;
    }
    ContextLevel::Normal
}

fn level_token(level: ContextLevel) -> &'static str {
    match level {
        ContextLevel::Normal => "statusLineContext",
        ContextLevel::Warning => "warning",
        ContextLevel::Purple => "thinkingHigh",
        ContextLevel::Error => "error",
    }
}

/// `context_pct` (`segments.ts:832-887`): `<pct>%/<window>`, `<window>`
/// alone when pct unknown, `<tokens>/?` when the window is unknown.
fn context_pct(ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    let window = ctx.st.context_window;
    let pct = ctx.st.context_percent;
    let level = context_level(pct, ctx.st.context_tokens);
    let mut token = level_token(level);
    // Auto-compact speculation blink (`segments.ts:861`): `armed` holds the
    // accent; `running` blinks between accent and muted.
    match ctx.st.speculation.as_ref() {
        Some(super::Speculation::Armed) => token = "accent",
        Some(super::Speculation::Running) => {
            token = if ctx.st.speculation_blink_on {
                "accent"
            } else {
                "muted"
            };
        }
        None => {}
    }
    let window_s = window.map(fmt::format_number);
    let text = match (pct, window_s) {
        (Some(p), Some(w)) => format!("{} {:.1}%/{w}", style::icon(theme, "icon.context", "◫"), p),
        (None, Some(w)) => {
            format!("{} {w}", style::icon(theme, "icon.context", "◫"))
        }
        (Some(p), None) => format!("{} {:.1}%/?", style::icon(theme, "icon.context", "◫"), p),
        (None, None) => return None,
    };
    Some(vec![Piece::new(token, text)])
}

/// `context_total` (`segments.ts:889-911`): `◫ <window>`.
fn context_total(ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    let window = ctx.st.context_window?;
    if window == 0 {
        return None;
    }
    Some(vec![Piece::new(
        "statusLineContext",
        format!(
            "{} {}",
            style::icon(theme, "icon.context", "◫"),
            fmt::format_number(window)
        ),
    )])
}

/// One provider quota window (`usage` segment, `segments.ts:1134-1275`).
#[derive(Debug, Clone, Copy)]
pub struct QuotaWindow {
    /// Short label (`5h`, `1d`, `7d`, `mo`).
    pub label: &'static str,
    /// Used percent (0–100+).
    pub percent: f64,
    /// Minutes until reset.
    pub reset_minutes: u64,
}

/// `usage` — provider quota display, joined with ` · `.
fn usage(ctx: &Ctx<'_>, theme: &Theme) -> Option<Vec<Piece>> {
    if ctx.st.usage_windows.is_empty() {
        return None;
    }
    let parts = ctx
        .st
        .usage_windows
        .iter()
        .map(|w| {
            let reset = if w.label == "5h" || w.label == "1d" {
                let h = w.reset_minutes / 60;
                let m = w.reset_minutes % 60;
                if h > 0 {
                    format!("{h}h {m}m")
                } else {
                    format!("{m}m")
                }
            } else {
                format!("{}h", w.reset_minutes / 60)
            };
            let pct = if w.label == "mo" {
                w.percent.floor()
            } else {
                w.percent.round()
            };
            format!("{} {pct}% ({reset})", w.label)
        })
        .collect::<Vec<_>>();
    let text = format!(
        "{} {}",
        style::icon(theme, "icon.time", "⏱"),
        parts.join(" · ")
    );
    Some(vec![ctx.maybe_accent("accent", text)])
}
