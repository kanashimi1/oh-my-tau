//! `/budget`: per-session cost/token budget with automatic stop.
//!
//! `/budget $2.50` caps total cost, `/budget 200k` caps input+output tokens
//! (`k`/`m` suffixes, plain number = tokens), `/budget off` clears the limit,
//! bare `/budget` prints spent vs limit. Once a limit is exceeded the agent is
//! aborted after the assistant turn and new prompts are refused until the
//! limit is raised or cleared. The live state lives in module-level statics so
//! the `runtime.rs` hook stays a two-line call.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use serde_json::{json, Value};

use crate::rpc::slash::Outcome;
use crate::runtime::Runtime;

/// A budget ceiling: total USD spent, or total input+output tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Limit {
    Cost(f64),
    Tokens(u64),
}

impl std::fmt::Display for Limit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Limit::Cost(c) => write!(f, "${c:.2}"),
            Limit::Tokens(t) => write!(f, "{}", format_count(*t)),
        }
    }
}

/// What the user asked `/budget` to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetAction {
    Set(Limit),
    Off,
    Status,
}

/// Parse a `/budget` argument. Empty means status.
pub fn parse_action(arg: &str) -> Result<BudgetAction, String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Ok(BudgetAction::Status);
    }
    if arg.eq_ignore_ascii_case("off") {
        return Ok(BudgetAction::Off);
    }
    parse_limit(arg).map(BudgetAction::Set)
}

/// Parse one limit value: `$12.50` (cost) or `200k` / `1.5m` / `5000` (tokens).
// The token-path casts are range-checked just below (`u64::MAX as f64` rounds
// up, so the bound is conservative); the `$` cost path intentionally keeps f64.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn parse_limit(arg: &str) -> Result<Limit, String> {
    if let Some(rest) = arg.strip_prefix('$') {
        let v: f64 = rest
            .trim()
            .parse()
            .map_err(|_| format!("Invalid cost amount: {arg} (try $2.50)"))?;
        if !v.is_finite() || v <= 0.0 {
            return Err(format!("Cost limit must be positive: {arg}"));
        }
        return Ok(Limit::Cost(v));
    }
    let (num, mult, rest) = suffix_value(arg)?;
    if !rest.is_empty() {
        return Err(format!("Invalid budget value: {arg} (try 200k or $2.50)"));
    }
    if num <= 0.0 {
        return Err(format!("Token limit must be positive: {arg}"));
    }
    let tokens = num * mult;
    // `u64::MAX as f64` rounds up, so this is a conservative upper bound; the
    // final cast is range-checked (finite, positive, <= u64::MAX) above.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    if tokens > u64::MAX as f64 {
        return Err(format!("Token limit too large: {arg}"));
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    Ok(Limit::Tokens(tokens as u64))
}

/// Split a trailing `k`/`m` suffix: returns (value, multiplier, leftover).
fn suffix_value(arg: &str) -> Result<(f64, f64, &str), String> {
    let trimmed = arg.trim();
    let (num_part, mult) = match trimmed.chars().last() {
        Some(c) if c == 'k' || c == 'K' => (&trimmed[..trimmed.len() - 1], 1_000.0),
        Some(c) if c == 'm' || c == 'M' => (&trimmed[..trimmed.len() - 1], 1_000_000.0),
        _ => (trimmed, 1.0),
    };
    let num: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| format!("Invalid budget value: {arg} (try 200k or $2.50)"))?;
    Ok((num, mult, ""))
}

/// Human-readable count: 1234 -> "1.2k".
// The casts are display-only approximations; full u64 precision is not needed.
#[allow(clippy::cast_precision_loss)]
fn format_count(v: u64) -> String {
    if v >= 1_000_000 && v.is_multiple_of(100_000) {
        format!("{}m", trim_num(v as f64 / 1_000_000.0))
    } else if v >= 10_000 && v.is_multiple_of(1_000) {
        format!("{}k", trim_num(v as f64 / 1_000.0))
    } else {
        v.to_string()
    }
}

fn trim_num(v: f64) -> String {
    let s = format!("{v:.1}");
    s.strip_suffix(".0").unwrap_or(&s).to_string()
}

static LIMIT: OnceLock<Mutex<Option<Limit>>> = OnceLock::new();
static TRIPPED: AtomicBool = AtomicBool::new(false);

fn limit_slot() -> &'static Mutex<Option<Limit>> {
    LIMIT.get_or_init(|| Mutex::new(None))
}

/// Current budget limit, if set.
pub fn current_limit() -> Option<Limit> {
    *limit_slot().lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether the budget was exceeded and prompts are being refused.
pub fn is_tripped() -> bool {
    TRIPPED.load(Ordering::SeqCst)
}

/// Sync check used by the `runtime.rs` hooks: a limit is set and should gate.
pub fn is_armed() -> bool {
    current_limit().is_some()
}

/// Refusal message for a new prompt while over budget, or `None` to admit it.
pub fn prompt_refusal() -> Option<String> {
    if is_tripped() {
        Some(
            "Budget limit reached: prompt refused. Raise the limit (`/budget <value>`) or clear it (`/budget off`)."
                .to_string(),
        )
    } else {
        None
    }
}

/// Extract (cost, input+output tokens) from a `session_stats` payload.
pub fn spent_from_stats(stats: &Value) -> (f64, u64) {
    let cost = stats["cost"].as_f64().unwrap_or(0.0);
    let tokens = stats["tokens"]["input"].as_u64().unwrap_or(0)
        + stats["tokens"]["output"].as_u64().unwrap_or(0);
    (cost, tokens)
}

/// Whether the measured spend exceeds the limit.
pub fn exceeds(limit: &Limit, cost: f64, tokens: u64) -> bool {
    match limit {
        Limit::Cost(c) => cost >= *c,
        Limit::Tokens(t) => tokens >= *t,
    }
}

/// After each assistant turn (`runtime.rs` hook): if the limit is now exceeded,
/// set the trip flag, abort the run and warn. Cheap no-op unless armed.
pub async fn check_after_turn(rt: &Arc<Runtime>) {
    let Some(limit) = current_limit() else {
        return;
    };
    if TRIPPED.load(Ordering::SeqCst) {
        return;
    }
    let stats = rt.session_stats().await;
    let (cost, tokens) = spent_from_stats(&stats);
    if !exceeds(&limit, cost, tokens) {
        return;
    }
    TRIPPED.store(true, Ordering::SeqCst);
    rt.abort().await;
    rt.emit_notice(
        "warning",
        &format!("Budget limit reached ({limit}): agent stopped; new prompts are refused until the limit is raised or cleared."),
        Some("budget"),
    );
}

fn format_spent(limit: &Limit, cost: f64, tokens: u64) -> String {
    match limit {
        Limit::Cost(c) => format!("${cost:.2} of ${c:.2}"),
        Limit::Tokens(t) => format!("{} of {} tokens", format_count(tokens), format_count(*t)),
    }
}

/// Slash-command entry point.
pub async fn command(rt: &Arc<Runtime>, args: &str) -> Outcome {
    let action = match parse_action(args) {
        Ok(a) => a,
        Err(e) => {
            rt.send(json!({"type": "command_output", "text": e}));
            return Outcome::Handled {
                agent_invoked: false,
            };
        }
    };
    let text = match action {
        BudgetAction::Off => {
            *limit_slot().lock().unwrap_or_else(PoisonError::into_inner) = None;
            TRIPPED.store(false, Ordering::SeqCst);
            "Budget cleared.".to_string()
        }
        BudgetAction::Set(limit) => {
            *limit_slot().lock().unwrap_or_else(PoisonError::into_inner) = Some(limit);
            // Raising a limit re-admits prompts.
            TRIPPED.store(false, Ordering::SeqCst);
            let stats = rt.session_stats().await;
            let (cost, tokens) = spent_from_stats(&stats);
            format!(
                "Budget set: {}. Spent: {}.",
                limit,
                format_spent(&limit, cost, tokens)
            )
        }
        BudgetAction::Status => match current_limit() {
            Some(limit) => {
                let stats = rt.session_stats().await;
                let (cost, tokens) = spent_from_stats(&stats);
                if TRIPPED.load(Ordering::SeqCst) {
                    format!(
                        "Budget EXCEEDED: {}. New prompts are refused until the limit is raised or cleared.",
                        format_spent(&limit, cost, tokens)
                    )
                } else {
                    format!("Budget: {}.", format_spent(&limit, cost, tokens))
                }
            }
            None => "Budget: no limit set. Usage: /budget $2.50 | /budget 200k | /budget off"
                .to_string(),
        },
    };
    rt.send(json!({"type": "command_output", "text": text}));
    Outcome::Handled {
        agent_invoked: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cost_limits() {
        assert_eq!(parse_limit("$2.50"), Ok(Limit::Cost(2.5)));
        assert_eq!(parse_limit("$10"), Ok(Limit::Cost(10.0)));
        assert!(parse_limit("$0").is_err());
        assert!(parse_limit("$-1").is_err());
        assert!(parse_limit("$abc").is_err());
    }

    #[test]
    fn parses_token_suffixes() {
        assert_eq!(parse_limit("200k"), Ok(Limit::Tokens(200_000)));
        assert_eq!(parse_limit("1.5m"), Ok(Limit::Tokens(1_500_000)));
        assert_eq!(parse_limit("5000"), Ok(Limit::Tokens(5_000)));
        assert_eq!(parse_limit(" 200K "), Ok(Limit::Tokens(200_000)));
        assert!(parse_limit("k").is_err());
        assert!(parse_limit("12x").is_err());
        assert!(parse_limit("-5k").is_err());
    }

    #[test]
    fn parses_actions() {
        assert_eq!(parse_action(""), Ok(BudgetAction::Status));
        assert_eq!(parse_action("  "), Ok(BudgetAction::Status));
        assert_eq!(parse_action("off"), Ok(BudgetAction::Off));
        assert_eq!(parse_action("OFF"), Ok(BudgetAction::Off));
        assert_eq!(parse_action("$3"), Ok(BudgetAction::Set(Limit::Cost(3.0))));
        assert!(parse_action("bogus").is_err());
    }

    #[test]
    fn limit_check_uses_input_plus_output() {
        let stats = json!({
            "cost": 1.0,
            "tokens": {"input": 90_000, "output": 20_000, "cacheRead": 500_000},
        });
        let (cost, tokens) = spent_from_stats(&stats);
        assert!((cost - 1.0).abs() < f64::EPSILON);
        assert_eq!(tokens, 110_000);
        assert!(exceeds(&Limit::Tokens(100_000), cost, tokens));
        assert!(!exceeds(&Limit::Tokens(110_001), cost, tokens));
        assert!(exceeds(&Limit::Cost(1.0), cost, tokens));
        assert!(!exceeds(&Limit::Cost(1.01), cost, tokens));
    }

    #[test]
    fn empty_stats_are_zero() {
        let (cost, tokens) = spent_from_stats(&json!({}));
        assert_eq!((cost, tokens), (0.0, 0));
    }

    #[test]
    fn refusal_only_when_tripped() {
        // Statics are process-global: keep this test order-independent by
        // resetting what it touches.
        TRIPPED.store(false, Ordering::SeqCst);
        assert!(prompt_refusal().is_none());
        TRIPPED.store(true, Ordering::SeqCst);
        assert!(prompt_refusal().is_some());
        TRIPPED.store(false, Ordering::SeqCst);
    }
}
