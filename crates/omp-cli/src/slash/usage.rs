//! `/usage` slash command: provider usage lines, resets and the TUI usage panel frame.

use super::*;

// =============================================================================
// /usage, /stats, /trace, /changelog (builtin-session.ts)
// =============================================================================

/// `/usage [show|reset ...]` — session token ledger (`session_stats`) plus, for the
/// anthropic provider, the OAuth rate-limit report polled from `/api/oauth/usage`
/// (TS `usage/claude.ts`) and saved reset credits discovered via the cedar/juniper
/// probes (`usage/claude-reset.ts`). `/usage reset` spends one server-selected credit.
pub(super) async fn usage_command(rt: &Runtime, args: &str) -> Outcome {
    let (verb, rest) = subcommand(args);
    match verb.as_str() {
        "" | "show" if rest.is_empty() => usage_show(rt, true, true).await,
        "session" if rest.is_empty() => usage_show(rt, true, false).await,
        "provider" if rest.is_empty() => usage_show(rt, false, true).await,
        "reset" => usage_reset(rt, &rest).await,
        _ => usage(
            rt,
            "Usage: /usage [show|session|provider|reset [provider/credential-id|provider/active]]",
        ),
    }
}

async fn usage_show(rt: &Runtime, session: bool, provider: bool) -> Outcome {
    let mut lines = Vec::new();
    if session {
        let stats = rt.session_stats().await;
        let tokens = &stats["tokens"];
        lines.extend([
            "Session token usage:".to_string(),
            format!("  input:       {}", tokens["input"].as_u64().unwrap_or(0)),
            format!("  output:      {}", tokens["output"].as_u64().unwrap_or(0)),
            format!(
                "  cache read:  {}",
                tokens["cacheRead"].as_u64().unwrap_or(0)
            ),
            format!(
                "  cache write: {}",
                tokens["cacheWrite"].as_u64().unwrap_or(0)
            ),
            format!("  total:       {}", tokens["total"].as_u64().unwrap_or(0)),
            format!(
                "  cost:        {:.4}",
                stats["cost"].as_f64().unwrap_or(0.0)
            ),
        ]);
        if let Some(window) = stats["contextUsage"]["contextWindow"].as_u64() {
            lines.push(format!("  context window: {window} tokens"));
        }
    }
    if provider {
        lines.extend(provider_usage_lines(rt).await);
    }
    output(rt, lines.join("\n").trim_start());
    Handled {
        agent_invoked: false,
    }
}

const USAGE_PROVIDERS: [&str; 6] = [
    "anthropic",
    "kimi-code",
    "zai",
    "openai-codex",
    "google-antigravity",
    "google-gemini-cli",
];

/// Active provider plus its rate-limit report: `Ok(None)` = provider has no
/// usage endpoint, `Err` = fetch/credential failure.
async fn fetch_active_usage(
    rt: &Runtime,
) -> (
    String,
    Result<Option<omp_providers::usage::UsageReport>, String>,
) {
    let provider = rt
        .model_info()
        .get("provider")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();
    if provider.is_empty() || !USAGE_PROVIDERS.contains(&provider.as_str()) {
        return (provider, Ok(None));
    }
    let Some(cred) = omp_providers::auth::credential_row_async(
        omp_providers::AgentDir::default(),
        provider.clone(),
    )
    .await
    else {
        return (provider, Err("no credential".to_string()));
    };
    let cred_input = omp_providers::usage::UsageCredential {
        access_token: Some(&cred.0.key),
        raw_key: Some(&cred.0.key),
        data: Some(&cred.1),
    };
    let report = omp_providers::usage::fetch_provider_usage(&provider, &cred_input, None)
        .await
        .map(Some)
        .map_err(|e| format!("{e:#}"));
    (provider, report)
}

fn unsupported_usage_note(provider: &str) -> String {
    format!(
        "Provider rate-limit usage for `{provider}` is not available (supported: {}).",
        USAGE_PROVIDERS.join(", ")
    )
}

/// Render the anthropic OAuth usage report + reset-credit inventory, one line per
/// window; empty when the active provider has no usage endpoint or credential.
// invariant: write!/writeln! to a String is infallible.
#[allow(clippy::expect_used)]
async fn provider_usage_lines(rt: &Runtime) -> Vec<String> {
    let (provider, report) = fetch_active_usage(rt).await;
    let mut lines = vec![String::new()];
    match report {
        Ok(None) => {
            lines.push(unsupported_usage_note(&provider));
            return lines;
        }
        Ok(Some(report)) => {
            lines.push(format!("Provider rate-limit usage ({provider}):"));
            for line in &report.lines {
                let mut s = line.label.clone();
                if let Some(p) = line.used_percent {
                    write!(s, ": {p:.0}%").expect("writing to a String is infallible");
                }
                if let Some(r) = line.remaining {
                    write!(s, ", {r} left").expect("writing to a String is infallible");
                }
                if let Some(ms) = line.resets_at {
                    s.push_str(&fmt_reset(ms));
                }
                if let Some(n) = &line.note {
                    write!(s, ": {n}").expect("writing to a String is infallible");
                }
                lines.push(format!("  {s}"));
            }
            for n in &report.notes {
                lines.push(format!("  note: {n}"));
            }
        }
        Err(e) => {
            lines.push(format!("Provider rate-limit usage ({provider}):"));
            lines.push(format!("  unavailable: {e}"));
        }
    }
    lines
}

/// Interactive `/usage` panel: a `session_info` frame titled "Usage"
/// (session tokens + per-window provider limits with a fill bar), opened by
/// the TUI in its info overlay (TS `handleUsageCommand` → usage dashboard).
pub(crate) async fn usage_panel_frame(rt: &Runtime) -> Value {
    let kv = |k: &str, v: String| json!({"k": k, "v": v});
    let stats = rt.session_stats().await;
    let tokens = &stats["tokens"];
    let n = |key: &str| tokens[key].as_u64().unwrap_or(0).to_string();
    let mut lines = vec![
        json!({"head": "Session tokens"}),
        kv("Input", n("input")),
        kv("Output", n("output")),
        kv("Cache read", n("cacheRead")),
        kv("Cache write", n("cacheWrite")),
        kv("Total", n("total")),
        kv(
            "Cost",
            format!("${:.4}", stats["cost"].as_f64().unwrap_or(0.0)),
        ),
    ];
    if let Some(window) = stats["contextUsage"]["contextWindow"].as_u64() {
        lines.push(kv("Context window", format!("{window} tokens")));
    }
    lines.push(json!({"text": ""}));
    let (provider, report) = fetch_active_usage(rt).await;
    match report {
        Ok(None) => lines.push(json!({"text": unsupported_usage_note(&provider), "dim": true})),
        Err(e) => {
            lines.push(json!({"head": format!("Provider limits ({provider})")}));
            lines.push(
                json!({"k": "Status", "vs": [{"t": format!("unavailable: {e}"), "c": "error"}]}),
            );
        }
        Ok(Some(report)) => {
            lines.push(json!({"head": format!("Provider limits ({provider})")}));
            for line in &report.lines {
                let mut spans = Vec::new();
                if let Some(p) = line.used_percent {
                    const BAR: usize = 20;
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss
                    )] // BAR is a 20-cell constant, exact in f64
                    let filled = ((p.clamp(0.0, 100.0) / 100.0) * BAR as f64).round() as usize;
                    let color = if p >= 90.0 {
                        "error"
                    } else if p >= 70.0 {
                        "warning"
                    } else {
                        "success"
                    };
                    spans.push(json!({"t": "█".repeat(filled), "c": color}));
                    spans.push(json!({"t": "░".repeat(BAR - filled), "c": "dim"}));
                    spans.push(json!({"t": format!(" {p:.0}%")}));
                }
                if let Some(r) = line.remaining {
                    spans.push(json!({"t": format!(" · {r} left")}));
                }
                if let Some(ms) = line.resets_at {
                    spans.push(json!({"t": fmt_reset(ms), "c": "dim"}));
                }
                if let Some(note) = &line.note {
                    spans.push(json!({"t": format!(" · {note}"), "c": "dim"}));
                }
                lines.push(json!({"k": line.label, "vs": spans}));
            }
            for note in &report.notes {
                lines.push(json!({"text": format!("note: {note}"), "dim": true}));
            }
        }
    }
    lines.push(json!({"text": ""}));
    lines.push(json!({"text": "/usage reset [provider/credential-id|provider/active] spends a saved reset", "dim": true}));
    json!({"type": "session_info", "title": "Usage", "lines": lines})
}

#[allow(clippy::many_single_char_names)] // Hinnant civil-from-days uses its canonical single-letter names
fn fmt_reset(ms: i64) -> String {
    let secs = ms / 1000;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, m, _s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // days-from-civil inverse, enough for display purposes.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, mo) = (
        doy - (153 * mp + 2) / 5 + 1,
        if mp < 10 { mp + 3 } else { mp - 9 },
    );
    let y = if mo <= 2 { y + 1 } else { y };
    format!(" (resets {y}-{mo:02}-{d:02} {h:02}:{m:02})")
}

/// `/usage reset [provider/...]` — spend the server-selected saved reset credit.
#[allow(clippy::too_many_lines)] // reset flow: pick provider, confirm, spend the credit, report
async fn usage_reset(rt: &Runtime, rest: &str) -> Outcome {
    let provider = rt
        .model_info()
        .get("provider")
        .and_then(|p| p.as_str())
        .unwrap_or("anthropic")
        .to_string();
    // The only argument form TS documents is a provider/credential selector; anything
    // naming another provider means "not available there".
    let requested = rest.split('/').next().unwrap_or("").trim().to_string();
    if !requested.is_empty() && requested != provider && requested != "active" {
        return usage(
            rt,
            &format!("Saved rate-limit resets are not available for `{requested}` (active provider is `{provider}`)."),
        );
    }
    if !omp_providers::usage::supports_reset_credits(&provider) {
        return usage(
            rt,
            &format!("Saved rate-limit resets are not supported for provider `{provider}`."),
        );
    }
    let Some(credential) =
        omp_providers::auth::credential_async(omp_providers::AgentDir::default(), provider.clone())
            .await
    else {
        return usage(
            rt,
            &format!("No credential for `{provider}`; login first (`/login`)."),
        );
    };
    let token = credential.key;
    let list = if provider == "anthropic" {
        omp_providers::usage::anthropic_reset_credits(&token, None).await
    } else {
        omp_providers::usage::codex_reset_credits(&token, None).await
    };
    let Ok(Some(list)) = list else {
        return usage(
            rt,
            "Could not discover saved reset credits (discovery failed).",
        );
    };
    let usable: Vec<&omp_providers::usage::ResetCredit> = list
        .credits
        .iter()
        .filter(|c| c.usable && c.remaining_count.unwrap_or(0) > 0)
        .collect();
    let credit = if crate::slash_extra::dialog::has_ui() && !usable.is_empty() {
        // TS `showResetUsageSelector`: pick which saved credit to spend (server pick first).
        use crate::slash_extra::dialog::{select_with, Choice, SelectStyle};
        let choices: Vec<Choice> = usable
            .iter()
            .map(|c| {
                let next = if list.next_credit_id.as_deref() == Some(c.id.as_str()) {
                    " · server pick"
                } else {
                    ""
                };
                Choice::with(
                    format!("{} ({})", c.id, c.program),
                    format!(
                        "{} left · clears {}{next}",
                        c.remaining_count.unwrap_or(0),
                        c.clears.join(", ")
                    ),
                )
            })
            .collect();
        let initial = usable
            .iter()
            .position(|c| list.next_credit_id.as_deref() == Some(c.id.as_str()))
            .unwrap_or(0);
        let style = SelectStyle {
            initial,
            ..SelectStyle::default()
        };
        let Some(index) = select_with(rt, "Spend a saved rate-limit reset", &choices, style).await
        else {
            return Handled {
                agent_invoked: false,
            };
        };
        usable[index].clone()
    } else {
        let selected = list
            .next_credit_id
            .as_ref()
            .and_then(|id| list.credits.iter().find(|c| &c.id == id).cloned())
            .or_else(|| usable.first().map(|c| (*c).clone()));
        let Some(credit) = selected else {
            return usage(rt, "No usable reset credit discovered.");
        };
        credit
    };
    let outcome = if provider == "anthropic" {
        let Some(org_id) = omp_providers::usage::resolve_organization(&token, None).await else {
            return usage(
                rt,
                "Could not resolve the account organization id from /api/oauth/profile.",
            );
        };
        omp_providers::usage::consume_anthropic_reset_credit(&token, None, &org_id, &credit).await
    } else {
        omp_providers::usage::consume_codex_reset_credit(&token, None, &credit).await
    };
    let message = match outcome.code.as_str() {
        "reset" => format!(
            "Rate limit reset spent ({}): cleared {}{}",
            credit.id,
            outcome.cleared.join(", "),
            outcome
                .reason
                .as_deref()
                .map(|r| format!(" — {r}"))
                .unwrap_or_default()
        ),
        "already_redeemed" => "That reset credit was already redeemed.".to_string(),
        "nothing_to_reset" => "No window is currently limited; nothing to reset.".to_string(),
        "rate_limited" => "Reset endpoint rate-limited; try again in a minute.".to_string(),
        "auth_error" => "Auth rejected by the reset endpoint; re-login.".to_string(),
        other => format!(
            "Reset failed ({other}).{}",
            outcome
                .reason
                .as_deref()
                .map(|r| format!(" {r}"))
                .unwrap_or_default()
        ),
    };
    output(rt, &message);
    Handled {
        agent_invoked: false,
    }
}
