//! `omp-rs stats` — TS `omp stats` (cli/stats-cli.ts + stats `index.ts`):
//! by default sync in the background and serve the dashboard; `--json` /
//! `--summary` sync fully and print a one-shot report.
//!
//! Also home of the in-process dashboard launcher shared by the `/stats` and
//! `/trace` slash commands (slash-commands/helpers/stats-dashboard.ts).

#![allow(clippy::cast_precision_loss)]

use std::fmt::Write as _;
use std::io::{IsTerminal, Write as _};
use std::sync::Mutex;

use anyhow::{bail, Context as _, Result};
use omp_stats::server::{
    format_stats_dashboard_url, start_server, StartServerOptions, StatsServerHandle, DEFAULT_PORT,
};
use omp_stats::types::DashboardStats;

const USAGE: &str =
    "Usage: omt stats [-p, --port <port>] [--host <host>] [-j, --json] [-s, --summary|--sync]";

/// Parsed `omp stats` flags.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsArgs {
    pub port: u16,
    pub host: String,
    pub json: bool,
    pub summary: bool,
}

/// TS `parseInt(value, 10)`: leading digits (after optional whitespace) are
/// the port, trailing junk is ignored; no digits at all is an error (TS would
/// try to listen on `NaN`).
fn parse_port(value: &str) -> Result<u16> {
    let digits: String = value
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        bail!("Invalid port: {value}");
    }
    digits
        .parse::<u16>()
        .map_err(|_| anyhow::anyhow!("Invalid port: {value}"))
}

/// Parse `omp stats` flags like TS `parseStandaloneStatsArgs`: `-p/--port`
/// (also attached `-p3000`), `--host`, `-j/--json`, `-s/--summary` with the TS
/// spelling `--sync` as an alias; positional arguments are accepted and ignored.
///
/// # Errors
/// Unknown flag, missing value, or an invalid port.
pub fn parse_args(args: &[String]) -> Result<StatsArgs> {
    let mut parsed = StatsArgs {
        port: DEFAULT_PORT,
        host: "127.0.0.1".to_string(),
        json: false,
        summary: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-j" | "--json" => parsed.json = true,
            "-s" | "--summary" | "--sync" => parsed.summary = true,
            "-p" | "--port" => {
                parsed.port = parse_port(it.next().context("--port needs a value")?)?
            }
            "--host" => it
                .next()
                .context("--host needs a value")?
                .clone_into(&mut parsed.host),
            other if other.starts_with("--port=") => {
                parsed.port = parse_port(&other["--port=".len()..])?
            }
            other if other.starts_with("--host=") => {
                parsed.host = other["--host=".len()..].to_string()
            }
            // node `parseArgs` short clusters: `-js`, `-jp3000`, `-sp 4000`.
            other if other.len() > 2 && other.starts_with('-') && !other.starts_with("--") => {
                let cluster = &other[1..];
                for (i, flag) in cluster.char_indices() {
                    match flag {
                        'j' => parsed.json = true,
                        's' => parsed.summary = true,
                        'p' => {
                            let rest = &cluster[i + 1..];
                            let value = if rest.is_empty() {
                                it.next().context("--port needs a value")?.as_str()
                            } else {
                                rest
                            };
                            parsed.port = parse_port(value)?;
                            break;
                        }
                        _ => bail!("Unknown option: -{flag}. {USAGE}"),
                    }
                }
            }
            other if !other.starts_with('-') || other == "-" => {}
            other => bail!("Unknown option: {other}. {USAGE}"),
        }
    }
    Ok(parsed)
}

/// Entry point: `omp-rs stats [options]`.
///
/// # Errors
/// Sync, db, or server start failure.
pub fn run(args: &[String]) -> Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(0);
    }
    let parsed = parse_args(args)?;
    if parsed.json || parsed.summary {
        return one_shot(&parsed);
    }
    // The dashboard starts immediately and ingests sessions in the background,
    // streaming progress to the page.
    super::stats_judge::register();
    super::usage_broker::register();
    let handle = start_server(parsed.port, &parsed.host, StartServerOptions::default())?;
    let url = format_stats_dashboard_url(&handle.hostname, handle.port);
    println!("Dashboard available at: {url}");
    if std::io::stdout().is_terminal() {
        open_in_browser(&url);
    }
    println!("Press Ctrl+C to stop\n");
    // The server runs on its own threads. Ctrl+C closes the db and exits like TS
    // (own thread + runtime: `run` is called from inside the CLI's runtime).
    std::thread::spawn(|| {
        let waited = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map(|rt| rt.block_on(tokio::signal::ctrl_c()));
        if matches!(waited, Ok(Ok(()))) {
            println!("\nShutting down...");
            omp_stats::db::close_db();
            std::process::exit(0);
        }
    });
    loop {
        std::thread::park();
    }
}

/// Best-effort browser open (TS `openPath`); failures are silent.
fn open_in_browser(url: &str) {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(opener)
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// One-shot reports need fully ingested, fully rolled-up data before printing.
fn one_shot(args: &StatsArgs) -> Result<i32> {
    let tty = std::io::stderr().is_terminal();
    eprintln!("Syncing session files...");
    let last_width = std::cell::Cell::new(0usize);
    let last_render = std::cell::Cell::new(
        std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or_else(std::time::Instant::now),
    );
    let progress = |event: &omp_stats::aggregator::SyncProgress| {
        if !tty {
            return;
        }
        let now = std::time::Instant::now();
        if event.current < event.total && now.duration_since(last_render.get()).as_millis() < 33 {
            return;
        }
        last_render.set(now);
        let file = event.session_file.as_str();
        let short = file
            .find("/sessions/")
            .map_or(file, |i| &file[i + "/sessions/".len()..]);
        let pct = (event.current as f64 / event.total.max(1) as f64 * 100.0).round();
        let line = format!("[{}/{}] {pct:>3}%  {short}", event.current, event.total);
        let clipped: String = line.chars().take(119).collect();
        eprint!("\r{clipped:<width$}", width = last_width.get());
        last_width.set(clipped.chars().count());
    };
    let options = omp_stats::aggregator::SyncOptions {
        on_progress: Some(&progress),
        ..Default::default()
    };
    let synced = omp_stats::aggregator::sync_all_sessions(&options)?;
    if tty {
        eprint!("\r{}\r", " ".repeat(120));
    }
    omp_stats::rollup::refresh_rollups()?;
    let total = omp_stats::aggregator::get_total_message_count()?;
    eprintln!(
        "Synced {} new entries from {} files ({total} total)\n",
        synced.processed, synced.files
    );
    let stats = omp_stats::aggregator::get_dashboard_stats(None)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        print_stats_summary(&stats);
    }
    omp_stats::db::close_db();
    Ok(0)
}

// ---------------------------------------------------------------------------
// printStatsSummary + formatters (pi-utils format.ts, client formatters)
// ---------------------------------------------------------------------------

fn trim1(n: f64) -> String {
    let s = format!("{n:.1}");
    s.strip_suffix(".0").map_or(s.clone(), str::to_string)
}

/// pi-utils `formatNumber`.
fn format_number(n: f64) -> String {
    if n < 1_000.0 {
        format!("{n}")
    } else if n < 10_000.0 {
        format!("{}K", trim1(n / 1e3))
    } else if n < 1e6 {
        format!("{}K", (n / 1e3).round())
    } else if n < 1e7 {
        format!("{}M", trim1(n / 1e6))
    } else if n < 1e9 {
        format!("{}M", (n / 1e6).round())
    } else if n < 1e10 {
        format!("{}B", trim1(n / 1e9))
    } else {
        format!("{}B", (n / 1e9).round())
    }
}

fn format_percent(ratio: f64, digits: usize) -> String {
    format!("{:.digits$}%", ratio * 100.0)
}

/// Client `formatErrorRate`: small nonzero rates must not round to zero.
fn format_error_rate(value: f64) -> String {
    let percent = value * 100.0;
    if percent > 0.0 && percent < 0.005 {
        return "<0.01%".to_string();
    }
    format_percent(value, if percent > 0.0 && percent < 0.1 { 2 } else { 1 })
}

/// pi-utils `formatDuration`.
fn format_duration(ms: f64) -> String {
    const SEC: f64 = 1000.0;
    const MIN: f64 = 60.0 * SEC;
    const HOUR: f64 = 60.0 * MIN;
    const DAY: f64 = 24.0 * HOUR;
    if !ms.is_finite() || ms <= 0.0 {
        return "0ms".to_string();
    }
    if ms < SEC {
        return format!("{}ms", ms.floor());
    }
    if ms < MIN {
        let tenths = (ms / 100.0).round();
        return if tenths < 600.0 {
            format!("{:.1}s", tenths / 10.0)
        } else {
            "1m".to_string()
        };
    }
    if ms < HOUR {
        let (mins, secs) = ((ms / MIN).floor(), ((ms % MIN) / SEC).floor());
        return if secs > 0.0 {
            format!("{mins}m{secs}s")
        } else {
            format!("{mins}m")
        };
    }
    if ms < DAY {
        let (hours, mins) = ((ms / HOUR).floor(), ((ms % HOUR) / MIN).floor());
        return if mins > 0.0 {
            format!("{hours}h{mins}m")
        } else {
            format!("{hours}h")
        };
    }
    let (days, hours) = ((ms / DAY).floor(), ((ms % DAY) / HOUR).floor());
    if hours > 0.0 {
        format!("{days}d{hours}h")
    } else {
        format!("{days}d")
    }
}

/// Format an API-equivalent estimate in dollars, or N/A for unpriced usage.
fn format_cost(n: f64, unpriced_requests: i64) -> String {
    if n == 0.0 && unpriced_requests > 0 {
        "N/A".to_string()
    } else if n < 0.01 {
        format!("${n:.4}")
    } else if n < 1.0 {
        format!("${n:.3}")
    } else {
        format!("${n:.2}")
    }
}

fn bold(text: &str) -> String {
    if std::io::stdout().is_terminal() {
        format!("\x1b[1m{text}\x1b[22m")
    } else {
        text.to_string()
    }
}

/// Print the dashboard summary (TS `printStatsSummary`).
fn print_stats_summary(stats: &DashboardStats) {
    let overall = &stats.overall;
    let out = &mut std::io::stdout().lock();
    let _ = writeln!(out, "{}", bold("\n=== AI Usage Statistics ===\n"));
    let _ = writeln!(out, "{}", bold("Overall:"));
    let _ = writeln!(
        out,
        "  Requests: {} ({} errors)",
        format_number(overall.total_requests as f64),
        format_number(overall.failed_requests as f64)
    );
    let _ = writeln!(
        out,
        "  Error Rate: {}",
        format_error_rate(overall.error_rate)
    );
    let _ = writeln!(
        out,
        "  Total Tokens: {}",
        format_number((overall.total_input_tokens + overall.total_output_tokens) as f64)
    );
    let _ = writeln!(
        out,
        "  Input Tokens: {}",
        format_number(overall.total_input_tokens as f64)
    );
    let _ = writeln!(
        out,
        "  Output Tokens: {}",
        format_number(overall.total_output_tokens as f64)
    );
    let _ = writeln!(
        out,
        "  Cache Rate: {}",
        format_percent(overall.cache_rate, 1)
    );
    let _ = writeln!(
        out,
        "  Cache Savings: {}",
        format_percent(overall.cache_savings, 1)
    );
    let _ = writeln!(
        out,
        "  API-equivalent estimate: {}",
        format_cost(overall.total_cost, overall.unpriced_requests)
    );
    let premium = ((overall.total_premium_requests + f64::EPSILON) * 100.0).round() / 100.0;
    let _ = writeln!(out, "  Premium Requests: {}", format_number(premium));
    let _ = writeln!(
        out,
        "  Avg Duration: {}",
        overall
            .avg_duration
            .map_or("-".to_string(), format_duration)
    );
    let _ = writeln!(
        out,
        "  Avg TTFT: {}",
        overall.avg_ttft.map_or("-".to_string(), format_duration)
    );
    if let Some(tps) = overall.avg_tokens_per_second {
        let _ = writeln!(out, "  Avg Tokens/s: {tps:.1}");
    }
    if !stats.by_model.is_empty() {
        let _ = writeln!(out, "{}", bold("\nBy Model (API-equivalent estimates):"));
        for m in stats.by_model.iter().take(10) {
            let b = &m.base;
            let _ = writeln!(
                out,
                "  {}: {} reqs, {}, {} cache rate, {} cache savings",
                m.model,
                format_number(b.total_requests as f64),
                format_cost(b.total_cost, b.unpriced_requests),
                format_percent(b.cache_rate, 1),
                format_percent(b.cache_savings, 1)
            );
        }
    }
    if !stats.by_folder.is_empty() {
        let _ = writeln!(out, "{}", bold("\nBy Folder (API-equivalent estimates):"));
        for f in stats.by_folder.iter().take(10) {
            let b = &f.base;
            let _ = writeln!(
                out,
                "  {}: {} reqs, {}",
                f.folder,
                format_number(b.total_requests as f64),
                format_cost(b.total_cost, b.unpriced_requests)
            );
        }
    }
    let _ = writeln!(out);
}

// ---------------------------------------------------------------------------
// In-process launcher for `/stats` and `/trace`
// ---------------------------------------------------------------------------

const DASHBOARD_USAGE: &str = "Usage: /stats [--port <port>] [--host <host>]";

/// Parse `/stats` arguments (TS `parseStatsDashboardArgs`).
///
/// # Errors
/// The TS error text (shown to the user).
pub fn parse_dashboard_args(args: &str) -> std::result::Result<(u16, String), String> {
    let mut port = DEFAULT_PORT;
    let mut host = "127.0.0.1".to_string();
    let port_of = |v: Option<&str>| -> std::result::Result<u16, String> {
        let v = v
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("Missing port. {DASHBOARD_USAGE}"))?;
        parse_port(v).map_err(|_| format!("Invalid port: {v}"))
    };
    let mut tokens = args.split_whitespace();
    while let Some(token) = tokens.next() {
        if token == "--port" || token == "-p" {
            port = port_of(tokens.next())?;
        } else if let Some(v) = token.strip_prefix("--port=") {
            port = port_of(Some(v))?;
        } else if token == "--host" {
            host = tokens
                .next()
                .ok_or_else(|| format!("Missing host. {DASHBOARD_USAGE}"))?
                .to_string();
        } else if let Some(v) = token.strip_prefix("--host=") {
            if v.is_empty() {
                return Err(format!("Missing host. {DASHBOARD_USAGE}"));
            }
            host = v.to_string();
        } else {
            return Err(format!("Unknown option: {token}. {DASHBOARD_USAGE}"));
        }
    }
    Ok((port, host))
}

static ACTIVE_SERVER: Mutex<Option<StatsServerHandle>> = Mutex::new(None);

/// Start (or reuse) the in-process dashboard. Blocking (port probe): call via
/// `spawn_blocking` from async code. Returns `(url, message)`.
///
/// # Errors
/// Server start failure.
pub fn launch_dashboard(port: u16, host: &str) -> Result<(String, String)> {
    // Like TS `launchStatsDashboard(args, judge)`: every launch (re)registers the host judge
    // and the broker usage source.
    super::stats_judge::register();
    super::usage_broker::register();
    let mut active = ACTIVE_SERVER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut ignored = false;
    let handle = if let Some(handle) = active.as_ref() {
        ignored = port != handle.port || host != handle.hostname;
        handle.clone()
    } else {
        let handle = start_server(port, host, StartServerOptions::default())?;
        // A reused (foreign) dashboard can die any time: probe again next launch.
        if !handle.is_reused() {
            *active = Some(handle.clone());
        }
        handle
    };
    open_in_browser(&format_stats_dashboard_url(&handle.hostname, handle.port));
    let url = format_stats_dashboard_url(&handle.hostname, handle.port);
    let line = if ignored {
        format!("Dashboard already running at: {url} (requested {host}:{port} ignored)")
    } else {
        format!("Dashboard available at: {url}")
    };
    Ok((url, format!("{line} (sessions sync in the background)")))
}

/// JS `encodeURIComponent`.
#[must_use]
pub fn encode_uri_component(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|a| (*a).to_string()).collect()
    }

    #[test]
    fn flags_parse_like_ts() {
        let p = parse_args(&args(&["-p", "0", "--host=::1", "-j"])).expect("parse");
        assert_eq!(
            (p.port, p.host.as_str(), p.json, p.summary),
            (0, "::1", true, false)
        );
        assert_eq!(parse_args(&[]).expect("defaults").port, 3847);
        assert!(parse_args(&args(&["--port", "x"])).is_err());
        assert!(parse_args(&args(&["--bogus"])).is_err());
        // TS spelling and attached short value; positionals are ignored.
        let p = parse_args(&args(&["--sync", "-p3000", "extra", "-j"])).expect("ts flags");
        assert_eq!((p.port, p.summary, p.json), (3000, true, true));
        assert!(parse_args(&args(&["-pabc"])).is_err());
        // node parseArgs short clusters (verified with bun): -js, -jp3000, -sp 4000.
        let p = parse_args(&args(&["-js"])).expect("cluster");
        assert_eq!((p.json, p.summary), (true, true));
        let p = parse_args(&args(&["-jp3000"])).expect("cluster with value");
        assert_eq!((p.json, p.port), (true, 3000));
        let p = parse_args(&args(&["-sp", "4000"])).expect("cluster, value next");
        assert_eq!((p.summary, p.port), (true, 4000));
        // TS parseInt keeps leading digits.
        assert_eq!(
            parse_args(&args(&["--port", "3000extra"]))
                .expect("parseInt")
                .port,
            3000
        );
        assert!(parse_args(&args(&["-jx"])).is_err());
    }

    #[test]
    fn dashboard_args_and_uri_encoding() {
        assert_eq!(
            parse_dashboard_args("--port 99 --host=h"),
            Ok((99, "h".to_string()))
        );
        assert_eq!(
            parse_dashboard_args("--port"),
            Err(format!("Missing port. {DASHBOARD_USAGE}"))
        );
        assert_eq!(
            parse_dashboard_args("--port 70000"),
            Err("Invalid port: 70000".to_string())
        );
        assert!(parse_dashboard_args("zzz").is_err());
        assert_eq!(encode_uri_component("/a b/c.jsonl"), "%2Fa%20b%2Fc.jsonl");
    }

    #[test]
    fn number_formatting_matches_pi_utils() {
        assert_eq!(format_number(999.0), "999");
        assert_eq!(format_number(1500.0), "1.5K");
        assert_eq!(format_number(12_345.0), "12K");
        assert_eq!(format_number(2_000_000.0), "2M");
        assert_eq!(format_duration(61_000.0), "1m1s");
        assert_eq!(format_duration(59_960.0), "1m");
        assert_eq!(format_error_rate(0.00001), "<0.01%");
        assert_eq!(format_cost(0.0, 3), "N/A");
        assert_eq!(format_cost(0.005, 0), "$0.0050");
    }
}
