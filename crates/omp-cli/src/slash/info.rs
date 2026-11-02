//! Info slash commands: `/stats`, `/trace`, `/changelog`.

use super::{output, usage, Handled, Outcome, PathBuf, Runtime};

/// `/stats [--port <port>] [--host <host>]` — start (or reuse) the in-process
/// dashboard; sessions sync in the background (TS `launchStatsDashboard`).
pub(super) async fn stats_command(rt: &Runtime, args: &str) -> Outcome {
    let (port, host) = match crate::subcmd::stats::parse_dashboard_args(args) {
        Ok(parsed) => parsed,
        Err(message) => return usage(rt, &message),
    };
    output(rt, "Syncing session files...");
    match tokio::task::spawn_blocking(move || crate::subcmd::stats::launch_dashboard(port, &host))
        .await
    {
        Ok(Ok((_url, message))) => output(rt, &message),
        Ok(Err(error)) => output(rt, &format!("Stats dashboard failed: {error:#}")),
        Err(error) => output(rt, &format!("Stats dashboard failed: {error}")),
    }
    Handled {
        agent_invoked: false,
    }
}

/// `/trace` — open this session's trace in the stats dashboard (prints the URL).
pub(super) async fn trace_command(rt: &Runtime) -> Outcome {
    let session_file = rt.session_file();
    if !session_file.is_file() {
        output(rt, "No session file yet — send a message first.");
        return Handled {
            agent_invoked: false,
        };
    }
    // TS `/trace`: bare `startServer()` (default port + hostname); not the
    // `/stats` launcher, so no browser open and no cached handle.
    // TS fetchUsageData resolves the broker on every call, so the `/trace`
    // dashboard shows broker-backed provider windows too.
    crate::subcmd::usage_broker::register();
    let started = tokio::task::spawn_blocking(|| {
        omp_stats::server::start_server(
            omp_stats::server::DEFAULT_PORT,
            omp_stats::port_conflict::STATS_DASHBOARD_HOSTNAME,
            omp_stats::server::StartServerOptions::default(),
        )
    })
    .await;
    match started {
        Ok(Ok(handle)) => output(
            rt,
            &format!(
                "{}/#/traces?s={}",
                omp_stats::server::format_stats_dashboard_url(&handle.hostname, handle.port),
                crate::subcmd::stats::encode_uri_component(&session_file.to_string_lossy())
            ),
        ),
        Ok(Err(error)) => output(rt, &format!("Failed to open trace: {error:#}")),
        Err(error) => output(rt, &format!("Failed to open trace: {error}")),
    }
    Handled {
        agent_invoked: false,
    }
}

/// Parse `/changelog [full|last [N]]` (TS `parseChangelogView`).
enum ChangelogView {
    Full,
    Recent(usize),
    Last(usize),
}

fn parse_changelog_view(args: &str) -> Result<ChangelogView, String> {
    const USAGE: &str = "Usage: /changelog [full|last [N]]";
    let trimmed = args.trim().to_lowercase();
    if trimmed.is_empty() {
        return Ok(ChangelogView::Recent(3));
    }
    if trimmed == "full" {
        return Ok(ChangelogView::Full);
    }
    let Some(rest) = trimmed.strip_prefix("last") else {
        return Err(USAGE.to_string());
    };
    let rest = rest.trim();
    if rest.is_empty() {
        return Ok(ChangelogView::Last(1));
    }
    match rest.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(ChangelogView::Last(n)),
        Ok(_) => Err(format!("{USAGE} (N must be a positive integer)")),
        Err(_) => Err(USAGE.to_string()),
    }
}

/// Split changelog markdown into `## `-headed entries (TS
/// `parseChangelogContent`), newest first as written.
fn parse_changelog_content(content: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut current: Option<(String, Vec<String>)> = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            if let Some((header, lines)) = current.take() {
                let text = lines.join("\n").trim().to_string();
                if !text.is_empty() {
                    entries.push(format!("{header}\n{text}"));
                }
            }
            current = Some((rest.trim().to_string(), Vec::new()));
        } else if let Some((_, lines)) = current.as_mut() {
            lines.push(line.to_string());
        }
    }
    if let Some((header, lines)) = current.take() {
        let text = lines.join("\n").trim().to_string();
        if !text.is_empty() {
            entries.push(format!("{header}\n{text}"));
        }
    }
    entries
}

/// Find a CHANGELOG.md next to the running executable (TS bundles one with
/// the package and reads `getChangelogPath()`).
fn changelog_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join("CHANGELOG.md");
    candidate.is_file().then_some(candidate)
}

/// `/changelog [full|last [N]]` (TS `handle` in builtin-session.ts).
pub(super) fn changelog_command(rt: &Runtime, args: &str) -> Outcome {
    let view = match parse_changelog_view(args) {
        Ok(v) => v,
        Err(e) => return usage(rt, &e),
    };
    let Some(path) = changelog_path() else {
        output(
            rt,
            "No changelog entries found. (No CHANGELOG.md next to the omt executable.)",
        );
        return Handled {
            agent_invoked: false,
        };
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        output(rt, "No changelog entries found.");
        return Handled {
            agent_invoked: false,
        };
    };
    let entries = parse_changelog_content(&content);
    let selected: Vec<&String> = match view {
        ChangelogView::Full => entries.iter().collect(),
        ChangelogView::Recent(n) | ChangelogView::Last(n) => entries.iter().take(n).collect(),
    };
    if selected.is_empty() {
        output(rt, "No changelog entries found.");
        return Handled {
            agent_invoked: false,
        };
    }
    // TS renders oldest-first (`renderChangelogEntries` default).
    let markdown = selected
        .iter()
        .rev()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    output(rt, &markdown);
    Handled {
        agent_invoked: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changelog_view_parsing_matches_ts() {
        assert!(matches!(
            parse_changelog_view(""),
            Ok(ChangelogView::Recent(3))
        ));
        assert!(matches!(
            parse_changelog_view("full"),
            Ok(ChangelogView::Full)
        ));
        assert!(matches!(
            parse_changelog_view("last"),
            Ok(ChangelogView::Last(1))
        ));
        assert!(matches!(
            parse_changelog_view("last 5"),
            Ok(ChangelogView::Last(5))
        ));
        assert!(parse_changelog_view("last 0").is_err());
        assert!(parse_changelog_view("bogus").is_err());
        let entries = parse_changelog_content(
            "# Changelog\n\n## [1.2.0]\n\n- Added A\n\n## [1.1.0]\n\n- Added B\n",
        );
        assert_eq!(entries.len(), 2);
        assert!(entries[0].starts_with("[1.2.0]"));
        assert!(entries[0].contains("- Added A"));
    }
}
