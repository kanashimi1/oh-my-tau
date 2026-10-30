//! Interactive command center shared by the informational slash commands.

use super::*;
use crate::slash_extra::dialog::{self, Choice, SelectStyle};

const ROOT: [(&str, &str, &str); 7] = [
    (
        "usage",
        "Usage & limits",
        "Session tokens, provider limits and saved resets",
    ),
    (
        "stats",
        "Statistics dashboard",
        "Open cross-session usage statistics",
    ),
    (
        "trace",
        "Session trace",
        "Open this session in the trace dashboard",
    ),
    (
        "changelog",
        "Changelog",
        "Show the three most recent releases",
    ),
    (
        "model",
        "Model & thinking",
        "Switch the model or reasoning level",
    ),
    (
        "mcp",
        "MCP servers",
        "Test, enable, disable, add or remove servers",
    ),
    ("todo", "Todo", "Show the current task list"),
];

/// Open the shared command menu. Bare informational commands start on their
/// own row; explicit arguments retain the original non-interactive behavior.
pub(crate) async fn command_center(rt: &std::sync::Arc<Runtime>, initial: &str) -> Outcome {
    let choices: Vec<Choice> = ROOT
        .iter()
        .map(|(_, label, description)| Choice::with(*label, *description))
        .collect();
    let initial = ROOT
        .iter()
        .position(|(name, _, _)| *name == initial)
        .unwrap_or(0);
    let Some(index) = dialog::select_with(
        rt,
        "Command center",
        &choices,
        SelectStyle {
            fullscreen: true,
            searchable: true,
            initial,
        },
    )
    .await
    else {
        return Handled {
            agent_invoked: false,
        };
    };
    match ROOT[index].0 {
        "usage" => usage_menu(rt).await,
        "stats" => stats_command(rt, "").await,
        "trace" => trace_command(rt).await,
        "changelog" => changelog_command(rt, ""),
        "model" => Box::pin(dispatch_builtin(rt, "model", ""))
            .await
            .unwrap_or(Handled {
                agent_invoked: false,
            }),
        "mcp" => mcp_command(rt, "").await,
        "todo" => todo_command(rt, "").await,
        _ => unreachable!("command center rows and dispatch are kept together"),
    }
}

async fn usage_menu(rt: &std::sync::Arc<Runtime>) -> Outcome {
    let choices = [
        Choice::with(
            "Session usage",
            "Input, output, cache, total tokens and cost",
        ),
        Choice::with(
            "Provider limits",
            "Rate-limit windows, remaining quota and reset times",
        ),
        Choice::with(
            "Spend saved reset",
            "Choose and consume an available rate-limit reset",
        ),
    ];
    let Some(index) = dialog::select_with(rt, "Usage", &choices, SelectStyle::default()).await
    else {
        return Handled {
            agent_invoked: false,
        };
    };
    usage_command(rt, ["session", "provider", "reset"][index]).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_center_rows_are_unique_and_usage_is_first() {
        let mut names = std::collections::HashSet::new();
        assert_eq!(ROOT[0].0, "usage");
        assert!(ROOT.iter().all(|(name, _, _)| names.insert(*name)));
        assert_eq!(ROOT.len(), 7);
    }
}
