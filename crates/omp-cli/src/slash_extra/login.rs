//! Interactive `/login` and `/logout` (TS `showOAuthSelector` → `#handleOAuthLogin` /
//! `#showOAuthLogoutAccountSelector`), driven through the TUI's extension dialogs:
//! provider/account pickers are `select` dialogs, paste prompts are `input` dialogs, the auth URL
//! and progress lines are `command_output` transcript blocks. Credential handling is
//! `omp_providers::login`, shared with `omp-rs login`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use omp_providers::login::{self, providers, Services, Target};
use omp_providers::AgentDir;
use pa_ai::oauth::{OAuthLoginUi, OAuthPrompt};
use serde_json::{json, Value};

use super::dialog;
use crate::runtime::Runtime;

fn output(rt: &Runtime, text: &str) {
    rt.send(json!({"type": "command_output", "text": text}));
}

/// A full-screen, type-to-search `select` dialog over `labels` (TS `OAuthSelectorComponent`);
/// the chosen index.
async fn select(rt: &Runtime, title: &str, labels: &[String]) -> Option<usize> {
    dialog::select_fullscreen(rt, title, labels).await
}

/// TS `LoginDialogComponent` as transcript lines + input dialogs.
struct TuiUi {
    rt: Arc<Runtime>,
}

impl OAuthLoginUi for TuiUi {
    fn on_auth(&self, url: &str, instructions: Option<&str>) {
        let mut text = format!("Open this URL in your browser:\n{url}");
        if let Some(instructions) = instructions.filter(|i| !i.is_empty()) {
            text.push('\n');
            text.push_str(instructions);
        }
        output(&self.rt, &text);
        crate::subcmd::login::open_url(url);
    }

    fn on_prompt(
        &self,
        prompt: &OAuthPrompt,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        let fields = json!({"title": prompt.message, "placeholder": prompt.placeholder});
        Box::pin(async move {
            let answer = dialog::request(&self.rt, "input", fields).await?;
            Some(
                answer
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        })
    }

    fn on_progress(&self, message: &str) {
        output(&self.rt, message);
    }

    /// The paste racing the browser callback (TS: `/login <redirect URL>` into the editor): an
    /// input dialog over the transcript's URL block. Esc cancels the login and frees the callback
    /// port; a callback that wins first withdraws the dialog.
    fn on_manual_code_input(
        &self,
    ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
        let fields = json!({
            "title": "Paste the authorization code or full redirect URL (or finish in the browser)",
            "placeholder": "waiting for the browser callback… esc cancels the login",
        });
        Some(Box::pin(async move {
            let answer = dialog::request(&self.rt, "input", fields).await?;
            Some(
                answer
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        }))
    }
}

const NOT_INTERACTIVE: &str =
    "needs an interactive UI (provider pickers are dialogs); run `omt login` / `omt logout` from a terminal";

/// `/login [provider]`.
pub async fn login_command(rt: &Arc<Runtime>, args: &str) {
    if !dialog::has_ui() {
        output(rt, &format!("/login: {NOT_INTERACTIVE}"));
        return;
    }
    let args = args.trim();
    let provider_id = if args.is_empty() {
        let labels: Vec<String> = providers::PROVIDERS
            .iter()
            .map(|(_, name)| (*name).to_string())
            .collect();
        let Some(index) = select(rt, "Select provider to login", &labels).await else {
            return;
        };
        providers::PROVIDERS[index].0.to_string()
    } else if providers::display_name(args).is_some() {
        args.to_string()
    } else {
        // TS: any other argument is a pasted redirect URL for a pending flow; omp-rs has no
        // paste-code flows, so nothing can be waiting for it.
        output(rt, "No OAuth login is waiting for a manual callback.");
        return;
    };
    handle_login(rt, &provider_id).await;
}

async fn handle_login(rt: &Arc<Runtime>, provider_id: &str) {
    output(rt, &format!("Logging in to {provider_id}…"));
    let dir = AgentDir::default();
    let target = match Target::resolve(&dir) {
        Ok(target) => target,
        Err(e) => return output(rt, &format!("Login failed: {e:#}")),
    };
    let ui = Arc::new(TuiUi { rt: rt.clone() });
    let identity = match login::login(&target, provider_id, ui, &Services::production()).await {
        Ok(identity) => identity,
        Err(message) if message == "Login cancelled" => return output(rt, "Login cancelled"),
        Err(message) => return output(rt, &format!("Login failed: {message}")),
    };
    let Some(identity) = identity else {
        return output(
            rt,
            &format!("No credentials were stored for {provider_id}."),
        );
    };
    // Provider-scoped discovery so the unlocked models appear in-session; never fails the login.
    let _ = omp_providers::refresh_async(&dir, Some(providers::store_as(provider_id))).await;
    let who = identity
        .describe()
        .map(|w| format!(" as {w}"))
        .unwrap_or_default();
    output(
        rt,
        &format!(
            "✓ Successfully logged in to {provider_id}{who}\n{}",
            target.saved_line()
        ),
    );
}

/// `/logout [provider]`.
pub async fn logout_command(rt: &Arc<Runtime>, args: &str) {
    if !dialog::has_ui() {
        output(rt, &format!("/logout: {NOT_INTERACTIVE}"));
        return;
    }
    let dir = AgentDir::default();
    let target = match Target::resolve(&dir) {
        Ok(target) => target,
        Err(e) => return output(rt, &format!("Could not load stored credentials: {e:#}")),
    };
    let args = args.trim();
    let provider_id = if args.is_empty() {
        let stored = match target.providers().await {
            Ok(stored) => stored,
            Err(e) => return output(rt, &format!("Could not load stored credentials: {e:#}")),
        };
        let logged_in: Vec<(&str, &str)> = providers::PROVIDERS
            .iter()
            .copied()
            .filter(|(id, _)| stored.iter().any(|s| s == id))
            .collect();
        if logged_in.is_empty() {
            return output(rt, "No stored provider credentials to log out. Remove env or config auth at its source.");
        }
        let labels: Vec<String> = logged_in
            .iter()
            .map(|(_, name)| (*name).to_string())
            .collect();
        let Some(index) = select(rt, "Select provider to logout", &labels).await else {
            return;
        };
        logged_in[index].0.to_string()
    } else if providers::display_name(args).is_some() {
        args.to_string()
    } else {
        return output(rt, &format!("Unknown OAuth provider: {args}"));
    };

    let rows = match target.list(Some(&provider_id)).await {
        Ok(rows) => rows,
        Err(e) => return output(rt, &format!("Could not load stored credentials: {e:#}")),
    };
    if rows.is_empty() {
        return output(
            rt,
            &format!("Logout skipped: no stored credentials for {provider_id}."),
        );
    }
    let mut labels: Vec<String> = rows.iter().map(login::StoredRow::label).collect();
    for i in 0..labels.len() {
        if labels.iter().filter(|l| **l == labels[i]).count() > 1 {
            labels[i] = format!("{} [#{}]", labels[i], rows[i].id);
        }
    }
    let name = providers::display_name(&provider_id).unwrap_or(&provider_id);
    let Some(index) = select(rt, &format!("Select {name} account to log out"), &labels).await
    else {
        return;
    };
    let (row, label) = (&rows[index], rows[index].label());
    match target.remove_by_id(&provider_id, row.id).await {
        Ok(true) => {
            let _ =
                omp_providers::refresh_async(&dir, Some(providers::store_as(&provider_id))).await;
            output(
                rt,
                &format!(
                    "✓ Successfully logged out {label} from {provider_id}\n{}",
                    target.removed_line()
                ),
            );
        }
        Ok(false) => output(
            rt,
            &format!("Logout skipped: {label} is no longer stored for {provider_id}."),
        ),
        Err(e) => output(rt, &format!("Logout failed: {e:#}")),
    }
}
