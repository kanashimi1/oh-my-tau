//! `omp-rs login [PROVIDER]` / `omp-rs logout [PROVIDER]` — the terminal counterparts of
//! `/login` and `/logout`.
//!
//! `login` mirrors TS `cli/login-cli.ts` + `cli/oauth-terminal.ts`: numbered provider picker when
//! PROVIDER is omitted, the auth URL printed (and opened best-effort), prompts read from stdin,
//! then `Logged in to <name>[ as <identity>]` and the credential location. Failures print
//! `Login failed: <message>` to stderr and exit 1. TS has no top-level `omp logout`; this one
//! mirrors `omp auth-broker logout` (soft-delete every credential of a provider with cause
//! `logged out by user`, `Logged out of <provider>`).

use std::future::Future;
use std::io::{BufRead as _, IsTerminal as _, Write as _};
use std::pin::Pin;
use std::sync::Arc;

use omp_providers::login::{self, providers, Services, Target};
use omp_providers::AgentDir;
use pa_ai::oauth::{OAuthLoginUi, OAuthPrompt};

const APP_NAME: &str = "omt";
const LOGIN_DESCRIPTION: &str = "Log in to a model provider (terminal counterpart of /login)";
const LOGOUT_DESCRIPTION: &str = "Log out of a model provider (removes its stored credentials)";

fn paint(code: &str, text: &str, stream_is_tty: bool) -> String {
    if stream_is_tty && std::env::var_os("NO_COLOR").is_none() {
        format!("\x1b[{code}m{text}\x1b[39m")
    } else {
        text.to_string()
    }
}

fn green(text: &str) -> String {
    paint("32", text, std::io::stdout().is_terminal())
}
fn yellow(text: &str) -> String {
    paint("33", text, std::io::stdout().is_terminal())
}
fn dim(text: &str) -> String {
    if std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
        format!("\x1b[2m{text}\x1b[22m")
    } else {
        text.to_string()
    }
}
fn red_err(text: &str) -> String {
    paint("31", text, std::io::stderr().is_terminal())
}

/// One stdin line after printing `question`; `None` when stdin is closed (TS: `Login cancelled`).
fn prompt_line(question: &str) -> Option<String> {
    let mut out = std::io::stdout();
    let _ = write!(out, "{question}");
    let _ = out.flush();
    let mut line = String::new();
    match std::io::stdin().lock().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim_end_matches(['\n', '\r']).to_string()),
    }
}

/// TS `pickIndex`: numbered stdin picker.
fn pick_index(title: &str, labels: &[String]) -> Result<usize, String> {
    let mut out = std::io::stdout();
    let _ = write!(out, "{title}\n\n");
    for (i, label) in labels.iter().enumerate() {
        let _ = writeln!(out, "  {}. {label}", i + 1);
    }
    let _ = writeln!(out);
    let choice =
        prompt_line(&format!("Enter number (1-{}): ", labels.len())).ok_or("Login cancelled")?;
    choice
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|n| (1..=labels.len()).contains(n))
        .map(|n| n - 1)
        .ok_or_else(|| format!("Invalid selection: {choice}"))
}

/// Open `url` with the platform opener, best effort (TS `openPath`). `$BROWSER` wins when set.
pub(crate) fn open_url(url: &str) {
    let candidates: Vec<String> = match std::env::var("BROWSER") {
        Ok(browser) if !browser.trim().is_empty() => vec![browser],
        _ => ["wslview", "xdg-open", "open"].map(String::from).to_vec(),
    };
    for program in candidates {
        let spawned = std::process::Command::new(&program)
            .arg(url)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        if let Ok(mut child) = spawned {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            return;
        }
    }
}

/// TS `runTerminalOAuthLogin`'s controller: URL block, progress lines, stdin prompts.
struct TermUi {
    open_browser: bool,
}

impl OAuthLoginUi for TermUi {
    fn on_auth(&self, url: &str, instructions: Option<&str>) {
        let mut out = std::io::stdout();
        let _ = write!(out, "\nOpen this URL in your browser:\n{url}\n");
        if let Some(instructions) = instructions.filter(|i| !i.is_empty()) {
            let _ = writeln!(out, "{instructions}");
        }
        let _ = writeln!(out);
        let _ = out.flush();
        if self.open_browser {
            open_url(url);
        }
    }

    fn on_prompt(
        &self,
        prompt: &OAuthPrompt,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        let question = format!(
            "{}{}: ",
            prompt.message,
            prompt
                .placeholder
                .as_deref()
                .filter(|p| !p.is_empty())
                .map_or_else(String::new, |p| format!(" ({p})"))
        );
        Box::pin(async move {
            tokio::task::spawn_blocking(move || prompt_line(&question))
                .await
                .ok()
                .flatten()
        })
    }

    fn on_progress(&self, message: &str) {
        println!("{message}");
    }

    fn on_manual_code_input(
        &self,
    ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
        // Only paste-code providers (fixed non-loopback redirect) get a manual prompt in TS;
        // none of the ported flows are paste-code providers.
        None
    }
}

/// Run an async block on a dedicated thread with its own runtime (this entry point runs inside
/// the process-wide tokio runtime, where nesting `block_on` panics).
fn run_async<T: Send>(future: impl Future<Output = T> + Send) -> T {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime")
                    .block_on(future)
            })
            .join()
            .expect("login thread panicked")
    })
}

fn help(name: &str, description: &str, arg_doc: &str, examples: &[&str]) {
    println!("{description}\n");
    println!("USAGE\n  $ {APP_NAME} {name} [PROVIDER]\n");
    println!("ARGUMENTS\n  PROVIDER  {arg_doc}\n");
    println!("EXAMPLES");
    for example in examples {
        println!("  {example}\n");
    }
}

/// Entry point: `omp-rs login [PROVIDER]`.
///
/// # Errors
/// Never; failures are reported TS-style on stderr and returned as exit code 1.
pub fn run_login(args: &[String]) -> anyhow::Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        help(
            "login",
            LOGIN_DESCRIPTION,
            "OAuth provider id (e.g. anthropic, openai-codex); omit to pick interactively",
            &[
                &format!("# Pick a provider interactively\n  {APP_NAME} login"),
                &format!("# Log in to a specific provider\n  {APP_NAME} login anthropic"),
            ],
        );
        return Ok(0);
    }
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if let Some(flag) = args.iter().find(|a| a.starts_with('-')) {
        eprintln!(
            "{}",
            red_err(&format!("Login failed: Nonexistent flag: {flag}"))
        );
        return Ok(1);
    }
    if positional.len() > 1 {
        eprintln!(
            "{}",
            red_err(&format!(
                "Login failed: Unexpected argument: {}",
                positional[1]
            ))
        );
        return Ok(1);
    }
    let provider = positional.first().map(|p| p.as_str());
    let code = run_async(async move {
        match login_command(provider).await {
            Ok(()) => 0,
            Err(message) => {
                eprintln!("{}", red_err(&format!("Login failed: {message}")));
                1
            }
        }
    });
    Ok(code)
}

async fn login_command(provider: Option<&str>) -> Result<(), String> {
    let dir = AgentDir::default();
    let target = Target::resolve(&dir).map_err(|e| format!("{e:#}"))?;
    let roster: Vec<(&str, &str)> = providers::PROVIDERS.to_vec();
    let provider_id = match provider {
        Some(id) => id.to_string(),
        None => {
            let labels: Vec<String> = roster.iter().map(|(_, name)| (*name).to_string()).collect();
            let index =
                tokio::task::spawn_blocking(move || pick_index("Select a provider:", &labels))
                    .await
                    .map_err(|e| e.to_string())??;
            roster[index].0.to_string()
        }
    };
    let Some(&(id, name)) = roster.iter().find(|(id, _)| *id == provider_id) else {
        return Err(format!(
            "Unknown OAuth provider '{provider_id}'. Run `{APP_NAME} login` to pick one."
        ));
    };

    let services = Services::production();
    let identity = login::login(
        &target,
        id,
        Arc::new(TermUi { open_browser: true }),
        &services,
    )
    .await?;
    let Some(identity) = identity else {
        print!(
            "{}",
            yellow(&format!("No credentials were stored for {name}.\n"))
        );
        return Ok(());
    };
    // TS then re-runs that provider's model discovery so the next session sees the unlocked
    // models; discovery failures never fail the login.
    let refresh_dir = dir.clone();
    let refresh_id = providers::store_as(id).to_string();
    let _ = tokio::task::spawn_blocking(move || {
        omp_providers::refresh(&refresh_dir, Some(&refresh_id))
    })
    .await;

    let who = identity
        .describe()
        .map(|w| format!(" as {w}"))
        .unwrap_or_default();
    print!("{}", green(&format!("\nLogged in to {name}{who}\n")));
    print!("{}", dim(&format!("{}\n", target.saved_line())));
    Ok(())
}

/// Entry point: `omp-rs logout [PROVIDER]`.
///
/// # Errors
/// Never; failures are reported on stderr and returned as exit code 1.
pub fn run_logout(args: &[String]) -> anyhow::Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        help(
            "logout",
            LOGOUT_DESCRIPTION,
            "provider id with stored credentials; omit to pick interactively",
            &[
                &format!("# Pick a stored provider interactively\n  {APP_NAME} logout"),
                &format!("# Log out of a specific provider\n  {APP_NAME} logout anthropic"),
            ],
        );
        return Ok(0);
    }
    if let Some(flag) = args.iter().find(|a| a.starts_with('-')) {
        eprintln!("Error: Nonexistent flag: {flag}");
        return Ok(1);
    }
    if args.len() > 1 {
        eprintln!("Error: Unexpected argument: {}", args[1]);
        return Ok(1);
    }
    let provider = args.first().map(String::as_str);
    let code = run_async(async move {
        match logout_command(provider).await {
            Ok(()) => 0,
            Err(message) => {
                eprintln!("Error: {message}");
                1
            }
        }
    });
    Ok(code)
}

async fn logout_command(provider: Option<&str>) -> Result<(), String> {
    let dir = AgentDir::default();
    let target = Target::resolve(&dir).map_err(|e| format!("{e:#}"))?;
    let provider_id = match provider {
        Some(id) => id.to_string(),
        None => {
            let stored = target.providers().await.map_err(|e| format!("{e:#}"))?;
            if stored.is_empty() {
                println!("No credentials stored.");
                return Ok(());
            }
            let labels = stored.clone();
            let index = tokio::task::spawn_blocking(move || {
                pick_index("Select a provider to logout:", &labels)
            })
            .await
            .map_err(|e| e.to_string())??;
            stored[index].clone()
        }
    };
    target
        .logout_provider(&provider_id, "logged out by user")
        .await
        .map_err(|e| format!("{e:#}"))?;
    println!("Logged out of {provider_id}");
    Ok(())
}
