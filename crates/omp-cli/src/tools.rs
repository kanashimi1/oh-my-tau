//! Built-in tools with omp's names, parameter schemas and result formats:
//! `read`, `edit`, `write`, `grep`, `glob`, `bash`.
//!
//! The edit engine is the vendored `pi-edit` crate (omp's native edit
//! sessions, hashline mode included) and the file walks are `pi-walker`.
//! Failures are returned as `Err` so the agent loop reports `isError: true`.

pub mod ask;
mod ast;
pub mod bash;
pub mod checkpoint;
mod edit;
mod eval;
pub(crate) mod eval_agents;
pub(crate) mod eval_bridge;
pub(crate) mod eval_journal;
mod find;
mod jobs;
pub mod mcp;
mod read;
pub(crate) mod relevance;
pub(crate) mod rlm;
pub(crate) mod rlm_combinators;
mod search;
mod task;
mod todo;
pub mod wasm;
pub use todo::{phases_from_entries, TodoPhase, TodoTool};
#[cfg(test)]
pub use todo::{Status, TodoItem};
mod goal;
mod lsp;
mod think;
mod tree;
mod util;
mod wait;
mod web_search;
mod yield_tool;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use pi_edit::{EditMode, EditStore};
use serde_json::{json, Value};

/// Job registry of the most recently built tool set (see [`take_last_jobs`]).
static LAST_JOBS: std::sync::Mutex<Option<jobs::JobRegistry>> = std::sync::Mutex::new(None);
/// The agent session's job registry (see [`set_session_jobs`]).
static SESSION_JOBS: std::sync::Mutex<Option<jobs::JobRegistry>> = std::sync::Mutex::new(None);

/// The registry of the tool set built last, for the caller that builds the session's tools to
/// capture right after (before anything else builds a throwaway set).
#[must_use]
pub fn take_last_jobs() -> Option<jobs::JobRegistry> {
    LAST_JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Publish the agent session's job registry (`ctx.getAsyncJobSnapshot()`).
pub fn set_session_jobs(jobs: Option<jobs::JobRegistry>) {
    *SESSION_JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = jobs;
}

/// The agent session's job registry, once published.
#[must_use]
pub fn session_jobs() -> Option<jobs::JobRegistry> {
    SESSION_JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Per-session state shared by the tools: working directory, the hashline
/// snapshot store (`read` mints tags, `edit` validates them) and edit mode.
pub struct Env {
    pub cwd: PathBuf,
    pub store: EditStore,
    pub edit_mode: EditMode,
    /// Hashline anchors in `read`/`grep` output (hashline edit mode only).
    pub hashline: bool,
    /// Background `bash` jobs (async mode) and the `proc://` read surface.
    pub jobs: jobs::JobRegistry,
    /// Request text of the running tool call (latest user message + assistant
    /// text), set per call from [`relevance::take`]; ranks the inline view of
    /// spilled `bash`/`read` output.
    pub query: parking_lot::Mutex<String>,
}

impl Env {
    #[must_use]
    pub fn with_mode(cwd: &Path, edit_mode: EditMode) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            store: EditStore::new(),
            edit_mode,
            hashline: edit_mode == EditMode::Hashline,
            jobs: jobs::JobRegistry::default(),
            query: parking_lot::Mutex::default(),
        }
    }

    /// The current call's request text (see [`Env::query`]).
    #[must_use]
    pub fn query(&self) -> String {
        self.query.lock().clone()
    }

    #[cfg(test)]
    #[must_use]
    pub fn new(cwd: &Path, hashline: bool) -> Self {
        Self::with_mode(
            cwd,
            if hashline {
                EditMode::Hashline
            } else {
                EditMode::Replace
            },
        )
    }
}

/// omp `resolveEditMode`: `PI_EDIT_VARIANT` pins the mode; otherwise hashline,
/// except for model families whose hashline output is unreliable (omp's
/// per-model fallback to `replace`).
#[must_use]
pub fn edit_mode_for(model_id: &str) -> EditMode {
    if let Some(mode) = std::env::var("PI_EDIT_VARIANT")
        .ok()
        .as_deref()
        .and_then(EditMode::parse)
    {
        return mode;
    }
    let id = model_id.to_lowercase();
    let glm_flash_53 = id.contains("glm") && id.contains("flash") && id.contains("5.3");
    let replace_family = ["kimi", "mimo", "minimax", "deepseek", "step", "codex-spark"]
        .iter()
        .any(|f| id.contains(f));
    if glm_flash_53 || replace_family {
        EditMode::Replace
    } else {
        EditMode::Hashline
    }
}

type Handler = Arc<
    dyn Fn(Arc<Env>, Value, AbortSignal) -> BoxFut<'static, Result<AgentToolResult>> + Send + Sync,
>;

struct FnTool {
    name: &'static str,
    description: String,
    schema: Value,
    env: Arc<Env>,
    run: Handler,
}

impl AgentTool for FnTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> &Value {
        &self.schema
    }
    fn execute(
        self: Arc<Self>,
        tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        if let Some(q) = relevance::take(&tool_call_id) {
            *self.env.query.lock() = q;
        }
        (self.run)(self.env.clone(), params, signal)
    }
}

const READ_DESCRIPTION: &str = "Use `read` for static web; browser only if needed.\n\nPath suffixes: :50 or :50- starts at line 50; :50-200 inclusive; :50+150 counts lines; :-60 last 60; commas join ranges (:5-16,960-973) or individual lines (:19,59). :raw verbatim without anchors/prefixes; combine :2-4:raw or :raw:2-4.\n\nSources:\n- Selected file: `[foo.ts#1A2B]` snapshot+lines. Copy `[FILENAME#TAG]` for anchored edits; NEVER invent tag.\n- Directory: complete root; child listings cap at 12 (`… N more`), read child; page via :N-M/:-N.";
const BASH_DESCRIPTION: &str = "Persistent shell: one fact command/pipeline; dependencies use `&&`.\nScripts/heredocs/`$(…)`/complex pipelines → `eval`.\n`cwd`, not `cd`; `pty` only interactive.\nInternal URIs work as paths for builtins/coreutils, redirects, globs.\n`async` defers finite results but keeps the deadline (default 300s); `timeout: 0` for watchers and long jobs.\nNo `head`/`tail`/redirection; output trunc by default, full result at `artifact://<id>`.\nLong-lived services: unique name; ready requires name; no async/timeout; pty defaults true. ready needs log regex or port (both if given); host defaults 127.0.0.1, ready.timeout 30s.\nBackground results auto-deliver.";
const GLOB_DESCRIPTION: &str = "Glob files/dirs: `;`-separated paths or internal URLs (`local://*.md`, `omp://**/*.md`); default workspace root.\n`gitignore` and `hidden` default true; ignored dotfiles need `gitignore: false`. Newest-first by directory; dirs end `/`.";
const GREP_DESCRIPTION: &str = "Regex: Rust, then PCRE2. `path`: `;`-separated file/dir/glob/URL; default `.`. Default case-sensitive, gitignore respected; `skip` paginates files.\nFile-only selector: `src/foo.ts:50-100`. Literal `\\n`/`\\\\n` enables cross-line.\nBare glob `*.ts` matches any depth; `dir/*.ts` only `dir`'s direct children (`dir/**/*.ts` recurses).";
const WRITE_DESCRIPTION: &str = "SHOULD `edit` existing files; `write` for required new files or whole-file replacement. NEVER create docs or emojis unless requested.";
const FIND_DESCRIPTION: &str = "Describe behavior, get implementing files and line ranges. MUST use first for unknown locations; known strings/regex/symbols → `grep`, names → `glob`.\n`query`: plain language, not regex; quoted phrases match whole. `grep_keywords`: likely verbatim terms, `[]` if unsure.\n`path`: one directory or file, host path or internal URL (`omp://`, `omp://<file>.md`, `skill://<name>`, `local://…`); omitted = workspace root. Scope known subsystem; batch related questions. Searches live files, no index; no `:start-end` selector (judges whole files).\nHits strongest first: `path:start-end score snippet` (workspace-relative, or URL under URL scope); read returned ranges. Scores: absolute comparable 0–1 probability; below ~0.4 = weak evidence, so widen query or use `grep` before concluding absence.";

/// All built-in tools, rooted at `cwd`.
#[must_use]
#[allow(clippy::too_many_lines)] // the tool registry table; order mirrors omp's builtinTools
pub fn builtin_tools(cwd: &Path, edit_mode: EditMode) -> Vec<Arc<dyn AgentTool>> {
    let env = Arc::new(Env::with_mode(cwd, edit_mode));
    *LAST_JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(env.jobs.clone());
    let mk = |name: &'static str,
              description: String,
              schema: Value,
              run: Handler|
     -> Arc<dyn AgentTool> {
        Arc::new(FnTool {
            name,
            description,
            schema,
            env: env.clone(),
            run,
        })
    };
    // Blocking tools run on the `spawn_blocking` pool (safe under the
    // current_thread runtime). Abort is honored around the blocking section:
    // checked before dispatch, watched concurrently (flipping the tool's
    // `CancelFlag` so long walker loops can bail early), and checked again
    // after completion so a cancelled call does not surface stale output.
    let blocking = |f: fn(&Env, &Value) -> Result<AgentToolResult>| -> Handler {
        Arc::new(move |env, args, signal| {
            Box::pin(async move {
                pa_agent::abort::throw_if_aborted(&signal)?;
                let result = tokio::task::spawn_blocking(move || f(&env, &args))
                    .await
                    .map_err(|e| anyhow::anyhow!("tool task failed: {e}"))?;
                // A cancelled turn must not surface stale output.
                if signal.is_aborted() {
                    return Err(pa_agent::abort::aborted_error());
                }
                result
            })
        })
    };
    // Same contract as `blocking`, plus a cooperative `CancelFlag` the tool
    // body polls inside long loops (walker chunks, per-file scoring) so an
    // aborted call stops burning CPU on the blocking pool promptly.
    let blocking_cancel =
        |f: fn(&Env, &Value, &util::CancelFlag) -> Result<AgentToolResult>| -> Handler {
            Arc::new(move |env, args, signal| {
                Box::pin(async move {
                    pa_agent::abort::throw_if_aborted(&signal)?;
                    let cancel = util::CancelFlag::default();
                    let flag = cancel.clone();
                    let watch_signal = signal.clone();
                    let watcher = tokio::spawn(async move {
                        watch_signal.aborted().await;
                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    });
                    let result = tokio::task::spawn_blocking(move || f(&env, &args, &cancel))
                        .await
                        .map_err(|e| anyhow::anyhow!("tool task failed: {e}"))?;
                    watcher.abort();
                    if signal.is_aborted() {
                        return Err(pa_agent::abort::aborted_error());
                    }
                    result
                })
            })
        };
    vec![
        mk(
            "read",
            READ_DESCRIPTION.to_string(),
            json!({"type":"object","properties":{"path":{"type":"string","description":"Local path, internal URI, or URL; selectors inline."}},"required":["path"],"additionalProperties":false}),
            blocking(read::read),
        ),
        mk(
            "bash",
            BASH_DESCRIPTION.to_string(),
            json!({"type":"object","properties":{
                "command":{"type":"string"},
                "timeout":{"type":"number","description":"timeout in seconds; 0 disables the command deadline; nonzero values are clamped to 1-3600"},
                "cwd":{"type":"string"},
                "pty":{"type":"boolean"},
                "async":{"type":"boolean"},
                "name":{"type":"string","maxLength":48},
                "ready":{"type":"object","properties":{
                    "log":{"type":"string"},
                    "port":{"type":"number"},
                    "host":{"type":"string"},
                    "timeout":{"type":"number"}
                }}
            },"required":["command"],"additionalProperties":false}),
            Arc::new(|env, args, signal| Box::pin(bash::bash(env, args, signal))),
        ),
        mk(
            "edit",
            pi_edit::description(edit_mode).to_string(),
            edit::schema(edit_mode),
            Arc::new(|env, args, _| Box::pin(async move { edit::edit(&env, &args).await })),
        ),
        mk(
            "glob",
            GLOB_DESCRIPTION.to_string(),
            json!({"type":"object","properties":{"path":{"type":"string"},"hidden":{"type":"boolean"},"gitignore":{"type":"boolean"},"limit":{"type":"number"}},"additionalProperties":false}),
            blocking(search::glob),
        ),
        mk(
            "grep",
            GREP_DESCRIPTION.to_string(),
            json!({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"case":{"type":"boolean"},"gitignore":{"type":"boolean"},"skip":{"anyOf":[{"type":"number"},{"type":"null"}]}},"required":["pattern"],"additionalProperties":false}),
            blocking_cancel(search::grep_with_cancel),
        ),
        mk(
            "write",
            WRITE_DESCRIPTION.to_string(),
            json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path"],"additionalProperties":false}),
            blocking(edit::write),
        ),
        mk(
            "find",
            FIND_DESCRIPTION.to_string(),
            json!({"type":"object","properties":{
                "query":{"type":"string"},
                "grep_keywords":{"type":"array","items":{"type":"string"}},
                "path":{"type":"string"}
            },"required":["query","grep_keywords"],"additionalProperties":false}),
            blocking_cancel(find::find_with_cancel),
        ),
        mk(
            "task",
            task::description(cwd),
            task::schema(),
            Arc::new(|env, args, _| Box::pin(task::task(env, args))),
        ),
        wait::tool(env.clone()),
        // Own AgentTool: eval needs the abort signal (cell cancel) and on_update
        // (bridge status events), which the `mk` closure signature drops.
        Arc::new(eval::EvalTool::new(env.clone())),
        mk(
            "ast_grep",
            ast::AST_GREP_DESCRIPTION.to_string(),
            ast::schema_ast_grep(),
            blocking_cancel(ast::ast_grep_with_cancel),
        ),
        mk(
            "ast_edit",
            ast::AST_EDIT_DESCRIPTION.to_string(),
            ast::schema_ast_edit(),
            blocking_cancel(ast::ast_edit_with_cancel),
        ),
        ask::tool(),
        todo::tool(),
        web_search::tool(),
        lsp::tool(env.clone()),
        checkpoint::checkpoint_tool(),
        checkpoint::rewind_tool(),
    ]
}

/// TS `isToolAllowed` setting gates for the plain built-ins: a tool whose
/// `<id>.enabled` setting is off is not advertised (`astGrep`/`checkpoint`
/// default off, the rest on). `find.enabled` is an enum: `off` hides it;
/// `auto` stays on (TS narrows `auto` to native-judge models).
#[must_use]
pub fn enabled_by_settings(settings: &omp_settings::Settings, name: &str) -> bool {
    if name == "find" {
        return settings.get_string("find.enabled").as_deref() != Some("off");
    }
    let id = match name {
        "bash" => "bash.enabled",
        "glob" => "glob.enabled",
        "grep" => "grep.enabled",
        "ast_grep" => "astGrep.enabled",
        "ast_edit" => "astEdit.enabled",
        "ask" => "ask.enabled",
        "todo" => "todo.enabled",
        "web_search" => "web_search.enabled",
        "lsp" => "lsp.enabled",
        "checkpoint" | "rewind" => "checkpoint.enabled",
        _ => return true,
    };
    settings.get_bool(id).unwrap_or(true)
}

/// [`builtin_tools`] minus the ones switched off by settings: the surface a
/// session advertises.
#[must_use]
pub fn session_builtin_tools(
    cwd: &Path,
    edit_mode: EditMode,
    settings: &omp_settings::Settings,
) -> Vec<Arc<dyn AgentTool>> {
    let mut tools = builtin_tools(cwd, edit_mode);
    tools.retain(|t| enabled_by_settings(settings, t.name()));
    tools
}

/// Active hidden tools for a session (TS `resolveBuiltinToolPlan`'s
/// `...(externalThinkingActive ? ["think"] : [])`, `...(includeYield ? ["yield"]
/// : [])`, `...(goalModeActive ? ["goal"] : [])` legs, in that order; never in
/// the visible built-in list). `think` needs the `externalThinking` setting and
/// a model whose transport supports external thinking; `yield` is included when
/// the session requires it (subagent runs); `goal` when goal mode is active
/// (`goal.enabled` setting + enabled goal-mode state).
#[must_use]
pub fn hidden_tools(
    settings: &omp_settings::Settings,
    model: Option<&pa_types::ai::Model>,
    require_yield_tool: bool,
    goal_mode_active: bool,
) -> Vec<Arc<dyn AgentTool>> {
    let mut tools: Vec<Arc<dyn AgentTool>> = Vec::new();
    let external_thinking = settings.get_bool("externalThinking").unwrap_or(false)
        && model.is_some_and(think::supports_external_thinking);
    if external_thinking {
        tools.push(think::tool());
    }
    if require_yield_tool {
        tools.push(yield_tool::tool());
    }
    if goal_mode_active {
        tools.push(goal::tool());
    }
    tools
}

/// Construct one hidden tool by name (omp `HIDDEN_TOOLS` factories), for
/// explicit `--tools <csv>` requests that name a hidden tool.
#[must_use]
pub fn hidden_tool_by_name(name: &str) -> Option<Arc<dyn AgentTool>> {
    match name {
        "think" => Some(think::tool()),
        "yield" => Some(yield_tool::tool()),
        "goal" => Some(goal::tool()),
        _ => None,
    }
}

pub use ask::{tool_with_ui, AskUi, SelectOption};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_mode_follows_omp_model_rule() {
        // PI_EDIT_VARIANT may be set by the harness; only assert the rule when unset.
        if std::env::var_os("PI_EDIT_VARIANT").is_some() {
            return;
        }
        assert_eq!(edit_mode_for("glm-5.3-flash"), EditMode::Replace);
        assert_eq!(edit_mode_for("kimi-k2.5"), EditMode::Replace);
        assert_eq!(edit_mode_for("claude-opus-5-5"), EditMode::Hashline);
        assert_eq!(edit_mode_for("glm-5.2"), EditMode::Hashline);
    }

    #[test]
    fn settings_gate_the_advertised_builtins() {
        let dir = tempfile::tempdir().unwrap();
        let names = |yaml: &str| -> Vec<String> {
            let cfg = dir.path().join("config.yml");
            std::fs::write(&cfg, yaml).unwrap();
            let settings = omp_settings::Settings::load_from(Some(&cfg), dir.path());
            session_builtin_tools(dir.path(), EditMode::Hashline, &settings)
                .iter()
                .map(|t| t.name().to_string())
                .collect()
        };
        // TS defaults: astGrep and checkpoint/rewind off, the rest on.
        let defaults = names("{}");
        for off in ["ast_grep", "checkpoint", "rewind"] {
            assert!(!defaults.iter().any(|n| n == off), "{off} on by default");
        }
        for on in ["read", "bash", "ast_edit", "lsp", "todo"] {
            assert!(defaults.iter().any(|n| n == on), "{on} off by default");
        }
        let custom = names("lsp:\n  enabled: false\nastGrep:\n  enabled: true\ncheckpoint:\n  enabled: true\nfind:\n  enabled: \"off\"\n");
        assert!(!custom.iter().any(|n| n == "lsp" || n == "find"));
        assert!(
            defaults.iter().any(|n| n == "find"),
            "find.enabled auto keeps find"
        );
        for on in ["ast_grep", "checkpoint", "rewind"] {
            assert!(custom.iter().any(|n| n == on), "{on} not enabled by config");
        }
    }

    #[test]
    fn tool_names_and_schemas_match_omp() {
        let tools = builtin_tools(Path::new("/tmp"), EditMode::Hashline);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            [
                "read",
                "bash",
                "edit",
                "glob",
                "grep",
                "write",
                "find",
                "task",
                "wait",
                "eval",
                "ast_grep",
                "ast_edit",
                "ask",
                "todo",
                "web_search",
                "lsp",
                "checkpoint",
                "rewind"
            ]
        );
        let edit = tools.iter().find(|t| t.name() == "edit").unwrap();
        assert_eq!(edit.parameters()["properties"]["input"]["type"], "string");
        assert!(edit
            .description()
            .starts_with("Hashline patches existing files"));
        let read = tools.iter().find(|t| t.name() == "read").unwrap();
        assert_eq!(read.parameters()["required"], json!(["path"]));
    }

    #[test]
    fn bash_has_async_name_ready_params() {
        let tools = builtin_tools(Path::new("/tmp"), EditMode::Replace);
        let bash = tools.iter().find(|t| t.name() == "bash").unwrap();
        let props = &bash.parameters()["properties"];
        assert_eq!(props["async"], json!({"type": "boolean"}));
        assert_eq!(props["name"], json!({"type": "string", "maxLength": 48}));
        assert_eq!(
            props["ready"],
            json!({"type": "object", "properties": {
                "log": {"type": "string"},
                "port": {"type": "number"},
                "host": {"type": "string"},
                "timeout": {"type": "number"}
            }})
        );
        assert_eq!(
            bash.description().lines().next().unwrap(),
            "Persistent shell: one fact command/pipeline; dependencies use `&&`."
        );
    }

    #[test]
    fn grep_skip_uses_omp_anyof_encoding() {
        let tools = builtin_tools(Path::new("/tmp"), EditMode::Replace);
        let grep = tools.iter().find(|t| t.name() == "grep").unwrap();
        assert_eq!(
            grep.parameters()["properties"]["skip"],
            json!({"anyOf": [{"type": "number"}, {"type": "null"}]})
        );
        assert_eq!(
            grep.description().lines().next().unwrap(),
            "Regex: Rust, then PCRE2. `path`: `;`-separated file/dir/glob/URL; default `.`. Default case-sensitive, gitignore respected; `skip` paginates files."
        );
        assert_eq!(
            tools.iter().find(|t| t.name() == "glob").unwrap().description().lines().next().unwrap(),
            "Glob files/dirs: `;`-separated paths or internal URLs (`local://*.md`, `omp://**/*.md`); default workspace root."
        );
    }

    #[test]
    fn find_schema_matches_omp() {
        let tools = builtin_tools(Path::new("/tmp"), EditMode::Replace);
        let find = tools.iter().find(|t| t.name() == "find").unwrap();
        assert_eq!(
            find.parameters()["required"],
            json!(["query", "grep_keywords"])
        );
        let props = &find.parameters()["properties"];
        assert_eq!(props["query"], json!({"type": "string"}));
        assert_eq!(
            props["grep_keywords"],
            json!({"type": "array", "items": {"type": "string"}})
        );
        assert_eq!(props["path"], json!({"type": "string"}));
        assert_eq!(
            find.description().lines().next().unwrap(),
            "Describe behavior, get implementing files and line ranges. MUST use first for unknown locations; known strings/regex/symbols → `grep`, names → `glob`."
        );
    }
}
