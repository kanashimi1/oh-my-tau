//! Built-in tools with omp's names, parameter schemas and result formats:
//! `read`, `edit`, `write`, `grep`, `glob`, `bash`.
//!
//! The edit engine is the vendored `pi-edit` crate (omp's native edit
//! sessions, hashline mode included) and the file walks are `pi-walker`.
//! Failures are returned as `Err` so the agent loop reports `isError: true`.

pub mod bash;
mod edit;
mod find;
mod jobs;
mod read;
mod search;
mod tree;
mod util;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use pi_edit::{EditMode, EditStore};
use serde_json::{json, Value};

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
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn new(cwd: &Path, hashline: bool) -> Self {
        Self::with_mode(cwd, if hashline { EditMode::Hashline } else { EditMode::Replace })
    }
}

/// omp `resolveEditMode`: `PI_EDIT_VARIANT` pins the mode; otherwise hashline,
/// except for model families whose hashline output is unreliable (omp's
/// per-model fallback to `replace`).
#[must_use]
pub fn edit_mode_for(model_id: &str) -> EditMode {
    if let Some(mode) = std::env::var("PI_EDIT_VARIANT").ok().as_deref().and_then(EditMode::parse) {
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

type Handler = Arc<dyn Fn(Arc<Env>, Value, AbortSignal) -> BoxFut<'static, Result<AgentToolResult>> + Send + Sync>;

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
        _tool_call_id: String,
        params: Value,
        signal: AbortSignal,
        _on_update: AgentToolUpdateCallback,
    ) -> BoxFut<'static, Result<AgentToolResult>> {
        (self.run)(self.env.clone(), params, signal)
    }
}

const READ_DESCRIPTION: &str = "Use `read` for static web; browser only if needed.\n\nPath suffixes: :50 or :50- starts at line 50; :50-200 inclusive; :50+150 counts lines; :-60 last 60; commas join ranges (:5-16,960-973) or individual lines (:19,59). :raw verbatim without anchors/prefixes; combine :2-4:raw or :raw:2-4.\n\nSources:\n- Selected file: `[foo.ts#1A2B]` snapshot+lines. Copy `[FILENAME#TAG]` for anchored edits; NEVER invent tag.\n- Directory: complete root; child listings cap at 12 (`… N more`), read child; page via :N-M/:-N.";
const BASH_DESCRIPTION: &str = "Persistent shell: one fact command/pipeline; dependencies use `&&`.\nScripts/heredocs/`$(…)`/complex pipelines → `eval`.\n`cwd`, not `cd`; `pty` only interactive.\nInternal URIs work as paths for builtins/coreutils, redirects, globs.\n`async` defers finite results but keeps the deadline (default 300s); `timeout: 0` for watchers and long jobs.\nNo `head`/`tail`/redirection; output trunc by default, full result at `artifact://<id>`.\nLong-lived services: unique name; ready requires name; no async/timeout; pty defaults true. ready needs log regex or port (both if given); host defaults 127.0.0.1, ready.timeout 30s.\nBackground results follow; NEVER poll; foreground wait unchanged.";
const GLOB_DESCRIPTION: &str = "Glob files/dirs: `;`-separated paths or internal URLs (`local://*.md`, `omp://**/*.md`); default workspace root.\n`gitignore` and `hidden` default true; ignored dotfiles need `gitignore: false`. Newest-first by directory; dirs end `/`.\nBehavior search → `find`.\nMulti-round discovery → Task + scout.";
const GREP_DESCRIPTION: &str = "Regex: Rust, then PCRE2. `path`: `;`-separated file/dir/glob/URL; default `.`. Default case-sensitive, gitignore respected; `skip` paginates files.\nFile-only selector: `src/foo.ts:50-100`. Literal `\\n`/`\\\\n` enables cross-line.\nBare glob `*.ts` matches any depth; `dir/*.ts` only `dir`'s direct children (`dir/**/*.ts` recurses).\nBehavior/unknown symbol → `find`; literals/regex → `grep`.\nMulti-round search MUST use Task + scout, not chained calls.";
const WRITE_DESCRIPTION: &str = "SHOULD `edit` existing files; `write` for required new files or whole-file replacement. NEVER create docs or emojis unless requested.";
const FIND_DESCRIPTION: &str = "Describe behavior, get implementing files and line ranges. MUST use first for unknown locations; known strings/regex/symbols → `grep`, names → `glob`.\n`query`: plain language, not regex; quoted phrases match whole. `grep_keywords`: likely verbatim terms, `[]` if unsure.\n`path`: one directory or file, host path or internal URL (`omp://`, `omp://<file>.md`, `skill://<name>`, `local://…`); omitted = workspace root. Scope known subsystem; batch related questions. Searches live files, no index; no `:start-end` selector (judges whole files).\nHits strongest first: `path:start-end score snippet` (workspace-relative, or URL under URL scope); read returned ranges. Scores: absolute comparable 0–1 probability; below ~0.4 = weak evidence, so widen query or use `grep` before concluding absence.";

/// All built-in tools, rooted at `cwd`.
#[must_use]
pub fn builtin_tools(cwd: &Path, edit_mode: EditMode) -> Vec<Arc<dyn AgentTool>> {
    let env = Arc::new(Env::with_mode(cwd, edit_mode));
    let mk = |name: &'static str, description: String, schema: Value, run: Handler| -> Arc<dyn AgentTool> {
        Arc::new(FnTool { name, description, schema, env: env.clone(), run })
    };
    let blocking = |f: fn(&Env, &Value) -> Result<AgentToolResult>| -> Handler {
        Arc::new(move |env, args, _| {
            Box::pin(async move {
                tokio::task::spawn_blocking(move || f(&env, &args))
                    .await
                    .map_err(|e| anyhow::anyhow!("tool task failed: {e}"))?
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
            blocking(search::grep),
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
            blocking(find::find),
        ),
    ]
}

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
    fn tool_names_and_schemas_match_omp() {
        let tools = builtin_tools(Path::new("/tmp"), EditMode::Hashline);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, ["read", "bash", "edit", "glob", "grep", "write", "find"]);
        let edit = tools.iter().find(|t| t.name() == "edit").unwrap();
        assert_eq!(edit.parameters()["properties"]["input"]["type"], "string");
        assert!(edit.description().starts_with("Hashline patches existing files"));
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
        assert_eq!(bash.description().lines().next().unwrap(), "Persistent shell: one fact command/pipeline; dependencies use `&&`.");
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
        assert_eq!(find.parameters()["required"], json!(["query", "grep_keywords"]));
        let props = &find.parameters()["properties"];
        assert_eq!(props["query"], json!({"type": "string"}));
        assert_eq!(props["grep_keywords"], json!({"type": "array", "items": {"type": "string"}}));
        assert_eq!(props["path"], json!({"type": "string"}));
        assert_eq!(
            find.description().lines().next().unwrap(),
            "Describe behavior, get implementing files and line ranges. MUST use first for unknown locations; known strings/regex/symbols → `grep`, names → `glob`."
        );
    }
}
