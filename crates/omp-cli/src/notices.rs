//! Hidden per-prompt notices: magic keywords (TS `modes/magic-keywords.ts` +
//! `AgentSession#createMagicKeywordNotices`) and the ultra modes.
//!
//! A notice is a `custom` message `{customType: "<id>-notice", content:
//! [{type: "text", text}], display: false, attribution: "user", timestamp}`
//! prepended to the user message it belongs to. The TUI never renders
//! `display: false` custom messages; `convert_to_llm` turns them into
//! provider-facing developer messages.

use std::path::PathBuf;

use omp_tui::magic::{self, Ultra};
use pa_agent::types::{AgentMessage, CustomAgentMessage};
use serde_json::{json, Value};

const ULTRATHINK: &str = include_str!("../prompts/notices/ultrathink-notice.md");
const ORCHESTRATE: &str = include_str!("../prompts/notices/orchestrate-notice.md");
const WORKFLOW: &str = include_str!("../prompts/notices/workflow-notice.md");
const JEVIFY: &str = include_str!("../prompts/notices/jevify-notice.md");
const ULTRACODE: &str = include_str!("../prompts/notices/ultracode-notice.md");
const ULTRADEEP_ADDENDUM: &str = include_str!("../prompts/notices/ultradeep-addendum.md");
const ULTRALONG_ADDENDUM: &str = include_str!("../prompts/notices/ultralong-addendum.md");

/// Session facts a notice renders against (TS `MagicKeywordContext`).
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Active tool names for the turn.
    pub tools: Vec<String>,
    /// `task.batch`.
    pub task_batch: bool,
    /// The `scout` agent can be dispatched.
    pub scout_available: bool,
    /// `eval.tools.enabled`.
    pub eval_tools: bool,
}

impl Context {
    /// Facts from the session's active tools and settings.
    #[must_use]
    pub fn new(tools: Vec<String>, settings: &omp_settings::Settings) -> Self {
        let scout_available = tools.iter().any(|t| t == "task")
            && !settings
                .get_string_array("task.disabledAgents")
                .iter()
                .any(|a| a == "scout");
        Self {
            tools,
            task_batch: settings.get_bool("task.batch").unwrap_or(true),
            scout_available,
            eval_tools: settings.get_bool("eval.tools.enabled").unwrap_or(true),
        }
    }

    fn data(&self) -> Value {
        json!({
            "tools": self.tools,
            "taskBatch": self.task_batch,
            "scoutAvailable": self.scout_available,
            "evalTools": self.eval_tools,
        })
    }

    fn has(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t == tool)
    }
}

/// Workspace checkpoint CLI the ultracode notice points at: `OMP_WS_CKPT`;
/// `None` when unset or the file is absent.
#[must_use]
pub fn ws_ckpt_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("OMP_WS_CKPT").filter(|v| !v.is_empty())?);
    path.is_file().then_some(path)
}

/// Rendered text of keyword `id`'s notice (TS `keyword.notice(context)`).
#[must_use]
pub fn keyword_notice(id: &str, ctx: &Context) -> Option<String> {
    let text = match id {
        "ultrathink" => ULTRATHINK.trim().to_owned(),
        "jevify" => JEVIFY.trim().to_owned(),
        "orchestrate" => crate::sysprompt::render(ORCHESTRATE, &ctx.data())
            .trim()
            .to_owned(),
        "workflow" => crate::sysprompt::render(WORKFLOW, &ctx.data())
            .trim()
            .to_owned(),
        "ultracode" => ultracode_notice(None, ctx, ws_ckpt_path().as_deref()),
        _ => return None,
    };
    Some(text)
}

/// The ultracode notice: for an active ultra `mode` (with its RLM / CLM
/// addenda), or for the `ultracode` keyword when `mode` is `None`.
#[must_use]
pub fn ultracode_notice(
    mode: Option<Ultra>,
    ctx: &Context,
    ws_ckpt: Option<&std::path::Path>,
) -> String {
    let name = mode.map_or("ultracode", Ultra::name);
    let addendum_data = json!({ "mode": name });
    let mut addenda = Vec::new();
    if mode.is_some_and(Ultra::rlm) {
        addenda.push(crate::sysprompt::render(ULTRADEEP_ADDENDUM, &addendum_data));
    }
    if mode.is_some_and(Ultra::clm) {
        addenda.push(crate::sysprompt::render(ULTRALONG_ADDENDUM, &addendum_data));
    }
    let mut data = ctx.data();
    data["keyword"] = json!(mode.is_none());
    data["mode"] = json!(name);
    data["wsCkpt"] = json!(ws_ckpt.map(|p| p.display().to_string()));
    data["addenda"] = json!(addenda.join("\n\n"));
    crate::sysprompt::render(ULTRACODE, &data).trim().to_owned()
}

/// One hidden notice message (CONTEXT contract 4).
#[must_use]
pub fn message(custom_type: &str, text: &str, timestamp: i64) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage {
        role: "custom".to_owned(),
        payload: json!({
            "customType": custom_type,
            "content": [{"type": "text", "text": text}],
            "display": false,
            "attribution": "user",
            "timestamp": timestamp,
        }),
    })
}

/// Notices for one user prompt, in keyword registry order: each enabled
/// keyword standing as prose in `text` whose required tools are all active,
/// plus the ultracode notice for an active ultra `mode` (never twice).
#[must_use]
pub fn build(
    text: &str,
    settings: &omp_settings::Settings,
    ctx: &Context,
    mode: Option<Ultra>,
    timestamp: i64,
) -> Vec<AgentMessage> {
    let mut out = Vec::new();
    let mut ultracode_sent = false;
    for k in magic::present(text) {
        if !magic::enabled(settings, k.id) || !k.requires.iter().all(|t| ctx.has(t)) {
            continue;
        }
        if k.id == "ultracode" {
            if mode.is_some() {
                continue; // the mode's own notice below covers it
            }
            ultracode_sent = true;
        }
        if let Some(body) = keyword_notice(k.id, ctx) {
            out.push(message(&format!("{}-notice", k.id), &body, timestamp));
        }
    }
    if let Some(m) = mode.filter(|_| !ultracode_sent) {
        let body = ultracode_notice(Some(m), ctx, ws_ckpt_path().as_deref());
        out.push(message("ultracode-notice", &body, timestamp));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(tools: &[&str]) -> Context {
        Context {
            tools: tools.iter().map(|t| (*t).to_owned()).collect(),
            task_batch: true,
            scout_available: true,
            eval_tools: true,
        }
    }

    fn defaults() -> omp_settings::Settings {
        let dir = tempfile::tempdir().unwrap();
        omp_settings::Settings::load_from(None, dir.path())
    }

    fn custom_type(m: &AgentMessage) -> String {
        match m {
            AgentMessage::Custom(c) => c.payload["customType"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            AgentMessage::Standard(_) => String::new(),
        }
    }

    fn text(m: &AgentMessage) -> String {
        match m {
            AgentMessage::Custom(c) => c.payload["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            AgentMessage::Standard(_) => String::new(),
        }
    }

    /// Every template renders without leftover handlebars for every flag combination.
    #[test]
    fn templates_render_without_handlebars() {
        let tool_sets: [&[&str]; 3] = [
            &[],
            &["task", "eval"],
            &[
                "task", "eval", "bash", "lsp", "edit", "write", "todo", "wait",
            ],
        ];
        for tools in tool_sets {
            for flags in 0..8u8 {
                let c = Context {
                    tools: tools.iter().map(|t| (*t).to_owned()).collect(),
                    task_batch: flags & 1 != 0,
                    scout_available: flags & 2 != 0,
                    eval_tools: flags & 4 != 0,
                };
                for k in magic::KEYWORDS {
                    let body = keyword_notice(k.id, &c).expect("every keyword has a notice");
                    assert!(!body.contains("{{"), "{}: {body}", k.id);
                    assert!(body.starts_with("<system-notice>"), "{}", k.id);
                }
                for u in Ultra::ALL {
                    for ck in [None, Some(std::path::Path::new("/x/ws_ckpt.py"))] {
                        let body = ultracode_notice(Some(u), &c, ck);
                        assert!(!body.contains("{{"), "{}: {body}", u.name());
                        assert!(body.ends_with("</system-notice>"), "{}", u.name());
                        assert_eq!(body.contains("lam_rlm"), u.rlm(), "{}", u.name());
                        assert_eq!(body.contains("context_edit"), u.clm(), "{}", u.name());
                        assert_eq!(body.contains("python3 /x/ws_ckpt.py save"), ck.is_some());
                    }
                }
            }
        }
        let c = ctx(&["task", "eval"]);
        let js = ultracode_notice(Some(Ultra::Code), &c, None);
        assert!(
            js.find("```js").unwrap() < js.find("```python").unwrap(),
            "JS example first"
        );
        assert!(js.contains("Ultra mode **ultracode** is active"));
        assert!(keyword_notice("ultracode", &c)
            .unwrap()
            .contains("User message contains **ultracode**"));
    }

    #[test]
    fn keyword_notices_respect_requirements_and_settings() {
        let s = defaults();
        let all = ctx(&["task", "eval"]);
        let got: Vec<String> = build(
            "please ultrathink and orchestrate, jevify too",
            &s,
            &all,
            None,
            1,
        )
        .iter()
        .map(custom_type)
        .collect();
        assert_eq!(
            got,
            ["ultrathink-notice", "orchestrate-notice", "jevify-notice"]
        );
        // A required tool missing: that notice is skipped.
        let no_eval = ctx(&["task"]);
        let got: Vec<String> = build("jevify orchestrate ultracode", &s, &no_eval, None, 1)
            .iter()
            .map(custom_type)
            .collect();
        assert_eq!(got, ["orchestrate-notice"]);
        // Code spans do not trigger.
        assert!(build("run `ultrathink`", &s, &all, None, 1).is_empty());
        // Disabled by setting.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.yml"),
            "magicKeywords:\n  orchestrate: false\n",
        )
        .unwrap();
        let off = omp_settings::Settings::load_from(None, dir.path());
        let got: Vec<String> = build("orchestrate ultrathink", &off, &all, None, 1)
            .iter()
            .map(custom_type)
            .collect();
        assert_eq!(got, ["ultrathink-notice"]);
        std::fs::write(
            dir.path().join("config.yml"),
            "magicKeywords:\n  enabled: false\n",
        )
        .unwrap();
        let off = omp_settings::Settings::load_from(None, dir.path());
        assert!(build("orchestrate ultrathink", &off, &all, None, 1).is_empty());
    }

    #[test]
    fn ultra_notice_injected_once() {
        let s = defaults();
        let c = ctx(&["task", "eval"]);
        // Keyword without a mode: the keyword notice.
        let k = build("ultracode this", &s, &c, None, 7);
        assert_eq!(
            k.iter().map(custom_type).collect::<Vec<_>>(),
            ["ultracode-notice"]
        );
        assert!(text(&k[0]).contains("User message contains **ultracode**"));
        // Mode active: exactly one ultracode notice, keyword or not, with the mode's addenda.
        for msg in ["plain question", "ultracode this"] {
            let m = build(msg, &s, &c, Some(Ultra::Hyper), 7);
            assert_eq!(
                m.iter().map(custom_type).collect::<Vec<_>>(),
                ["ultracode-notice"],
                "{msg}"
            );
            assert!(text(&m[0]).contains("**hypercode**"));
        }
        let AgentMessage::Custom(cm) = &k[0] else {
            panic!("custom")
        };
        assert_eq!(cm.role, "custom");
        assert_eq!(cm.payload["display"], json!(false));
        assert_eq!(cm.payload["attribution"], json!("user"));
        assert_eq!(cm.payload["timestamp"], json!(7));
    }
}
