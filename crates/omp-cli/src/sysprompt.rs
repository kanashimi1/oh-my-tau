//! System prompt assembly, ported from omp's `system-prompt.ts` +
//! `prompts/system/*.md`: base prompt selection (user/project `SYSTEM.md`
//! override, `--system-prompt`/`--append-system-prompt` flags), project
//! context discovery (`AGENTS.md`/`CLAUDE.md` walk-up), skills listing, the
//! tool inventory section, and the workstation environment block.
//!
//! Rendering uses a small Handlebars-subset interpreter that covers exactly
//! the helpers omp's bundled templates use (`if`, `unless`, `each`, `list`,
//! `ifAny`, `has`, `when`, `includes`, `not`, `else`). The templates ship in
//! `omp-cli/prompts/` (copied verbatim from the TS distribution).

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

const CUSTOM_TEMPLATE: &str = include_str!("../prompts/custom-system-prompt.md");
const BUNDLED_TEMPLATE: &str = include_str!("../prompts/system-prompt.md");
const COMPACT_TEMPLATE: &str = include_str!("../prompts/system-prompt-compact.md");
const PROJECT_TEMPLATE: &str = include_str!("../prompts/project-prompt.md");
const USER_APPEND_TEMPLATE: &str = include_str!("../prompts/user-append.md");

/// User home directory.
fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// omp's agent dir (`getAgentDir()` for the default profile).
fn agent_dir() -> PathBuf {
    home_dir().join(".omp/agent")
}

/// Settings the prompt renders from.
#[allow(clippy::struct_excessive_bools)] // mirrors omp's PromptConfig shape; one struct beats six params
struct PromptConfig {
    skills_enabled: bool,
    include_model_in_prompt: bool,
    inline_tool_descriptors: bool,
    reactions: bool,
    /// `systemPrompt.compact`: render the compact bundled template.
    compact: bool,
}

/// Prompt-rendered settings via `omp-settings` (TS registry layering over the
/// agent dir's `config.yml`). The port predates omp-settings and defaults
/// `includeModelInPrompt` / `inlineToolDescriptors` / `reactions` to off where
/// the TS registry defaults them on, so those three keep their port defaults
/// unless the user explicitly set the key (`Provenance::Default` = unset).
fn load_config(agent: &Path) -> PromptConfig {
    // No project layer: the prompt historically reads the agent dir only.
    let project_none = agent.join(".no-project-layer");
    let settings =
        omp_settings::Settings::load_from(Some(&agent.join("config.yml")), &project_none);
    let set = |id: &str| {
        settings
            .provenance(id)
            .is_some_and(|p| p != omp_settings::Provenance::Default)
    };
    let mut cfg = PromptConfig {
        skills_enabled: true,
        include_model_in_prompt: false,
        inline_tool_descriptors: false,
        reactions: false,
        compact: settings.get_bool("systemPrompt.compact").unwrap_or(false),
    };
    if set("skills.enabled") {
        cfg.skills_enabled = settings.get_bool("skills.enabled").unwrap_or(true);
    }
    if set("includeModelInPrompt") {
        cfg.include_model_in_prompt = settings.get_bool("includeModelInPrompt").unwrap_or(false);
    }
    if set("inlineToolDescriptors") {
        cfg.inline_tool_descriptors = match settings.get("inlineToolDescriptors") {
            serde_json::Value::String(v) => v == "on",
            serde_json::Value::Bool(true) => true, // YAML 1.1 `on` unquoted
            _ => false,
        };
    }
    if set("tui.reactions") {
        cfg.reactions = settings.get_bool("tui.reactions").unwrap_or(false);
    }
    cfg
}

/// Workstation facts, matching `getEnvironmentInfo()` (Node `os.platform()`
///`os.release()`/`os.arch()`).
pub fn environment_info() -> Vec<(String, String)> {
    let release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let os = if release.is_empty() {
        std::env::consts::OS.to_string()
    } else {
        format!("{} {release}", std::env::consts::OS)
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    vec![("OS".into(), os), ("Arch".into(), arch.into())]
}

/// Discover the git worktree root by walking up for a `.git` entry
/// (equivalent of the repo root the TS discovery passes as boundary).
fn git_root(cwd: &Path, _home: &Path) -> Option<PathBuf> {
    let mut cur = Some(cwd);
    while let Some(dir) = cur {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        cur = dir.parent();
    }
    None
}

/// Walk-up boundary for context-file discovery, mirroring
/// `loadStandaloneContextFiles`: `(boundary, include_boundary, exclude_home)`.
fn scan_boundary(cwd: &Path, home: &Path) -> (PathBuf, bool, bool) {
    let cwd_under_home = cwd.starts_with(home);
    let repo_root = git_root(cwd, home);
    let repo_is_home = repo_root.as_deref() == Some(home);
    let repo_under_home = repo_root
        .as_deref()
        .is_some_and(|r| r.starts_with(home) && r != home);
    let scan_to_home = repo_root.is_some() && cwd_under_home && repo_under_home;
    let boundary = if scan_to_home {
        home.to_path_buf()
    } else {
        match repo_root.clone() {
            Some(r) => r,
            None if cwd_under_home => home.to_path_buf(),
            None => cwd.ancestors().last().unwrap_or(cwd).to_path_buf(),
        }
    };
    let include_boundary = match repo_root {
        None => cwd_under_home,
        Some(_) => boundary != home || repo_is_home,
    };
    (boundary, include_boundary, scan_to_home)
}

/// One discovered project context file (`AGENTS.md`/`CLAUDE.md`), ordered
/// farthest-from-cwd first per the TS depth-descending sort.
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
    pub depth: usize,
}

/// `loadStandaloneContextFiles` for `AGENTS.md` then `CLAUDE.md`: at each
/// directory from `cwd` up to the boundary, the first non-empty file claims
/// that depth scope. Hidden directories (`.foo/AGENTS.md`) are skipped.
pub fn context_files(cwd: &Path, home: &Path) -> Vec<ContextFile> {
    let (boundary, include_boundary, exclude_home) = scan_boundary(cwd, home);
    let mut out: Vec<ContextFile> = Vec::new();
    let mut current = cwd.to_path_buf();
    loop {
        let at_boundary = current == boundary;
        let at_home = exclude_home && current == home;
        if !at_home && (!at_boundary || include_boundary) {
            for name in ["AGENTS.md", "CLAUDE.md"] {
                let candidate = current.join(name);
                if let Ok(content) = std::fs::read_to_string(&candidate) {
                    if !content.is_empty()
                        && !current
                            .file_name()
                            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                    {
                        let depth = cwd
                            .ancestors()
                            .position(|a| a == current.as_path())
                            .unwrap_or(0);
                        out.push(ContextFile {
                            path: candidate,
                            content,
                            depth,
                        });
                        break; // one file per depth scope
                    }
                }
            }
        }
        if at_boundary {
            break;
        }
        match current.parent() {
            Some(p) => current = p.to_path_buf(),
            None => break,
        }
    }
    out.sort_by_key(|f| std::cmp::Reverse(f.depth));
    out
}

/// A discovered system prompt override (SYSTEM.md text or `SYSTEM_TEMPLATE.md`
/// template), project scope beating user scope as in
/// `discoverSystemPromptOverride`.
#[derive(Debug)]
pub enum Override {
    Text(String),
    Template(String),
}

/// Walk from `cwd` up to the git root (or filesystem root) for SYSTEM.md /
/// `SYSTEM_TEMPLATE.md` (`getProjectPathCandidates` analog), falling back to the
/// user-level files in the agent dir.
pub fn discover_override(cwd: &Path, agent: &Path) -> Option<Override> {
    let mut dirs: Vec<PathBuf> = cwd.ancestors().map(Path::to_path_buf).collect();
    if let Some(root) = git_root(cwd, &home_dir()) {
        if !dirs.contains(&root) {
            dirs.push(root.clone());
        }
        dirs.retain(|d| root.starts_with(d) || d.starts_with(&root));
    }
    for dir in dirs {
        for (name, template) in [("SYSTEM.md", false), ("SYSTEM_TEMPLATE.md", true)] {
            if let Ok(content) = std::fs::read_to_string(dir.join(name)) {
                if content.is_empty() {
                    continue;
                }
                if template {
                    if content.trim().is_empty() {
                        continue;
                    }
                    return Some(Override::Template(content));
                }
                return Some(Override::Text(content));
            }
        }
    }
    for (name, template) in [("SYSTEM.md", false), ("SYSTEM_TEMPLATE.md", true)] {
        if let Ok(content) = std::fs::read_to_string(agent.join(name)) {
            if content.is_empty() {
                continue;
            }
            if template {
                if content.trim().is_empty() {
                    continue;
                }
                return Some(Override::Template(content));
            }
            return Some(Override::Text(content));
        }
    }
    None
}

/// A skill entry as rendered into the prompt (`name` + `description`).
pub struct Skill {
    pub name: String,
    pub description: String,
    pub hide: bool,
}

/// Parse a SKILL.md frontmatter block: `---` fenced `key: value` pairs.
fn parse_frontmatter(content: &str) -> (Map<String, Value>, String) {
    let mut meta = Map::new();
    let mut body = content;
    if let Some(rest) = content.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            for line in rest[..end].lines() {
                if let Some((k, v)) = line.split_once(':') {
                    meta.insert(
                        k.trim().to_string(),
                        Value::String(v.trim().trim_matches('"').to_string()),
                    );
                }
            }
            body = &rest[end + 4..];
        }
    }
    (meta, body.trim().to_string())
}

/// Load skills from a `skills/` dir (`~/.omp/agent/skills`, project
/// `.omp/skills`): one directory per skill with a SKILL.md.
fn load_skills_dir(dir: &Path, skills: &mut Vec<Skill>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let file = path.join("SKILL.md");
        let Ok(content) = std::fs::read_to_string(&file) else {
            continue;
        };
        let (meta, _body) = parse_frontmatter(&content);
        let name = meta.get("name").and_then(Value::as_str).map_or_else(
            || entry.file_name().to_string_lossy().into_owned(),
            Into::into,
        );
        let description = meta
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let hide = meta
            .get("hide")
            .and_then(Value::as_str)
            .is_some_and(|v| v == "true");
        skills.push(Skill {
            name,
            description,
            hide,
        });
    }
}

/// All visible skills for the prompt (TS: `loadSkills` from the agent dir's
/// `skills/`, `hide: true` dropped from the render).
pub fn load_skills(agent: &Path, cwd: &Path, enabled: bool) -> Vec<Skill> {
    if !enabled {
        return Vec::new();
    }
    let mut skills = Vec::new();
    // Project `.omp/skills` walk-up, closest first.
    for dir in cwd.ancestors() {
        load_skills_dir(&dir.join(".omp/skills"), &mut skills);
    }
    load_skills_dir(&agent.join("skills"), &mut skills);
    skills.retain(|s| !s.hide);
    skills
}

/// `resolvePromptInput`: multi-line input is literal; otherwise try reading
/// the value as a file, falling back to the literal string.
pub fn resolve_prompt_input(input: &str) -> String {
    if input.contains('\n') {
        return input.to_string();
    }
    match std::fs::read_to_string(input) {
        Ok(content) => content,
        Err(_) => input.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Handlebars-subset renderer for omp's bundled templates.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Arg {
    Str(String),
    Num(f64),
    Path(String),
    Call(String, Vec<Arg>),
}

#[derive(Debug)]
enum Node {
    Text(String),
    Var(Arg),
    Block {
        helper: String,
        args: Vec<Arg>,
        hash: Vec<(String, String)>,
        body: Vec<Node>,
        else_body: Vec<Node>,
    },
}

fn tokenize(template: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let (before, tag_start) = rest.split_at(start);
        if !before.is_empty() {
            tags.push(format!("\0{before}"));
        }
        if let Some(end) = tag_start[2..].find("}}") {
            tags.push(tag_start[..end + 4].to_string());
            rest = &tag_start[end + 4..];
        } else {
            tags.push(tag_start.to_string());
            rest = "";
        }
    }
    if !rest.is_empty() {
        tags.push(format!("\0{rest}"));
    }
    tags
}

/// Split on whitespace outside quotes, keeping quoted segments whole.
fn split_tokens(rest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in rest.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => {
                quote = None;
                cur.push(c);
            }
            (None, '"' | '\'') => {
                quote = Some(c);
                cur.push(c);
            }
            (None, c) if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            (_, c) => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Interpret the escape sequences template authors write inside `join=`/`prefix=`.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_arg(tok: &str) -> Arg {
    let t = tok.trim();
    if let Some(inner) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let mut parts = inner.split_whitespace();
        let name = parts.next().unwrap_or("").to_string();
        return Arg::Call(name, parts.map(parse_arg).collect());
    }
    if t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')))
    {
        return Arg::Str(t[1..t.len() - 1].to_string());
    }
    if let Ok(n) = t.parse::<f64>() {
        return Arg::Num(n);
    }
    Arg::Path(t.to_string())
}

fn parse_nodes(tags: &[String], pos: &mut usize) -> (Vec<Node>, Option<String>) {
    let mut nodes = Vec::new();
    while *pos < tags.len() {
        let tag = &tags[*pos];
        if let Some(text) = tag.strip_prefix('\0') {
            nodes.push(Node::Text(text.to_string()));
            *pos += 1;
            continue;
        }
        let inner = tag.trim_start_matches("{{").trim_end_matches("}}").trim();
        if let Some(close) = inner.strip_prefix('/') {
            return (nodes, Some(close.trim().to_string()));
        }
        if inner == "else" {
            return (nodes, Some("else".into()));
        }
        if let Some(rest) = inner.strip_prefix('#') {
            let toks = split_tokens(rest);
            let helper = toks.first().map_or("", String::as_str).to_string();
            let mut args = Vec::new();
            let mut hash = Vec::new();
            for t in &toks[1..] {
                // Quoted tokens may contain '=' (comparison operators like
                // "=="); only bare `key=value` tokens are hashes.
                let quoted = t.starts_with('"') || t.starts_with('\'');
                if !quoted && !t.starts_with('(') {
                    if let Some((k, v)) = t.split_once('=') {
                        hash.push((k.to_string(), unescape(v.trim_matches('"'))));
                        continue;
                    }
                }
                args.push(parse_arg(t));
            }
            *pos += 1;
            let (body, term) = parse_nodes(tags, pos);
            let mut else_body = Vec::new();
            match term.as_deref() {
                Some("else") => {
                    // Consume the {{else}} tag, then parse the inverse branch.
                    *pos += 1;
                    let (eb, t2) = parse_nodes(tags, pos);
                    else_body = eb;
                    if t2.is_some() {
                        // Consume the {{/helper}} close tag.
                        *pos += 1;
                    }
                }
                Some(name) => {
                    debug_assert_eq!(name, helper, "mismatched block close: {name} vs {helper}");
                    // Consume the {{/helper}} close tag.
                    *pos += 1;
                }
                None => {}
            }
            nodes.push(Node::Block {
                helper,
                args,
                hash,
                body,
                else_body,
            });
            continue;
        }
        // Inline variable or inline helper (subexpressions are handled by parse_arg).
        nodes.push(Node::Var(parse_arg(inner)));
        *pos += 1;
    }
    (nodes, None)
}

fn lookup(ctx: &[&Value], path: &str) -> Value {
    if path == "this" || path == "." || path.is_empty() {
        return ctx.last().copied().cloned().unwrap_or(Value::Null);
    }
    // `a.length` on arrays.
    if let Some((base, "length")) = path.rsplit_once('.') {
        if let Value::Array(a) = lookup(ctx, base) {
            return json!(a.len());
        }
    }
    for frame in ctx.iter().rev() {
        let mut cur = *frame;
        let mut ok = true;
        for part in path.split('.') {
            if let Value::Object(m) = cur {
                if let Some(v) = m.get(part) {
                    cur = v;
                } else {
                    ok = false;
                    break;
                }
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            return cur.clone();
        }
    }
    Value::Null
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
    }
}

fn eval_arg(arg: &Arg, ctx: &[&Value]) -> Value {
    match arg {
        Arg::Str(s) => Value::String(s.clone()),
        Arg::Num(n) => json!(n),
        Arg::Path(p) => lookup(ctx, p),
        Arg::Call(name, args) => eval_helper_value(name, args, ctx),
    }
}

fn eval_helper_value(name: &str, args: &[Arg], ctx: &[&Value]) -> Value {
    match name {
        "includes" | "has" => {
            let hay = eval_arg(&args[0], ctx);
            let needle = eval_arg(&args[1], ctx);
            Value::Bool(hay.as_array().is_some_and(|a| a.contains(&needle)))
        }
        "not" => Value::Bool(!truthy(&eval_arg(&args[0], ctx))),
        _ => Value::Null,
    }
}

fn value_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[allow(clippy::too_many_lines)] // template node walker; one loop per node kind, splitting adds indirection
fn render_nodes(nodes: &[Node], ctx: &mut Vec<&Value>, out: &mut String) {
    for node in nodes {
        match node {
            Node::Text(t) => out.push_str(t),
            Node::Var(arg) => {
                let v = eval_arg(arg, ctx);
                out.push_str(&value_string(&v));
            }
            Node::Block {
                helper,
                args,
                hash,
                body,
                else_body,
            } => match helper.as_str() {
                "if" => {
                    let cond = truthy(&eval_arg(&args[0], ctx));
                    render_nodes(if cond { body } else { else_body }, ctx, out);
                }
                "unless" => {
                    let cond = truthy(&eval_arg(&args[0], ctx));
                    render_nodes(if cond { else_body } else { body }, ctx, out);
                }
                "ifAny" => {
                    let any = args.iter().any(|a| truthy(&eval_arg(a, ctx)));
                    render_nodes(if any { body } else { else_body }, ctx, out);
                }
                "has" => {
                    let hay = eval_arg(&args[0], ctx);
                    let needle = eval_arg(&args[1], ctx);
                    let hit = hay.as_array().is_some_and(|a| a.contains(&needle));
                    render_nodes(if hit { body } else { else_body }, ctx, out);
                }
                "when" => {
                    // Guard malformed comparisons (TS renders the else branch).
                    if args.len() < 3 {
                        render_nodes(else_body, ctx, out);
                        continue;
                    }
                    let l = eval_arg(&args[0], ctx);
                    let op = eval_arg(&args[1], ctx);
                    let r = eval_arg(&args[2], ctx);
                    let lnum = l.as_f64().or_else(|| l.as_str()?.parse().ok());
                    let rnum = r.as_f64().or_else(|| r.as_str()?.parse().ok());
                    let eq = value_string(&l) == value_string(&r);
                    let cmp = match (lnum, rnum) {
                        (Some(a), Some(b)) => Some(a.partial_cmp(&b)),
                        _ => None,
                    };
                    let op = value_string(&op);
                    let pass = match op.as_str() {
                        "==" => eq,
                        "!=" => !eq,
                        ">" => cmp.is_some_and(|c| c == Some(std::cmp::Ordering::Greater)),
                        "<" => cmp.is_some_and(|c| c == Some(std::cmp::Ordering::Less)),
                        ">=" => cmp.is_some_and(|c| c != Some(std::cmp::Ordering::Less)),
                        "<=" => cmp.is_some_and(|c| c != Some(std::cmp::Ordering::Greater)),
                        _ => false,
                    };
                    render_nodes(if pass { body } else { else_body }, ctx, out);
                }
                "each" | "list" => {
                    let items = eval_arg(&args[0], ctx);
                    let Value::Array(items) = items else {
                        render_nodes(else_body, ctx, out);
                        continue;
                    };
                    if items.is_empty() {
                        render_nodes(else_body, ctx, out);
                        continue;
                    }
                    let join = hash
                        .iter()
                        .find(|(k, _)| k == "join")
                        .map_or("", |(_, v)| v.as_str());
                    let prefix = hash
                        .iter()
                        .find(|(k, _)| k == "prefix")
                        .map_or("", |(_, v)| v.as_str());
                    let rendered: Vec<String> = items
                        .iter()
                        .map(|item| {
                            let mut s = String::new();
                            let mut sub: Vec<&Value> = ctx.clone();
                            sub.push(item);
                            render_nodes(body, &mut sub, &mut s);
                            s
                        })
                        .collect();
                    if helper == "list" {
                        for (i, part) in rendered.iter().enumerate() {
                            if i > 0 {
                                out.push_str(join);
                            }
                            out.push_str(prefix);
                            out.push_str(part);
                        }
                    } else {
                        out.push_str(&rendered.concat());
                    }
                }
                other => {
                    // Unknown block helper: render body as-is (best effort).
                    let _ = other;
                    render_nodes(body, ctx, out);
                }
            },
        }
    }
}

const BLOCK_HELPERS: &[&str] = &[
    "if", "unless", "each", "list", "ifAny", "has", "when", "with",
];

/// Handlebars "standalone" semantics (mirrors `stripStandalone` in
/// pi-utils/template.ts): a line consisting solely of one block/else/close
/// tag plus whitespace is replaced by the trimmed tag alone — the leading
/// whitespace and the trailing newline disappear, the tag stays.
fn strip_standalone(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    for line in template.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = body.trim();
        let decor = trimmed
            .strip_prefix("{{")
            .and_then(|l| l.strip_suffix("}}"))
            .filter(|inner| !inner.contains("{{") && !inner.contains("}}"))
            .is_some_and(|inner| {
                let name = inner
                    .trim_start_matches('#')
                    .trim_start_matches('/')
                    .split_whitespace()
                    .next()
                    .unwrap_or("");
                inner.trim() == "else" || BLOCK_HELPERS.contains(&name)
            });
        if decor {
            out.push_str(trimmed); // tag kept; line newline dropped
        } else {
            out.push_str(line);
        }
    }
    out
}

/// `prompt.format` with `renderPhase: "post-render"`: trimEnd lines, collapse
/// runs of blank lines (keep at most one, and only between content), pop
/// blank lines before closing XML tags, strip trailing blanks. Code fences
/// (`` ``` `` / `` ~~~ ``) disable blank handling inside them.
fn format_post_render(content: &str) -> String {
    fn is_closing_tag_line(line: &str) -> bool {
        let Some(rest) = line.trim_start().strip_prefix("</") else {
            return false;
        };
        let name_len = rest
            .chars()
            .take_while(|c| c.is_ascii_lowercase() || *c == '-' || *c == '_')
            .count();
        name_len > 0 && rest[name_len..] == *">"
    }

    let lines: Vec<&str> = content.split('\n').collect();
    let mut result: Vec<String> = Vec::with_capacity(lines.len());
    let mut in_code = false;
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let line = raw.trim_end();
        let indentless = line.trim_start();
        if indentless.starts_with("```") || indentless.starts_with("~~~") {
            in_code = !in_code;
            result.push(line.to_string());
            i += 1;
            continue;
        }
        if in_code {
            result.push(line.to_string());
            i += 1;
            continue;
        }
        if line.trim().is_empty() {
            let next_blank = i + 1 >= lines.len() || lines[i + 1].trim().is_empty();
            if next_blank {
                // Run of 2+ blanks (or trailing): strip entirely.
                while result.last().is_some_and(std::string::String::is_empty) {
                    result.pop();
                }
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                i = j;
                continue;
            }
            if result.last().is_none_or(std::string::String::is_empty) {
                i += 1;
                continue;
            }
            result.push(String::new());
            i += 1;
            continue;
        }
        if is_closing_tag_line(line) {
            while result.last().is_some_and(std::string::String::is_empty) {
                result.pop();
            }
        }
        result.push(line.to_string());
        i += 1;
    }
    while result.last().is_some_and(std::string::String::is_empty) {
        result.pop();
    }
    result.join("\n")
}

/// Render a template against a JSON data object: Handlebars-subset render
/// followed by the `prompt.format` post-render pass, matching
/// `prompt.render(template, context)`.
pub fn render(template: &str, data: &Value) -> String {
    let template = strip_standalone(template);
    let tags = tokenize(&template);
    let mut pos = 0;
    let (nodes, _) = parse_nodes(&tags, &mut pos);
    let mut out = String::new();
    let mut ctx: Vec<&Value> = vec![data];
    render_nodes(&nodes, &mut ctx, &mut out);
    format_post_render(&out)
}

// ---------------------------------------------------------------------------
// System prompt assembly (`buildSystemPrompt`).
// ---------------------------------------------------------------------------

/// Inputs the CLI gathers for the system prompt.
#[derive(Default)]
pub struct Options {
    /// `--system-prompt <path|text>` (replaces the default prompt).
    pub system_prompt: Option<String>,
    /// `--append-system-prompt <path|text>` (appended in the project block).
    pub append_system_prompt: Option<String>,
    /// `--tools <csv>`: restrict enabled tools (omp semantics).
    pub tools_csv: Option<String>,
    /// Prompt is for a subagent session.
    pub subagent: bool,
    /// `--compact-prompt`: select the compact bundled template.
    pub compact_prompt: bool,
    /// Active model identifier (surfaced only when the config enables it).
    pub model: String,
    /// Generated append blocks (e.g. MCP server instructions), joined like
    /// `composeAppendPrompt`.
    pub generated_append: Vec<String>,
}

/// Result of building the provider-facing system prompt blocks.
pub struct Built {
    /// Ordered blocks; providers preserve them as distinct messages.
    pub blocks: Vec<String>,
    /// Tool names enabled after the `--tools` restriction.
    #[allow(dead_code)] // read by tests and prompt debug dumps; kept with `Built`
    pub tool_names: Vec<String>,
}

/// A tool's prompt metadata: internal name plus display label.
pub type ToolMeta = (String, String);

/// Join generated append parts with the user's append prompt
/// (`composeAppendPrompt`).
fn compose_append(generated: &[String], user: Option<&str>) -> String {
    let generated = if generated.is_empty() {
        None
    } else {
        Some(generated.join("\n\n"))
    };
    let user = user.filter(|u| !u.trim().is_empty());
    match (generated, user) {
        (Some(g), Some(u)) => render(
            USER_APPEND_TEMPLATE,
            &json!({ "generatedAppend": g, "userAppend": u }),
        ),
        (Some(g), None) => g,
        (None, Some(u)) => u.to_string(),
        (None, None) => String::new(),
    }
}

/// Build the system prompt blocks for `cwd` (env-injected dirs for tests).
#[allow(clippy::too_many_lines)] // sequential prompt assembly; mirrors omp's buildSystemPrompt order
pub fn build_with(
    cwd: &Path,
    home: &Path,
    agent: &Path,
    opts: &Options,
    available_tools: &[ToolMeta],
) -> Built {
    let cfg = load_config(agent);

    // Resolve the custom prompt: explicit flag > discovered SYSTEM.md /
    // SYSTEM_TEMPLATE.md override (resolvePromptInput + discovery).
    let mut custom_prompt: Option<String> = None;
    let mut template: Option<String> = None;
    if let Some(flag) = opts.system_prompt.as_deref() {
        custom_prompt = Some(resolve_prompt_input(flag));
    } else if let Some(ov) = discover_override(cwd, agent) {
        match ov {
            Override::Text(t) => custom_prompt = Some(t),
            Override::Template(t) => template = Some(t),
        }
    }
    let caller_controls = custom_prompt.is_some() || template.is_some();
    // `loadSystemPromptFiles` customization: only consulted when the caller
    // does not control the prompt; same discovery feed, usually null.
    let customization: Option<String> = if caller_controls {
        None
    } else {
        discover_override(cwd, agent).and_then(|ov| match ov {
            Override::Text(t) => Some(t),
            Override::Template(_) => None,
        })
    };

    let user_append = opts
        .append_system_prompt
        .as_deref()
        .map(resolve_prompt_input);
    let append = compose_append(&opts.generated_append, user_append.as_deref());

    // Tool surface: explicit list > the full registry. `--tools` entries that
    // match no registered tool are ignored (omp `normalizeToolNames`).
    let available_names: Vec<String> = available_tools.iter().map(|(n, _)| n.clone()).collect();
    let tool_names: Vec<String> = match opts.tools_csv.as_deref() {
        Some(csv) => csv
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .filter(|n| available_names.iter().any(|a| a == *n))
            .map(ToString::to_string)
            .collect(),
        None => available_names,
    };

    let skills = load_skills(agent, cwd, cfg.skills_enabled);
    let has_skill_reader = tool_names.iter().any(|n| n == "read");
    let has_skill_uri_access = has_skill_reader && !skills.is_empty();

    let env: Vec<Value> = environment_info()
        .into_iter()
        .map(|(label, value)| json!({ "label": label, "value": value }))
        .collect();
    let files: Vec<Value> = context_files(cwd, home)
        .into_iter()
        .map(|f| {
            json!({
                "path": f.path.to_string_lossy(),
                "content": f.content,
                "depth": f.depth,
            })
        })
        .collect();
    let tool_refs: Map<String, Value> = tool_names
        .iter()
        .map(|n| (n.clone(), Value::String(n.clone())))
        .collect();
    let tool_info: Vec<Value> = available_tools
        .iter()
        .filter(|(n, _)| tool_names.contains(n))
        .map(|(n, label)| json!({ "name": n, "label": label }))
        .collect();
    let skill_values: Vec<Value> = skills
        .iter()
        .map(|s| json!({ "name": s.name, "description": s.description }))
        .collect();

    let data = json!({
        "systemPromptCustomization": customization,
        "customPrompt": custom_prompt,
        "appendPrompt": append,
        "tools": tool_names,
        "toolInfo": tool_info,
        "toolInventory": "",
        "inlineToolDescriptors": cfg.inline_tool_descriptors,
        "toolListMode": !cfg.inline_tool_descriptors,
        "toolRefs": Value::Object(tool_refs),
        "environment": env,
        "contextFiles": files,
        "agentsMdSearch": { "files": [] },
        "workspaceTree": { "rootPath": cwd, "rendered": "", "truncated": false },
        "hasSkillUriAccess": has_skill_uri_access,
        "internalUrls": [],
        "skills": skill_values,
        "rules": [],
        "alwaysApplyRules": [],
        "cwd": cwd.to_string_lossy(),
        "additionalWorkspaceRoots": [],
        "model": if cfg.include_model_in_prompt { opts.model.clone() } else { String::new() },
        "delegationBias": "eager",
        "personality": "",
        "intentTracing": false,
        "intentField": "",
        "eagerTasks": false,
        "eagerTasksAlways": false,
        "inlineFirstDelegation": false,
        "taskBatch": true,
        "MAX_CONCURRENCY": 0,
        "scoutAvailable": true,
        "taskIrcEnabled": false,
        "secretsEnabled": false,
        "browserEnabled": false,
        "computerEnabled": false,
    });
    let more = json!({
        "includeWorkspaceTree": false,
        "renderMermaid": true,
        "reactions": cfg.reactions,
        "xdevTools": [],
        "hasDynamicXdevTools": false,
        "xdevDocs": "",
        "autoQaEnabled": false,
        "writeTransportOnly": false,
        "subagent": opts.subagent,
        "activeRepoContext": "",
    });
    let mut data = data;
    if let (Some(dst), Some(src)) = (data.as_object_mut(), more.as_object()) {
        dst.extend(src.clone());
    }

    // Compact selection: explicit flag > `OMT_COMPACT_PROMPT=1` >
    // `systemPrompt.compact` in the agent config. A caller-controlled prompt
    // (custom text or template) always wins.
    let compact = opts.compact_prompt
        || std::env::var("OMT_COMPACT_PROMPT").is_ok_and(|v| v == "1")
        || cfg.compact;
    let selected = if custom_prompt.is_some() {
        CUSTOM_TEMPLATE
    } else if compact {
        COMPACT_TEMPLATE
    } else {
        template.as_deref().unwrap_or(BUNDLED_TEMPLATE)
    };
    let rendered = render(selected, &data).trim().to_string();

    let mut blocks = vec![rendered];
    let project = render(PROJECT_TEMPLATE, &data).trim().to_string();
    if !project.is_empty() {
        blocks.push(project);
    }
    Built { blocks, tool_names }
}

/// Build the system prompt using the process home/agent dirs.
#[must_use]
pub fn build(cwd: &Path, opts: &Options, available_tools: &[ToolMeta]) -> Built {
    build_with(cwd, &home_dir(), &agent_dir(), opts, available_tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omp-rs-sysprompt-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Project SYSTEM.md beats user-level, mirroring
    /// `discoverSystemPromptOverride` scope priority.
    #[test]
    fn project_system_md_beats_user() {
        let root = tempdir("ov");
        let proj = root.join("proj");
        let agent = root.join("agent");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::write(agent.join("SYSTEM.md"), "user prompt").unwrap();
        std::fs::write(proj.join("SYSTEM.md"), "project prompt").unwrap();
        match discover_override(&proj, &agent).unwrap() {
            Override::Text(t) => assert_eq!(t, "project prompt"),
            Override::Template(_) => panic!("expected text override, got template"),
        }
        // No project file: user-level wins.
        std::fs::remove_file(proj.join("SYSTEM.md")).unwrap();
        match discover_override(&proj, &agent).unwrap() {
            Override::Text(t) => assert_eq!(t, "user prompt"),
            Override::Template(_) => panic!("expected text override, got template"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// AGENTS.md/CLAUDE.md walk-up: closest file claims the depth scope, only
    /// one file per directory, empty files ignored, results depth-descending.
    #[test]
    fn context_files_walk_up() {
        let root = tempdir("ctx");
        let cwd = root.join("a/b");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(root.join("AGENTS.md"), "root rules").unwrap();
        std::fs::write(cwd.join("CLAUDE.md"), "cwd rules").unwrap();
        let home = root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let files = context_files(&cwd, &home);
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["AGENTS.md", "CLAUDE.md"]);
        assert!(files[0].depth > files[1].depth);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Skills: `enabled: false` in config.yml suppresses the listing; hide:
    /// true entries are dropped from the render.
    #[test]
    fn skills_respect_config_and_hide() {
        let root = tempdir("skills");
        let agent = root.join("agent");
        let sk = agent.join("skills/demo");
        std::fs::create_dir_all(&sk).unwrap();
        std::fs::write(agent.join("config.yml"), "skills:\n  enabled: false\n").unwrap();
        std::fs::write(
            sk.join("SKILL.md"),
            "---\nname: demo\ndescription: Demo skill.\n---\nbody\n",
        )
        .unwrap();
        assert!(load_skills(&agent, &root, false).is_empty());
        // Enabled: visible skill loads with frontmatter fields.
        std::fs::write(agent.join("config.yml"), "skills:\n  enabled: true\n").unwrap();
        let skills = load_skills(&agent, &root, true);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "demo");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `--tools <csv>` keeps only registered names, in the given order.
    #[test]
    fn tools_csv_restricts_surface() {
        let root = tempdir("tools");
        let agent = root.join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        let available: Vec<ToolMeta> = [("read", "read"), ("bash", "bash"), ("edit", "edit")]
            .into_iter()
            .map(|(a, b)| (a.into(), b.into()))
            .collect();
        let opts = Options {
            tools_csv: Some("bash,read,unknown".into()),
            ..Options::default()
        };
        let built = build_with(&root, &root, &agent, &opts, &available);
        assert_eq!(built.tool_names, vec!["bash", "read"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `--compact-prompt` selects the compact template; without it the full
    /// bundled prompt is used.
    #[test]
    fn compact_prompt_selection() {
        let root = tempdir("compact");
        let agent = root.join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        let tools: Vec<ToolMeta> = [("read", "read"), ("bash", "bash")]
            .into_iter()
            .map(|(a, b)| (a.into(), b.into()))
            .collect();
        let built = build_with(&root, &root, &agent, &Options::default(), &tools);
        assert!(!built.blocks[0].contains("Edit discipline"));
        let opts = Options {
            compact_prompt: true,
            ..Options::default()
        };
        let built = build_with(&root, &root, &agent, &opts, &tools);
        assert!(built.blocks[0].contains("Edit discipline"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `systemPrompt.compact: true` in the agent config selects the compact
    /// template without the flag.
    #[test]
    fn compact_prompt_setting() {
        let root = tempdir("compact-setting");
        let agent = root.join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::write(agent.join("config.yml"), "systemPrompt:\n  compact: true\n").unwrap();
        let built = build_with(&root, &root, &agent, &Options::default(), &[]);
        assert!(built.blocks[0].contains("Edit discipline"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `--system-prompt` literal (no file) replaces the discovered override,
    /// and `--append-system-prompt` lands at the end of the project block.
    #[test]
    fn flags_replace_and_append() {
        let root = tempdir("flags");
        let agent = root.join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::write(agent.join("SYSTEM.md"), "discovered").unwrap();
        let opts = Options {
            system_prompt: Some("literal custom".into()),
            append_system_prompt: Some("extra guidance".into()),
            ..Options::default()
        };
        let built = build_with(&root, &root, &agent, &opts, &[]);
        assert_eq!(built.blocks[0], "literal custom");
        assert!(built.blocks[1].ends_with("extra guidance"));
        // Discovery path: user SYSTEM.md becomes block 0 verbatim (trimmed).
        let opts = Options::default();
        let built = build_with(&root, &root, &agent, &opts, &[]);
        assert_eq!(built.blocks[0], "discovered");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Renderer: standalone block lines disappear, `list` honors join/prefix
    /// escapes, nested if/else and each render correctly.
    #[test]
    fn renderer_semantics() {
        let tmpl = "A\n{{#if ok}}\nYES\n{{else}}\nNO\n{{/if}}\nB\n{{#list items prefix=\"- \" join=\"\\n\"}}{{this}}{{/list}}";
        let out = render(tmpl, &json!({"ok": true, "items": ["x", "y"]}));
        assert_eq!(out, "A\nYES\nB\n- x\n- y");
        let out = render(tmpl, &json!({"ok": false, "items": []}));
        // `prompt.format` strips trailing blank lines.
        assert_eq!(out, "A\nNO\nB");
    }

    /// Regression: `{{#when x "==" "y"}}` — the quoted operator must parse as
    /// an argument, not a `key=value` hash (same-line nesting included).
    #[test]
    fn when_operator_with_equals_is_an_arg() {
        let tmpl =
            "{{#when delegationBias \"==\" \"eager\"}}E{{/when}}{{#when n \">\" 1}}G{{/when}}";
        assert_eq!(
            render(tmpl, &json!({"delegationBias": "eager", "n": 2})),
            "EG"
        );
        assert_eq!(
            render(tmpl, &json!({"delegationBias": "gated", "n": 0})),
            ""
        );
        // Same-line nested open/close (system-prompt.md:160 shape).
        let nested = "{{#if scout}}{{#when bias \"==\" \"eager\"}}; scout allowed{{/when}}{{/if}}.";
        assert_eq!(
            render(nested, &json!({"scout": true, "bias": "eager"})),
            "; scout allowed."
        );
        assert_eq!(
            render(nested, &json!({"scout": true, "bias": "gated"})),
            "."
        );
    }

    /// `resolvePromptInput`: multi-line stays literal; an existing path is
    /// read; a missing path falls back to the literal string.
    #[test]
    fn resolve_prompt_input_semantics() {
        assert_eq!(resolve_prompt_input("a\nb"), "a\nb");
        let root = tempdir("rpi");
        let f = root.join("prompt.md");
        std::fs::write(&f, "from file\n").unwrap();
        assert_eq!(resolve_prompt_input(f.to_str().unwrap()), "from file\n");
        assert_eq!(resolve_prompt_input("no such file"), "no such file");
        let _ = std::fs::remove_dir_all(&root);
    }
}
