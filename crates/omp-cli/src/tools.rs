//! Built-in tools: bash, read, write, edit, grep, glob.
//!
//! Tool names and parameter names follow omp where practical. Failures are
//! returned as `Err` so the agent loop reports them with `isError: true`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use pa_agent::abort::AbortSignal;
use pa_agent::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use pa_agent::BoxFut;
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;

/// Output cap for every tool result.
pub const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const DEFAULT_READ_LINES: usize = 2000;
const DEFAULT_BASH_TIMEOUT_SECS: f64 = 120.0;
const MAX_GREP_MATCHES: usize = 500;
const MAX_GLOB_RESULTS: usize = 1000;

type Handler = fn(PathBuf, Value, AbortSignal) -> BoxFut<'static, Result<AgentToolResult>>;

struct FnTool {
    name: &'static str,
    description: &'static str,
    schema: Value,
    cwd: PathBuf,
    run: Handler,
}

impl AgentTool for FnTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        self.description
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
        (self.run)(self.cwd.clone(), params, signal)
    }
}

/// All built-in tools, rooted at `cwd`.
#[must_use]
pub fn builtin_tools(cwd: &Path) -> Vec<Arc<dyn AgentTool>> {
    let mk = |name, description, schema, run: Handler| -> Arc<dyn AgentTool> {
        Arc::new(FnTool {
            name,
            description,
            schema,
            cwd: cwd.to_path_buf(),
            run,
        })
    };
    vec![
        mk(
            "bash",
            "Run a shell command (bash). stdout and stderr are merged. Non-zero exit is reported as an error. Output is tail-truncated to 50KB.",
            json!({"type":"object","properties":{
                "command":{"type":"string","description":"Command to run"},
                "cwd":{"type":"string","description":"Working directory (default: session cwd)"},
                "timeout":{"type":"number","description":"Timeout in seconds (default 120)"}
            },"required":["command"]}),
            |c, a, s| Box::pin(bash(c, a, s)),
        ),
        mk(
            "read",
            "Read a text file. Optional offset (1-indexed start line) and limit (max lines, default 2000).",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"File path"},
                "offset":{"type":"integer","description":"1-indexed line to start from"},
                "limit":{"type":"integer","description":"Maximum number of lines"}
            },"required":["path"]}),
            |c, a, _| Box::pin(async move { read(&c, &a) }),
        ),
        mk(
            "write",
            "Write content to a file, creating parent directories and overwriting existing content.",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"File path"},
                "content":{"type":"string","description":"Full file content"}
            },"required":["path","content"]}),
            |c, a, _| Box::pin(async move { write(&c, &a) }),
        ),
        mk(
            "edit",
            "Replace one exact occurrence of old_string with new_string in a file. old_string must match exactly once.",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"File path"},
                "old_string":{"type":"string","description":"Exact text to replace (must be unique)"},
                "new_string":{"type":"string","description":"Replacement text"}
            },"required":["path","old_string","new_string"]}),
            |c, a, _| Box::pin(async move { edit(&c, &a) }),
        ),
        mk(
            "grep",
            "Search file contents with a regex (respects .gitignore). Returns path:line:text.",
            json!({"type":"object","properties":{
                "pattern":{"type":"string","description":"Regex pattern"},
                "path":{"type":"string","description":"File or directory to search (default: cwd)"},
                "glob":{"type":"string","description":"Only search files matching this glob"},
                "i":{"type":"boolean","description":"Case-insensitive"}
            },"required":["pattern"]}),
            |c, a, _| {
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || grep(&c, &a))
                        .await
                        .context("grep task failed")?
                })
            },
        ),
        mk(
            "glob",
            "Find files by glob pattern (respects .gitignore), newest first.",
            json!({"type":"object","properties":{
                "pattern":{"type":"string","description":"Glob pattern, e.g. **/*.rs"},
                "path":{"type":"string","description":"Directory to search (default: cwd)"}
            },"required":["pattern"]}),
            |c, a, _| {
                Box::pin(async move {
                    tokio::task::spawn_blocking(move || glob(&c, &a))
                        .await
                        .context("glob task failed")?
                })
            },
        ),
    ]
}

// ---------------------------------------------------------------- helpers

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string argument `{key}`"))
}

fn resolve(cwd: &Path, p: &str) -> PathBuf {
    let expanded = if let Some(rest) = p.strip_prefix("~/") {
        std::env::var_os("HOME").map_or_else(|| PathBuf::from(p), |h| PathBuf::from(h).join(rest))
    } else {
        PathBuf::from(p)
    };
    if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    }
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Keep the last `max` bytes (UTF-8 safe), prefixing a marker when cut.
#[must_use]
pub fn truncate_tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let start = ceil_boundary(s, s.len() - max);
    format!(
        "[output truncated: showing last {} of {} bytes]\n{}",
        s.len() - start,
        s.len(),
        &s[start..]
    )
}

/// Keep the first `max` bytes (UTF-8 safe), appending a marker when cut.
#[must_use]
pub fn truncate_head(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let end = floor_boundary(s, max);
    format!(
        "{}\n[output truncated: showing first {} of {} bytes]",
        &s[..end],
        end,
        s.len()
    )
}

// ------------------------------------------------------------------- bash

async fn bash(cwd: PathBuf, args: Value, signal: AbortSignal) -> Result<AgentToolResult> {
    let command = str_arg(&args, "command")?.to_string();
    let dir = args
        .get("cwd")
        .and_then(Value::as_str)
        .map_or_else(|| cwd.clone(), |d| resolve(&cwd, d));
    let timeout = args
        .get("timeout")
        .and_then(Value::as_f64)
        .filter(|t| *t > 0.0)
        .unwrap_or(DEFAULT_BASH_TIMEOUT_SECS);

    let script = format!("exec 2>&1\n{command}");
    let mut child = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(script)
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("failed to spawn bash in {}", dir.display()))?;
    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;

    // Keep roughly the last MAX_OUTPUT_BYTES; memory stays bounded.
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 8192];
    let deadline = tokio::time::sleep(Duration::from_secs_f64(timeout));
    tokio::pin!(deadline);
    let mut outcome = None; // Some("timeout"|"aborted")
    loop {
        tokio::select! {
            n = stdout.read(&mut chunk) => {
                let n = n.context("reading command output")?;
                if n == 0 { break; }
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > MAX_OUTPUT_BYTES * 4 {
                    let cut = buf.len() - MAX_OUTPUT_BYTES * 2;
                    buf.drain(..cut);
                }
            }
            () = &mut deadline => { outcome = Some("timeout"); break; }
            () = signal.aborted() => { outcome = Some("aborted"); break; }
        }
    }
    if let Some(reason) = outcome {
        let _ = child.kill().await;
        let text = truncate_tail(&String::from_utf8_lossy(&buf), MAX_OUTPUT_BYTES);
        bail!(
            "{text}\n[command {}]",
            if reason == "timeout" {
                format!("timed out after {timeout}s")
            } else {
                "aborted".to_string()
            }
        );
    }
    let status = child.wait().await.context("waiting for command")?;
    let text = truncate_tail(&String::from_utf8_lossy(&buf), MAX_OUTPUT_BYTES);
    match status.code() {
        Some(0) => Ok(AgentToolResult::text(if text.is_empty() {
            "(no output)".to_string()
        } else {
            text
        })),
        Some(code) => bail!("{text}\n[exit code {code}]"),
        None => bail!("{text}\n[terminated by signal]"),
    }
}

// ------------------------------------------------------------------- read

fn read(cwd: &Path, args: &Value) -> Result<AgentToolResult> {
    let path = resolve(cwd, str_arg(args, "path")?);
    let bytes = std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .map_or(1, |o| usize::try_from(o).unwrap_or(usize::MAX).max(1));
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(DEFAULT_READ_LINES, |l| usize::try_from(l).unwrap_or(usize::MAX));
    if total == 0 {
        return Ok(AgentToolResult::text("(empty file)"));
    }
    if offset > total {
        bail!("offset {offset} is beyond end of file ({total} lines)");
    }
    let start = offset - 1;
    let end = start.saturating_add(limit).min(total);
    let body = truncate_head(&lines[start..end].join("\n"), MAX_OUTPUT_BYTES);
    let mut out = body;
    if end < total {
        out.push_str(&format!(
            "\n[showing lines {offset}-{end} of {total}; use offset={} to continue]",
            end + 1
        ));
    }
    Ok(AgentToolResult::text(out))
}

// ------------------------------------------------------------------ write

fn write(cwd: &Path, args: &Value) -> Result<AgentToolResult> {
    let path = resolve(cwd, str_arg(args, "path")?);
    let content = str_arg(args, "content")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    std::fs::write(&path, content).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(AgentToolResult::text(format!(
        "Wrote {} bytes to {}",
        content.len(),
        path.display()
    )))
}

// ------------------------------------------------------------------- edit

fn edit(cwd: &Path, args: &Value) -> Result<AgentToolResult> {
    let path = resolve(cwd, str_arg(args, "path")?);
    let old = str_arg(args, "old_string")?;
    let new = str_arg(args, "new_string")?;
    if old.is_empty() {
        bail!("old_string must not be empty");
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let count = text.matches(old).count();
    match count {
        0 => bail!("old_string not found in {}", path.display()),
        1 => {}
        n => bail!(
            "old_string matches {n} times in {}; include more context to make it unique",
            path.display()
        ),
    }
    let updated = text.replacen(old, new, 1);
    std::fs::write(&path, updated).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(AgentToolResult::text(format!("Edited {}", path.display())))
}

// ------------------------------------------------------------------- grep

fn display_path(cwd: &Path, p: &Path) -> String {
    p.strip_prefix(cwd).unwrap_or(p).display().to_string()
}

fn grep(cwd: &Path, args: &Value) -> Result<AgentToolResult> {
    let pattern = str_arg(args, "pattern")?;
    let root = args
        .get("path")
        .and_then(Value::as_str)
        .map_or_else(|| cwd.to_path_buf(), |p| resolve(cwd, p));
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(args.get("i").and_then(Value::as_bool).unwrap_or(false))
        .build()
        .map_err(|e| anyhow!("invalid regex: {e}"))?;
    let glob = match args.get("glob").and_then(Value::as_str) {
        Some(g) => Some(
            globset::Glob::new(g)
                .map_err(|e| anyhow!("invalid glob: {e}"))?
                .compile_matcher(),
        ),
        None => None,
    };
    let mut out = String::new();
    let mut matches = 0usize;
    'files: for entry in ignore::WalkBuilder::new(&root).build().flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if let Some(g) = &glob {
            let rel = path.strip_prefix(&root).unwrap_or(path);
            if !g.is_match(rel) && !g.is_match(path.file_name().unwrap_or_default()) {
                continue;
            }
        }
        let Ok(bytes) = std::fs::read(path) else { continue };
        if bytes.contains(&0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                out.push_str(&format!("{}:{}:{}\n", display_path(cwd, path), i + 1, line));
                matches += 1;
                if matches >= MAX_GREP_MATCHES || out.len() > MAX_OUTPUT_BYTES {
                    out.push_str("[match limit reached]\n");
                    break 'files;
                }
            }
        }
    }
    if matches == 0 {
        return Ok(AgentToolResult::text("No matches found"));
    }
    Ok(AgentToolResult::text(truncate_head(out.trim_end(), MAX_OUTPUT_BYTES)))
}

// ------------------------------------------------------------------- glob

fn glob(cwd: &Path, args: &Value) -> Result<AgentToolResult> {
    let pattern = str_arg(args, "pattern")?;
    let root = args
        .get("path")
        .and_then(Value::as_str)
        .map_or_else(|| cwd.to_path_buf(), |p| resolve(cwd, p));
    let matcher = globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map_err(|e| anyhow!("invalid glob: {e}"))?
        .compile_matcher();
    // A pattern without a separator matches at any depth, like gitignore/ripgrep.
    let any_depth = !pattern.contains('/');
    let mut found: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in ignore::WalkBuilder::new(&root).build().flatten() {
        let path = entry.path();
        if path == root {
            continue;
        }
        let rel = path.strip_prefix(&root).unwrap_or(path);
        let hit = matcher.is_match(rel)
            || (any_depth && rel.file_name().is_some_and(|n| matcher.is_match(n)));
        if hit {
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            found.push((mtime, display_path(cwd, path)));
        }
    }
    if found.is_empty() {
        return Ok(AgentToolResult::text("No files found"));
    }
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let total = found.len();
    let mut out: Vec<String> = found
        .into_iter()
        .take(MAX_GLOB_RESULTS)
        .map(|(_, p)| p)
        .collect();
    if total > MAX_GLOB_RESULTS {
        out.push(format!("[showing {MAX_GLOB_RESULTS} of {total} results]"));
    }
    Ok(AgentToolResult::text(truncate_head(&out.join("\n"), MAX_OUTPUT_BYTES)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(r: &AgentToolResult) -> String {
        r.content
            .iter()
            .map(|c| match c {
                pa_agent::types::ToolResultContent::Text(t) => t.text.clone(),
                pa_agent::types::ToolResultContent::Image(_) => String::new(),
            })
            .collect()
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("omprs-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn edit_requires_unique_match() {
        let d = tmp("edit");
        std::fs::write(d.join("f.txt"), "a b a").unwrap();
        let e = edit(&d, &json!({"path":"f.txt","old_string":"a","new_string":"x"})).unwrap_err();
        assert!(e.to_string().contains("2 times"), "{e}");
        let e = edit(&d, &json!({"path":"f.txt","old_string":"zzz","new_string":"x"})).unwrap_err();
        assert!(e.to_string().contains("not found"));
        edit(&d, &json!({"path":"f.txt","old_string":"b","new_string":"x"})).unwrap();
        assert_eq!(std::fs::read_to_string(d.join("f.txt")).unwrap(), "a x a");
    }

    #[test]
    fn read_ranges() {
        let d = tmp("read");
        std::fs::write(d.join("f.txt"), "1\n2\n3\n4\n5\n").unwrap();
        let r = read(&d, &json!({"path":"f.txt","offset":2,"limit":2})).unwrap();
        let t = text_of(&r);
        assert!(t.starts_with("2\n3"), "{t}");
        assert!(t.contains("offset=4"), "{t}");
        let r = read(&d, &json!({"path":"f.txt","offset":5})).unwrap();
        assert_eq!(text_of(&r), "5");
        assert!(read(&d, &json!({"path":"f.txt","offset":9})).is_err());
    }

    #[test]
    fn write_creates_parents() {
        let d = tmp("write");
        write(&d, &json!({"path":"a/b/c.txt","content":"hi"})).unwrap();
        assert_eq!(std::fs::read_to_string(d.join("a/b/c.txt")).unwrap(), "hi");
    }

    #[tokio::test]
    async fn bash_exit_code_and_output() {
        let d = tmp("bash");
        let ok = bash(d.clone(), json!({"command":"echo out; echo err >&2"}), AbortSignal::never())
            .await
            .unwrap();
        let t = text_of(&ok);
        assert!(t.contains("out") && t.contains("err"), "{t}");
        let e = bash(d.clone(), json!({"command":"echo boom; exit 3"}), AbortSignal::never())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("boom") && e.contains("exit code 3"), "{e}");
        let e = bash(d, json!({"command":"sleep 5","timeout":0.2}), AbortSignal::never())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("timed out"), "{e}");
    }

    #[test]
    fn truncation_keeps_tail() {
        let s = "x".repeat(100) + "END";
        let t = truncate_tail(&s, 10);
        assert!(t.ends_with("END") && t.contains("truncated"));
    }

    #[test]
    fn grep_and_glob_respect_gitignore() {
        let d = tmp("grep");
        std::fs::create_dir_all(d.join(".git")).unwrap();
        std::fs::write(d.join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(d.join("keep.txt"), "needle here\n").unwrap();
        std::fs::write(d.join("ignored.txt"), "needle there\n").unwrap();
        let t = text_of(&grep(&d, &json!({"pattern":"needle"})).unwrap());
        assert!(t.contains("keep.txt:1:needle here") && !t.contains("ignored.txt"), "{t}");
        let t = text_of(&glob(&d, &json!({"pattern":"*.txt"})).unwrap());
        assert!(t.contains("keep.txt") && !t.contains("ignored.txt"), "{t}");
    }
}
