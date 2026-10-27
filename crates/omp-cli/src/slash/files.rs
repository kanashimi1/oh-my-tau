//! File-based slash commands (`commands/*.md` templates).

use super::{Mutex, Path, PathBuf};

pub(super) static FILE_COMMANDS: Mutex<Vec<FileCommand>> = Mutex::new(Vec::new());

/// A file-based slash command (`commands/*.md`).
#[derive(Debug, Clone)]
pub struct FileCommand {
    pub name: String,
    pub description: String,
    pub argument_hint: Option<String>,
    pub content: String,
}

fn command_dirs(cwd: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push((PathBuf::from(home).join(".omp/agent/commands"), "user"));
    }
    dirs.push((cwd.join(".omp/commands"), "project"));
    dirs
}

/// Load markdown commands from the user and project `commands/` directories
/// (TS `loadSlashCommands` builtin provider; frontmatter description /
/// `argument-hint` honored, first body line as fallback description).
pub fn load_file_commands(cwd: &Path) -> Vec<FileCommand> {
    let mut out: Vec<FileCommand> = Vec::new();
    for (dir, _level) in command_dirs(cwd) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "md"))
            .collect();
        paths.sort();
        for path in paths {
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if out.iter().any(|c| c.name == name) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let (frontmatter, body) = split_frontmatter(&content);
            let mut description = String::new();
            let mut argument_hint = None;
            for line in frontmatter.lines() {
                if let Some(v) = line.strip_prefix("description:") {
                    description = v.trim().to_string();
                }
                if let Some(v) = line
                    .strip_prefix("argument-hint:")
                    .or_else(|| line.strip_prefix("argumentHint:"))
                {
                    argument_hint = Some(v.trim().to_string());
                }
            }
            if description.is_empty() {
                if let Some(first) = body.lines().find(|l| !l.trim().is_empty()) {
                    description = first.trim().chars().take(60).collect();
                    if first.trim().chars().count() > 60 {
                        description.push_str("...");
                    }
                }
            }
            out.push(FileCommand {
                name: name.to_string(),
                description,
                argument_hint,
                content: body.to_string(),
            });
        }
    }
    out
}

/// Minimal frontmatter split: `---` fenced block at the top of the file.
fn split_frontmatter(content: &str) -> (&str, &str) {
    if let Some(rest) = content.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---") {
            let fm = &rest[..end];
            let body = rest[end + 4..]
                .strip_prefix('\n')
                .unwrap_or(&rest[end + 4..]);
            return (fm, body);
        }
    }
    ("", content)
}

/// Reload the cached file commands from disk (cheap; safe to call from async context).
pub fn refresh_file_commands(cwd: &Path) {
    let cmds = load_file_commands(cwd);
    *FILE_COMMANDS.lock() = cmds;
}

/// Expand `/name args` through a file template (`expandSlashCommand`): the
/// raw args string is appended when the template has no `$ARGUMENTS`/`$1`
/// placeholders; `$ARGUMENTS`/`$ARGS` render the args text, `$1..$9` the
/// positional words.
pub(super) fn expand_file_command(name: &str, args: &str, cwd: &Path) -> Option<String> {
    let cmds = load_file_commands(cwd);
    let cmd = cmds.into_iter().find(|c| c.name == name)?;
    let positional: Vec<&str> = args.split_whitespace().collect();
    let args_text = positional.join(" ");
    let mut body = cmd.content;
    let mut used = false;
    for (placeholder, value) in [
        ("$ARGUMENTS", args_text.as_str()),
        ("$ARGS", args_text.as_str()),
    ] {
        if body.contains(placeholder) {
            used = true;
            body = body.replace(placeholder, value);
        }
    }
    for (i, word) in positional.iter().enumerate().take(9) {
        let placeholder = format!("${}", i + 1);
        if body.contains(&placeholder) {
            used = true;
            body = body.replace(&placeholder, word);
        }
    }
    if !used && !args_text.is_empty() {
        body = body.trim_end().to_string();
        body.push('\n');
        body.push_str(&args_text);
    }
    Some(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_commands_load_and_expand() {
        let dir = std::env::temp_dir().join(format!("omp-slash-test-{}", std::process::id()));
        let cmds = dir.join(".omp/commands");
        std::fs::create_dir_all(&cmds).unwrap();
        std::fs::write(
            cmds.join("greet.md"),
            "---\ndescription: Greet someone\nargument-hint: <name>\n---\nHello $ARGUMENTS!\n",
        )
        .unwrap();
        let loaded = load_file_commands(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "greet");
        assert_eq!(loaded[0].description, "Greet someone");
        assert_eq!(loaded[0].argument_hint.as_deref(), Some("<name>"));
        let expanded = expand_file_command("greet", "World", &dir).unwrap();
        assert_eq!(expanded.trim_end(), "Hello World!");
        // No placeholder: args appended inline (TS appendInlineArgsFallback).
        std::fs::write(cmds.join("plain.md"), "Just text\n").unwrap();
        let expanded = expand_file_command("plain", "extra bits", &dir).unwrap();
        assert_eq!(expanded.trim_end(), "Just text\nextra bits");
        assert_eq!(expand_file_command("nope", "", &dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
