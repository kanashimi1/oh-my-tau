//! `/todo` slash command: phase/task markdown round trips and mutations.

use super::*;

const TODO_HELP: &str = "Usage: /todo <verb> [args]
  /todo                              Show current todos
  /todo edit                         (TUI only) open in $EDITOR
  /todo copy                         Print todos as Markdown
  /todo expand                       (TUI only) expand the sticky HUD
  /todo collapse                     (TUI only) collapse the sticky HUD
  /todo export [<path>]              Write todos to file (default: TODO.md)
  /todo import [<path>]              Replace todos from file (default: TODO.md)
  /todo append [<phase>] <task...>   Append a task
  /todo start  <task>                Mark task in_progress (fuzzy match)
  /todo done   [<task|phase>]        Mark task/phase/all completed
  /todo drop   [<task|phase>]        Mark task/phase/all abandoned
  /todo rm     [<task|phase>]        Remove task/phase/all";

fn todo_marker(status: &str) -> &'static str {
    match status {
        "in_progress" => "/",
        "completed" => "x",
        "abandoned" => "-",
        "blocked" => "!",
        _ => " ",
    }
}

/// TS `phasesToMarkdown`: `# <phase>` headings, `- [<marker>] <content>`.
/// Phases are handled as raw JSON values (the concrete `TodoItem`/`Status`
/// types are crate-internal to `tools::todo`).
fn phases_to_markdown(phases: &[Value]) -> String {
    let mut out = String::new();
    if phases.is_empty() {
        return "# Todos\n".to_string();
    }
    for (i, phase) in phases.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let name = phase.get("name").and_then(Value::as_str).unwrap_or("");
        let _ = writeln!(out, "# {name}");
        for task in phase
            .get("tasks")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
        {
            let content = task.get("content").and_then(Value::as_str).unwrap_or("");
            let status = task
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending");
            let note = if status == "blocked" {
                task.get("blocker")
                    .and_then(Value::as_str)
                    .map_or(String::new(), |b| format!(" <!-- blocker: {b} -->"))
            } else {
                String::new()
            };
            let _ = writeln!(out, "- [{}] {content}{note}", todo_marker(status));
        }
    }
    out
}

/// TS `resolveTodoMarkdownPath`: explicit path or `TODO.md` under cwd.
fn todo_markdown_path(input: &str, cwd: &Path) -> PathBuf {
    let raw = input.trim();
    if raw.is_empty() {
        return cwd.join("TODO.md");
    }
    let p = PathBuf::from(raw);
    if p.is_absolute() {
        p
    } else {
        cwd.join(p)
    }
}

fn phase_name(phase: &Value) -> &str {
    phase.get("name").and_then(Value::as_str).unwrap_or("")
}

fn task_content(task: &Value) -> &str {
    task.get("content").and_then(Value::as_str).unwrap_or("")
}

fn task_status(task: &Value) -> &str {
    task.get("status")
        .and_then(Value::as_str)
        .unwrap_or("pending")
}

fn tasks_of(phase: &Value) -> &[Value] {
    phase
        .get("tasks")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
}

/// TS `findPhaseFuzzy`: exact, then unique prefix, then unique substring.
fn find_phase_fuzzy<'a>(phases: &'a [Value], query: &str) -> Option<&'a Value> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return None;
    }
    if let Some(p) = phases.iter().find(|p| phase_name(p).to_lowercase() == q) {
        return Some(p);
    }
    let prefix: Vec<_> = phases
        .iter()
        .filter(|p| phase_name(p).to_lowercase().starts_with(&q))
        .collect();
    if prefix.len() == 1 {
        return prefix.first().copied();
    }
    let sub: Vec<_> = phases
        .iter()
        .filter(|p| phase_name(p).to_lowercase().contains(&q))
        .collect();
    if sub.len() == 1 {
        return sub.first().copied();
    }
    None
}

/// TS `findTaskFuzzy`: exact, then unique substring (preferring open tasks).
fn find_task_fuzzy<'a>(phases: &'a [Value], query: &str) -> Option<&'a Value> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return None;
    }
    for phase in phases {
        for task in tasks_of(phase) {
            if task_content(task).to_lowercase() == q {
                return Some(task);
            }
        }
    }
    let hits: Vec<&Value> = phases
        .iter()
        .flat_map(|p| tasks_of(p).iter())
        .filter(|t| task_content(t).to_lowercase().contains(&q))
        .collect();
    if hits.len() == 1 {
        return hits.first().copied();
    }
    let open: Vec<_> = hits
        .iter()
        .copied()
        .filter(|t| matches!(task_status(t), "in_progress" | "pending"))
        .collect();
    if open.len() == 1 {
        return open.first().copied();
    }
    None
}

/// TS `commitTodos`: persist the new phases as a `user_todo_edit` custom entry
/// and re-arm the live todo tool.
async fn commit_todos(rt: &Runtime, phases: &[Value]) {
    let entry = json!({"customType": "user_todo_edit", "data": {"phases": phases}});
    let _ = rt.append_entry("custom", entry.clone());
    let mut full = json!({"type": "custom"});
    if let (Some(dst), Some(src)) = (full.as_object_mut(), entry.as_object()) {
        dst.extend(src.clone());
    }
    rt.rehydrate_todo_tool(&[full]).await;
}

/// Current todo phases as raw JSON (TS `currentPhases` =
/// `getLatestTodoPhasesFromEntries` over the branch).
fn current_todo_phases(rt: &Runtime) -> Vec<Value> {
    let entries = rt.entries().unwrap_or_default();
    crate::tools::phases_from_entries(&entries)
        .iter()
        .filter_map(|p| serde_json::to_value(p).ok())
        .collect()
}

fn new_task(content: &str, status: &str) -> Value {
    json!({"content": content, "status": status})
}

fn title_case_words(text: &str) -> String {
    text.split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_case_sentence(text: &str) -> String {
    let trimmed = text.trim();
    let mut c = trimmed.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => trimmed.to_string(),
    }
}

/// Minimal TS `markdownToPhases` round-trip (what `/todo export` + `/todo
/// import` produce): `# phase` headings and `- [marker] task` lines.
// invariant: phases are built in this loop with a tasks array on every phase.
#[allow(clippy::expect_used)]
fn markdown_to_phases(content: &str) -> Result<Vec<Value>, String> {
    let mut phases: Vec<Value> = Vec::new();
    for line in content.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("# ") {
            phases.push(json!({"name": rest.trim(), "tasks": []}));
            continue;
        }
        if let Some(rest) = line.strip_prefix("- [") {
            let Some((marker, content)) = rest.split_once("] ") else {
                continue;
            };
            if phases.is_empty() {
                phases.push(json!({"name": "Todos", "tasks": []}));
            }
            let status = match marker.trim() {
                "x" | "X" => "completed",
                "/" | ">" => "in_progress",
                "-" => "abandoned",
                "!" => "blocked",
                _ => "pending",
            };
            let mut body = content.trim();
            let mut blocker: Option<&str> = None;
            if let Some(pos) = body.find("<!-- blocker:") {
                let (main, comment) = body.split_at(pos);
                body = main.trim_end();
                blocker = comment
                    .strip_prefix("<!-- blocker:")
                    .and_then(|c| c.strip_suffix("-->"))
                    .map(str::trim);
            }
            let mut task = new_task(body, status);
            if let Some(b) = blocker {
                task["blocker"] = json!(b);
            }
            phases
                .last_mut()
                .and_then(|p| p.get_mut("tasks"))
                .and_then(Value::as_array_mut)
                .expect("phase tasks just ensured")
                .push(task);
        }
    }
    let task_count: usize = phases.iter().map(|p| tasks_of(p).len()).sum();
    if task_count == 0 {
        return Err("no todo entries found".to_string());
    }
    Ok(phases)
}

/// TS `normalize_in_progress`: at most one in-progress task (the earliest);
/// with none, the earliest pending starts.
fn normalize_todo(phases: &mut [Value]) {
    let mut seen = false;
    for phase in phases.iter_mut() {
        let Some(tasks) = phase.get_mut("tasks").and_then(Value::as_array_mut) else {
            continue;
        };
        for task in tasks.iter_mut() {
            let status = task
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending");
            if status == "in_progress" {
                if seen {
                    task["status"] = json!("pending");
                } else {
                    seen = true;
                }
            } else if !seen && status == "pending" {
                task["status"] = json!("in_progress");
                seen = true;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TodoVerb {
    Done,
    Drop,
    Rm,
}

/// TS `handleTodoMutationCommand` for one verb.
#[allow(clippy::too_many_lines)] // one todo verb per arm: parse the target, mutate, persist, report
async fn todo_mutate(rt: &Runtime, verb: TodoVerb, arg: &str, current: &[Value]) -> Outcome {
    let target_status = match verb {
        TodoVerb::Done => "completed",
        TodoVerb::Drop => "abandoned",
        TodoVerb::Rm => "",
    };
    if arg.trim().is_empty() {
        if verb == TodoVerb::Rm {
            commit_todos(rt, &[]).await;
            output(rt, "Cleared all todos.");
            return Handled {
                agent_invoked: false,
            };
        }
        let next: Vec<Value> = current
            .iter()
            .map(|p| {
                json!({
                    "name": phase_name(p),
                    "tasks": tasks_of(p)
                        .iter()
                        .map(|t| {
                            let mut t = t.clone();
                            t["status"] = json!(target_status);
                            t
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        commit_todos(rt, &next).await;
        output(
            rt,
            if verb == TodoVerb::Done {
                "Marked all tasks completed."
            } else {
                "Marked all tasks abandoned."
            },
        );
        return Handled {
            agent_invoked: false,
        };
    }
    if let Some(task) = find_task_fuzzy(current, arg) {
        let content = task_content(task).to_string();
        let mut next: Vec<Value> = current.to_vec();
        for phase in &mut next {
            let Some(tasks) = phase.get_mut("tasks").and_then(Value::as_array_mut) else {
                continue;
            };
            for t in tasks.iter_mut() {
                if task_content(t) == content && !target_status.is_empty() {
                    t["status"] = json!(target_status);
                }
            }
            if verb == TodoVerb::Rm {
                tasks.retain(|t| task_content(t) != content);
            }
        }
        let next: Vec<_> = next
            .into_iter()
            .filter(|p| !tasks_of(p).is_empty())
            .collect();
        commit_todos(rt, &next).await;
        let label = match verb {
            TodoVerb::Done => "Marked completed",
            TodoVerb::Drop => "Marked abandoned",
            TodoVerb::Rm => "Removed",
        };
        output(rt, &format!("{label}: {content}"));
        return Handled {
            agent_invoked: false,
        };
    }
    if let Some(phase) = find_phase_fuzzy(current, arg) {
        let name = phase_name(phase).to_string();
        let mut next: Vec<Value> = current.to_vec();
        match verb {
            TodoVerb::Rm => next.retain(|p| phase_name(p) != name),
            TodoVerb::Done | TodoVerb::Drop => {
                for p in &mut next {
                    if phase_name(p) == name {
                        if let Some(tasks) = p.get_mut("tasks").and_then(Value::as_array_mut) {
                            for t in tasks.iter_mut() {
                                t["status"] = json!(target_status);
                            }
                        }
                    }
                }
            }
        }
        let next: Vec<_> = next
            .into_iter()
            .filter(|p| !tasks_of(p).is_empty())
            .collect();
        commit_todos(rt, &next).await;
        let message = match verb {
            TodoVerb::Done => format!("Marked phase {name} completed."),
            TodoVerb::Drop => format!("Marked phase {name} abandoned."),
            TodoVerb::Rm => format!("Removed phase: {name}"),
        };
        output(rt, &message);
        return Handled {
            agent_invoked: false,
        };
    }
    usage(rt, &format!("No task or phase matched \"{}\".", arg.trim()))
}

/// TS `handleTodoAcp`.
#[allow(clippy::too_many_lines)] // the /todo verb table: one arm per subcommand, shared phase state
pub(super) async fn todo_command(rt: &Runtime, args: &str) -> Outcome {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        let phases = current_todo_phases(rt);
        output(
            rt,
            &if phases.is_empty() {
                "No todos. Use /todo append <task> to start one.".to_string()
            } else {
                phases_to_markdown(&phases).trim_end().to_string()
            },
        );
        return Handled {
            agent_invoked: false,
        };
    }
    let (verb, rest) = subcommand(trimmed);
    match verb.as_str() {
        "copy" => {
            let phases = current_todo_phases(rt);
            let markdown = if phases.is_empty() {
                String::new()
            } else {
                phases_to_markdown(&phases).trim_end().to_string()
            };
            if crate::slash_extra::dialog::has_ui() {
                if markdown.is_empty() {
                    output(rt, "No todos to copy.");
                } else {
                    crate::slash_extra::dialog::copy_to_clipboard(rt, &markdown);
                    output(rt, "Copied todos to clipboard.");
                }
                return Handled {
                    agent_invoked: false,
                };
            }
            output(
                rt,
                &format!(
                    "Copy not available in ACP mode; printing instead:\n\n{}",
                    if markdown.is_empty() {
                        "No todos."
                    } else {
                        &markdown
                    }
                ),
            );
            Handled {
                agent_invoked: false,
            }
        }
        "export" => {
            let phases = current_todo_phases(rt);
            if phases.is_empty() {
                output(rt, "No todos to export.");
                return Handled {
                    agent_invoked: false,
                };
            }
            let target = todo_markdown_path(&rest, &rt.cwd);
            if let Err(e) = std::fs::write(&target, phases_to_markdown(&phases)) {
                return usage(rt, &format!("Failed to write todos: {e}"));
            }
            output(rt, &format!("Wrote todos to {}", target.display()));
            Handled {
                agent_invoked: false,
            }
        }
        "import" => {
            let target = todo_markdown_path(&rest, &rt.cwd);
            let content = match std::fs::read_to_string(&target) {
                Ok(c) => c,
                Err(e) => return usage(rt, &format!("Failed to read todos: {e}")),
            };
            match markdown_to_phases(&content) {
                Ok(phases) => {
                    let task_count: usize = phases.iter().map(|p| tasks_of(p).len()).sum();
                    commit_todos(rt, &phases).await;
                    output(
                        rt,
                        &format!(
                            "Imported {} phase(s), {task_count} task(s) from {}.",
                            phases.len(),
                            target.display()
                        ),
                    );
                    Handled {
                        agent_invoked: false,
                    }
                }
                Err(e) => usage(rt, &format!("Could not parse {}: {e}", target.display())),
            }
        }
        "append" => {
            let tokens: Vec<&str> = rest.split_whitespace().collect();
            if tokens.is_empty() {
                return usage(rt, "Usage: /todo append [<phase>] <task...>");
            }
            let current = current_todo_phases(rt);
            let mut next = current.clone();
            let (phase_query, content) = if tokens.len() == 1 {
                (None, tokens[0].to_string())
            } else {
                (Some(tokens[0].to_string()), tokens[1..].join(" "))
            };
            let target_name = match phase_query {
                Some(q) => match find_phase_fuzzy(&next, &q) {
                    Some(p) => phase_name(p).to_string(),
                    None => title_case_words(&q),
                },
                None => next
                    .last()
                    .map_or_else(|| "Todos".to_string(), |p| phase_name(p).to_string()),
            };
            let final_content = title_case_sentence(&content);
            let existing = next
                .iter_mut()
                .find(|p| phase_name(p) == target_name)
                .and_then(|p| p.get_mut("tasks"))
                .and_then(Value::as_array_mut);
            match existing {
                Some(tasks) => tasks.push(new_task(&final_content, "pending")),
                None => next.push(json!({
                    "name": target_name,
                    "tasks": [new_task(&final_content, "pending")],
                })),
            }
            normalize_todo(&mut next);
            commit_todos(rt, &next).await;
            output(rt, &format!("Appended to {target_name}: {final_content}"));
            Handled {
                agent_invoked: false,
            }
        }
        "start" => {
            if rest.is_empty() {
                return usage(rt, "Usage: /todo start <task>");
            }
            let current = current_todo_phases(rt);
            let Some(task) = find_task_fuzzy(&current, &rest) else {
                return usage(
                    rt,
                    &format!("No task matched \"{rest}\". Use /todo to list current tasks."),
                );
            };
            let content = task_content(task).to_string();
            let mut next = current.clone();
            for phase in &mut next {
                let Some(tasks) = phase.get_mut("tasks").and_then(Value::as_array_mut) else {
                    continue;
                };
                for t in tasks.iter_mut() {
                    let status = t.get("status").and_then(Value::as_str).unwrap_or("pending");
                    if task_content(t) == content {
                        t["status"] = json!("in_progress");
                    } else if status == "in_progress" {
                        t["status"] = json!("pending");
                    }
                }
            }
            commit_todos(rt, &next).await;
            output(rt, &format!("Started: {content}"));
            Handled {
                agent_invoked: false,
            }
        }
        "done" => todo_mutate(rt, TodoVerb::Done, &rest, &current_todo_phases(rt)).await,
        "drop" => todo_mutate(rt, TodoVerb::Drop, &rest, &current_todo_phases(rt)).await,
        "rm" => todo_mutate(rt, TodoVerb::Rm, &rest, &current_todo_phases(rt)).await,
        "edit" => todo_edit(rt).await,
        "expand" | "collapse" => usage(
            rt,
            &format!("/todo {verb} controls the interactive HUD and is unavailable in this mode."),
        ),
        "help" | "?" => {
            output(rt, TODO_HELP);
            Handled {
                agent_invoked: false,
            }
        }
        _ => usage(
            rt,
            "Unknown /todo subcommand. Use append, start, done, drop, rm, copy, export, import.",
        ),
    }
}

/// `/todo edit` (TS `handleTui`): the todo list as Markdown in an editor dialog; the saved
/// text replaces the list.
async fn todo_edit(rt: &Runtime) -> Outcome {
    if !crate::slash_extra::dialog::has_ui() {
        return usage(
            rt,
            "/todo edit requires an interactive editor; use /todo export then /todo import for non-interactive edits.",
        );
    }
    let phases = current_todo_phases(rt);
    let markdown = phases_to_markdown(&phases);
    let Some(edited) =
        crate::slash_extra::dialog::editor(rt, "Edit todos (Markdown)", &markdown).await
    else {
        return Handled {
            agent_invoked: false,
        };
    };
    if edited.trim() == markdown.trim() {
        output(rt, "Todos unchanged.");
        return Handled {
            agent_invoked: false,
        };
    }
    match markdown_to_phases(&edited) {
        Ok(next) => {
            let task_count: usize = next.iter().map(|p| tasks_of(p).len()).sum();
            commit_todos(rt, &next).await;
            output(
                rt,
                &format!(
                    "Updated todos: {} phase(s), {task_count} task(s).",
                    next.len()
                ),
            );
            Handled {
                agent_invoked: false,
            }
        }
        Err(e) => usage(rt, &format!("Could not parse the edited todos: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn todo_markdown_round_trips() {
        use crate::tools::{Status, TodoItem, TodoPhase};
        let phases = vec![
            TodoPhase {
                name: "Setup".into(),
                tasks: vec![
                    TodoItem {
                        content: "Install deps".into(),
                        status: Status::Completed,
                        blocker: None,
                    },
                    TodoItem {
                        content: "Write tests".into(),
                        status: Status::InProgress,
                        blocker: None,
                    },
                ],
            },
            TodoPhase {
                name: "Ship".into(),
                tasks: vec![TodoItem {
                    content: "Blocked task".into(),
                    status: Status::Blocked,
                    blocker: Some("waiting on review".into()),
                }],
            },
        ];
        let values: Vec<Value> = serde_json::to_value(&phases)
            .expect("serialize phases")
            .as_array()
            .expect("phase array")
            .clone();
        let markdown = phases_to_markdown(&values);
        assert!(markdown.contains("# Setup"));
        assert!(markdown.contains("- [x] Install deps"));
        assert!(markdown.contains("- [/] Write tests"));
        assert!(markdown.contains("<!-- blocker: waiting on review -->"));
        let parsed = markdown_to_phases(&markdown).expect("round-trip");
        assert_eq!(parsed, values);
    }
}
