//! Workspace-edit application (omp `lsp/edits.ts`): bottom-up text edits with
//! overlap/snippet validation, resource operations, and rename-with-rollback.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::types::{DocumentChange, Range, TextEdit, WorkspaceEdit};
use crate::utils::{format_path_relative_to_cwd, uri_to_file};

/// Error type mirroring omp's `ToolError` for edit validation failures.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EditError(pub String);

fn compare_position(a: &crate::types::Position, b: &crate::types::Position) -> std::cmp::Ordering {
    a.line
        .cmp(&b.line)
        .then_with(|| a.character.cmp(&b.character))
}

fn positions_equal(a: &crate::types::Position, b: &crate::types::Position) -> bool {
    a.line == b.line && a.character == b.character
}

fn ranges_equal(a: &Range, b: &Range) -> bool {
    positions_equal(&a.start, &b.start) && positions_equal(&a.end, &b.end)
}

fn is_empty_range(range: &Range) -> bool {
    positions_equal(&range.start, &range.end)
}

fn format_range(range: &Range) -> String {
    format!(
        "{}:{}-{}:{}",
        range.start.line + 1,
        range.start.character + 1,
        range.end.line + 1,
        range.end.character + 1
    )
}

/// True when two ranges overlap (share any position other than a touching boundary).
#[must_use]
pub fn ranges_overlap(a: &Range, b: &Range) -> bool {
    compare_position(&a.start, &b.end) == std::cmp::Ordering::Less
        && compare_position(&b.start, &a.end) == std::cmp::Ordering::Less
}

/// Sort edits bottom-to-top for in-place application and reject overlaps.
///
/// # Errors
///
/// `EditError` on snippet-formatted edits or overlapping ranges.
pub fn sort_and_validate_text_edits(edits: &mut Vec<TextEdit>) -> Result<Vec<TextEdit>, EditError> {
    for edit in &*edits {
        if edit.insert_text_format == Some(2) {
            return Err(EditError(
                "snippet-formatted LSP edits are unsupported".to_string(),
            ));
        }
    }
    let mut indexed: Vec<(usize, TextEdit)> = edits.drain(..).enumerate().collect();
    indexed.sort_by(|(ai, a), (bi, b)| {
        b.range
            .start
            .line
            .cmp(&a.range.start.line)
            .then_with(|| b.range.start.character.cmp(&a.range.start.character))
            .then_with(|| bi.cmp(ai))
    });
    // Collapse byte-identical duplicate non-empty edits.
    let mut unique: Vec<TextEdit> = Vec::new();
    for (_, edit) in indexed {
        if let Some(prev) = unique.last() {
            if !is_empty_range(&edit.range)
                && ranges_equal(&prev.range, &edit.range)
                && prev.new_text == edit.new_text
            {
                continue;
            }
        }
        unique.push(edit);
    }
    for i in 0..unique.len().saturating_sub(1) {
        let later = &unique[i].range;
        let earlier = &unique[i + 1].range;
        if compare_position(&earlier.end, &later.start) == std::cmp::Ordering::Greater {
            return Err(EditError(format!(
                "overlapping LSP edits: {} conflicts with {}; LSP produced inconsistent edits",
                format_range(earlier),
                format_range(later)
            )));
        }
    }
    Ok(unique)
}

/// Apply text edits to a string, bottom-up (omp `applyTextEditsToString`).
///
/// # Errors
///
/// Propagates validation errors from [`sort_and_validate_text_edits`].
pub fn apply_text_edits_to_string(content: &str, edits: &[TextEdit]) -> Result<String, EditError> {
    let mut lines: Vec<String> = content.split('\n').map(ToString::to_string).collect();
    let sorted = sort_and_validate_text_edits(&mut edits.to_vec())?;
    for edit in &sorted {
        let start = &edit.range.start;
        let end = &edit.range.end;
        let start_line = start.line as usize;
        let end_line = end.line as usize;
        if start_line == end_line {
            let line = lines.get(start_line).cloned().unwrap_or_default();
            let (a, b) = (start.character as usize, end.character as usize);
            lines[start_line] = format!(
                "{}{}{}",
                &line[..line.len().min(a)],
                edit.new_text,
                &line[b.min(line.len())..]
            );
        } else {
            let start_line_text = lines.get(start_line).cloned().unwrap_or_default();
            let end_line_text = lines.get(end_line).cloned().unwrap_or_default();
            let (a, b) = (start.character as usize, end.character as usize);
            let new_content = format!(
                "{}{}{}",
                &start_line_text[..start_line_text.len().min(a)],
                edit.new_text,
                &end_line_text[b.min(end_line_text.len())..]
            );
            let splice: Vec<String> = new_content.split('\n').map(ToString::to_string).collect();
            let removed = end_line - start_line + 1;
            lines.splice(start_line..start_line + removed, splice);
        }
    }
    Ok(lines.join("\n"))
}

/// Apply text edits to a file on disk.
///
/// # Errors
///
/// IO errors and validation errors.
pub async fn apply_text_edits(file_path: &Path, edits: &[TextEdit]) -> Result<(), EditError> {
    let content = tokio::fs::read_to_string(file_path)
        .await
        .map_err(|e| EditError(format!("failed to read {}: {e}", file_path.display())))?;
    let result = apply_text_edits_to_string(&content, edits)?;
    tokio::fs::write(file_path, result)
        .await
        .map_err(|e| EditError(format!("failed to write {}: {e}", file_path.display())))
}

/// Flatten a `WorkspaceEdit`'s text edits into a uri → edits map (omp
/// `flattenWorkspaceTextEdits`). Resource operations are ignored.
#[must_use]
pub fn flatten_workspace_text_edits(edit: &WorkspaceEdit) -> Vec<(String, Vec<TextEdit>)> {
    let mut out: Vec<(String, Vec<TextEdit>)> = Vec::new();
    let mut push = |uri: &str, edits: Vec<TextEdit>| {
        if edits.is_empty() {
            return;
        }
        if let Some(entry) = out.iter_mut().find(|(u, _)| u == uri) {
            entry.1.extend(edits);
        } else {
            out.push((uri.to_string(), edits));
        }
    };
    if let Some(changes) = &edit.changes {
        for (uri, edits_value) in changes {
            if let Ok(edits) = serde_json::from_value::<Vec<TextEdit>>(edits_value.clone()) {
                push(uri, edits);
            }
        }
    }
    if let Some(document_changes) = &edit.document_changes {
        for change in document_changes {
            if let DocumentChange::TextDocumentEdit(tdc) = change {
                push(&tdc.text_document.uri, tdc.edits.clone());
            }
        }
    }
    out
}

/// A reference file and the text edits a rename computed for it.
pub struct RenameReferenceEdit {
    pub file_path: PathBuf,
    pub edits: Vec<TextEdit>,
}

/// Apply reference edits and then move `source` → `dest` as one unit; a failed
/// move rolls every edited file back to its snapshot (omp `applyEditsThenRename`).
///
/// # Errors
///
/// IO/validation errors, with rollback performed on move failure.
pub async fn apply_edits_then_rename(
    references: &[RenameReferenceEdit],
    source: &Path,
    dest: &Path,
) -> Result<(), EditError> {
    let mut backups: Vec<(PathBuf, String)> = Vec::new();
    for reference in references {
        let original = tokio::fs::read_to_string(&reference.file_path)
            .await
            .map_err(|e| {
                EditError(format!(
                    "failed to read {}: {e}",
                    reference.file_path.display()
                ))
            })?;
        backups.push((reference.file_path.clone(), original));
        apply_text_edits(&reference.file_path, &reference.edits).await?;
    }
    if let Some(parent) = dest.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            rollback(&backups).await;
            return Err(EditError(format!(
                "failed to create {}: {e}",
                parent.display()
            )));
        }
    }
    if let Err(e) = tokio::fs::rename(source, dest).await {
        rollback(&backups).await;
        return Err(EditError(format!(
            "failed to rename {} to {}: {e}",
            source.display(),
            dest.display()
        )));
    }
    Ok(())
}

async fn rollback(backups: &[(PathBuf, String)]) {
    for (path, original) in backups {
        let _ = tokio::fs::write(path, original).await;
    }
}

#[derive(Debug)]
enum WorkspaceEditOp {
    Text {
        uri: String,
        edits: Vec<TextEdit>,
    },
    Create {
        uri: String,
        options: Option<Value>,
    },
    Rename {
        old_uri: String,
        new_uri: String,
        options: Option<Value>,
    },
    Delete {
        uri: String,
        options: Option<Value>,
    },
}

fn opt_bool(options: Option<&Value>, key: &str) -> bool {
    options
        .and_then(|o| o.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn uri_related(op_uri: &str, uri: &str) -> bool {
    op_uri == uri || op_uri.starts_with(&format!("{uri}/"))
}

/// Plan documentChanges into an ordered op list (simplified `planDocumentChanges`):
/// text edits accumulate per URI and flush before any resource op touching the
/// same URI (or a descendant), so resource ops see correct prior state.
fn plan_document_changes(document_changes: &[DocumentChange]) -> Vec<WorkspaceEditOp> {
    let mut ops: Vec<WorkspaceEditOp> = Vec::new();
    let mut pending_text: Vec<(String, Vec<TextEdit>)> = Vec::new();
    let flush_uri =
        |ops: &mut Vec<WorkspaceEditOp>, pending: &mut Vec<(String, Vec<TextEdit>)>, uri: &str| {
            let mut i = 0;
            while i < pending.len() {
                if uri_related(uri, &pending[i].0) {
                    let (u, edits) = pending.remove(i);
                    ops.push(WorkspaceEditOp::Text { uri: u, edits });
                } else {
                    i += 1;
                }
            }
        };
    for change in document_changes {
        match change {
            DocumentChange::TextDocumentEdit(tdc) => {
                if let Some(entry) = pending_text
                    .iter_mut()
                    .find(|(u, _)| *u == tdc.text_document.uri)
                {
                    entry.1.extend(tdc.edits.iter().cloned());
                } else {
                    pending_text.push((tdc.text_document.uri.clone(), tdc.edits.clone()));
                }
            }
            DocumentChange::CreateFile(c) => {
                flush_uri(&mut ops, &mut pending_text, &c.uri);
                ops.push(WorkspaceEditOp::Create {
                    uri: c.uri.clone(),
                    options: c.options.clone(),
                });
            }
            DocumentChange::RenameFile(r) => {
                flush_uri(&mut ops, &mut pending_text, &r.old_uri);
                flush_uri(&mut ops, &mut pending_text, &r.new_uri);
                ops.push(WorkspaceEditOp::Rename {
                    old_uri: r.old_uri.clone(),
                    new_uri: r.new_uri.clone(),
                    options: r.options.clone(),
                });
            }
            DocumentChange::DeleteFile(d) => {
                flush_uri(&mut ops, &mut pending_text, &d.uri);
                ops.push(WorkspaceEditOp::Delete {
                    uri: d.uri.clone(),
                    options: d.options.clone(),
                });
            }
        }
    }
    for (uri, edits) in pending_text {
        ops.push(WorkspaceEditOp::Text { uri, edits });
    }
    ops
}

/// Apply a workspace edit to disk (omp `applyWorkspaceEdit`). Text edits are
/// validated for every file before any file is written.
///
/// # Errors
///
/// `EditError` on validation or IO failures; earlier writes are not rolled back.
pub async fn apply_workspace_edit(
    edit: &WorkspaceEdit,
    cwd: &Path,
) -> Result<Vec<String>, EditError> {
    let mut applied: Vec<String> = Vec::new();
    if let Some(document_changes) = &edit.document_changes {
        let ops = plan_document_changes(document_changes);
        for op in &ops {
            if let WorkspaceEditOp::Text { edits, .. } = op {
                let mut edits = edits.clone();
                sort_and_validate_text_edits(&mut edits)?;
            }
        }
        for op in ops {
            match op {
                WorkspaceEditOp::Text { uri, edits } => {
                    let file_path = uri_to_file(&uri);
                    apply_text_edits(&file_path, &edits).await?;
                    applied.push(format!(
                        "Applied {} edit(s) to {}",
                        edits.len(),
                        format_path_relative_to_cwd(&file_path, cwd)
                    ));
                }
                WorkspaceEditOp::Create { uri, options } => {
                    match create_file(&uri, options.as_ref()).await {
                        Ok(created) => {
                            if created {
                                applied.push(format!(
                                    "Created {}",
                                    format_path_relative_to_cwd(&uri_to_file(&uri), cwd)
                                ));
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                WorkspaceEditOp::Rename {
                    old_uri,
                    new_uri,
                    options,
                } => {
                    rename_file(&old_uri, &new_uri, options.as_ref(), cwd).await?;
                    applied.push(format!(
                        "Renamed {} {} {}",
                        format_path_relative_to_cwd(&uri_to_file(&old_uri), cwd),
                        crate::utils::NAV_CURSOR,
                        format_path_relative_to_cwd(&uri_to_file(&new_uri), cwd)
                    ));
                }
                WorkspaceEditOp::Delete { uri, options } => {
                    delete_file(&uri, options.as_ref()).await?;
                    applied.push(format!(
                        "Deleted {}",
                        format_path_relative_to_cwd(&uri_to_file(&uri), cwd)
                    ));
                }
            }
        }
    } else if let Some(changes) = &edit.changes {
        // Legacy changes-map path: validate every file's edits before writing any.
        for edits_value in changes.values() {
            let mut edits: Vec<TextEdit> = match serde_json::from_value(edits_value.clone()) {
                Ok(e) => e,
                Err(e) => return Err(EditError(format!("invalid text edits: {e}"))),
            };
            sort_and_validate_text_edits(&mut edits)?;
        }
        for (uri, edits_value) in changes {
            let text_edits: Vec<TextEdit> = match serde_json::from_value(edits_value.clone()) {
                Ok(e) => e,
                Err(e) => return Err(EditError(format!("invalid text edits: {e}"))),
            };
            if text_edits.is_empty() {
                continue;
            }
            let file_path = uri_to_file(uri);
            apply_text_edits(&file_path, &text_edits).await?;
            applied.push(format!(
                "Applied {} edit(s) to {}",
                text_edits.len(),
                format_path_relative_to_cwd(&file_path, cwd)
            ));
        }
    }
    Ok(applied)
}

async fn create_file(uri: &str, options: Option<&Value>) -> Result<bool, EditError> {
    let file_path = uri_to_file(uri);
    if let Some(parent) = file_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| EditError(format!("failed to create {}: {e}", parent.display())))?;
    }
    let overwrite = opt_bool(options, "overwrite");
    if overwrite {
        tokio::fs::write(&file_path, b"")
            .await
            .map_err(|e| EditError(format!("failed to create {}: {e}", file_path.display())))?;
        return Ok(true);
    }
    match tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&file_path)
        .await
    {
        Ok(_) => Ok(true),
        Err(e)
            if e.kind() == std::io::ErrorKind::AlreadyExists
                && opt_bool(options, "ignoreIfExists") =>
        {
            Ok(false)
        }
        Err(e) => Err(EditError(format!(
            "failed to create {}: {e}",
            file_path.display()
        ))),
    }
}

async fn rename_file(
    old_uri: &str,
    new_uri: &str,
    options: Option<&Value>,
    cwd: &Path,
) -> Result<(), EditError> {
    let old_path = uri_to_file(old_uri);
    let new_path = uri_to_file(new_uri);
    if old_path == new_path {
        return Ok(());
    }
    if let Some(parent) = new_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| EditError(format!("failed to create {}: {e}", parent.display())))?;
    }
    if new_path.symlink_metadata().is_ok() {
        if !opt_bool(options, "overwrite") {
            if opt_bool(options, "ignoreIfExists") {
                return Ok(());
            }
            return Err(EditError(format!(
                "rename target already exists: {}",
                format_path_relative_to_cwd(&new_path, cwd)
            )));
        }
        tokio::fs::remove_file(&new_path)
            .await
            .map_err(|e| EditError(format!("failed to overwrite {}: {e}", new_path.display())))?;
    }
    tokio::fs::rename(&old_path, &new_path).await.map_err(|e| {
        EditError(format!(
            "failed to rename {} to {}: {e}",
            old_path.display(),
            new_path.display()
        ))
    })
}

async fn delete_file(uri: &str, options: Option<&Value>) -> Result<(), EditError> {
    let file_path = uri_to_file(uri);
    let recursive = opt_bool(options, "recursive");
    match tokio::fs::symlink_metadata(&file_path).await {
        Ok(md) if md.is_dir() && !md.file_type().is_symlink() && !recursive => {
            tokio::fs::remove_dir(&file_path)
                .await
                .map_err(|e| EditError(format!("failed to delete {}: {e}", file_path.display())))
        }
        Ok(_) => tokio::fs::remove_dir_all(&file_path)
            .await
            .map_err(|e| EditError(format!("failed to delete {}: {e}", file_path.display()))),
        Err(e)
            if e.kind() == std::io::ErrorKind::NotFound
                && opt_bool(options, "ignoreIfNotExists") =>
        {
            Ok(())
        }
        Err(e) => Err(EditError(format!(
            "failed to delete {}: {e}",
            file_path.display()
        ))),
    }
}

/// Parse a loose `WorkspaceEdit` value and apply it (used for server-initiated
/// `workspace/applyEdit` requests).
///
/// # Errors
///
/// Propagates parse and application errors.
pub async fn apply_workspace_edit_value(edit: &Value, cwd: &Path) -> anyhow::Result<Vec<String>> {
    let parsed: WorkspaceEdit = serde_json::from_value(edit.clone())?;
    Ok(apply_workspace_edit(&parsed, cwd).await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Position, Range, TextEdit};

    fn pos(line: u32, character: u32) -> Position {
        Position { line, character }
    }

    fn edit(start: (u32, u32), end: (u32, u32), new_text: &str) -> TextEdit {
        TextEdit {
            range: Range {
                start: pos(start.0, start.1),
                end: pos(end.0, end.1),
            },
            new_text: new_text.to_string(),
            insert_text_format: None,
        }
    }

    #[test]
    fn applies_single_line_edit_bottom_up() {
        let result =
            apply_text_edits_to_string("hello world\n", &[edit((0, 6), (0, 11), "there")]).unwrap();
        assert_eq!(result, "hello there\n");
    }

    #[test]
    fn applies_multi_line_splice() {
        let result = apply_text_edits_to_string("a\nb\nc\n", &[edit((0, 0), (1, 1), "X")]).unwrap();
        assert_eq!(result, "X\nc\n");
    }

    #[test]
    fn sorts_edits_so_later_positions_apply_first() {
        let edits = vec![edit((0, 0), (0, 1), "A"), edit((1, 0), (1, 1), "B")];
        let result = apply_text_edits_to_string("x\ny\n", &edits).unwrap();
        assert_eq!(result, "A\nB\n");
    }

    #[test]
    fn rejects_overlapping_edits() {
        let edits = vec![edit((0, 0), (0, 3), "A"), edit((0, 1), (0, 4), "B")];
        assert!(apply_text_edits_to_string("abcdef", &edits).is_err());
    }

    #[test]
    fn rejects_snippet_edits() {
        let mut e = edit((0, 0), (0, 0), "x");
        e.insert_text_format = Some(2);
        let mut edits = vec![e];
        assert!(sort_and_validate_text_edits(&mut edits).is_err());
    }

    #[test]
    fn collapses_identical_duplicate_edits() {
        let edits = vec![edit((0, 0), (0, 1), "A"), edit((0, 0), (0, 1), "A")];
        let result = apply_text_edits_to_string("x", &edits).unwrap();
        assert_eq!(result, "A");
    }

    #[test]
    fn detects_touching_ranges_as_non_overlapping() {
        let edits = vec![edit((0, 0), (0, 1), "A"), edit((0, 1), (0, 2), "B")];
        assert!(apply_text_edits_to_string("xy", &edits).is_ok());
    }
}
