//! Recursive track tree (`buildTrackTree` in trace.ts): build the track for
//! one transcript and recurse into its artifacts directory, appending tracks
//! in DFS order and placing subagent spans on parents.

use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::Path;

use crate::types::{TraceSpan, TraceSpanKind, TraceTrack};

use super::scan::{scan_transcript, TrackScan};
use super::{
    head_text, parse_entry_values, read_dir_names, read_transcript, transcript_stem, DETAIL_MAX,
    LABEL_MAX, MAX_TRACK_DEPTH,
};

/// Base name of a transcript stem (`path.basename` in TS).
fn basename(path_str: &str) -> String {
    path_str.rsplit('/').next().unwrap_or(path_str).to_string()
}

/// Hardening beyond TS (m6): a child transcript that symlinks outside the
/// sessions root is skipped rather than read.
fn within_sessions_root(path: &Path) -> bool {
    let Ok(target) = std::fs::canonicalize(path) else {
        return true; // Lexical containment was already checked; TS reads it.
    };
    let roots = crate::paths::session_roots();
    let canon: Vec<_> = roots
        .iter()
        .filter_map(|r| std::fs::canonicalize(r).ok())
        .collect();
    // No root exists on disk: nothing to compare against (TS reads it).
    canon.is_empty() || canon.iter().any(|root| target.starts_with(root))
}

/// Insertion-ordered children map: TS `Map` preserves insertion order, which
/// is the tie-break for equal start/end subagent spans.
struct OrderedChildren {
    index: HashMap<String, usize>,
    order: Vec<(String, usize)>,
}

impl OrderedChildren {
    fn insert(&mut self, key: String, value: usize) {
        self.index.insert(key.clone(), value);
        self.order.push((key, value));
    }

    fn get(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    fn iter(&self) -> impl Iterator<Item = &(String, usize)> {
        self.order.iter()
    }
}

/**
 * Build the track for one transcript and recurse into its artifacts directory.
 * Appends tracks to `out` and scans to `scans` in TS DFS order (root pushed
 * before recursion), and places subagent spans on parents. Returns the scan's
 * index in `scans`.
 */
pub(crate) fn build_track_tree(
    file: &str,
    track_id: &str,
    parent_track_id: Option<&str>,
    depth: usize,
    visited: &mut HashSet<String>,
    out: &mut Vec<TraceTrack>,
    scans: &mut Vec<TrackScan>,
) -> anyhow::Result<Option<usize>> {
    if !visited.insert(file.to_string()) {
        return Ok(None);
    }

    let bytes = match read_transcript(Path::new(file)) {
        Ok(bytes) => bytes,
        Err(err)
            if err
                .downcast_ref::<std::io::Error>()
                .map(|e| e.kind() == ErrorKind::NotFound)
                .unwrap_or(false) =>
        {
            return Ok(None)
        }
        Err(err) => return Err(err),
    };

    // TS pushes each scan before recursing, so the order is root, C, G — the
    // order in which tool_stats ties and costTotal summation break.
    let mut scan = scan_transcript(parse_entry_values(&bytes), track_id, parent_track_id, file);
    let track_index = out.len();
    out.push(scan.track.clone());
    let scan_idx = scans.len();
    scans.push(scan.clone());

    // Recurse into the artifacts directory (transcript stem = directory name).
    let mut children_by_id = OrderedChildren {
        index: HashMap::new(),
        order: Vec::new(),
    };
    if depth < MAX_TRACK_DEPTH {
        let artifacts_dir = transcript_stem(file);
        if let Ok(mut names) = read_dir_names(Path::new(&artifacts_dir)) {
            names.sort();
            for name in names {
                // `.jsonl.*.bak` backups fail both suffix checks and are skipped.
                if !name.ends_with(".jsonl") && !name.ends_with(".jsonl.gz") {
                    continue;
                }
                let child_file = format!("{artifacts_dir}/{name}");
                if !within_sessions_root(Path::new(&child_file)) {
                    continue;
                }
                // The child is pushed first, at exactly this index — even when
                // it has its own descendants (M1).
                let child_track_idx = out.len();
                let stem = basename(&transcript_stem(&child_file));
                let child_track_id = if track_id == "main" {
                    stem.clone()
                } else {
                    format!("{track_id}/{stem}")
                };
                let Some(child_scan_idx) = build_track_tree(
                    &child_file,
                    &child_track_id,
                    Some(track_id),
                    depth + 1,
                    visited,
                    out,
                    scans,
                )?
                else {
                    continue;
                };
                let is_advisor = name == "__advisor.jsonl"
                    || (name.starts_with("__advisor.") && name.ends_with(".jsonl"));
                let agent = out[child_track_idx].agent.clone();
                let label = if is_advisor {
                    "Advisor".to_string()
                } else {
                    head_text(
                        &format!("{} {stem}", agent.as_deref().unwrap_or("agent")),
                        LABEL_MAX,
                    )
                };
                out[child_track_idx].label = label.clone();
                // TS shares one object: the label must reach the scans copy too.
                scans[child_scan_idx].track.label = label;
                children_by_id.insert(stem, child_scan_idx);
            }
        }
    }

    // Subagent spans on this track, one per task SingleResult.
    let mut linked_children: HashSet<String> = HashSet::new();
    for result in &scan.task_results {
        let child = children_by_id.get(&result.child_id);
        if child.is_some() {
            linked_children.insert(result.child_id.clone());
        }
        let (start, end) =
            match child.and_then(|i| scans[i].first_activity.zip(scans[i].last_activity)) {
                Some(bounds) => bounds,
                None => match result.duration_ms {
                    Some(d) => (result.tool_span_end.saturating_sub(d), result.tool_span_end),
                    None => continue,
                },
            };
        let mut span = TraceSpan {
            id: format!(
                "{track_id}:{}:{}",
                result.result_entry_id.as_deref().unwrap_or(""),
                result.child_id
            ),
            kind: TraceSpanKind::Subagent,
            start,
            end: end.max(start),
            label: head_text(
                result.agent.as_deref().unwrap_or(&result.child_id),
                LABEL_MAX,
            ),
            ..TraceSpan::default()
        };
        // TS `if (result.task)` — an empty task string is skipped.
        span.detail = result
            .task
            .as_ref()
            .filter(|t| !t.is_empty())
            .map(|t| head_text(t, DETAIL_MAX));
        span.entry_id = result.result_entry_id.clone();
        if let Some(i) = child {
            span.child_track_id = Some(scans[i].track.id.clone());
            // TS `if (child.track.model)` — an empty model string is skipped.
            span.model = scans[i].track.model.clone().filter(|m| !m.is_empty());
        }
        scan.track.spans.push(span);
    }
    // Children with no structured task result — async task jobs (results are
    // delivered as formatted text), crashed parents, advisors — still get a
    // parent span from their own activity bounds so the Agents lane shows them.
    for (stem, i) in children_by_id.iter() {
        let i = *i;
        if linked_children.contains(stem) {
            continue;
        }
        let (Some(first), Some(last)) = (scans[i].first_activity, scans[i].last_activity) else {
            continue;
        };
        let mut span = TraceSpan {
            id: format!("{track_id}:sub:{stem}"),
            kind: TraceSpanKind::Subagent,
            start: first,
            end: last.max(first),
            label: head_text(
                scans[i]
                    .track
                    .agent
                    .as_deref()
                    .unwrap_or(&scans[i].track.label),
                LABEL_MAX,
            ),
            child_track_id: Some(scans[i].track.id.clone()),
            ..TraceSpan::default()
        };
        span.model = scans[i].track.model.clone().filter(|m| !m.is_empty());
        scan.track.spans.push(span);
    }
    scan.track
        .spans
        .sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

    // Mirror the finished spans/markers onto the pushed track and scan.
    out[track_index].spans = scan.track.spans.clone();
    out[track_index].markers = scan.track.markers.clone();
    scans[scan_idx] = scan;

    Ok(Some(scan_idx))
}
