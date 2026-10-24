//! `eval_journal`: per-session, append-only journal of eval setup cells so a
//! kernel lost to a kill/reset or a fresh process resuming the session can
//! restore its setup with `%replay` (omp `eval/state.ts` generation tracking +
//! `prompts/system/eval-state-context.md` kernel-loss wording).
//!
//! The journal lives at `eval-journal.jsonl` inside the session artifacts dir
//! (`omp_urls::artifacts_directory_for` of the global session file). Without
//! a session file (tests, `--no-session`) it falls back to an in-memory map;
//! nothing persists, which matches the TS host-retained-only registry.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use omp_eval::CellStatus;

/// Kernel language the journal distinguishes (Py/Js kernels are separate
/// runtimes with separate generations, TS `EvalStateLanguage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lang {
    Py,
    Js,
}

/// One append-only JSONL record in `eval-journal.jsonl`.
#[derive(Debug, Serialize, Deserialize)]
struct Event {
    lang: Lang,
    cwd: String,
    #[serde(flatten)]
    kind: EventKind,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum EventKind {
    /// A fresh kernel process started for `(cwd, lang)` — a new generation.
    KernelStarted,
    /// A successful setup-classified cell; `code` is replayed by `%replay`.
    Cell { code: String },
}

/// Rebuilt journal state for one `(session, cwd, lang)` key.
#[derive(Default, Debug)]
struct Journal {
    /// Number of `KernelStarted` events seen — i.e. the current generation.
    kernels: u32,
    /// Setup cells recorded this session, in execution order.
    setup: Vec<String>,
}

impl Journal {
    fn apply(&mut self, kind: &EventKind) {
        match kind {
            EventKind::KernelStarted => self.kernels += 1,
            EventKind::Cell { code } => self.setup.push(code.clone()),
        }
    }

    /// TS generation logic (`eval/state.ts` `updateEvalState`: a new identity
    /// in the same lineage starts at `predecessor.generation + 1`): the first
    /// kernel is generation 1 and is not a loss; every later fresh kernel
    /// (kill, reset, or a new process resuming the journaled session) means
    /// the previous generation's state is gone.
    fn on_kernel_started(&mut self) -> Option<String> {
        self.kernels += 1;
        if self.kernels <= 1 {
            return None;
        }
        Some(kernel_loss_notice(self.kernels, self.setup.len()))
    }
}

/// Kernel-loss notice, TS wording from `prompts/system/eval-state-context.md`
/// ("Eval variables, imports, and kernels referenced in earlier session
/// history were not restored; re-run required setup before using them."),
/// plus how many setup cells `%replay` would re-run.
fn kernel_loss_notice(generation: u32, setup_cells: usize) -> String {
    let cells = if setup_cells == 1 { "cell" } else { "cells" };
    format!(
        "Kernel state was lost (fresh kernel, generation {generation}). \
         Eval variables, imports, and kernels referenced in earlier session \
         history were not restored; re-run required setup before using them. \
         `%replay` would re-run {setup_cells} recorded setup {cells}."
    )
}

/// Record one executed cell. Only successful setup-classified cells are
/// journaled — everything else is neither replayable nor counts toward the
/// loss notice (conservative: replaying side-effecting cells would be wrong).
pub fn record_cell(cwd: &Path, lang: Lang, code: &str, result: &omp_eval::CellResult) {
    if result.status != CellStatus::Ok {
        return;
    }
    let setup = match lang {
        Lang::Py => is_setup_py(code),
        Lang::Js => is_setup_js(code),
    };
    if !setup {
        return;
    }
    with_event(cwd, lang, true, |journal| {
        journal.setup.push(code.to_string());
        (
            (),
            Some(EventKind::Cell {
                code: code.to_string(),
            }),
        )
    });
}

/// A fresh kernel for `(cwd, lang)` started. `None` for the first kernel of
/// a session (nothing lost); on any later fresh kernel returns the kernel-loss
/// notice with the number of setup cells `%replay` would re-run.
pub fn kernel_started(cwd: &Path, lang: Lang) -> Option<String> {
    with_event(cwd, lang, true, |journal| {
        let notice = journal.on_kernel_started();
        (notice, Some(EventKind::KernelStarted))
    })
}

/// Setup cells journaled for `(cwd, lang)`, in execution order — the codes
/// `%replay` re-runs on a fresh kernel.
pub fn replay_setup(cwd: &Path, lang: Lang) -> Vec<String> {
    with_event(cwd, lang, false, |journal| (journal.setup.clone(), None))
}

// ---------------------------------------------------------------------------
// Storage: append-only JSONL in the session artifacts dir, in-memory fallback.
// ---------------------------------------------------------------------------

/// Journal key: which session file (None = in-memory only), cwd, language.
#[derive(Hash, PartialEq, Eq)]
struct Key {
    session: Option<PathBuf>,
    cwd: PathBuf,
    lang: Lang,
}

fn journal_path() -> Option<PathBuf> {
    let session = omp_urls::global_session_file()?;
    Some(omp_urls::artifacts_directory_for(&session)?.join("eval-journal.jsonl"))
}

/// Resume writes a NEW session file seeded from the old one: carry the resumed
/// session's journal over, so the kernel-loss notice and `%replay` see the
/// previous process's kernel generations and setup cells.
pub fn carry_over(from_session: &Path) {
    let (Some(src_dir), Some(dst)) = (
        omp_urls::artifacts_directory_for(from_session),
        journal_path(),
    ) else {
        return;
    };
    let src = src_dir.join("eval-journal.jsonl");
    if src == dst {
        return;
    }
    let Ok(data) = std::fs::read(&src) else {
        return;
    };
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dst)
        .and_then(|mut f| f.write_all(&data));
}

/// In-memory fallback for runs without a session file.
static MEM: LazyLock<Mutex<HashMap<Key, Journal>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn load_events(path: &Path) -> Vec<Event> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str::<Event>(&line).ok())
        .collect()
}

fn append_event(path: &Path, event: &Event) {
    // The artifacts dir is created lazily by whoever writes first; the very
    // first kernel of a session usually precedes any artifact.
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    if let Ok(line) = serde_json::to_string(event) {
        let _ = writeln!(file, "{line}");
    }
}

/// Run `op` against the journal for `(cwd, lang)`, persisting the event `op`
/// yields when `persist` and a journal file exists (in-memory map otherwise).
fn with_event<T>(
    cwd: &Path,
    lang: Lang,
    persist: bool,
    op: impl FnOnce(&mut Journal) -> (T, Option<EventKind>),
) -> T {
    let cwd = cwd.to_path_buf();
    let session = omp_urls::global_session_file();
    let key = Key {
        session: session.clone(),
        cwd: cwd.clone(),
        lang,
    };
    let path = journal_path();

    let mut journal = Journal::default();
    if let Some(path) = &path {
        for event in load_events(path) {
            if event.lang == lang && event.cwd == cwd.to_string_lossy() {
                journal.apply(&event.kind);
            }
        }
    } else {
        let mem = MEM.lock().expect("eval journal lock");
        if let Some(saved) = mem.get(&key) {
            journal = Journal {
                kernels: saved.kernels,
                setup: saved.setup.clone(),
            };
        }
    }

    let (out, event) = op(&mut journal);

    if persist {
        if let (Some(path), Some(kind)) = (&path, event) {
            append_event(
                path,
                &Event {
                    lang,
                    cwd: cwd.to_string_lossy().into_owned(),
                    kind,
                },
            );
        } else if path.is_none() {
            MEM.lock().expect("eval journal lock").insert(key, journal);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Setup classifier — deliberately conservative: a cell is setup only when
// every top-level statement is a definition, an import, or a constant/loading
// assignment. Anything executable (calls with side effects other than loads,
// control flow, prints) disqualifies the whole cell.
// ---------------------------------------------------------------------------

/// Split Python source into top-level statements: newline-separated at
/// bracket depth 0, honoring `#` comments and string literals (incl. `"""`).
/// Returns `None` when the source looks unbalanced (unterminated string or
/// bracket) — conservatively not setup.
fn py_statements(code: &str) -> Option<Vec<String>> {
    #[derive(PartialEq)]
    enum Str {
        None,
        Single(char),
        Triple(char),
    }
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut depth: i32 = 0;
    let mut str = Str::None;
    let mut chars = code.chars().peekable();
    while let Some(c) = chars.next() {
        match str {
            Str::Single(q) => {
                cur.push(c);
                if c == '\\' {
                    cur.push(chars.next()?);
                } else if c == q {
                    str = Str::None;
                }
                continue;
            }
            Str::Triple(q) => {
                cur.push(c);
                if c == q && chars.peek() == Some(&q) {
                    cur.push(chars.next()?);
                    cur.push(chars.next()?);
                    str = Str::None;
                }
                continue;
            }
            Str::None => {}
        }
        match c {
            '\'' | '"' => {
                let triple = chars.peek() == Some(&c) && {
                    // look at second char too
                    let mut clone = chars.clone();
                    clone.next();
                    clone.peek() == Some(&c)
                };
                if triple {
                    cur.push(c);
                    cur.push(chars.next()?);
                    cur.push(chars.next()?);
                    str = Str::Triple(c);
                } else {
                    cur.push(c);
                    str = Str::Single(c);
                }
            }
            '#' => {
                while let Some(&n) = chars.peek() {
                    if n == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
                cur.push(c);
            }
            '\n' if depth == 0 => {
                push_py_statement(&mut stmts, std::mem::take(&mut cur));
            }
            '\\' if chars.peek() == Some(&'\n') => {
                chars.next();
            }
            _ => cur.push(c),
        }
    }
    if depth != 0 || str != Str::None {
        return None;
    }
    push_py_statement(&mut stmts, cur);
    Some(stmts)
}

/// An indented line continues the previous compound statement (a `def`/`class`
/// body or a block) instead of starting a new top-level one.
fn push_py_statement(stmts: &mut Vec<String>, line: String) {
    if line.trim().is_empty() {
        return;
    }
    match stmts.last_mut() {
        Some(prev) if line.starts_with([' ', '\t']) => {
            prev.push('\n');
            prev.push_str(&line);
        }
        _ => stmts.push(line),
    }
}

/// A top-level single `=` (not `==`, `<=`, `>=`, `!=`, `:=`, `+=` …) in
/// `stmt`, splitting into (lhs, rhs). Only valid at bracket depth 0.
fn split_assignment(stmt: &str) -> Option<(&str, &str)> {
    let bytes = stmt.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'=' if depth == 0 => {
                let prev = bytes.get(i.wrapping_sub(1)).copied().unwrap_or(b' ');
                let next = bytes.get(i + 1).copied().unwrap_or(b' ');
                if !matches!(
                    prev,
                    b'=' | b'!'
                        | b'<'
                        | b'>'
                        | b'+'
                        | b'-'
                        | b'*'
                        | b'/'
                        | b'%'
                        | b':'
                        | b'&'
                        | b'|'
                        | b'^'
                ) && next != b'='
                {
                    return Some((&stmt[..i], &stmt[i + 1..]));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// LHS of a setup assignment: plain names, optionally dotted targets or
/// comma-separated names — nothing else (no subscripts, no calls).
fn is_assignable_lhs(lhs: &str) -> bool {
    let lhs = lhs.trim();
    if lhs.is_empty() {
        return false;
    }
    lhs.split(',').all(|part| {
        let part = part.trim();
        !part.is_empty()
            && part.split('.').all(|seg| {
                !seg.is_empty()
                    && seg.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && !seg.chars().next().unwrap().is_ascii_digit()
            })
    })
}

/// RHS constants: scalar literals (numbers, strings, bools, None, ellipsis).
fn is_literal_rhs(rhs: &str) -> bool {
    let rhs = rhs.trim();
    let quoted = rhs.len() >= 2
        && ['"', '\'']
            .iter()
            .any(|&q| rhs.starts_with(q) && rhs.ends_with(q))
        && !rhs[1..rhs.len() - 1].contains(['"', '\'']);
    let num = rhs.strip_prefix('-').unwrap_or(rhs).trim();
    quoted || num.parse::<f64>().is_ok() || matches!(rhs, "True" | "False" | "None" | "...")
}

/// RHS data loads: `json/pickle/np.load(…)`, `pd.read_*`, `open(…).read()`.
fn is_load_rhs(rhs: &str) -> bool {
    let r = rhs.trim();
    let bare_open_read = r.starts_with("open(") && (r.contains(".read()") || r.contains(".read("));
    bare_open_read
        || r.contains("json.load(")
        || r.contains("pickle.load(")
        || r.contains("np.load(")
        || r.contains("numpy.load(")
        || r.contains("pd.read_")
        || r.contains("pandas.read_")
}

fn is_setup_py(code: &str) -> bool {
    let Some(stmts) = py_statements(code) else {
        return false;
    };
    !stmts.is_empty()
        && stmts.iter().all(|stmt| {
            let t = stmt.trim();
            let head = t.split_whitespace().next().unwrap_or("");
            if head == "import"
                || (head == "from" && t.contains(" import "))
                || head == "def"
                || head == "class"
                || head.starts_with('@')
                || (head == "async" && t[4..].trim_start().starts_with("def "))
            {
                return true;
            }
            let Some((lhs, rhs)) = split_assignment(t) else {
                return false;
            };
            is_assignable_lhs(lhs) && (is_literal_rhs(rhs) || is_load_rhs(rhs))
        })
}

/// Split JS source into top-level statements at `;` or newlines at bracket
/// depth 0 (string-aware enough for quotes and template literals).
fn js_statements(code: &str) -> Option<Vec<String>> {
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut chars = code.chars().peekable();
    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            cur.push(c);
            if c == '\\' {
                cur.push(chars.next()?);
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '\'' | '"' | '`' => {
                cur.push(c);
                quote = Some(c);
            }
            '/' if chars.peek() == Some(&'/') => {
                while let Some(&n) = chars.peek() {
                    if n == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
                cur.push(c);
            }
            ';' | '\n' if depth == 0 => {
                if !cur.trim().is_empty() {
                    stmts.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if depth != 0 || quote.is_some() {
        return None;
    }
    if !cur.trim().is_empty() {
        stmts.push(cur);
    }
    Some(stmts)
}

fn is_setup_js(code: &str) -> bool {
    let Some(stmts) = js_statements(code) else {
        return false;
    };
    !stmts.is_empty()
        && stmts.iter().all(|stmt| {
            let t = stmt.trim();
            let head = t.split_whitespace().next().unwrap_or("");
            if head == "import" || head == "function" || head == "class" {
                return true;
            }
            if head == "async" && t[5..].trim_start().starts_with("function ") {
                return true;
            }
            // `const X = await import(…) / require(…) / load…(…)`
            let Some(rest) = t.strip_prefix("const ") else {
                return false;
            };
            let Some(eq) = rest.find('=') else {
                return false;
            };
            let (lhs, rhs) = (&rest[..eq], rest[eq + 1..].trim());
            let ok_lhs = !lhs.trim().is_empty()
                && lhs
                    .trim()
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '$');
            let ok_rhs = rhs.starts_with("await import(")
                || rhs.starts_with("import(")
                || rhs.starts_with("require(")
                || rhs.starts_with("await load");
            ok_lhs && ok_rhs
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_result() -> omp_eval::CellResult {
        omp_eval::CellResult {
            output: String::new(),
            status: CellStatus::Ok,
            cancelled: false,
            execution_count: 1,
            error: None,
            displays: Vec::new(),
            timed_out: false,
        }
    }

    // ---- classifier: Python ----

    #[test]
    fn py_imports_defs_are_setup() {
        assert!(is_setup_py("import json\nfrom pathlib import Path\ndef f(x):\n    return x + 1\nclass C:\n    pass"));
    }

    #[test]
    fn py_load_assignments_are_setup() {
        assert!(is_setup_py("data = json.load(open('a.json'))\ndf = pd.read_csv('x.csv')\narr = np.load('a.npy')\nblob = open('f.txt').read()"));
    }

    #[test]
    fn py_constants_are_setup() {
        assert!(is_setup_py("LIMIT = 10\nNAME = 'x'\nRATIO = -0.5"));
    }

    #[test]
    fn py_side_effect_calls_not_setup() {
        assert!(!is_setup_py("import json\nprint('hi')"));
        assert!(!is_setup_py("x = compute()"));
        assert!(!is_setup_py("import os\nos.makedirs('d')"));
    }

    #[test]
    fn py_control_flow_not_setup() {
        assert!(!is_setup_py("import json\nfor i in range(3):\n    pass"));
        assert!(!is_setup_py("if True:\n    x = 1"));
    }

    #[test]
    fn py_failed_classifier_edge_cases() {
        assert!(!is_setup_py("x[0] = 1"));
        assert!(!is_setup_py("x += 1"));
        // Defining a function never runs its body: replaying the def is safe.
        assert!(is_setup_py("def f():\n    open('x').write('y')"));
        assert!(!is_setup_py("")); // empty cell: nothing to replay
    }

    #[test]
    fn py_multiline_paren_import_is_setup() {
        assert!(is_setup_py(
            "from typing import (\n    Optional,\n    List,\n)"
        ));
    }

    // ---- classifier: JS ----

    #[test]
    fn js_imports_and_decls_are_setup() {
        assert!(is_setup_js(
            "import { x } from 'm';\nfunction f() {}\nclass C {}"
        ));
    }

    #[test]
    fn js_const_dynamic_imports_are_setup() {
        assert!(is_setup_js(
            "const m = await import('m');\nconst r = require('r');"
        ));
    }

    #[test]
    fn js_calls_and_control_flow_not_setup() {
        assert!(!is_setup_js("const x = compute();"));
        assert!(!is_setup_js("await tool.read('x');"));
        assert!(!is_setup_js("let y = 1;"));
        assert!(!is_setup_js("const f = () => 1;"));
    }

    // ---- generation logic ----

    #[test]
    fn first_kernel_is_not_a_loss() {
        let mut j = Journal::default();
        assert!(j.on_kernel_started().is_none());
    }

    #[test]
    fn later_kernel_notices_with_replay_count() {
        let mut j = Journal::default();
        let _ = j.on_kernel_started();
        j.setup.push("import json".to_string());
        j.setup.push("df = pd.read_csv('x')".to_string());
        let notice = j.on_kernel_started().expect("loss notice");
        assert!(notice.contains("not restored"), "{notice}");
        assert!(notice.contains("re-run required setup"), "{notice}");
        assert!(notice.contains("2 recorded setup cells"), "{notice}");
        assert!(notice.contains("generation 2"), "{notice}");
    }

    #[test]
    fn third_kernel_increments_generation() {
        let mut j = Journal::default();
        let _ = j.on_kernel_started();
        let _ = j.on_kernel_started();
        let notice = j.on_kernel_started().expect("loss notice");
        assert!(notice.contains("generation 3"), "{notice}");
    }

    #[test]
    fn single_setup_cell_grammar() {
        let notice = kernel_loss_notice(2, 1);
        assert!(notice.contains("1 recorded setup cell"), "{notice}");
    }

    #[test]
    fn record_cell_skips_failed_and_non_setup() {
        let mut j = Journal::default();
        // Simulate record_cell decisions without touching storage:
        // only Ok + setup-classified cells land in the journal.
        let ok = ok_result();
        assert_eq!(ok.status, CellStatus::Ok);
        assert!(is_setup_py("import json"));
        assert!(!is_setup_py("print(1)"));
        j.apply(&EventKind::Cell {
            code: "import json".into(),
        });
        assert_eq!(j.setup.len(), 1);
    }
}
