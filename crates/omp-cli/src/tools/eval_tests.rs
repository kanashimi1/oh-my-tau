//! Unit tests for the eval tool loop: the pre-port parity tests plus the
//! new loop behavior (default timeout, cancel, `is_error`, per-language reset,
//! output cap, kernel-loss notice, bridge-backed helpers).

use super::*;

/// Unique cwd per test: parallel tests must not share a kernel slot (a
/// JS timeout kill is global to the slot's kernel).
fn test_env(tag: &str) -> Arc<Env> {
    let dir = std::env::temp_dir().join(format!("omp-rs-eval-test-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Arc::new(Env::new(&dir, false))
}

fn ctx() -> CallCtx {
    CallCtx {
        tool_call_id: "t1".into(),
        signal: pa_agent::abort::AbortSignal::never(),
        on_update: None,
    }
}

async fn run(env: &Arc<Env>, args: Value) -> AgentToolResult {
    eval(env.clone(), args, ctx()).await.unwrap()
}

fn text(r: &AgentToolResult) -> String {
    let pa_agent::types::ToolResultContent::Text(content) = &r.content[0] else {
        panic!("unexpected content");
    };
    content.text.clone()
}

/// Prerequisite gate for tests that need the Bun JS runtime: prints the one
/// skip line and returns false when `bun` is missing, so a clean machine
/// (no bun) stays green while machines that have bun run the real test.
fn require_bun(test: &str) -> bool {
    if bun_on_path() {
        return true;
    }
    eprintln!(
        "skipping {test}: bun is not on PATH (install bun from https://bun.sh and make the `bun` executable resolve on PATH)"
    );
    false
}

/// `bun` resolvable on PATH: each PATH entry is probed for an executable
/// file named `bun` (nothing is spawned).
fn bun_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| is_executable(&dir.join("bun")))
}

#[cfg(unix)]
fn is_executable(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: std::path::PathBuf) -> bool {
    std::fs::metadata(&path).is_ok_and(|m| m.is_file())
}

#[test]
fn schema_matches_ts_reference() {
    let s = schema();
    let props = s["properties"].as_object().unwrap();
    assert_eq!(props["language"]["enum"], json!(["py", "js"]));
    assert_eq!(s["required"], json!(["language", "code"]));
    assert_eq!(props.len(), 5);
}

#[test]
fn timeout_defaults_to_30s_zero_disables_and_clamps() {
    assert_eq!(resolve_timeout(&json!({})), Some(Duration::from_secs(30)));
    assert_eq!(resolve_timeout(&json!({"timeout": 0})), None);
    assert_eq!(
        resolve_timeout(&json!({"timeout": 0.2})),
        Some(Duration::from_secs(1))
    );
    assert_eq!(
        resolve_timeout(&json!({"timeout": 1e9})),
        Some(Duration::from_hours(1))
    );
}

#[test]
fn small_output_is_not_capped() {
    assert_eq!(cap_output("hello\n"), "hello\n");
}

#[tokio::test]
async fn missing_language_is_rejected_like_ts_validation() {
    let env = test_env("lang");
    let err = eval(env.clone(), json!({"code": "1"}), ctx())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("language"), "got: {err}");
}

#[tokio::test]
async fn python_state_persists_and_errors_are_flagged() {
    let env = test_env("py");
    run(&env, json!({"language": "py", "code": "x = 40 + 2"})).await;
    let r = run(&env, json!({"language": "py", "code": "x"})).await;
    assert!(text(&r).contains("42"), "got: {}", text(&r));
    assert_eq!(r.is_error, None);

    let r = run(
        &env,
        json!({"language": "py", "code": "raise ValueError('bad')"}),
    )
    .await;
    assert_eq!(r.is_error, Some(true), "raised cell must be is_error");
    assert!(text(&r).contains("ValueError: bad"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn python_failed_cell_rolls_back_new_names() {
    let env = test_env("txn");
    let r = run(
        &env,
        json!({"language": "py", "code": "leaked = 1\nraise ValueError('boom')"}),
    )
    .await;
    assert_eq!(r.is_error, Some(true));
    assert!(
        text(&r).contains("[omp-txn] rolled back: leaked"),
        "got: {}",
        text(&r)
    );
    let r = run(
        &env,
        json!({"language": "py", "code": "'leaked' in globals()"}),
    )
    .await;
    assert!(
        text(&r).contains("False"),
        "name must be rolled back, got: {}",
        text(&r)
    );
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn reset_wipes_only_the_requested_language() {
    if !require_bun("reset_wipes_only_the_requested_language") {
        return;
    }
    let env = test_env("reset");
    run(&env, json!({"language": "py", "code": "p = 1"})).await;
    run(&env, json!({"language": "js", "code": "globalThis.j = 2"})).await;
    // Python reset: the fresh kernel has no `p` (and reports the state loss) ...
    let r = run(
        &env,
        json!({"language": "py", "code": "'p' in globals()", "reset": true}),
    )
    .await;
    assert!(text(&r).contains("False"), "got: {}", text(&r));
    // ... while the JS kernel keeps its state.
    let r = run(&env, json!({"language": "js", "code": "j"})).await;
    assert!(
        text(&r).contains('2'),
        "js state survives a py reset: {}",
        text(&r)
    );
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn fresh_kernel_generation_gets_a_kernel_loss_notice() {
    let env = test_env("notice");
    let first = run(&env, json!({"language": "py", "code": "1"})).await;
    assert!(
        !text(&first).contains("Kernel state was lost"),
        "first kernel: {}",
        text(&first)
    );
    let second = run(&env, json!({"language": "py", "code": "2", "reset": true})).await;
    assert!(
        text(&second).contains("Kernel state was lost"),
        "second generation: {}",
        text(&second)
    );
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn abort_cancels_a_running_cell() {
    let env = test_env("abort");
    let controller = pa_agent::abort::AbortController::new();
    let signal = controller.signal();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        controller.abort();
    });
    let started = std::time::Instant::now();
    let r = eval(
        env.clone(),
        json!({"language": "py", "code": "import time\ntime.sleep(20)", "timeout": 0}),
        CallCtx {
            tool_call_id: "t2".into(),
            signal,
            on_update: None,
        },
    )
    .await
    .unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "cancel must not wait for the cell"
    );
    assert_eq!(r.is_error, Some(true));
    // The Python kernel survives an interrupt.
    let r = run(&env, json!({"language": "py", "code": "'alive'"})).await;
    assert!(text(&r).contains("alive"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn python_prelude_tool_bridge_reaches_session_tools() {
    let env = test_env("bridge");
    std::fs::write(env.cwd.join("note.txt"), "bridged\n").unwrap();
    // `tool.read` goes cell → host frame → EvalBridge → the read tool.
    let r = run(
        &env,
        json!({"language": "py", "code": "r = await tool.read({'path': 'note.txt'})\nprint(r)"}),
    )
    .await;
    assert!(text(&r).contains("bridged"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn js_state_persists_and_top_level_await_works() {
    if !require_bun("js_state_persists_and_top_level_await_works") {
        return;
    }
    let env = test_env("js-state");
    run(
        &env,
        json!({"language": "js", "code": "let acc = await Promise.resolve(40)"}),
    )
    .await;
    let r = run(&env, json!({"language": "js", "code": "acc + 2"})).await;
    assert!(text(&r).contains("42"), "got: {}", text(&r));
    run(
        &env,
        json!({"language": "js", "code": "acc", "reset": true}),
    )
    .await;
    let r = run(&env, json!({"language": "js", "code": "typeof acc"})).await;
    assert!(
        text(&r).contains("\"undefined\""),
        "state wiped: {}",
        text(&r)
    );
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn js_error_and_timeout_annotations_match_omp() {
    if !require_bun("js_error_and_timeout_annotations_match_omp") {
        return;
    }
    let env = test_env("js-timeout");
    let r = run(
        &env,
        json!({"language": "js", "code": "throw new Error('boom')"}),
    )
    .await;
    assert!(text(&r).contains("Error: boom"), "got: {}", text(&r));
    assert_eq!(r.is_error, Some(true));

    let r = run(
        &env,
        json!({"language": "js", "code": "while (true) {}", "timeout": 1}),
    )
    .await;
    assert!(
        text(&r).contains("The JS worker was force-killed and its VM state was reset"),
        "got: {}",
        text(&r)
    );
    let r = run(&env, json!({"language": "js", "code": "'alive'"})).await;
    assert!(text(&r).contains("alive"), "got: {}", text(&r));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn nested_eval_from_a_cell_is_refused_not_deadlocked() {
    let env = test_env("nested");
    let started = std::time::Instant::now();
    let code = "try:\n    await tool.eval({'language': 'py', 'code': '1'})\nexcept Exception as e:\n    print('refused:', e)";
    let r = tokio::time::timeout(
        Duration::from_secs(30),
        run(&env, json!({"language": "py", "code": code})),
    )
    .await
    .expect("nested eval must not deadlock on the kernel lock");
    assert!(
        text(&r).contains("cannot run from inside an eval cell"),
        "got: {}",
        text(&r)
    );
    assert!(started.elapsed() < Duration::from_secs(30));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn kernel_wait_is_raced_against_abort() {
    let env = test_env("lock-abort");
    let slot = slot(&env.cwd);
    let held = slot.lock().await;
    let controller = pa_agent::abort::AbortController::new();
    let ctx = CallCtx {
        tool_call_id: "t2".into(),
        signal: controller.signal(),
        on_update: None,
    };
    let pending = tokio::spawn(eval(
        env.clone(),
        json!({"language": "py", "code": "1"}),
        ctx,
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    controller.abort();
    let err = tokio::time::timeout(Duration::from_secs(5), pending)
        .await
        .expect("an aborted eval must stop waiting for the kernel")
        .unwrap()
        .unwrap_err();
    assert!(err.to_string().contains("aborted"), "got: {err}");
    drop(held);
    shutdown_slot(&env.cwd);
}

// ===== opt-in observation pack / same-session dedup (OMP_EVAL_OBSERVATION_PACK) =====

/// Run one cell with the pack gate pinned through the test-only override (no
/// process-env mutation, so concurrently running kernel tests see no flips).
async fn run_packed(env: &Arc<Env>, args: Value, enabled: bool) -> String {
    let _guard = eval_observation::GATE_ENV_LOCK.lock();
    *eval_observation::PACK_OVERRIDE.lock() = Some(enabled);
    let r = run(env, args).await;
    *eval_observation::PACK_OVERRIDE.lock() = None;
    text(&r)
}

#[tokio::test]
async fn pack_gate_off_and_unknown_output_are_byte_identical() {
    let env = test_env("p6-off");
    let code = json!({"language": "py", "code": "print('hello world')\nprint('second line')"});
    let off = run_packed(&env, code.clone(), false).await;
    assert!(off.contains("hello world"));
    assert!(!off.contains("[eval observation"));
    let on_unknown = run_packed(&env, code, true).await;
    assert_eq!(
        on_unknown, off,
        "unknown format must fall back byte-identical"
    );
    // Same for over-budget output on the elision path.
    let big = json!({"language": "py", "code": "print('x' * 60000)"});
    let off_big = run_packed(&env, big.clone(), false).await;
    let on_big = run_packed(&env, big, true).await;
    assert_eq!(
        on_big, off_big,
        "big unknown output must fall back byte-identical"
    );
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn json_output_packs_when_enabled() {
    let env = test_env("p6-json");
    let code = json!({"language": "py", "code": r#"print('{"a": 1, "b": [1, 2]}')"#});
    let off = run_packed(&env, code.clone(), false).await;
    assert!(off.contains(r#"{"a": 1, "b": [1, 2]}"#), "got: {off}");
    let on = run_packed(&env, code, true).await;
    assert!(on.contains("[eval observation: json]"), "got: {on}");
    assert!(on.contains("keys: a, b"), "got: {on}");
    assert!(
        on.contains("full output: "),
        "pack must link the full output: {on}"
    );
    assert!(!on.contains(r#"{"a": 1"#), "pack replaces the body: {on}");
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn dedup_collapses_identical_output_and_shows_changes() {
    let env = test_env("p6-dedup");
    let a = json!({"language": "py", "code": "print('same-payload-42')"});
    let first = run_packed(&env, a.clone(), true).await;
    assert!(
        first.contains("same-payload-42"),
        "first occurrence is shown: {first}"
    );
    assert!(!first.contains("identical to previous cell"));
    let second = run_packed(&env, a, true).await;
    assert!(
        second.contains("identical to previous cell"),
        "got: {second}"
    );
    assert!(
        second.contains("full output: "),
        "dedup still links full output: {second}"
    );
    let changed = run_packed(
        &env,
        json!({"language": "py", "code": "print('changed-payload-43')"}),
        true,
    )
    .await;
    assert!(
        changed.contains("changed-payload-43"),
        "changes are never hidden: {changed}"
    );
    assert!(!changed.contains("identical to previous cell"));
    shutdown_slot(&env.cwd);
}

#[tokio::test]
async fn error_cells_never_pack_or_dedup() {
    let env = test_env("p6-err");
    let _ = run_packed(
        &env,
        json!({"language": "py", "code": "print('payload-x')"}),
        true,
    )
    .await;
    let raise = json!({"language": "py", "code": "raise ValueError('bad')"});

    let first = run_packed(&env, raise.clone(), true).await;
    assert!(first.contains("ValueError: bad"), "got: {first}");
    assert!(
        !first.contains("[eval observation"),
        "errors are never packed: {first}"
    );
    assert!(!first.contains("identical to previous cell"));
    let second = run_packed(&env, raise, true).await;
    assert!(
        second.contains("ValueError: bad"),
        "repeated errors stay visible: {second}"
    );
    assert!(
        !second.contains("identical to previous cell"),
        "errors are never deduped"
    );
    shutdown_slot(&env.cwd);
}

/// Whole-read slice helper mirroring the prelude's line semantics exactly:
/// split on /\r?\n/, window [offset, offset+limit), join with "\n".
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // offsets are small non-negative
fn js_slice(text: &str, offset: i64, limit: Option<usize>) -> String {
    if !(offset > 1 || limit.is_some()) {
        return text.to_owned();
    }
    if limit.is_some_and(|l| l == 0) {
        return String::new();
    }
    let lines: Vec<&str> = text
        .split("\r\n")
        .flat_map(|part| part.split('\n'))
        .collect();
    let start = (offset.saturating_sub(1).max(0)) as usize;
    if start >= lines.len() {
        return String::new();
    }
    let end = limit.map_or(lines.len(), |l| (start + l).min(lines.len()));
    lines[start..end].join("\n")
}

/// Single-quoted JS string literal (expected values embedded in cells).
fn js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// Ranged plain `read()` equals slicing the whole text (LF, CRLF, lone CR,
/// no trailing newline, empty, multibyte at the 1 MiB chunk boundary, big
/// file) and keeps whole-read failure semantics for `limit <= 0` and missing
/// files. Expected values are computed in Rust; each cell is one simple
/// `&&`-chain ending in a boolean, so the test exercises `read`, not the
/// cell transform.
#[tokio::test]
async fn js_ranged_read_matches_whole_read_semantics() {
    if !require_bun("js_ranged_read_matches_whole_read_semantics") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("omp-rs-p6-read-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut big = String::new();
    for i in 1..=70000 {
        big.push_str(&format!("entry {i}\n"));
    }
    let mut lf = String::new();
    for i in 1..=500 {
        lf.push_str(&format!("line {i}\n"));
    }
    let mut crlf = String::new();
    for i in 1..=300 {
        crlf.push_str(&format!("row {i}\r\n"));
    }
    let contents: Vec<(&str, String)> = vec![
        ("lf.txt", lf),
        ("crlf.txt", crlf),
        ("lonecr.txt", "x\ry\r\nz\r\nw".to_owned()),
        ("notrail.txt", "a\nb".to_owned()),
        ("empty.txt", String::new()),
        ("mb.txt", format!("{}é\nbé\ncé\n", "a".repeat(1048575))),
        ("big.txt", big),
    ];
    for (name, content) in &contents {
        std::fs::write(dir.join(name), content).unwrap();
    }
    let env = test_env("p6-jsread");
    // Pin the pack gate off for the whole test: sibling kernel tests flip the
    // test-only override concurrently in the same process.
    let _gate = eval_observation::GATE_ENV_LOCK.lock();
    *eval_observation::PACK_OVERRIDE.lock() = Some(false);
    let path = |name: &str| js_string(&format!("{}/{name}", dir.display()));

    let cases: Vec<(&str, i64, Option<usize>)> = vec![
        ("lf.txt", 1, Some(3)),
        ("lf.txt", 10, Some(2)),
        ("lf.txt", 498, Some(100)),
        ("lf.txt", 499, None),
        ("lf.txt", 2000, Some(5)),
        ("crlf.txt", 10, Some(2)),
        ("crlf.txt", 2, None),
        ("crlf.txt", 1, Some(1)),
        ("lonecr.txt", 1, Some(2)),
        ("lonecr.txt", 2, None),
        ("lonecr.txt", 1, Some(4)),
        ("notrail.txt", 2, None),
        ("notrail.txt", 1, Some(5)),
        ("empty.txt", 1, Some(3)),
        ("big.txt", 70000, Some(1)),
        ("big.txt", 69998, Some(3)),
        ("big.txt", 70001, Some(1)),
    ];
    // Named cases with Rust-computed expectations; each cell writes a JSON
    // row set ({name, pass, got, want}) to a temp file so a failure reports
    // the exact case instead of one opaque `false`.
    let mut named: Vec<(String, String, i64, Option<usize>, String)> = cases
        .iter()
        .map(|(file, offset, limit)| {
            let body = &contents.iter().find(|(n, _)| n == file).unwrap().1;
            let want = js_slice(body, *offset, *limit);
            (
                format!("{file}@{offset}+{limit:?}"),
                path(file),
                *offset,
                *limit,
                js_string(&want),
            )
        })
        .collect();
    // Multibyte chars straddling the 1 MiB chunk boundary decode correctly.
    // The 1 MiB first line is expressed as a repeat expression: embedding the
    // literal in the cell source would put its own é at a 1 MiB stdin chunk
    // edge, which the kernel transport (not read) mangles.
    named.push((
        "mb.txt@2+1".into(),
        path("mb.txt"),
        2,
        Some(1),
        js_string("bé"),
    ));
    named.push((
        "mb.txt@3+1".into(),
        path("mb.txt"),
        3,
        Some(1),
        js_string("cé"),
    ));
    named.push((
        "mb.txt@1+1".into(),
        path("mb.txt"),
        1,
        Some(1),
        "('a'.repeat(1048575) + 'é')".into(),
    ));

    let read_rows = |rows_path: &std::path::Path| -> Vec<Value> {
        serde_json::from_str(&std::fs::read_to_string(rows_path).unwrap()).unwrap()
    };
    let rows_path = dir.join("rows.json");
    let assert_rows = |rows: Vec<Value>| {
        for row in rows {
            assert!(
                row["pass"].as_bool().unwrap_or(false),
                "case {} failed: glen {} tail {}",
                row["name"].as_str().unwrap_or("?"),
                row["glen"],
                row["tail"],
            );
        }
    };
    for group in named.chunks(4) {
        let items = group
            .iter()
            .map(|(name, path_js, offset, limit, want)| {
                format!(
                    "(async () => {{ const got = await read({path_js}, {offset}, {limit}); \
                     return {{ name: {name}, got: JSON.stringify(got), glen: got.length, tail: JSON.stringify(got.slice(-2)), want: String({want}), pass: got === ({want}) }}; }})()",
                    path_js = path_js,
                    offset = offset,
                    limit = limit.map_or("undefined".to_owned(), |l| l.to_string()),
                    name = js_string(name),
                    want = want, // already a JS expression (literal or repeat-expr)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let code = format!(
            "await write({}, JSON.stringify(await Promise.all([{}])))",
            js_string(&rows_path.display().to_string()),
            items
        );
        let r = run(&env, json!({"language": "js", "code": code})).await;
        assert!(r.is_error.is_none(), "cell failed: {}", text(&r));
        assert_rows(read_rows(&rows_path));
    }

    // Edge semantics: `limit <= 0` yields "" (but keeps whole-read failure
    // behavior), a missing file throws in both the ranged and whole forms.
    let edge_items = format!(
        "(async () => {{ const got = await read({empty}, 1, 0); return {{ name: 'limit0', \
         pass: got === '', got: JSON.stringify(got) }}; }})(), \
         (async () => {{ const got = await read({lf}, 5, -2); return {{ name: 'neg-limit', \
         pass: got === '', got: JSON.stringify(got) }}; }})(), \
         (async () => {{ const got = await read({missing}, 1, 2).then(v => 'read', e => 'throw'); \
         return {{ name: 'missing-ranged', pass: got === 'throw', got }}; }})(), \
         (async () => {{ const got = await read({missing}).then(v => 'read', e => 'throw'); \
         return {{ name: 'missing-whole', pass: got === 'throw', got }}; }})()",
        empty = path("empty.txt"),
        lf = path("lf.txt"),
        missing = path("missing.txt"),
    );
    let code = format!(
        "await write({}, JSON.stringify(await Promise.all([{}])))",
        js_string(&rows_path.display().to_string()),
        edge_items
    );
    let r = run(&env, json!({"language": "js", "code": code})).await;
    assert!(r.is_error.is_none(), "edge cell failed: {}", text(&r));
    assert_rows(read_rows(&rows_path));

    *eval_observation::PACK_OVERRIDE.lock() = None;
    shutdown_slot(&env.cwd);
    let _ = std::fs::remove_dir_all(&dir);
}
