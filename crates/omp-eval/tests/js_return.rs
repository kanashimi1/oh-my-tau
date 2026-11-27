//! Integration tests: a trailing top-level `return <expr>;` publishes the
//! value like a trailing expression does, inside the async wrapper and
//! without it; `return;` and a `return` followed by a line break stay as
//! written (both return undefined).
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use omp_eval::js::{default_runtime, JsKernel};
use omp_eval::CellStatus;

mod common;

fn kernel(tag: &str) -> JsKernel {
    let dir = std::env::temp_dir().join(format!("omp-eval-js-return-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("cwd");
    JsKernel::start(default_runtime(), &dir).expect("kernel starts; bun must be on PATH")
}

fn output(k: &JsKernel, code: &str) -> (CellStatus, String) {
    let r = k.execute(code, None).expect("cell");
    (r.status, r.output.clone())
}

#[test]
fn trailing_return_publishes_the_value() {
    if !common::require_bun("trailing_return_publishes_the_value") {
        return;
    }
    let k = kernel("value");
    // After other statements and an await: the cell runs inside the async wrapper.
    let (status, out) = output(&k, "const a = 41\nreturn a + 1;");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, "42\n"));
    // As the only statement there is no await and so no wrapper: a bare
    // top-level `return` would be a syntax error.
    let (status, out) = output(&k, "return 1 + 1;");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, "2\n"));
    // An awaited value, an object, and a comment after the semicolon.
    let (status, out) = output(&k, "const v = await Promise.resolve(7)\nreturn { v }; // done");
    assert_eq!(status, CellStatus::Ok, "{out}");
    assert!(out.contains("7") && out.contains("v"), "{out}");
    // A block with an early `return` before it (inside the async wrapper):
    // the final statement is still the `return`.
    let (status, out) = output(
        &k,
        "if (false) { return 1 }\nconst c = await Promise.resolve(0)\nreturn 2;",
    );
    assert_eq!((status, out.as_str()), (CellStatus::Ok, "2\n"));
    // After a statement on the same line, and indented: still the trailing statement.
    let (status, out) = output(&k, "const f = await Promise.resolve(6); return f * 2;");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, "12\n"));
    let (status, out) = output(&k, "  const g = await Promise.resolve(5)\n  return g + 1;");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, "6\n"));
    // Values are published exactly like a trailing expression publishes them.
    let (_, plain) = output(&k, "const d = 5\nd * 2");
    let (_, returned) = output(&k, "const e = 5\nreturn e * 2;");
    assert_eq!(plain, returned);
}

#[test]
fn returns_without_a_value_and_inside_functions_are_unchanged() {
    if !common::require_bun("returns_without_a_value_and_inside_functions_are_unchanged") {
        return;
    }
    let k = kernel("bare");
    let (status, out) = output(&k, "const a = 1\nreturn;");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, ""));
    // Automatic semicolon insertion: `return` ends the statement, the next
    // line is never reached as a value.
    let (status, out) = output(&k, "const b = await Promise.resolve(2)\nreturn\nb + 1");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, ""));
    // `return` inside a function body is not a top-level statement.
    let (status, out) = output(&k, "const f = () => { return 9 }\nf()");
    assert_eq!((status, out.as_str()), (CellStatus::Ok, "9\n"));
}
