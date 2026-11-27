//! Safe Rust wrapper around the goose-compiled tool-output dedup kernel.
//!
//! The kernel (`goose/omt_dedup.goose`, generated to `generated/omt_dedup.c`)
//! decides, for each tool result, whether it is a first occurrence or an exact
//! duplicate of an earlier one, using SipHash-1-3 over the decoded text —
//! byte-identical to the `std::hash::DefaultHasher`-based reference in
//! `crates/omp-cli/src/dedup.rs`.
//!
//! The only `unsafe` in this crate is the single FFI call below; everything
//! else is safe. Inputs are passed as caller-provided slices (pointer + length)
//! and results come back in caller-provided buffers, so no memory crosses the
//! boundary.

// The FFI surface is index-typed (i64/usize), so the `as` casts below are
// deliberate; they only run on lengths/indices we control.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::ffi::c_char;

use std::sync::Once;

/// Mirror of the generated `sl_i64` slice struct.
#[repr(C)]
struct SlI64 {
    data: *mut i64,
    len: i64,
}

/// Mirror of the generated `sl_u8` slice struct.
#[repr(C)]
struct SlU8 {
    data: *mut u8,
    len: i64,
}

extern "C" {
    fn goose_init(argc: i32, argv: *mut *mut c_char);
    fn omt_goose_dedup_classify(
        n: i64,
        text_len: SlI64,
        text_off: SlI64,
        skip: SlU8,
        texts: SlU8,
        out_kind: SlI64,
        out_target: SlI64,
    );
}

/// `goose_init` runs the runtime and is required before any exported call. The
/// goose runtime is single-threaded (matches omt's tokio current_thread
/// runtime), so this is initialized once on the first classification call.
static INIT: Once = Once::new();
fn ensure_goose_init() {
    INIT.call_once(|| unsafe { goose_init(0, std::ptr::null_mut()) });
}

/// True for entries that participate in duplicate detection. `texts` is used in
/// lockstep with `eligible`; callers build both from the same messages.
pub fn dedup_classify(texts: &[&str], eligible: &[bool]) -> Vec<Option<usize>> {
    let n = texts.len();
    debug_assert_eq!(n, eligible.len());

    // Flatten only the eligible texts into one buffer; ineligible entries keep a
    // zero length so the kernel (which skips them) never reads them.
    let mut flat: Vec<u8> = Vec::with_capacity(
        texts
            .iter()
            .zip(eligible)
            .map(|(t, &e)| if e { t.len() } else { 0 })
            .sum(),
    );
    let mut off: Vec<i64> = Vec::with_capacity(n);
    let mut len: Vec<i64> = Vec::with_capacity(n);
    for (i, t) in texts.iter().enumerate() {
        off.push(flat.len() as i64);
        if eligible[i] {
            len.push(t.len() as i64);
            flat.extend_from_slice(t.as_bytes());
        } else {
            len.push(0);
        }
    }
    // The kernel's `skip` flag is `eligible`'s negation.
    let mut skip: Vec<u8> = eligible.iter().map(|&e| u8::from(!e)).collect();

    let mut out_kind = vec![0i64; n];
    let mut out_target = vec![0i64; n];

    ensure_goose_init();

    // Safety: every vector outlives the call and its pointer is valid for its
    // length (dangling but non-null for empty vectors, which the kernel never
    // reads). The goose runtime is single-threaded and used only here.
    unsafe {
        omt_goose_dedup_classify(
            n as i64,
            SlI64 {
                data: len.as_mut_ptr(),
                len: n as i64,
            },
            SlI64 {
                data: off.as_mut_ptr(),
                len: n as i64,
            },
            SlU8 {
                data: skip.as_mut_ptr(),
                len: n as i64,
            },
            SlU8 {
                data: flat.as_mut_ptr(),
                len: flat.len() as i64,
            },
            SlI64 {
                data: out_kind.as_mut_ptr(),
                len: n as i64,
            },
            SlI64 {
                data: out_target.as_mut_ptr(),
                len: n as i64,
            },
        );
    }

    (0..n)
        .map(|i| {
            if out_kind[i] != 0 {
                Some(out_target[i] as usize)
            } else {
                None
            }
        })
        .collect()
}
