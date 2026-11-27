// omt dedup kernel, written in goose and generated to C.
//
// Semantics mirror `crates/omp-cli/src/dedup.rs` exactly: for each tool result
// entry, decide whether it is a first occurrence (stay full) or an exact
// duplicate of an earlier one (become a stub pointing at the first occurrence).
// The duplicate key is a SipHash-1-3 over the decoded text followed by the
// 0xff string separator, with zero keys — byte-identical to how
// `std::hash::DefaultHasher` hashes a `&str` (DefaultHasher::new() is
// `SipHasher13::new_with_keys(0, 0)`, and `Hash for str` appends 0xff).
//
// Nothing is allocated across the C boundary: inputs arrive as caller-provided
// slices (pointer + length), results are written into caller-provided buffers.
// The open-addressing first-seen table is a single resizable goose array, so it
// lives entirely on goose's own data stack for the duration of one call.
//
// Regenerate with `./regenerate.sh` (see that script for the exact compiler
// invocation). This source is not required to build omt — only the generated C
// in `../generated/` is compiled (by `../build.rs` via the `cc` crate).
import std;

// One open-addressing slot: key (hash), value (first-occurrence index), tag.
struct Slot { key: u64, val: i64, tag: u8 }

fn npow2(x: i64) -> i64 {
    var p: i64 = 1;
    while p < x { p <<= 1; }
    return p;
}

fn rotl(x: u64, k: u64) -> u64 {
    return (x << k) | (x >> (64 - k));
}

fn sipround(v0: u64&, v1: u64&, v2: u64&, v3: u64&) {
    v0 += v1;
    v1 = rotl(v1, 13);
    v1 ^= v0;
    v0 = rotl(v0, 32);
    v2 += v3;
    v3 = rotl(v3, 16);
    v3 ^= v2;
    v0 += v3;
    v3 = rotl(v3, 21);
    v3 ^= v0;
    v2 += v1;
    v1 = rotl(v1, 17);
    v1 ^= v2;
    v2 = rotl(v2, 32);
}

// SipHash-1-3 over `data` followed by the 0xff string separator, with zero
// keys. Finalization is c=1 round after `v3 ^= b` and d=3 rounds after
// `v2 ^= 0xff` (the SipHasher13 layout, not the 4+4 of SipHash-2-4).
fn siphash13(data: const u8[:]) -> u64 {
    var v0: u64 = 0x736f6d6570736575;
    var v1: u64 = 0x646f72616e646f6d;
    var v2: u64 = 0x6c7967656e657261;
    var v3: u64 = 0x7465646279746573;
    let total = data.len + 1;
    var b: u64 = ((total as u64) & 0xff) << 56;
    let nfull = total >> 3;
    var i: i64 = 0;
    while i < nfull {
        var m: u64 = 0;
        var kb: i64 = 0;
        while kb < 8 {
            let bv: u64 = data[i * 8 + kb];
            m |= bv << ((kb as u64) * 8);
            kb += 1;
        }
        v3 ^= m;
        sipround(v0, v1, v2, v3);
        v0 ^= m;
        i += 1;
    }
    var tail: u64 = 0;
    let start = nfull * 8;
    let rem = total - start;
    var j: i64 = 0;
    while j < rem {
        var byte_val: u64 = 0;
        if start + j < data.len {
            byte_val = data[start + j];
        } else {
            byte_val = 0xff;
        }
        tail |= byte_val << ((j as u64) * 8);
        j += 1;
    }
    b |= tail;
    v3 ^= b;
    sipround(v0, v1, v2, v3);
    v0 ^= b;
    v2 ^= 0xff;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}

// Classify `n` entries. For entry i:
//   - skip[i] != 0  → never stubbed, never recorded (error / image / empty /
//     below-minimum, decided by the caller); out_kind[i] = 0.
//   - else hash texts[text_off[i] .. text_off[i] + text_len[i]] (in the caller
//     provided flattened `texts` buffer) and check it against the first-seen
//     table. First occurrence → out_kind[i] = 0 and record it; duplicate →
//     out_kind[i] = 1 and out_target[i] = the first-occurrence index.
export "omt_goose_dedup_classify" fn classify(
    n: i64,
    text_len: const i64[:],
    text_off: const i64[:],
    skip: const u8[:],
    texts: const u8[:],
    out_kind: i64[:],
    out_target: i64[:],
) {
    var cap = npow2((n * 3 + 1) / 2);
    if cap < 8 { cap = 8; }
    var slots: Slot[>..] = [];
    slots.resize(cap, Slot { 0, 0, 0 });
    let mask = cap - 1;
    var i: i64 = 0;
    while i < n {
        var kind: i64 = 0;
        var target: i64 = 0;
        if i < skip.len && skip[i] == 0 {
            let off = text_off[i];
            let len = text_len[i];
            if off >= 0 && len >= 0 && off + len <= texts.len {
                let h = siphash13(texts[off .. off + len]);
                var sl = (h & (mask as u64)) as i64;
                loop {
                    if slots[sl].tag == 0 {
                        slots[sl] = Slot { h, i, 1 };
                        break;
                    }
                    if slots[sl].key == h {
                        kind = 1;
                        target = slots[sl].val;
                        break;
                    }
                    sl = (sl + 1) & mask;
                }
            }
        }
        out_kind[i] = kind;
        out_target[i] = target;
        i += 1;
    }
}
