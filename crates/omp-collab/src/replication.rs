//! Hard-cap helper for host→guest collab frames.
//!
//! Port of
//! `vendor-src/oh-my-pi/packages/coding-agent/src/collab/replication-shrink.ts`
//! (the shrink/bounding layer) plus the byte-bounded snapshot packing
//! primitive host.ts hangs off it (`#snapshotChunks`,
//! `vendor-src/.../src/collab/host.ts:898-926` — the ~3s/512 KB chunking
//! constants live in host.ts comments there; host.rs owns the frame-emitting
//! caller, this module owns the pure slicing).
//!
//! The host wraps every [`crate::protocol`] frame in an AES-GCM envelope and
//! ships it through the relay's WebSocket. WebSocket servers enforce a
//! per-frame `maxPayloadLength` (Bun's default is 16 MB; many proxies cap
//! lower). A single oversized payload — typically a `read`/`bash`/`search`
//! tool result captured as one multi-megabyte string — would otherwise ship
//! as its own oversized frame and trip that limit, killing the host's
//! WebSocket with `1006 Received too big message`. `CollabSocket` treats 1006
//! as transient and reconnects, the next guest hello triggers the same
//! oversized send, and the loop never breaks (issue #3739).
//!
//! That ceiling has to be *enforced*, not approached (issue #11433). The TS
//! module guards three shapes that escaped a best-effort version: size in
//! object keys, nesting past the engine's serialization limit, and CJK
//! payloads measured in UTF-16 code units. All three are handled below.
//!
//! TS-vs-Rust domain mapping (intentional simplifications):
//!
//! - The walk operates on [`serde_json::Value`], the wire domain the rest of
//!   the crate speaks. TS values carrying `toJSON` (a `Date`) or an own
//!   `__proto__` key are already plain JSON by the time they reach a
//!   `Value`, so both PR #11999 concerns collapse into "the clone must
//!   serialize to the same bytes" — which [`to_json_string`] guarantees.
//! - A `Value` cannot be cyclic, so TS's `WeakSet` ancestor guard and
//!   [`CYCLE_ELISION_MARKER`] are unreachable; the marker constant is kept
//!   for wire-format parity only.
//! - TS's `JSON.stringify` throws `RangeError` at ~40 000 nesting levels and
//!   the throw routes deep payloads into the shrink passes. Rust has no such
//!   engine limit, but `serde_json::to_string` recurses per level (a deep
//!   `Value` would overflow the stack) and the guest re-parses frames under
//!   `serde_json`'s default 128-level recursion limit — so
//!   [`json_under_ceiling`] treats nesting >= [`MAX_REPLICATED_DEPTH`] as
//!   unserializable and degrades it exactly like the TS engine throw would.
//!   Net effect for the TS test suite's shapes (50 000 deep): identical
//!   bytes. (The limit is named `serde_json`'s, not an engine's, on purpose:
//!   it is the downstream re-parser that would break first.)
//! - `clipString` slices by UTF-16 code units and can split a surrogate
//!   pair; Rust slices by `char`. Identical for ASCII and BMP text; only a
//!   lone-surrogate split differs, which is unrepresentable in UTF-8.
//! - `replicationByteLength` returns `null` for unserializable input
//!   (cyclic/`BigInt`/engine `RangeError`). A `Value` is always serializable,
//!   so [`replication_byte_length`] returns the length directly; the `null`
//!   branch folds into the depth bound of [`json_under_ceiling`].

use serde_json::{Map, Value};

/// Per-payload ceiling for host→guest frames (replication-shrink.ts:49).
/// Bun's default WebSocket `maxPayloadLength` is 16 MB; a generous margin
/// keeps the AES-GCM envelope (+ IV + tag), the 4-byte peer header, and the
/// outer wire wrapper comfortably under that on every reasonable relay.
pub const MAX_REPLICATED_PAYLOAD_BYTES: usize = 1024 * 1024;

/// `customType` of the entry a guest receives in place of one that cannot be
/// shrunk (replication-shrink.ts:52).
pub const COLLAB_ENTRY_OMITTED_CUSTOM_TYPE: &str = "collab-entry-too-large";

/// Deepest nesting the walk emits (replication-shrink.ts:79). The engine, not
/// this helper, is the limit being defended; 1 000 keeps a margin under the
/// serialization limits of every consumer while staying far above any real
/// entry. Deeper payloads are elided with a marker rather than dropped, so
/// ids and shallow metadata still reach the guest.
pub const MAX_REPLICATED_DEPTH: usize = 1000;

/// Payload replacing a branch nested past [`MAX_REPLICATED_DEPTH`]
/// (replication-shrink.ts:112).
pub const DEPTH_ELISION_MARKER: &str = "…[deeper levels elided for collab session]";

/// Payload that would replace a cyclic reference (replication-shrink.ts:113).
/// Unreachable for [`Value`] — it cannot contain a cycle — kept so the exact
/// TS marker bytes stay pinned for any future log/diagnostic parity.
pub const CYCLE_ELISION_MARKER: &str = "…[cyclic reference elided for collab session]";

/// Strings keep this many chars of headroom below their pass cap for the
/// elision marker (replication-shrink.ts:117 `STRING_ELISION_RESERVE`).
const STRING_ELISION_RESERVE: usize = 80;

/// Soft byte cap per `snapshot-chunk` frame (host.ts:128-134). The first MB
/// of a snapshot takes ~3s through the default relay, so a 512 KB chunk lands
/// well under the guest's 30 s per-chunk progress timeout; oversized single
/// entries still ship in a chunk of their own. Measured in UTF-8 bytes — the
/// unit the relay and the seal step care about — not UTF-16 code units
/// (#11433).
pub const SNAPSHOT_CHUNK_BYTES: usize = 512 * 1024;

/// Progressive shrink passes (replication-shrink.ts:100-108): each pass
/// tightens both the per-string cap and the per-array head limit; the loop
/// stops at the first pass whose output fits
/// [`MAX_REPLICATED_PAYLOAD_BYTES`]. The final pass clamps every string to
/// 64 B and every array to one element. Payloads that still exceed the
/// ceiling after it are handled by the typed placeholders in
/// [`serialize_replicated_entry`] / [`serialize_replicated_event`].
const SHRINK_PASSES: [(usize, usize); 7] = [
    (64 * 1024, 256),
    (16 * 1024, 128),
    (4 * 1024, 64),
    (1024, 32),
    (256, 16),
    (256, 4),
    (64, 1),
];

/// A bounded replication payload together with its JSON form
/// (replication-shrink.ts:158-162), so the frame carrying it is serialized
/// exactly once. `value` is the input itself when it already fits, otherwise
/// a shrunk clone or a typed placeholder.
///
/// The TS struct is generic over the entry/event unions; the Rust wire domain
/// is [`Value`], so the port is not.
#[derive(Debug, Clone, PartialEq)]
pub struct SerializedReplication {
    pub value: Value,
    pub json: String,
}

// ── Serialization ──────────────────────────────────────────────────────────

/// Compact JSON serialization of `value`, byte-identical to
/// `serde_json::to_string` (asserted in tests) but iterative, so a deeply
/// nested `Value` cannot overflow the stack the way the recursive
/// `Serialize` impl would. This module must never be the code that crashes
/// on a pathological payload — that is the whole point of issue #11433.
fn to_json_string(value: &Value) -> String {
    /// Serializer stack frames. Iterators borrow the input tree, so no
    /// child is ever cloned to be walked.
    enum Job<'a> {
        Array(std::slice::Iter<'a, Value>),
        Object(serde_json::map::Iter<'a>),
    }

    let mut out = String::new();
    let mut stack: Vec<Job<'_>> = Vec::new();
    let mut current = value;
    loop {
        match current {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            // Leaf scalars go through serde_json's own formatter so number
            // rounding (`float_roundtrip`) and string escaping match the
            // rest of the crate byte for byte.
            Value::Number(n) => out.push_str(&serde_json::to_string(n).unwrap_or_default()),
            Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
            Value::Array(items) => {
                let mut iter = items.iter();
                if let Some(first) = iter.next() {
                    out.push('[');
                    stack.push(Job::Array(iter));
                    current = first;
                    continue;
                }
                out.push_str("[]");
            }
            Value::Object(map) => {
                let mut iter = map.iter();
                if let Some((key, first)) = iter.next() {
                    out.push('{');
                    out.push_str(&serde_json::to_string(key).unwrap_or_default());
                    out.push(':');
                    stack.push(Job::Object(iter));
                    current = first;
                    continue;
                }
                out.push_str("{}");
            }
        }
        // A value finished emitting: unwind, adding separators/closers
        // until the next pending child (or the root completes).
        loop {
            match stack.last_mut() {
                None => return out,
                Some(Job::Array(iter)) => match iter.next() {
                    Some(next) => {
                        out.push(',');
                        current = next;
                        break;
                    }
                    None => out.push(']'),
                },
                Some(Job::Object(iter)) => match iter.next() {
                    Some((key, next)) => {
                        out.push(',');
                        out.push_str(&serde_json::to_string(key).unwrap_or_default());
                        out.push(':');
                        current = next;
                        break;
                    }
                    None => out.push('}'),
                },
            }
            stack.pop();
        }
    }
}

/// UTF-8 byte length of `value`'s JSON form (replication-shrink.ts:142-155)
/// — the metric the relay actually enforces, and therefore the only one this
/// module compares against the ceiling. The TS `null` return for
/// unserializable input is unrepresentable for a [`Value`]; the too-deep
/// case is handled by [`json_under_ceiling`]'s depth bound instead.
#[must_use]
pub fn replication_byte_length(value: &Value) -> usize {
    to_json_string(value).len()
}

/// `value`'s JSON form when it serializes and fits the ceiling, else `None`
/// (replication-shrink.ts:165-177 `jsonUnderCeiling`).
///
/// The TS fast path multiplies the UTF-16 length by 3 as a conservative
/// UTF-8 bound before paying for `Buffer.byteLength`; Rust's `str::len` is
/// already bytes, so the single `<=` check is the same decision.
///
/// Nesting at or beyond [`MAX_REPLICATED_DEPTH`] returns `None` before any
/// serialization: this is the port's stand-in for the TS engine's
/// `RangeError` at ~40 000 levels (see the module docs for why the bound is
/// tighter here).
fn json_under_ceiling(value: &Value) -> Option<String> {
    if depth_at_or_beyond(value, MAX_REPLICATED_DEPTH) {
        return None;
    }
    let json = to_json_string(value);
    (json.len() <= MAX_REPLICATED_PAYLOAD_BYTES).then_some(json)
}

/// True when any container sits at nesting depth `>= limit` (the root is at
/// depth 0, mirroring `shrinkWalk`'s `visit(value, 0)` entry). Iterative —
/// this runs on payloads whose depth is exactly what we are probing.
fn depth_at_or_beyond(value: &Value, limit: usize) -> bool {
    let mut stack = vec![(value, 0_usize)];
    while let Some((current, depth)) = stack.pop() {
        match current {
            Value::Array(items) => {
                if depth >= limit {
                    return true;
                }
                for item in items {
                    stack.push((item, depth + 1));
                }
            }
            Value::Object(map) => {
                if depth >= limit {
                    return true;
                }
                for child in map.values() {
                    stack.push((child, depth + 1));
                }
            }
            _ => {}
        }
    }
    false
}

// ── The walk ───────────────────────────────────────────────────────────────

/// Head-truncate one string leaf to `string_cap`, appending a marker that
/// reports the exact number of dropped characters so a guest can tell "this
/// was bigger" (replication-shrink.ts:184-190 `clipString`). Strings that
/// already fit are returned unchanged. TS slices by UTF-16 code units; the
/// port slices by `char` (see module docs).
fn clip_string(value: &str, string_cap: usize) -> String {
    use std::fmt::Write as _;
    let total = value.chars().count();
    if total <= string_cap {
        return value.to_owned();
    }
    let head_len = string_cap.saturating_sub(STRING_ELISION_RESERVE);
    let mut out = String::with_capacity(value.len() + 48);
    out.extend(value.chars().take(head_len));
    let elided = total - head_len;
    // `String` writes are infallible.
    let _ = write!(out, "\n…[{elided} chars elided for collab session]");
    out
}

/// Where a surviving child lands inside its parent's destination.
#[derive(Clone)]
enum Slot {
    Array(usize),
    Key(String),
}

fn attach(dst: &mut Value, slot: Slot, child: Value) {
    match (dst, slot) {
        (Value::Array(items), Slot::Array(idx)) => items[idx] = child,
        (Value::Object(map), Slot::Key(key)) => {
            map.insert(key, child);
        }
        _ => unreachable!("slot always matches the destination container it was built for"),
    }
}

enum Src<'a> {
    Array(&'a [Value]),
    Object(&'a Map<String, Value>),
}

impl<'a> Src<'a> {
    /// The surviving child this slot reads from the source container. The
    /// returned reference points into the *input* tree (`'a`), not into
    /// `self`, so frames never keep the stack borrowed while attaching.
    fn child(&self, slot: &Slot) -> &'a Value {
        match (self, slot) {
            (Src::Array(items), Slot::Array(idx)) => &items[*idx],
            (Src::Object(map), Slot::Key(key)) => map.get(key).unwrap_or(&Value::Null),
            _ => unreachable!("slot always matches the source container it was built for"),
        }
    }
}

/// One container being rebuilt. `keys` holds only the *surviving* children
/// (array head or all object keys), visited one per loop tick, so the stack
/// holds one frame per nesting level — never one per sibling — which is what
/// keeps a 200 000-key object from allocating 200 000 frames
/// (replication-shrink.ts:192-224 `WalkFrame`).
struct Frame<'a> {
    src: Src<'a>,
    dst: Value,
    keys: Vec<Slot>,
    next: usize,
    /// Depth its children are visited at (parent container depth + 1).
    children_depth: usize,
    /// Where `dst` attaches when this frame completes.
    parent: Option<(usize, Slot)>,
}

enum Step<'a> {
    Leaf(Value),
    Container {
        src: Src<'a>,
        dst: Value,
        keys: Vec<Slot>,
    },
}

/// One `visit` from `shrinkWalk` (replication-shrink.ts:226-286): strings
/// are clipped first (even past the depth bound — TS checks string before
/// depth), plain leaves pass through, containers beyond
/// [`MAX_REPLICATED_DEPTH`] degrade to [`DEPTH_ELISION_MARKER`], and arrays
/// get their tails head-clipped with a trailing `…[N items elided…]`
/// summary element. There is no cycle check: a [`Value`] cannot be cyclic
/// (the TS `ancestors` `WeakSet` is unreachable here).
fn visit(value: &Value, depth: usize, string_cap: usize, array_limit: usize) -> Step<'_> {
    match value {
        Value::String(s) => Step::Leaf(Value::String(clip_string(s, string_cap))),
        Value::Null | Value::Bool(_) | Value::Number(_) => Step::Leaf(value.clone()),
        Value::Array(items) => {
            if depth >= MAX_REPLICATED_DEPTH {
                return Step::Leaf(Value::String(DEPTH_ELISION_MARKER.to_owned()));
            }
            let keep = items.len().min(array_limit);
            let elided = items.len() - keep;
            let mut dst = Vec::with_capacity(keep + usize::from(elided > 0));
            dst.resize(keep, Value::Null);
            if elided > 0 {
                // TS wording has no singular form: `…[1 items elided…]`.
                dst.push(Value::String(format!(
                    "…[{elided} items elided for collab session]"
                )));
            }
            Step::Container {
                src: Src::Array(items),
                dst: Value::Array(dst),
                keys: (0..keep).map(Slot::Array).collect(),
            }
        }
        Value::Object(map) => {
            if depth >= MAX_REPLICATED_DEPTH {
                return Step::Leaf(Value::String(DEPTH_ELISION_MARKER.to_owned()));
            }
            // `Object.keys` order — serde_json's `preserve_order` Map keeps
            // insertion order, matching what `JSON.stringify` emits. Object
            // keys are identity and are never clipped (issue #11433).
            Step::Container {
                src: Src::Object(map),
                dst: Value::Object(Map::new()),
                keys: map.keys().map(|k| Slot::Key(k.clone())).collect(),
            }
        }
    }
}

/// Rebuild `value` as a fresh clone with long strings head-truncated, long
/// array tails head-clipped, and nesting past [`MAX_REPLICATED_DEPTH`]
/// elided. Iterative by construction: recursion is what used to turn a deep
/// entry into a `RangeError` instead of an oversized payload
/// (replication-shrink.ts:229 `shrinkWalk`).
fn shrink_walk(root: &Value, string_cap: usize, array_limit: usize) -> Value {
    let mut stack: Vec<Frame<'_>> = Vec::new();
    match visit(root, 0, string_cap, array_limit) {
        Step::Leaf(v) => return v,
        Step::Container { src, dst, keys } => {
            if keys.is_empty() {
                return dst;
            }
            stack.push(Frame {
                src,
                dst,
                keys,
                next: 0,
                children_depth: 1,
                parent: None,
            });
        }
    }
    loop {
        let top = stack.len() - 1;
        let (slot, children_depth) = {
            let frame = &mut stack[top];
            let slot = frame.keys[frame.next].clone();
            frame.next += 1;
            (slot, frame.children_depth)
        };
        let child = stack[top].src.child(&slot);
        match visit(child, children_depth, string_cap, array_limit) {
            Step::Leaf(v) => attach(&mut stack[top].dst, slot, v),
            Step::Container { src, dst, keys } => {
                if keys.is_empty() {
                    attach(&mut stack[top].dst, slot, dst);
                } else {
                    stack.push(Frame {
                        src,
                        dst,
                        keys,
                        next: 0,
                        children_depth: children_depth + 1,
                        parent: Some((top, slot)),
                    });
                }
            }
        }
        // Completed frames pop and attach upward; the root frame (no
        // parent) ends the walk.
        while stack
            .last()
            .is_some_and(|frame| frame.next >= frame.keys.len())
        {
            let frame = stack.pop().expect("length checked above");
            match frame.parent {
                None => return frame.dst,
                Some((parent_idx, slot)) => attach(&mut stack[parent_idx].dst, slot, frame.dst),
            }
        }
    }
}

/// Deep-copy `value` under the same depth bound the shrink passes use,
/// without clipping a single string or array
/// (replication-shrink.ts:337 `copyForReplication`). The host uses this for
/// a snapshot entry that does not serialize as-is and must have its images
/// stripped (host.ts:875-877): `structuredClone` would throw past the
/// engine's recursion limit and strand a joining guest without its `final`
/// chunk (issue #11433).
///
/// Bounded, not lossless: nesting past [`MAX_REPLICATED_DEPTH`] becomes
/// [`DEPTH_ELISION_MARKER`], exactly as in the shrink passes. Container
/// identity (`__proto__` own keys as data, `toJSON` honored) is automatic in
/// the [`Value`] domain — see the module docs.
#[must_use]
pub fn copy_for_replication(value: &Value) -> Value {
    shrink_walk(value, usize::MAX, usize::MAX)
}

/// Best-effort shape-preserving shrink: long strings head-truncated, long
/// array tails head-clipped, nesting past [`MAX_REPLICATED_DEPTH`] elided
/// (replication-shrink.ts:356-369 `shrinkPayloadShape`). Returns the input
/// itself when it already fits. `None` json means no pass fits
/// [`MAX_REPLICATED_PAYLOAD_BYTES`] — the size lives in object keys, which
/// are identity and nothing here may drop — which is why
/// [`serialize_replicated_entry`] / [`serialize_replicated_event`] own the
/// ceiling guarantee. Every pass walks the *original* (never a previously
/// shrunk output), so elision counts always report against the source.
fn shrink_payload_shape(value: &Value) -> (Value, Option<String>) {
    if let Some(json) = json_under_ceiling(value) {
        return (value.clone(), Some(json));
    }
    for &(string_cap, array_limit) in &SHRINK_PASSES {
        let shrunk = shrink_walk(value, string_cap, array_limit);
        if let Some(json) = json_under_ceiling(&shrunk) {
            return (shrunk, Some(json));
        }
    }
    (value.clone(), None)
}

// ── Typed placeholders ─────────────────────────────────────────────────────

/// `entry.type` as the TS template literals render it — a missing type
/// stringifies as `"undefined"` there.
fn type_of(value: &Value) -> &str {
    value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("undefined")
}

/// Describe a payload that could not be shrunk, for the guest-facing marker
/// (replication-shrink.ts:372-374 `omittedDetail`). The TS `bytes === null`
/// branch is unrepresentable for a [`Value`] (see
/// [`replication_byte_length`]).
fn omitted_detail(entry_type: &str, bytes: usize) -> String {
    format!("{entry_type} entry, {bytes} bytes")
}

/// Bound one replicated session entry under [`MAX_REPLICATED_PAYLOAD_BYTES`],
/// returning it with its JSON so callers can embed it in a frame without
/// serializing the entry a second time
/// (replication-shrink.ts:405 `serializeReplicatedEntry`).
///
/// Shrinking is shape-preserving whenever it can be. When it cannot — size
/// in the keys, or a payload that is not serializable at all — the entry is
/// replaced by a `custom_message` that keeps `id`/`parentId`/`timestamp`, so
/// the guest's branch chain stays connected and the substitution is visible
/// in the replica. A `custom_message` is the carrier because the guest's
/// live entry path only special-cases `message`/`compaction`/
/// `branch_summary`, and because it survives the replica JSONL load that the
/// snapshot path performs. On that path the placeholder does reach the
/// guest's model context — deliberate: the guest's model is told the content
/// was dropped instead of silently reasoning over a gap in the transcript.
///
/// Never fails: it is called from the host's `onEntryAppended` chokepoint
/// (host.ts:478) and from the snapshot chunker, where a failure would strand
/// the guest without a `final` chunk.
///
/// TS drops a placeholder key when the source entry lacks it
/// (`JSON.stringify` omits `undefined`); the port inserts `id`/`parentId`/
/// `timestamp` only when present, producing the same bytes.
#[must_use]
pub fn serialize_replicated_entry(entry: &Value) -> SerializedReplication {
    let (value, json) = shrink_payload_shape(entry);
    if let Some(json) = json {
        return SerializedReplication { value, json };
    }
    let detail = omitted_detail(type_of(entry), replication_byte_length(entry));
    let mut placeholder = Map::new();
    placeholder.insert("type".to_owned(), Value::from("custom_message"));
    for key in ["id", "parentId", "timestamp"] {
        if let Some(found) = entry.get(key) {
            placeholder.insert(key.to_owned(), found.clone());
        }
    }
    placeholder.insert(
        "customType".to_owned(),
        Value::from(COLLAB_ENTRY_OMITTED_CUSTOM_TYPE),
    );
    placeholder.insert("display".to_owned(), Value::from(true));
    placeholder.insert(
        "content".to_owned(),
        Value::from(format!(
            "…[{detail} omitted for collab session: too large to replicate]"
        )),
    );
    let value = Value::Object(placeholder);
    let json = to_json_string(&value);
    SerializedReplication { value, json }
}

/// Emit the guest-visible notice that accompanies a live oversized-entry
/// substitution (replication-shrink.ts:427 `oversizedEntryNotice`). Mirrors
/// the wording of the placeholder so a guest that only applies `message`
/// entries to its live agent context still learns an entry was dropped
/// (PR #11999 review) — notices never enter agent state and never reach the
/// model, so this is display-only.
#[must_use]
pub fn oversized_entry_notice(entry_type: &str) -> Value {
    serde_json::json!({
        "type": "notice",
        "level": "warning",
        "source": "collab",
        "message": format!("Host entry omitted: too large to replicate ({entry_type})."),
    })
}

/// Bound one replicated agent event under [`MAX_REPLICATED_PAYLOAD_BYTES`],
/// returning it with its JSON — the host's per-token streaming path
/// serializes each `message_update` once (replication-shrink.ts:447
/// `serializeReplicatedEvent`; the host call site is host.ts:699).
///
/// Events carry no identity to preserve, so an event that survives the
/// shape-preserving passes is replaced by a `notice` naming the omitted type
/// — notices never enter agent state and never reach the model, which keeps
/// the substitution out of the guest's context instead of silently diverging
/// from the host's view.
#[must_use]
pub fn serialize_replicated_event(event: &Value) -> SerializedReplication {
    let (value, json) = shrink_payload_shape(event);
    if let Some(json) = json {
        return SerializedReplication { value, json };
    }
    let value = serde_json::json!({
        "type": "notice",
        "level": "warning",
        "source": "collab",
        "message": format!("Host event omitted: too large to replicate ({}).", type_of(event)),
    });
    let json = to_json_string(&value);
    SerializedReplication { value, json }
}

// ── Snapshot packing (host.ts) ─────────────────────────────────────────────

/// One byte-bounded batch of pre-serialized snapshot entries
/// (host.ts:898-926 `#snapshotChunks`); the caller encodes each batch as a
/// `snapshot-chunk` frame (encoding belongs to `crate::protocol`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotChunk {
    /// The batch's entry JSON strings, in snapshot order.
    pub entries: Vec<String>,
    /// `final` on the wire. Always emitted on the last batch — even for an
    /// empty snapshot — so the guest never blocks on a missing terminator.
    pub is_final: bool,
}

/// Slice pre-serialized entries into byte-bounded batches (host.ts:908-926).
/// Every batch carries at least one entry — the `i > start` guard, so an
/// oversized single entry ships in a chunk of its own rather than tripping
/// the relay's per-frame `maxPayloadLength` (issue #3739) — and the last
/// batch is tagged `is_final`. An empty snapshot still emits one final
/// batch. `bytes[i]` missing degrades to 0 (`bytes[i] ?? 0` in TS); the TS
/// `entries.fill("", start, i)` string release has no Rust equivalent — each
/// batch's strings are copied out instead, which does not change the frames.
#[must_use]
pub fn snapshot_chunks(entries: &[String], bytes: &[usize]) -> Vec<SnapshotChunk> {
    if entries.is_empty() {
        return vec![SnapshotChunk {
            entries: Vec::new(),
            is_final: true,
        }];
    }
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let start = i;
        let mut batch_bytes = 0;
        while i < entries.len() {
            let entry_bytes = bytes.get(i).copied().unwrap_or(0);
            if i > start && batch_bytes + entry_bytes > SNAPSHOT_CHUNK_BYTES {
                break;
            }
            batch_bytes += entry_bytes;
            i += 1;
        }
        let is_final = i >= entries.len();
        chunks.push(SnapshotChunk {
            entries: entries[start..i].to_vec(),
            is_final,
        });
    }
    chunks
}

// ── Tests ──────────────────────────────────────────────────────────────────
//
// Golden vectors are generated by running the actual TS module with bun
// 1.3.14 against `vendor-src/.../replication-shrink.ts` (`serialize…`,
// `copyForReplication`, `replicationByteLength` are pure and the module's
// imports are type-only, so it runs standalone). Shapes and expected
// outcomes follow the TS suite
// `vendor-src/oh-my-pi/packages/coding-agent/test/collab/
// replication-shrink.test.ts` (issues #3739/#11433, PR #11999 review).

#[cfg(test)]
mod tests {
    use super::*;
    use Value::String as S;

    /// Dropping a deeply nested `Value` recurses once per level and can
    /// overflow the test stack, so deep fixtures are dismantled iteratively.
    fn drop_deeply_nested(root: Value) {
        let mut stack: Vec<Value> = vec![root];
        while let Some(mut value) = stack.pop() {
            match &mut value {
                Value::Array(items) => stack.append(items),
                Value::Object(map) => stack.extend(map.values_mut().map(std::mem::take)),
                _ => {}
            }
            drop(value);
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        ring::digest::digest(&ring::digest::SHA256, bytes)
            .as_ref()
            .iter()
            .fold(String::new(), |mut out, byte| {
                let _ = write!(out, "{byte:02x}");
                out
            })
    }

    fn user_message(id: &str, parent_id: &Value, timestamp: &str, content: &Value) -> Value {
        serde_json::json!({
            "type": "message",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "message": { "role": "user", "content": content, "timestamp": 0 },
        })
    }

    /// An entry whose bulk lives in a `blob` object under a single giant
    /// string key (keys are identity — unshrinkable).
    fn giant_key_blob_entry(id: &str, parent_id: &str, key_len: usize) -> Value {
        let mut blob = Map::new();
        blob.insert("k".repeat(key_len), Value::from(1));
        serde_json::json!({
            "type": "message",
            "id": id,
            "parentId": parent_id,
            "timestamp": "2026-09-13T00:00:00Z",
            "message": { "role": "user", "content": "", "timestamp": 0, "blob": blob },
        })
    }

    /// The small synthetic session the TS oracle ran: every entry fits the
    /// ceiling, so each ships verbatim and the port's JSON must equal
    /// `JSON.stringify`'s byte for byte.
    fn synthetic_session() -> Vec<Value> {
        vec![
            user_message(
                "e1",
                &Value::Null,
                "2026-09-13T00:00:00Z",
                &S("hello world".into()),
            ),
            serde_json::json!({
                "type": "message",
                "id": "e2",
                "parentId": "e1",
                "timestamp": "2026-09-13T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": "hi there" },
                        { "type": "thinking", "thinking": "hmm", "checksum": "abc" },
                    ],
                    "timestamp": 1_726_176_001_000_u64,
                    "model": "claude",
                    "usage": { "input": 10, "output": 20, "cost": 0.5 },
                },
            }),
            serde_json::json!({
                "type": "custom_message",
                "id": "e3",
                "parentId": "e2",
                "timestamp": "2026-09-13T00:00:02Z",
                "customType": "my-ext",
                "display": true,
                "details": {
                    "at": "2026-09-13T12:00:00.000Z",
                    "label": "x",
                    "nested": { "a": [1, 2, 3] },
                },
            }),
            serde_json::json!({
                "type": "model_change",
                "id": "e4",
                "parentId": "e3",
                "timestamp": "2026-09-13T00:00:03Z",
                "model": "gpt",
            }),
        ]
    }

    const GOLDEN_SESSION: [&str; 4] = [
        r#"{"type":"message","id":"e1","parentId":null,"timestamp":"2026-09-13T00:00:00Z","message":{"role":"user","content":"hello world","timestamp":0}}"#,
        r#"{"type":"message","id":"e2","parentId":"e1","timestamp":"2026-09-13T00:00:01Z","message":{"role":"assistant","content":[{"type":"text","text":"hi there"},{"type":"thinking","thinking":"hmm","checksum":"abc"}],"timestamp":1726176001000,"model":"claude","usage":{"input":10,"output":20,"cost":0.5}}}"#,
        r#"{"type":"custom_message","id":"e3","parentId":"e2","timestamp":"2026-09-13T00:00:02Z","customType":"my-ext","display":true,"details":{"at":"2026-09-13T12:00:00.000Z","label":"x","nested":{"a":[1,2,3]}}}"#,
        r#"{"type":"model_change","id":"e4","parentId":"e3","timestamp":"2026-09-13T00:00:03Z","model":"gpt"}"#,
    ];

    /// TS: "does not substitute a placeholder for an entry that already
    /// fits" + the golden byte layout of a small synthetic session,
    /// verified against a bun run of the TS module.
    #[test]
    fn fitting_entries_ship_verbatim_with_ts_byte_layout() {
        for (entry, golden) in synthetic_session().into_iter().zip(GOLDEN_SESSION) {
            let bounded = serialize_replicated_entry(&entry);
            assert_eq!(bounded.json, golden);
            assert_eq!(bounded.value, entry, "fitting entry ships by reference");
            // Serialize-exactly-once contract: the frame JSON re-parses to
            // the same payload (unpack is the exact inverse).
            let unpacked: Value = serde_json::from_str(&bounded.json).expect("frame json parses");
            assert_eq!(unpacked, entry);
        }
    }

    /// The iterative serializer must agree with `serde_json`'s recursive one
    /// on every ordinary payload — the crate's other JSON surfaces go
    /// through `serde_json` directly.
    #[test]
    fn iterative_serializer_matches_serde_json() {
        let entries = synthetic_session();
        let event = serde_json::json!({ "type": "agent_end", "messages": [] });
        for value in entries.iter().chain(std::iter::once(&event)) {
            assert_eq!(
                to_json_string(value),
                serde_json::to_string(value).expect("serializable")
            );
        }
    }

    /// TS: "does not substitute a notice for an event that already fits".
    #[test]
    fn fitting_event_ships_verbatim() {
        let event = serde_json::json!({ "type": "agent_end", "messages": [] });
        let shrunk = serialize_replicated_event(&event);
        assert_eq!(shrunk.value, event);
        assert_eq!(shrunk.json, r#"{"type":"agent_end","messages":[]}"#);
        assert!(!shrunk.json.contains("Host event omitted"));
    }

    /// TS: "clamps a single giant string under the ceiling with an elision
    /// marker" — 5 MB of `x` head-truncated to the first pass's 64 KB cap;
    /// sha256 over the exact UTF-8 bytes bun produced.
    #[test]
    fn giant_string_clamped_under_ceiling() {
        let entry = user_message(
            "m1",
            &Value::Null,
            "2026-09-13T00:00:00Z",
            &S("x".repeat(5 * 1024 * 1024)),
        );
        let shrunk = serialize_replicated_entry(&entry);
        assert_ne!(shrunk.value, entry);
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
        // 65536 − 80 reserve = 65456 kept; 5242880 − 65456 = 5177424 elided.
        assert!(shrunk
            .json
            .contains("…[5177424 chars elided for collab session]"));
        assert_eq!(shrunk.json.len(), 65_634);
        assert_eq!(
            sha256_hex(shrunk.json.as_bytes()),
            "1458f7b9948630361c9e6b68859b3fdc0363917bb9055cf107d1152865f77c01"
        );
        assert!(shrunk.json.starts_with(
            r#"{"type":"message","id":"m1","parentId":null,"timestamp":"2026-09-13T00:00:00Z","message":{"role":"user","content":"xxxxxxxx"#,
        ));
    }

    /// TS: "clamps a payload built of many short strings (no individual
    /// oversized)" — 100 000 small blocks; the array head-clips to the first
    /// pass's 256 with the elision summary. sha256 over bun's exact bytes.
    #[test]
    fn many_short_strings_clamp_the_array_not_the_strings() {
        let blocks: Vec<Value> = (0..100_000)
            .map(|i| serde_json::json!({ "type": "text", "text": format!("block-{i}") }))
            .collect();
        let entry = serde_json::json!({
            "type": "message",
            "id": "m1",
            "parentId": null,
            "timestamp": "2026-09-13T00:00:00Z",
            "message": { "role": "toolResult", "content": blocks, "timestamp": 0 },
        });
        assert!(replication_byte_length(&entry) > MAX_REPLICATED_PAYLOAD_BYTES);
        let shrunk = serialize_replicated_entry(&entry);
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
        assert!(shrunk
            .json
            .contains("…[99744 items elided for collab session]"));
        assert_eq!(shrunk.json.len(), 9_032);
        assert_eq!(
            sha256_hex(shrunk.json.as_bytes()),
            "66074d7f44a59c8d9f6e9a369fbf7dff74d3ed975f8be7875dba20e960a59cbc"
        );
        assert!(shrunk.json.ends_with(
            r#","text":"block-255"},"…[99744 items elided for collab session]"],"timestamp":0}}"#,
        ));
    }

    /// TS: "replaces an entry whose size lives in one giant object key" —
    /// keys are identity, every pass returns the payload untouched, so the
    /// typed placeholder keeps `id`/`parentId` (golden: exact placeholder
    /// bytes).
    #[test]
    fn giant_object_key_replaced_by_placeholder() {
        // TS fixture used a 17 MiB key; the unshrinkability boundary is
        // "over the ceiling after every pass", which 2 MiB clears.
        let entry = giant_key_blob_entry("m1", "parent-1", 2 * 1024 * 1024);
        let shrunk = serialize_replicated_entry(&entry);
        assert_eq!(
            shrunk.json,
            r#"{"type":"custom_message","id":"m1","parentId":"parent-1","timestamp":"2026-09-13T00:00:00Z","customType":"collab-entry-too-large","display":true,"content":"…[message entry, 2097304 bytes omitted for collab session: too large to replicate]"}"#,
        );
        assert_eq!(
            shrunk.value["customType"],
            S(COLLAB_ENTRY_OMITTED_CUSTOM_TYPE.into())
        );
        assert_eq!(shrunk.value["id"], S("m1".into()));
        assert_eq!(shrunk.value["parentId"], S("parent-1".into()));
        assert_eq!(shrunk.value["display"], Value::Bool(true));
    }

    /// TS: "replaces an entry whose size lives in the number of its keys" —
    /// 200 000 short keys (~3.1 MB) ship as the placeholder.
    #[test]
    fn many_keys_replaced_by_placeholder() {
        let mut blob = Map::new();
        for i in 0..200_000 {
            blob.insert(format!("k{i}"), Value::from(i));
        }
        let entry = serde_json::json!({
            "type": "message",
            "id": "m2",
            "parentId": "parent-2",
            "timestamp": "2026-09-13T00:00:00Z",
            "message": { "role": "user", "content": "", "timestamp": 0, "blob": blob },
        });
        assert_eq!(replication_byte_length(&entry), 3_177_927);
        let shrunk = serialize_replicated_entry(&entry);
        assert_eq!(
            shrunk.json,
            r#"{"type":"custom_message","id":"m2","parentId":"parent-2","timestamp":"2026-09-13T00:00:00Z","customType":"collab-entry-too-large","display":true,"content":"…[message entry, 3177927 bytes omitted for collab session: too large to replicate]"}"#,
        );
        assert_eq!(shrunk.value["id"], S("m2".into()));
    }

    /// TS: "degrades a 50 000-deep entry instead of blowing the stack" —
    /// pre-fix the recursive walk threw `RangeError` and every caller
    /// swallowed it, silently dropping the entry (or stranding the guest
    /// without a `final` chunk). The port degrades the deep branch at the
    /// same nesting level (depth 1000) with the same marker, so the sha256
    /// over bun's bytes matches.
    #[test]
    fn deep_entry_degrades_instead_of_blowing_the_stack() {
        // Built with plain `Map` moves: the `json!` macro serializes its
        // interpolated expressions, which recurses once per level and is
        // exactly the stack hazard this test exists to pin.
        let mut deep = Value::String("leaf".into());
        for _ in 0..50_000 {
            let mut level = Map::new();
            level.insert("a".into(), deep);
            deep = Value::Object(level);
        }
        let mut message = Map::new();
        message.insert("role".into(), Value::from("user"));
        message.insert("content".into(), Value::from(""));
        message.insert("timestamp".into(), Value::from(0));
        message.insert("blob".into(), deep);
        let mut entry_map = Map::new();
        entry_map.insert("type".into(), Value::from("message"));
        entry_map.insert("id".into(), Value::from("m3"));
        entry_map.insert("parentId".into(), Value::from("parent-3"));
        entry_map.insert("timestamp".into(), Value::from("2026-09-13T00:00:00Z"));
        entry_map.insert("message".into(), Value::Object(message));
        let entry = Value::Object(entry_map);
        let shrunk = serialize_replicated_entry(&entry);
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
        assert_eq!(shrunk.value["id"], S("m3".into()));
        assert_eq!(shrunk.value["parentId"], S("parent-3".into()));
        assert!(shrunk.json.contains(DEPTH_ELISION_MARKER));
        // Not one of the typed placeholders: the entry survives as a
        // message and its depth is re-serializable.
        assert_eq!(shrunk.value["type"], S("message".into()));
        assert_eq!(
            sha256_hex(shrunk.json.as_bytes()),
            "385486a8d8578a7a46f5f3ddd061bbeba7823781183af86e7d8a6312be798ae3",
        );
        drop_deeply_nested(shrunk.value);
        drop_deeply_nested(entry);
    }

    /// TS: "measures the ceiling in UTF-8 bytes, not UTF-16 code units" —
    /// 400 000 CJK chars are ~400 KB of code units (under the ceiling for
    /// the old check) but ~1.2 MB of UTF-8 (over it for the relay).
    #[test]
    fn ceiling_measured_in_utf8_bytes() {
        let entry = user_message(
            "m4",
            &Value::Null,
            "2026-09-13T00:00:00Z",
            &S("中".repeat(400_000)),
        );
        let utf8_len = replication_byte_length(&entry);
        assert_eq!(utf8_len, 1_200_132);
        assert!(utf8_len > MAX_REPLICATED_PAYLOAD_BYTES);
        let shrunk = serialize_replicated_entry(&entry);
        assert_ne!(shrunk.value, entry);
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
        assert_eq!(
            sha256_hex(shrunk.json.as_bytes()),
            "f69ee1fb397a1fe6f5ea85e90053c743fefda45f72b2572321807249814f7ef6",
        );
    }

    /// TS: "degrades a cyclic payload to a marker instead of looping
    /// forever". A `serde_json::Value` cannot be cyclic, so the TS `null`
    /// measurement and marker are unrepresentable; the port pins the
    /// observable contract that survives — measurement never fails, and the
    /// cycle marker constant keeps its exact TS bytes.
    #[test]
    fn cycle_marker_constant_pins_ts_bytes() {
        let entry = user_message("m5", &Value::Null, "2026-09-13T00:00:00Z", &S("hi".into()));
        assert_eq!(
            replication_byte_length(&entry),
            to_json_string(&entry).len()
        );
        assert_eq!(
            CYCLE_ELISION_MARKER,
            "…[cyclic reference elided for collab session]"
        );
    }

    /// TS: "keeps the guest's branch connected across an omitted entry" —
    /// the module-level half (id/parentId survival; the `SessionManager`
    /// branch walk lives in omp-session): prior → placeholder → successor
    /// stays a chain a replica loader can walk.
    #[test]
    fn branch_chain_survives_an_omitted_entry() {
        let prior = user_message(
            "prior",
            &Value::Null,
            "2026-09-13T00:00:00Z",
            &S("before".into()),
        );
        let successor = user_message(
            "successor",
            &S("omitted".into()),
            "2026-09-13T00:00:01Z",
            &S("after".into()),
        );
        let placeholder =
            serialize_replicated_entry(&giant_key_blob_entry("omitted", "prior", 2 * 1024 * 1024))
                .value;
        assert_eq!(placeholder["id"], S("omitted".into()));
        assert_eq!(placeholder["parentId"], S("prior".into()));
        assert_eq!(placeholder["parentId"], prior["id"]);
        assert_eq!(successor["parentId"], placeholder["id"]);
    }

    /// TS: "still shrinks an event whose size lives in a string leaf".
    #[test]
    fn oversized_event_string_leaf_shrinks_in_place() {
        let event = serde_json::json!({
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "read",
            "result": { "text": "x".repeat(8 * 1024 * 1024) },
        });
        let shrunk = serialize_replicated_event(&event);
        assert_ne!(shrunk.value, event);
        assert_eq!(shrunk.value["type"], S("tool_execution_end".into()));
        assert_eq!(shrunk.value["toolCallId"], S("call-1".into()));
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
        assert!(!shrunk.json.contains("Host event omitted"));
    }

    /// TS: "replaces an unshrinkable event with a notice naming the omitted
    /// type" (golden: exact notice bytes).
    #[test]
    fn unshrinkable_event_replaced_by_notice() {
        let mut result = Map::new();
        result.insert("k".repeat(2 * 1024 * 1024), Value::from(1));
        let event = serde_json::json!({
            "type": "tool_execution_end",
            "toolCallId": "call-2",
            "toolName": "read",
            "result": result,
        });
        let shrunk = serialize_replicated_event(&event);
        assert_eq!(
            shrunk.json,
            r#"{"type":"notice","level":"warning","source":"collab","message":"Host event omitted: too large to replicate (tool_execution_end)."}"#,
        );
        assert_eq!(shrunk.value["level"], S("warning".into()));
    }

    /// Golden for `oversizedEntryNotice` (host.ts:479-486 sends it
    /// alongside a placeholder substitution on the live path).
    #[test]
    fn oversized_entry_notice_matches_ts_bytes() {
        let notice = oversized_entry_notice("message");
        assert_eq!(
            to_json_string(&notice),
            r#"{"type":"notice","level":"warning","source":"collab","message":"Host entry omitted: too large to replicate (message)."}"#,
        );
    }

    /// TS PR #11999 review: "preserves own `__proto__` keys as data" and
    /// "honors own `toJSON` on snapshot leaves". In the `Value` domain both
    /// collapse to "the clone serializes to the same bytes" — the port pins
    /// bun's exact output for the two TS golden shapes.
    #[test]
    fn copy_for_replication_json_contract() {
        let proto = serde_json::json!({
            "__proto__": { "injected": true },
            "normal": "metadata",
        });
        assert_eq!(
            to_json_string(&copy_for_replication(&proto)),
            r#"{"__proto__":{"injected":true},"normal":"metadata"}"#,
        );
        let nested = serde_json::json!({ "details": proto });
        assert!(to_json_string(&copy_for_replication(&nested)).contains("injected"));

        // The TS fixture ships a `Date` through its `toJSON`; a `Value`
        // already holds the serialized string, and the bytes must match.
        let date_details =
            serde_json::json!({ "details": { "at": "2026-09-13T12:00:00.000Z", "label": "x" } });
        assert_eq!(
            to_json_string(&copy_for_replication(&date_details)),
            r#"{"details":{"at":"2026-09-13T12:00:00.000Z","label":"x"}}"#,
        );
    }

    /// Threshold boundary, exactly at the ceiling: bun pinned
    /// `padLen = 1048444`, total JSON = 1 048 576 bytes = the ceiling →
    /// ships verbatim; one char over (1 048 577) → the first pass clips.
    #[test]
    fn ceiling_boundaries_exactly_at_and_one_over() {
        let empty = user_message(
            "m1",
            &Value::Null,
            "2026-09-13T00:00:00Z",
            &S(String::new()),
        );
        let pad_len = MAX_REPLICATED_PAYLOAD_BYTES - replication_byte_length(&empty);
        assert_eq!(pad_len, 1_048_444);

        let exact = user_message(
            "m1",
            &Value::Null,
            "2026-09-13T00:00:00Z",
            &S("x".repeat(pad_len)),
        );
        assert_eq!(
            replication_byte_length(&exact),
            MAX_REPLICATED_PAYLOAD_BYTES
        );
        let bounded = serialize_replicated_entry(&exact);
        assert_eq!(
            bounded.value, exact,
            "exactly at the ceiling still ships by reference"
        );
        assert!(!bounded.json.contains("elided for collab session"));

        let over = user_message(
            "m1",
            &Value::Null,
            "2026-09-13T00:00:00Z",
            &S("x".repeat(pad_len + 1)),
        );
        assert_eq!(
            replication_byte_length(&over),
            MAX_REPLICATED_PAYLOAD_BYTES + 1
        );
        let shrunk = serialize_replicated_entry(&over);
        assert_ne!(shrunk.value, over);
        // 1048445 chars in, 65456 kept → 982989 elided, first pass.
        assert!(shrunk
            .json
            .contains("…[982989 chars elided for collab session]"));
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
    }

    /// String-cap boundary inside the shrink schedule: a string of exactly
    /// `stringCap` is returned by reference (`<=`), one char over clips with
    /// the exact dropped count, and each pass measures against the original
    /// (never a previously clipped output). Observed through pass 2 (16 KB
    /// cap) by padding the payload over the ceiling with such strings.
    #[test]
    fn string_cap_boundaries_exactly_at_and_one_over() {
        let strings_entry = |count: usize, unit_len: usize| {
            let content: Vec<Value> = (0..count).map(|_| S("x".repeat(unit_len))).collect();
            serde_json::json!({
                "type": "message",
                "id": "m1",
                "parentId": null,
                "timestamp": "2026-09-13T00:00:00Z",
                "message": { "role": "user", "content": content, "timestamp": 0 },
            })
        };
        // 20 × 65536 bytes of strings: over the ceiling, but every string
        // equals pass 1's 64 KB cap (untouched — `<=`) and 20 ≤ 256 items,
        // so only pass 2 (16 KB) clips: 65536 − (16384 − 80) = 49232.
        let entry = strings_entry(20, 64 * 1024);
        assert!(replication_byte_length(&entry) > MAX_REPLICATED_PAYLOAD_BYTES);
        let shrunk = serialize_replicated_entry(&entry);
        let marker = "…[49232 chars elided for collab session]";
        assert_eq!(shrunk.json.matches(marker).count(), 20);
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);

        // 64 × 16385: pass 1 leaves them under its 64 KB cap and the payload
        // over the ceiling; pass 2 clips exactly one char each → 81 elided.
        let entry = strings_entry(64, 16 * 1024 + 1);
        assert!(replication_byte_length(&entry) > MAX_REPLICATED_PAYLOAD_BYTES);
        let shrunk = serialize_replicated_entry(&entry);
        assert_eq!(
            shrunk
                .json
                .matches("…[81 chars elided for collab session]")
                .count(),
            64
        );
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);

        // Passes restart from the original: pass 2 must count 49233 elided
        // from the 65537-char source, not 49152 from pass 1's clipped
        // 65456-char output.
        let entry = strings_entry(20, 64 * 1024 + 1);
        let shrunk = serialize_replicated_entry(&entry);
        assert_eq!(
            shrunk
                .json
                .matches("…[49233 chars elided for collab session]")
                .count(),
            20
        );
        assert!(!shrunk.json.contains("49152 chars elided"));
    }

    /// Array-limit boundary inside the shrink schedule: exactly
    /// `arrayLimit` items survive with no summary element; the next pass
    /// head-clips and appends the exact elided count. Observed on pass 2
    /// (limit 128) by padding the payload over the ceiling with 256 strings
    /// small enough that only the array ever needs clipping.
    #[test]
    fn array_limit_boundaries_exactly_at_and_one_over() {
        let strings_entry = |count: usize, unit_len: usize| {
            let content: Vec<Value> = (0..count).map(|_| S("x".repeat(unit_len))).collect();
            serde_json::json!({
                "type": "message",
                "id": "m1",
                "parentId": null,
                "timestamp": "2026-09-13T00:00:00Z",
                "message": { "role": "user", "content": content, "timestamp": 0 },
            })
        };
        // 256 × 5 KB: over the ceiling; pass 1's array limit is exactly 256
        // → every item kept, no `items elided` marker possible, and 5 KB
        // strings fit every cap — so only pass 2 (limit 128) clips.
        let entry = strings_entry(256, 5 * 1024);
        assert!(replication_byte_length(&entry) > MAX_REPLICATED_PAYLOAD_BYTES);
        let shrunk = serialize_replicated_entry(&entry);
        assert!(shrunk
            .json
            .contains("…[128 items elided for collab session]"));
        assert!(
            !shrunk.json.contains("chars elided for collab session"),
            "5 KB strings fit every cap"
        );
        let content = shrunk.value["message"]["content"]
            .as_array()
            .expect("array content");
        assert_eq!(content.len(), 129); // 128 kept + summary element
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);

        // 257 × 4080: one item over pass 1's limit and over the ceiling;
        // the clipped 256-item output fits, so pass 1 alone clips exactly
        // one item (TS wording has no singular form).
        let entry = strings_entry(257, 4080);
        assert!(replication_byte_length(&entry) > MAX_REPLICATED_PAYLOAD_BYTES);
        let shrunk = serialize_replicated_entry(&entry);
        assert!(
            shrunk.json.contains("…[1 items elided for collab session]"),
            "TS has no singular form"
        );
        assert!(replication_byte_length(&shrunk.value) <= MAX_REPLICATED_PAYLOAD_BYTES);
    }

    /// host.ts:898-926 `#snapshotChunks` — the ~3s/512 KB soft cap. Every
    /// batch carries at least one entry (the `i > start` guard), the last
    /// batch is final, an empty snapshot still emits one final chunk, and a
    /// single oversized entry ships in a chunk of its own.
    #[test]
    fn snapshot_chunk_boundaries() {
        let entry_json = |n: usize| "e".repeat(n);
        // Empty snapshot → one final terminator.
        assert_eq!(
            snapshot_chunks(&[], &[]),
            vec![SnapshotChunk {
                entries: vec![],
                is_final: true
            }]
        );

        // Exactly at the cap: 524287 + 1 = 524288 stays one batch.
        let entries = vec![entry_json(524_287), entry_json(1)];
        let bytes: Vec<usize> = entries.iter().map(String::len).collect();
        let chunks = snapshot_chunks(&entries, &bytes);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
        assert_eq!(chunks[0].entries, entries);

        // One byte over: 524288 + 1 splits; only the last batch is final.
        let entries = vec![entry_json(524_288), entry_json(1)];
        let bytes: Vec<usize> = entries.iter().map(String::len).collect();
        let chunks = snapshot_chunks(&entries, &bytes);
        assert_eq!(chunks.len(), 2);
        assert!(!chunks[0].is_final);
        assert_eq!(chunks[0].entries, vec![entry_json(524_288)]);
        assert!(chunks[1].is_final);
        assert_eq!(chunks[1].entries, vec![entry_json(1)]);

        // An oversized single entry ships alone in its own chunk (issue
        // #3739): 600 KB forces its own batch; the two 10 B entries then
        // share the final batch.
        let entries = vec![entry_json(600_000), entry_json(10), entry_json(10)];
        let bytes: Vec<usize> = entries.iter().map(String::len).collect();
        let chunks = snapshot_chunks(&entries, &bytes);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].entries, vec![entry_json(600_000)]);
        assert!(!chunks[0].is_final);
        assert_eq!(chunks[1].entries, vec![entry_json(10), entry_json(10)]);
        assert!(chunks[1].is_final);

        // Three 300 KB entries: nothing ever fits twice.
        let entries = vec![
            entry_json(300_000),
            entry_json(300_000),
            entry_json(300_000),
        ];
        let bytes: Vec<usize> = entries.iter().map(String::len).collect();
        let chunks = snapshot_chunks(&entries, &bytes);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.entries.len())
                .collect::<Vec<_>>(),
            vec![1, 1, 1],
        );
        assert_eq!(chunks.last().map(|chunk| chunk.is_final), Some(true));

        // Missing byte counts degrade to 0 (`bytes[i] ?? 0`).
        let entries = vec![entry_json(4), entry_json(4)];
        let chunks = snapshot_chunks(&entries, &[]);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
    }

    /// `copyForReplication` bounds depth without clipping anything: a chain
    /// past [`MAX_REPLICATED_DEPTH`] degrades to the marker, a fitting
    /// payload comes back byte-identical, and the copy's depth is
    /// re-serializable (the point of host.ts:868-877's walk before
    /// `serializeReplicatedEntry`).
    #[test]
    fn copy_for_replication_bounds_depth_only() {
        let value = serde_json::json!({ "a": [1, "x".repeat(300), { "b": null }] });
        let copy = copy_for_replication(&value);
        assert_eq!(to_json_string(&copy), to_json_string(&value));

        // Same construction constraint as the deep-entry test above: plain
        // `Map` moves, never `json!` interpolation of a deep value.
        let mut deep = Value::String("leaf".into());
        for _ in 0..50_000 {
            let mut level = Map::new();
            level.insert("a".into(), deep);
            deep = Value::Object(level);
        }
        let copy = copy_for_replication(&deep);
        assert!(to_json_string(&copy).contains(DEPTH_ELISION_MARKER));
        assert!(!depth_at_or_beyond(&copy, MAX_REPLICATED_DEPTH));
        drop_deeply_nested(copy);
        drop_deeply_nested(deep);
    }

    /// Downstream seal/JSONL writers rely on `SerializedReplication.json`
    /// equaling what `serde_json` emits for `value` — for verbatim ships and
    /// placeholders alike.
    #[test]
    fn serialized_json_equals_serde_json_of_value() {
        let entry = &synthetic_session()[1];
        let bounded = serialize_replicated_entry(entry);
        assert_eq!(
            bounded.json,
            serde_json::to_string(&bounded.value).expect("serializable")
        );
        let shrunk = serialize_replicated_entry(&giant_key_blob_entry("m1", "p", 2 * 1024 * 1024));
        assert_eq!(
            shrunk.json,
            serde_json::to_string(&shrunk.value).expect("serializable")
        );
    }
}
