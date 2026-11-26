//! Acceptance tests for the interface-projection and
//! location-relocation phase: coherence bucket isolation, positive
//! and negative cache behaviour, content-anchored negative
//! invalidation, cycle-key isolation, cfg/macro guard propagation,
//! the write-once escaped wire arena, and the observable-vs-diagnostic
//! location split with late symbolic resolution.

use omp_ladder::{
    cycle_key, hash_str, loc_dep_hash, normalize_error_sig, parse_top_level, project,
    project_interface, relocate, resolve_use, scan_loc_uses, BucketKind, CfgContext, CycleStore,
    GuardMiss, GuardStore, GuardVerdict, InterfaceCache, ItemKind, LocClass, LocMacro, Span,
    TrackedSpan, Visibility, WireArena, WireError,
};

fn ctx_with_feature(on: bool) -> CfgContext {
    let mut c = CfgContext::default();
    if on {
        c.kv.insert("feature".to_owned(), "x".to_owned());
    }
    c
}

fn project_src(src: &str, ctx: &CfgContext, workspace: &str) -> omp_ladder::InterfaceProjection {
    let cst = parse_top_level(src);
    let index = project(&cst, ctx);
    project_interface(&index, &cst, hash_str(workspace))
}

// ---------- coherence bucket isolation ----------

#[test]
fn only_unconditional_public_items_enter_buckets() {
    let src = "\
pub struct PublicTy;
struct PrivateTy;
pub(crate) struct RestrictedTy;
crate struct CrateVisTy;
pub enum PublicEnum {}
fn private_fn() {}
pub fn public_fn() {}
pub const PUBLIC_CONST: u8 = 0;
const PRIVATE_CONST: u8 = 0;
pub trait PublicTrait {}
trait PrivateTrait {}
pub mod public_mod {}
mod private_mod {}
impl PublicTy {}
";
    let proj = project_src(src, &CfgContext::default(), "ws-a");

    let types = proj.bucket(BucketKind::Types).expect("types bucket");
    let names: Vec<&str> = types.entries.iter().filter_map(|e| e.name.as_deref()).collect();
    assert_eq!(names, ["PublicTy", "PublicEnum"]);
    assert!(
        !proj.has_name(BucketKind::Types, "PrivateTy"),
        "private item leaked into global projection"
    );
    assert!(
        !proj.has_name(BucketKind::Types, "RestrictedTy"),
        "pub(crate) item leaked into global projection"
    );
    assert!(!proj.has_name(BucketKind::Types, "CrateVisTy"));

    let values = proj.bucket(BucketKind::Values).expect("values bucket");
    let vnames: Vec<&str> = values.entries.iter().filter_map(|e| e.name.as_deref()).collect();
    assert_eq!(vnames, ["public_fn", "PUBLIC_CONST"]);

    let traits = proj.bucket(BucketKind::Traits).expect("traits bucket");
    assert_eq!(traits.entries.len(), 1);
    assert_eq!(traits.entries[0].name.as_deref(), Some("PublicTrait"));

    // impl blocks are unnamed but coherence-relevant: they enter the
    // Impls bucket keyed by content, not by name.
    let impls = proj.bucket(BucketKind::Impls).expect("impls bucket");
    assert_eq!(impls.entries.len(), 1);
    assert_eq!(impls.entries[0].name, None);
    assert_eq!(impls.entries[0].kind, ItemKind::Impl);
}

#[test]
fn visibility_is_recorded_syntactically() {
    let cst = parse_top_level("pub fn a() {}\npub(crate) fn b() {}\nfn c() {}\ncrate fn d() {}\n");
    let index = project(&cst, &CfgContext::default());
    let vis = |n: &str| {
        index
            .items
            .iter()
            .find(|i| i.name.as_deref() == Some(n))
            .map(|i| i.visibility)
    };
    assert_eq!(vis("a"), Some(Visibility::Pub));
    assert_eq!(vis("b"), Some(Visibility::PubRestricted));
    assert_eq!(vis("c"), Some(Visibility::Inherited));
    assert_eq!(vis("d"), Some(Visibility::PubRestricted));
}

#[test]
fn pub_use_marks_projection_incomplete_and_guards_fail_closed() {
    let src = "pub struct A;\npub use std::fmt::Debug;\n";
    let proj = project_src(src, &CfgContext::default(), "ws-a");
    assert!(proj.incomplete, "pub use must mark the projection incomplete");
    // The name is verifiably present, yet no absence may be asserted
    // anywhere: guards refuse to record and refuse to answer.
    assert!(proj.has_name(BucketKind::Types, "A"));
    let mut guards = GuardStore::new();
    assert_eq!(guards.record(&proj, BucketKind::Values, "anything"), None);
    assert_eq!(
        guards.check(&proj, BucketKind::Values, "anything"),
        GuardVerdict::Miss(GuardMiss::Incomplete)
    );
}

// ---------- negative guards: hits, misses, added-name invalidation ----------

#[test]
fn negative_guard_hit_and_added_name_invalidation() {
    let ws = "ws-a";
    let before = project_src("pub struct A;\n", &CfgContext::default(), ws);
    let mut guards = GuardStore::new();

    // Record a verified absence.
    let g = guards
        .record(&before, BucketKind::Types, "Missing")
        .expect("absence is verifiable");
    assert_eq!(g.bucket, BucketKind::Types);
    assert_eq!(g.name, "Missing");
    assert_eq!(
        guards.check(&before, BucketKind::Types, "Missing"),
        GuardVerdict::Hit
    );

    // No guard recorded -> miss.
    assert_eq!(
        guards.check(&before, BucketKind::Types, "Other"),
        GuardVerdict::Miss(GuardMiss::NoGuard)
    );

    // Adding a *matching* item invalidates the guard (content anchor
    // moves), and the name is now positively present.
    let after = project_src("pub struct A;\npub struct Missing;\n", &CfgContext::default(), ws);
    assert_eq!(
        guards.check(&after, BucketKind::Types, "Missing"),
        GuardVerdict::Miss(GuardMiss::NameNowPresent)
    );

    // Adding an unrelated item also moves the bucket anchor: the
    // guard is stale and must be re-derived, not silently trusted.
    let edited = project_src("pub struct A;\npub struct B;\n", &CfgContext::default(), ws);
    assert_eq!(
        guards.check(&edited, BucketKind::Types, "Missing"),
        GuardVerdict::Miss(GuardMiss::AnchorChanged)
    );

    // A guard from another workspace never answers here.
    let other_ws = project_src("pub struct A;\n", &CfgContext::default(), "ws-b");
    assert_eq!(
        guards.check(&other_ws, BucketKind::Types, "Missing"),
        GuardVerdict::Miss(GuardMiss::WrongWorkspace)
    );
}

#[test]
fn guard_record_refuses_present_names() {
    let proj = project_src("pub struct A;\n", &CfgContext::default(), "ws-a");
    let mut guards = GuardStore::new();
    assert_eq!(guards.record(&proj, BucketKind::Types, "A"), None);
    assert!(guards.is_empty());
}

// ---------- interface cache: positive hits/misses, cfg/macro/workspace/loc ----------

#[test]
fn interface_cache_hits_and_guard_propagation() {
    let src = "#[cfg(feature = \"x\")]\npub struct Gated;\npub struct Plain;\n";
    let ws = hash_str("ws-a");
    let mut cache = InterfaceCache::new(16);

    let off = cache.get_or_project(&parse_top_level(src), &ctx_with_feature(false), ws);
    assert_eq!(cache.stats().misses, 1);
    // cfg guard propagated: the cfg-disabled public item is not in
    // the global projection.
    assert!(!off.has_name(BucketKind::Types, "Gated"));
    assert!(off.has_name(BucketKind::Types, "Plain"));

    // Same everything -> warm hit, same allocation.
    let cst = parse_top_level(src);
    let warm = cache.get_or_project(&cst, &ctx_with_feature(false), ws);
    assert_eq!(cache.stats().hits, 1);
    assert!(std::sync::Arc::ptr_eq(&off, &warm));

    // cfg context flip -> miss; the gated item now appears.
    let on = cache.get_or_project(&cst, &ctx_with_feature(true), ws);
    assert_eq!(cache.stats().misses, 2);
    assert!(on.has_name(BucketKind::Types, "Gated"));

    // Same text, different workspace -> miss (no cross-workspace
    // collision).
    let _other = cache.get_or_project(&cst, &ctx_with_feature(false), hash_str("ws-b"));
    assert_eq!(cache.stats().misses, 3);
}

#[test]
fn macro_env_change_invalidates_interface_cache() {
    let ws = hash_str("ws-a");
    let mut cache = InterfaceCache::new(16);
    let src1 = "macro_rules! m { () => {} }\npub fn f() {}\n";
    let src2 = "macro_rules! m { () => { let _ = 1; } }\npub fn f() {}\n";
    let _a = cache.get_or_project(&parse_top_level(src1), &CfgContext::default(), ws);
    let _b = cache.get_or_project(&parse_top_level(src2), &CfgContext::default(), ws);
    assert_eq!(
        cache.stats().misses,
        2,
        "a macro-definition edit must re-project, not hit"
    );
}

// ---------- observable locations: symbolic, late-resolved, invalidating ----------

#[test]
fn observable_uses_are_symbolic_and_late_resolved() {
    let src = "pub fn f() {\n    let _ = line!();\n    let _ = file!();\n}\n";
    let cst = parse_top_level(src);
    let uses = scan_loc_uses(&cst);
    assert_eq!(uses.len(), 2);
    assert_eq!(uses[0].which, LocMacro::Line);
    assert_eq!(uses[1].which, LocMacro::File);
    let item = uses[0].symbolic.item.expect("enclosed in item f");
    assert_eq!(cst.nodes[item].name.as_deref(), Some("f"));

    // Late resolution against the current source.
    let r = resolve_use(&cst, &uses[0], Some("src/lib.rs"));
    assert_eq!((r.line, r.col), (2, 13));
    assert_eq!(r.file.as_deref(), Some("src/lib.rs"));

    // Edit above the use: the symbolic identity (item anchor uses
    // item-relative offsets) is stable only if the item text is
    // unchanged; the *resolved* value follows the source.
    let moved = format!("// header\n{src}");
    let cst2 = parse_top_level(&moved);
    let uses2 = scan_loc_uses(&cst2);
    let r2 = resolve_use(&cst2, &uses2[0], Some("src/lib.rs"));
    assert_eq!(r2.line, 3, "late resolution sees the shifted line");
    assert_eq!(
        loc_dep_hash(&uses),
        loc_dep_hash(&uses2),
        "item-relative anchor: a pure whole-item shift is the same dependency"
    );
}

#[test]
fn relocating_an_observable_use_invalidates() {
    let a = "pub fn f() {\n    let _ = line!();\n}\n";
    let b = "pub fn f() {\n\n    let _ = line!();\n}\n";
    let ha = loc_dep_hash(&scan_loc_uses(&parse_top_level(a)));
    let hb = loc_dep_hash(&scan_loc_uses(&parse_top_level(b)));
    assert_ne!(
        ha, hb,
        "moving an observable use within its item must change the dependency hash"
    );

    // And the cache enforces it: same item text except the line!
    // position -> miss, never a warm hit on a relocated observable use.
    let ws = hash_str("ws-a");
    let mut cache = InterfaceCache::new(16);
    let _ = cache.get_or_project(&parse_top_level(a), &CfgContext::default(), ws);
    let _ = cache.get_or_project(&parse_top_level(b), &CfgContext::default(), ws);
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 0);
}

#[test]
fn not_every_bang_is_a_location_macro() {
    // `a != b` and non-location macros must not be detected.
    let src = "pub fn f() { let a = 1; let b = 2; let _ = a != b; foo!(); let _ = columns!(); }\n";
    let uses = scan_loc_uses(&parse_top_level(src));
    assert!(uses.is_empty(), "false positives: {uses:?}");
}

#[test]
fn diagnostic_only_spans_relocate_without_semantic_invalidation() {
    let src = "pub fn f() {}\n";
    let cst = parse_top_level(src);
    let dep_before = loc_dep_hash(&scan_loc_uses(&cst));

    let mut diag = TrackedSpan {
        symbolic: omp_ladder::SymbolicSpan {
            item: Some(0),
            span_id: 0,
            anchor: hash_str("diag"),
        },
        class: LocClass::DiagnosticOnly,
        span: Span { start: 0, end: 4 },
    };
    relocate(&mut diag, Span { start: 10, end: 14 }).expect("diagnostic spans move freely");
    assert_eq!(diag.span.start, 10);
    assert_eq!(
        loc_dep_hash(&scan_loc_uses(&cst)),
        dep_before,
        "diagnostic relocation must not touch the observable dependency"
    );

    // Observable spans are pinned: relocating one is refused, so a
    // diagnostic cache can never be warmed by moving a value-bearing
    // location.
    let mut obs = TrackedSpan {
        symbolic: omp_ladder::SymbolicSpan {
            item: Some(0),
            span_id: 1,
            anchor: hash_str("obs"),
        },
        class: LocClass::Observable,
        span: Span { start: 0, end: 5 },
    };
    assert_eq!(
        relocate(&mut obs, Span { start: 3, end: 8 }),
        Err(omp_ladder::RelocateError::ObservablePinned)
    );
    assert_eq!(obs.span.start, 0, "refused relocation leaves the span in place");
}

// ---------- cycle keys ----------

#[test]
fn cycle_key_isolation_and_normalization() {
    let err_a = "error[E0308]: mismatched types at src/main.rs:12:5\nexpected u32, found i32";
    let err_b = "error[E0308]:  mismatched   types at src/main.rs:99:14\nexpected u32, found i32";
    // Cosmetic differences (paths, line numbers, spacing) normalize away.
    assert_eq!(normalize_error_sig(err_a), normalize_error_sig(err_b));

    let k1 = cycle_key("ws-a", 1, err_a);
    let k2 = cycle_key("ws-b", 1, err_a); // different workspace
    let k3 = cycle_key("ws-a", 2, err_a); // different subgoal
    let k4 = cycle_key("ws-a", 1, "error[E9999]: other"); // different error
    assert_ne!(k1, k2);
    assert_ne!(k1, k3);
    assert_ne!(k1, k4);

    let mut store: CycleStore<&str> = CycleStore::new();
    store.insert(k1, "fix-a");
    assert_eq!(store.get(&k1), Some(&"fix-a"));
    assert_eq!(store.get(&k2), None, "identical error text in another workspace must not collide");
    assert_eq!(store.get(&k3), None, "identical error text under another subgoal must not collide");

    // Content-anchored negative constraints.
    let anchor1 = hash_str("projection-v1");
    let anchor2 = hash_str("projection-v2");
    store.record_negative(k2, "Missing", anchor1);
    assert!(store.negative_holds(&k2, "Missing", anchor1));
    assert!(
        !store.negative_holds(&k2, "Missing", anchor2),
        "a changed content anchor must invalidate the negative constraint"
    );
    assert!(
        !store.negative_holds(&k1, "Missing", anchor1),
        "negative constraints are scoped to their full cycle key"
    );
    // A positive insert disproves the negative constraint for that key.
    store.insert(k2, "fix-b");
    assert!(!store.negative_holds(&k2, "Missing", anchor1));
}

// ---------- write-once escaped wire arena ----------

#[test]
fn wire_arena_escapes_and_preserves_json_primitives() {
    let mut arena = WireArena::new(4096);
    let v = serde_json::json!({
        "s": "quote\" backslash\\ newline\n tab\t ctrl\u{1} unicode ✓",
        "t": true,
        "f": false,
        "zero": 0,
        "neg": -1,
        "big": u64::MAX,
        "n": null,
        "arr": [0, false, null, ""],
        "nested": { "k": "v" }
    });
    arena.write_value(&v).unwrap();
    arena.freeze();
    assert!(arena.is_frozen());

    // Round-trip: the wire text parses back to the identical value,
    // with false/0/null preserved.
    let back: serde_json::Value = serde_json::from_str(arena.as_str()).unwrap();
    assert_eq!(back, v);
    assert_eq!(back["f"], serde_json::Value::Bool(false));
    assert_eq!(back["zero"], serde_json::json!(0));
    assert_eq!(back["n"], serde_json::Value::Null);
    assert!(arena.as_str().contains("\\u0001"), "control chars escaped");
    assert!(arena.as_str().contains("✓"), "non-ASCII passes through as UTF-8");

    // Write-once: no append after freeze.
    assert_eq!(arena.write_raw("x"), Err(WireError::Frozen));
    assert_eq!(arena.write_value(&serde_json::json!(1)), Err(WireError::Frozen));
}

#[test]
fn wire_arena_is_bounded_and_transactional() {
    let mut arena = WireArena::new(16);
    arena.write_raw("12345678").unwrap();
    // Overflow: refused, buffer untouched.
    let err = arena.write_escaped_str("this will not fit").unwrap_err();
    assert!(matches!(err, WireError::Overflow { .. }));
    assert_eq!(arena.as_str(), "12345678");
    // A value that would overflow midway is rolled back completely.
    let err = arena.write_value(&serde_json::json!({"k": "way too long to fit"})).unwrap_err();
    assert!(matches!(err, WireError::Overflow { .. }));
    assert_eq!(arena.as_str(), "12345678");
    // Remaining budget still usable.
    arena.write_raw("90abcdef").unwrap();
    assert_eq!(arena.remaining(), 0);
    assert_eq!(arena.write_raw("x"), Err(WireError::Overflow { needed: 1, remaining: 0 }));
}

#[test]
fn projection_serializes_through_wire_arena_losslessly() {
    let proj = project_src("pub struct A;\npub fn f() {}\n", &CfgContext::default(), "ws-a");
    let mut arena = WireArena::new(1 << 16);
    proj.write_wire(&mut arena).unwrap();
    arena.freeze();
    let back: serde_json::Value = serde_json::from_str(arena.as_str()).unwrap();
    assert_eq!(back["workspace"], serde_json::json!(proj.workspace.0));
    assert_eq!(back["incomplete"], serde_json::Value::Bool(false));
}

// ---------- cold/warm projection measurement ----------

#[test]
fn cold_warm_projection_measurement() {
    // A moderately sized synthetic source: 200 public items with
    // bodies, attributes, and one observable location use.
    use std::fmt::Write as _;
    let mut src = String::from("#[cfg(feature = \"x\")]\npub struct Cold;\n");
    for i in 0..200 {
        let _ = write!(
            src,
            "/// doc for item {i}\n#[inline]\npub fn item_{i}(x: u64) -> u64 {{ let _ = column!(); x + {i} }}\n"
        );
    }
    let ws = hash_str("ws-bench");
    let ctx = ctx_with_feature(true);
    let mut cache = InterfaceCache::new(8);
    let cst = parse_top_level(&src);

    let t0 = std::time::Instant::now();
    let cold = cache.get_or_project(&cst, &ctx, ws);
    let cold_d = t0.elapsed();

    let t1 = std::time::Instant::now();
    let warm = cache.get_or_project(&cst, &ctx, ws);
    let warm_d = t1.elapsed();

    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
    assert!(std::sync::Arc::ptr_eq(&cold, &warm), "warm path reuses the projection");
    assert_eq!(cold.buckets.iter().map(|b| b.entries.len()).sum::<usize>(), 201);
    eprintln!(
        "cold/warm interface projection: cold={cold_d:?} warm={warm_d:?} items=201"
    );
}

// ---------- cache collision hardening (forced equal keys) ----------

#[test]
fn cst_cache_collision_is_miss_and_replace() {
    use std::sync::Arc;
    let real = "fn real() {}\n";
    let impostor = "fn impostor() {}\n"; // different bytes
    let mut cache = omp_ladder::CstCache::new(1 << 20);
    // Simulate a hash collision: impostor CST stored under real's key.
    cache.force_insert_for_test(hash_str(real), parse_top_level(impostor));

    let cst = cache.get_or_parse(real);
    assert_eq!(cache.stats().hits, 0, "byte mismatch must never be a hit");
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cst.source.as_ref(), real, "miss-and-replace re-parsed the real source");

    // The replacement entry now verifies and hits.
    let again = cache.get_or_parse(real);
    assert_eq!(cache.stats().hits, 1);
    assert!(Arc::ptr_eq(&cst, &again));
}

#[test]
fn semantic_cache_verifies_source_bytes_and_full_cfg_context() {
    use std::sync::Arc;
    let real = "pub fn real() {}\n";
    let ctx = CfgContext::default();
    let cst = parse_top_level(real);
    let mut cache = omp_ladder::SemanticCache::new(16);

    // Collision with different stored source bytes under the same key.
    let bogus = project(&parse_top_level("fn other() {}\n"), &ctx);
    cache.force_insert_for_test(&cst, &ctx, Arc::from("fn other() {}\n"), ctx.clone(), bogus);
    let idx = cache.get_or_project(&cst, &ctx);
    assert_eq!(cache.stats().hits, 0, "source-byte mismatch must never be a hit");
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(idx.items[0].name.as_deref(), Some("real"));
    let _ = cache.get_or_project(&cst, &ctx);
    assert_eq!(cache.stats().hits, 1);

    // Collision with same source but a different stored CfgContext.
    let mut other_ctx = CfgContext::default();
    other_ctx.flags.insert("impostor".to_owned());
    let bogus_ctx_index = project(&cst, &other_ctx);
    cache.force_insert_for_test(&cst, &ctx, Arc::from(real), other_ctx, bogus_ctx_index);
    let _ = cache.get_or_project(&cst, &ctx);
    assert_eq!(cache.stats().hits, 1, "cfg-context mismatch must never be a hit");
    assert_eq!(cache.stats().misses, 2);
}

// ---------- fail-closed angle-bracket ambiguity ----------

#[test]
fn angle_bracket_ambiguity_fails_closed_as_opaque() {
    let src = "pub struct A;\npub fn weird<T: Qux<N < M>;\npub struct B;\n";
    let cst = parse_top_level(src);
    let index = project(&cst, &CfgContext::default());
    let weird = index
        .items
        .iter()
        .find(|i| i.name.as_deref() == Some("weird"))
        .expect("weird item present");
    assert_eq!(
        weird.status,
        omp_ladder::ItemStatus::Opaque,
        "ambiguous signature must fail closed as Opaque"
    );
    assert_eq!(weird.body_span, None, "no trusted body span for an ambiguous item");

    // Neighbouring unambiguous items are unaffected.
    for name in ["A", "B"] {
        let it = index.items.iter().find(|i| i.name.as_deref() == Some(name)).unwrap();
        assert_eq!(it.status, omp_ladder::ItemStatus::Active);
    }

    // The opaque item never leaks into the public interface projection.
    let proj = project_interface(&index, &cst, hash_str("ws-a"));
    assert!(!proj.has_name(BucketKind::Values, "weird"));
    assert!(proj.has_name(BucketKind::Types, "A"));
    assert!(proj.has_name(BucketKind::Types, "B"));

    // A legitimate const-generic brace is NOT ambiguous.
    let ok = parse_top_level("pub fn g<const N: usize>() -> Foo<{ N }> {}\n");
    let ok_index = project(&ok, &CfgContext::default());
    assert_eq!(ok_index.items[0].status, omp_ladder::ItemStatus::Active);
    assert!(ok_index.items[0].body_span.is_some());
}
