//! Cache reuse, bounded invalidation, and proof that the syntax cache
//! never suppresses semantic invalidation.

use omp_ladder::{
    parse_top_level, project, CfgContext, CstCache, Dep, ItemStatus, SemanticCache,
};

fn ctx_with_feature(on: bool) -> CfgContext {
    let mut c = CfgContext::default();
    if on {
        c.kv.insert("feature".to_owned(), "x".to_owned());
    }
    c
}

#[test]
fn cst_cache_reuses_unchanged_content() {
    let src = "fn a() {}\nstruct B;\n";
    let mut cache = CstCache::new(1 << 20);
    let cold = cache.get_or_parse(src);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 0);
    let warm = cache.get_or_parse(src);
    assert_eq!(cache.stats().hits, 1);
    assert!(std::sync::Arc::ptr_eq(&cold, &warm), "warm parse must reuse the CST");

    // Different content -> miss; the old entry remains (content-addressed).
    let _other = cache.get_or_parse("fn c() {}\n");
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.len(), 2);
    let again = cache.get_or_parse(src);
    assert!(std::sync::Arc::ptr_eq(&cold, &again));
}

#[test]
fn cst_hit_does_not_suppress_semantic_invalidation() {
    let src = "#[cfg(feature = \"x\")]\nfn gated() {}\nfn plain() {}\n";
    let mut csts = CstCache::new(1 << 20);
    let mut sems = SemanticCache::new(16);

    let cst = csts.get_or_parse(src);
    let on = sems.get_or_project(&cst, &ctx_with_feature(true));
    assert_eq!(sems.stats().misses, 1);

    // Same source, same cfg context: both layers hit.
    let _ = sems.get_or_project(&cst, &ctx_with_feature(true));
    assert_eq!(sems.stats().hits, 1);

    // Same source (CST cache hit — parse is NOT re-run), different cfg
    // context: the semantic layer must miss and re-evaluate statuses.
    let reused = csts.get_or_parse(src);
    assert_eq!(csts.stats().hits, 1, "CST must be reused");
    assert!(std::sync::Arc::ptr_eq(&cst, &reused));
    let off = sems.get_or_project(&reused, &ctx_with_feature(false));
    assert_eq!(sems.stats().misses, 2, "cfg change must invalidate semantics");
    let gated_on = on
        .items
        .iter()
        .find(|i| i.name.as_deref() == Some("gated"))
        .map(|i| i.status);
    let gated_off = off
        .items
        .iter()
        .find(|i| i.name.as_deref() == Some("gated"))
        .map(|i| i.status);
    assert_eq!(gated_on, Some(ItemStatus::Active));
    assert_eq!(gated_off, Some(ItemStatus::CfgDisabled));
    // The semantic key differs in the cfg component only.
    let k_on = SemanticCache::key_for(&cst, &ctx_with_feature(true));
    let k_off = SemanticCache::key_for(&cst, &ctx_with_feature(false));
    assert_eq!(k_on.0, k_off.0);
    assert_ne!(k_on.1, k_off.1);
}

#[test]
fn macro_env_change_invalidates_macro_calls_conservatively() {
    let v1 = "macro_rules! m { () => { fn a() {} } }\nm!();\n";
    let v2 = "macro_rules! m { () => { fn b() {} } }\nm!();\n";
    let c1 = parse_top_level(v1);
    let c2 = parse_top_level(v2);
    assert_ne!(c1.source_hash, c2.source_hash);
    assert_ne!(
        omp_ladder::macro_env_hash(&c1),
        omp_ladder::macro_env_hash(&c2),
        "macro body edit must change the macro-env hash"
    );
    let i1 = project(&c1, &CfgContext::empty());
    let i2 = project(&c2, &CfgContext::empty());
    let call1 = i1.items.iter().find(|i| i.kind == omp_ladder::ItemKind::MacroCall);
    let call2 = i2.items.iter().find(|i| i.kind == omp_ladder::ItemKind::MacroCall);
    let (Some(c1), Some(c2)) = (call1, call2) else {
        panic!("both versions must index the macro call");
    };
    assert_eq!(c1.status, ItemStatus::Opaque);
    assert_eq!(c2.status, ItemStatus::Opaque);
    // The MacroEnv dependency hash differs -> downstream must rebuild.
    let env_dep = |i: &omp_ladder::Item| {
        i.deps.iter().find_map(|d| match d {
            Dep::MacroEnv(h) => Some(*h),
            Dep::MacroInvoke(_) => None,
        })
    };
    assert_ne!(env_dep(c1), env_dep(c2));
    // The macro *definitions* are indexed as MacroDef, not expanded:
    // `fn a` / `fn b` inside bodies are never items.
    assert!(!i1.items.iter().any(|i| i.name.as_deref() == Some("a")));
    assert!(!i2.items.iter().any(|i| i.name.as_deref() == Some("b")));
}

#[test]
fn caches_are_bounded_and_evict_lru() {
    let mut cache = CstCache::new(64); // tiny: one small file fits
    let a = cache.get_or_parse("fn a() {}\n");
    let _b = cache.get_or_parse("fn bbbbbbbbbbbbbbbbbbbbbbbbb() {}\n");
    let _c = cache.get_or_parse("fn c() {}\n");
    assert!(cache.stats().evictions > 0, "bounded cache must evict");
    // Re-requesting the evicted content is a miss, not stale data.
    let _ = a;
    let misses_before = cache.stats().misses;
    let _ = cache.get_or_parse("fn a() {}\n");
    assert_eq!(cache.stats().misses, misses_before + 1);

    let mut sems = SemanticCache::new(1);
    let cst1 = parse_top_level("fn one() {}\n");
    let cst2 = parse_top_level("fn two() {}\n");
    let _ = sems.get_or_project(&cst1, &CfgContext::empty());
    let _ = sems.get_or_project(&cst2, &CfgContext::empty());
    assert_eq!(sems.len(), 1);
    assert!(sems.stats().evictions > 0);
}

#[test]
fn explicit_invalidation_drops_entries() {
    let src = "fn a() {}\n";
    let mut cache = CstCache::new(1 << 20);
    let cst = cache.get_or_parse(src);
    cache.invalidate(cst.source_hash);
    assert_eq!(cache.len(), 0);
    let _ = cache.get_or_parse(src);
    assert_eq!(cache.stats().misses, 2);

    let mut sems = SemanticCache::new(4);
    let _ = sems.get_or_project(&cst, &CfgContext::empty());
    sems.invalidate_source(cst.source_hash);
    assert_eq!(sems.len(), 0);
}

#[test]
fn json_metadata_preserves_false_zero_and_null() {
    let src = "fn a() {}\n";
    let cst = parse_top_level(src);
    let mut idx = project(&cst, &CfgContext::empty());
    let item = idx.items.first_mut().map(|i| &mut i.metadata);
    let Some(meta) = item else { panic!("no item") };
    meta.insert("flag".to_owned(), serde_json::Value::Bool(false));
    meta.insert("count".to_owned(), serde_json::Value::from(0));
    meta.insert("note".to_owned(), serde_json::Value::Null);
    meta.insert(
        "nested".to_owned(),
        serde_json::json!({"off": false, "n": 0, "nil": null}),
    );

    let v = idx.to_json();
    let text = serde_json::to_string(&v).unwrap_or_default();
    assert!(text.contains("\"flag\":false"), "false dropped: {text}");
    assert!(text.contains("\"count\":0"), "zero dropped: {text}");
    assert!(text.contains("\"note\":null"), "null dropped: {text}");

    let back = omp_ladder::SemanticIndex::from_json(&v).map(|i| i.items);
    let back = back.unwrap_or_default();
    let meta = &back[0].metadata;
    assert_eq!(meta.get("flag"), Some(&serde_json::Value::Bool(false)));
    assert_eq!(meta.get("count"), Some(&serde_json::Value::from(0)));
    assert_eq!(meta.get("note"), Some(&serde_json::Value::Null));
    // Full index equality after round-trip.
    let back_idx = omp_ladder::SemanticIndex::from_json(&v);
    assert_eq!(back_idx.as_ref(), Some(&idx));
}
