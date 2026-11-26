//! Scratch/durable lifetime policy: no accidental promotion, explicit
//! promotion only, LRU byte-budget eviction, per-object caps, policy
//! persistence across store reopen.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly


use omp_cas::{CachePolicy, Store, Tier};

#[test]
fn scratch_put_never_creates_durable_state_and_get_never_promotes() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();

    let a = store.put_scratch(b"alpha").unwrap();
    let b = store.put_scratch(b"beta").unwrap();

    // Reads (any number) must not move bytes between tiers.
    for _ in 0..5 {
        store.get(&a.digest()).unwrap();
        store.get(&b.digest()).unwrap();
    }
    assert!(store.tier_path(Tier::Scratch, &a.digest()).is_file());
    assert!(!store.tier_path(Tier::Durable, &a.digest()).is_file());
    assert!(store.tier_path(Tier::Scratch, &b.digest()).is_file());
    assert!(!store.tier_path(Tier::Durable, &b.digest()).is_file());
}

#[test]
fn durable_put_never_creates_scratch_state() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let id = store.put_durable(b"durable only").unwrap();
    assert!(store.tier_path(Tier::Durable, &id.digest()).is_file());
    assert!(!store.tier_path(Tier::Scratch, &id.digest()).is_file());
    assert_eq!(store.scratch_stats().objects, 0);
    assert_eq!(store.scratch_stats().bytes, 0);
}

#[test]
fn promote_is_the_only_scratch_to_durable_path() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();

    let promoted = store.put_scratch(b"to keep").unwrap();
    let evicted_later = store.put_scratch(b"to drop").unwrap();

    assert!(store.promote(&promoted.digest()).unwrap());
    assert!(store.tier_path(Tier::Durable, &promoted.digest()).is_file());
    // Promotion copies: scratch copy remains and is still evictable.
    assert!(store.tier_path(Tier::Scratch, &promoted.digest()).is_file());

    // Promoting a digest that is durable-only reports false and does not
    // touch scratch.
    let durable_only = store.put_durable(b"never scratch").unwrap();
    assert!(!store.promote(&durable_only.digest()).unwrap());

    // Promoting a digest in neither tier reports false too.
    let ghost = omp_cas::Digest::of(b"missing");
    assert!(!store.promote(&ghost).unwrap());

    // The unpromoted scratch object is still scratch-only.
    assert!(store.tier_path(Tier::Scratch, &evicted_later.digest()).is_file());
    assert!(!store.tier_path(Tier::Durable, &evicted_later.digest()).is_file());
}

#[test]
fn byte_budget_evicts_lru_scratch_only_and_keeps_durable() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    store
        .set_scratch_policy(CachePolicy {
            max_bytes: Some(300),
            max_object_bytes: None,
        })
        .unwrap();

    let keep_durable = store.put_durable(b"d".repeat(200).as_slice()).unwrap();

    let first = store.put_scratch(&[1_u8; 100]).unwrap(); // tick 1 (oldest)
    let second = store.put_scratch(&[2_u8; 100]).unwrap(); // tick 2
    let third = store.put_scratch(&[3_u8; 100]).unwrap(); // tick 3

    assert_eq!(store.scratch_stats().bytes, 300);

    // Touch the second object so `first` becomes the LRU victim.
    store.get(&second.digest()).unwrap();

    // Overflow: 300 tracked + 100 new > 300 budget -> evict until fit.
    let fourth = store.put_scratch(&[4_u8; 100]).unwrap();

    assert!(!store.tier_path(Tier::Scratch, &first.digest()).is_file());
    assert!(store.tier_path(Tier::Scratch, &second.digest()).is_file());
    assert!(store.tier_path(Tier::Scratch, &third.digest()).is_file());
    assert!(store.tier_path(Tier::Scratch, &fourth.digest()).is_file());
    // Durable object untouched by scratch eviction.
    assert_eq!(store.get(&keep_durable.digest()).unwrap(), vec![b'd'; 200]);
    assert!(store.scratch_stats().bytes <= 300);
    assert_eq!(store.scratch_stats().evictions, 1);
}

#[test]
fn budget_applies_on_reopen_via_disk_seeded_accounting() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let store = Store::open(dir.path()).unwrap();
        store
            .set_scratch_policy(CachePolicy {
                max_bytes: Some(150),
                max_object_bytes: None,
            })
            .unwrap();
        id = store.put_scratch(&[7_u8; 100]).unwrap();
    }
    // Reopen: accounting must come from disk, not start at zero.
    {
        let store = Store::open(dir.path()).unwrap();
        assert_eq!(store.scratch_stats().objects, 1);
        assert_eq!(store.scratch_stats().bytes, 100);
        let id2 = store.put_scratch(&[8_u8; 100]).unwrap();
        // 200 > 150: the older (first) object is evicted.
        assert!(!store.tier_path(Tier::Scratch, &id.digest()).is_file());
        assert!(store.tier_path(Tier::Scratch, &id2.digest()).is_file());
    }
}

#[test]
fn per_object_cap_rejects_oversized_scratch_without_state() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    store
        .set_scratch_policy(CachePolicy {
            max_bytes: None,
            max_object_bytes: Some(10),
        })
        .unwrap();
    let err = store.put_scratch(&[0_u8; 11]).unwrap_err();
    assert!(matches!(err, omp_cas::Error::Limit(_)));
    // Nothing was accounted or written.
    assert_eq!(store.scratch_stats().objects, 0);
    let small = store.put_scratch(&[0_u8; 10]).unwrap();
    assert!(store.tier_path(Tier::Scratch, &small.digest()).is_file());
    // Durable ignores the scratch cap entirely.
    store.put_durable(&[0_u8; 64]).unwrap();
}

#[test]
fn get_reports_missing_object_with_digest() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let ghost = omp_cas::Digest::of(b"absent");
    let err = store.get(&ghost).unwrap_err();
    match err {
        omp_cas::Error::MissingObject { digest } => assert_eq!(digest, ghost.hex()),
        other => panic!("expected MissingObject, got {other:?}"),
    }
}

#[test]
fn remove_is_explicit_and_tier_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let id = store.put_scratch(b"gone soon").unwrap();
    assert!(store.remove(Tier::Scratch, &id.digest()).unwrap());
    assert!(!store.remove(Tier::Scratch, &id.digest()).unwrap());
    assert_eq!(store.scratch_stats().objects, 0);
    assert_eq!(store.scratch_stats().bytes, 0);
}


#[test]
fn repeated_scratch_put_counts_the_live_object_once() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let id = store.put_scratch(b"same bytes").unwrap();
    let before = store.scratch_stats();
    store.put_scratch(b"same bytes").unwrap();
    let after = store.scratch_stats();
    assert_eq!(after.objects, 1);
    assert_eq!(after.bytes, before.bytes);
    store.remove(Tier::Scratch, &id.digest()).unwrap();
    assert_eq!(store.scratch_stats().objects, 0);
    assert_eq!(store.scratch_stats().bytes, 0);
}
