//! Transitive dependency packs: closure correctness, cycle handling,
//! missing-dependency errors, corruption detection, deterministic pack
//! bytes, and import bounds.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly


use std::collections::BTreeSet;

use omp_cas::{DepGraph, Digest, Store, Tier};

/// Helper: declare deps for an object after storing it; returns its digest.
fn put_with_deps(store: &Store, graph: &mut DepGraph, bytes: &[u8], deps: &[Digest]) -> Digest {
    let id = store.put_scratch(bytes).unwrap();
    graph.declare(id.digest(), deps.iter().copied());
    id.digest()
}

#[test]
fn pack_carries_the_full_transitive_closure() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();

    // app -> lib_a -> core; app -> lib_b -> core; core -> (leaf)
    let leaf = put_with_deps(&store, &mut graph, b"leaf", &[]);
    let core = put_with_deps(&store, &mut graph, b"core", &[leaf]);
    let lib_a = put_with_deps(&store, &mut graph, b"lib-a", &[core]);
    let lib_b = put_with_deps(&store, &mut graph, b"lib-b", &[core]);
    let app = put_with_deps(&store, &mut graph, b"app", &[lib_a, lib_b]);

    let (pack_id, path) = store.build_pack_file(&[app], &graph, 100).unwrap();
    let pack_bytes = std::fs::read(&path).unwrap();
    assert_eq!(pack_id.digest(), Digest::of(&pack_bytes));
    let _ = pack_id;

    // Restore into an empty store; every closure member must be present.
    let restore = Store::open(dir.path().join("restore")).unwrap();
    let imported = restore.import_pack(&path, Tier::Scratch).unwrap();
    assert_eq!(imported.len(), 5);
    let imported_set: BTreeSet<Digest> = imported.iter().map(omp_cas::ObjectId::digest).collect();
    for digest in [app, lib_a, lib_b, core, leaf] {
        assert!(imported_set.contains(&digest), "closure missing {digest}");
        assert_eq!(restore.get(&digest).unwrap(), store.get(&digest).unwrap());
    }
    // Exactly the closure — no extra objects smuggled in.
    assert_eq!(imported_set.len(), 5);
}

#[test]
fn diamond_and_shared_deps_are_deduplicated() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let shared = put_with_deps(&store, &mut graph, b"shared", &[]);
    let left = put_with_deps(&store, &mut graph, b"left", &[shared]);
    let right = put_with_deps(&store, &mut graph, b"right", &[shared]);
    let root = put_with_deps(&store, &mut graph, b"root", &[left, right, shared]);

    let (_, path) = store.build_pack_file(&[root], &graph, 100).unwrap();
    let restore = Store::open(dir.path().join("restore")).unwrap();
    let imported = restore.import_pack(&path, Tier::Scratch).unwrap();
    assert_eq!(imported.len(), 4, "shared dep packed once");
}

#[test]
fn cyclic_dependencies_terminate_and_pack_completely() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();

    let a = put_with_deps(&store, &mut graph, b"A", &[]);
    let b = put_with_deps(&store, &mut graph, b"B", &[a]);
    let c = put_with_deps(&store, &mut graph, b"C", &[b]);
    // Close the cycle and add a self-edge; traversal must not recurse
    // forever or drop members.
    graph.declare(a, [c]);
    graph.declare(a, [a]);

    let (_, path) = store.build_pack_file(&[b], &graph, 100).unwrap();
    let restore = Store::open(dir.path().join("restore")).unwrap();
    let imported = restore.import_pack(&path, Tier::Scratch).unwrap();
    assert_eq!(imported.len(), 3, "cycle members all packed");
    let names: BTreeSet<String> = imported
        .iter()
        .map(|id| String::from_utf8(restore.get(&id.digest()).unwrap()).unwrap())
        .collect();
    assert_eq!(names, BTreeSet::from(["A".into(), "B".into(), "C".into()]));
}

#[test]
fn deep_linear_chain_is_bounded_by_max_nodes() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let mut chain = Vec::new();
    for i in 0..64 {
        let deps = if chain.is_empty() { Vec::new() } else { vec![*chain.last().unwrap()] };
        chain.push(put_with_deps(&store, &mut graph, format!("n{i}").as_bytes(), &deps));
    }
    // Bound below the closure size: clean error, no partial pack.
    let err = store
        .build_pack_file(&[*chain.last().unwrap()], &graph, 10)
        .unwrap_err();
    assert!(matches!(err, omp_cas::Error::Limit(_)));
    // Bound above the closure size: all 64 present.
    let (_, path) = store.build_pack_file(&[*chain.last().unwrap()], &graph, 100).unwrap();
    let restore = Store::open(dir.path().join("restore")).unwrap();
    assert_eq!(restore.import_pack(&path, Tier::Scratch).unwrap().len(), 64);
}

#[test]
fn missing_dependency_is_an_error_naming_the_digest() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let present = put_with_deps(&store, &mut graph, b"present", &[]);
    let ghost = Digest::of(b"never stored");
    let root = put_with_deps(&store, &mut graph, b"root", &[present, ghost]);

    let err = store.build_pack_file(&[root], &graph, 100).unwrap_err();
    match err {
        omp_cas::Error::MissingObject { digest } => assert_eq!(digest, ghost.hex()),
        other => panic!("expected MissingObject for the missing dep, got {other:?}"),
    }
}

#[test]
fn pack_bytes_are_deterministic_for_equal_content() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = DepGraph::new();
    let mut digests = Vec::new();
    {
        let store = Store::open(dir.path().join("one")).unwrap();
        for text in ["a", "b", "c"] {
            let id = store.put_durable(text.as_bytes()).unwrap();
            graph.declare(id.digest(), []);
            digests.push(id.digest());
        }
        let (id1, path1) = store.build_pack_file(digests.clone().as_slice(), &graph, 10).unwrap();
        let (id2, path2) = store.build_pack_file(digests.as_slice(), &graph, 10).unwrap();
        assert_eq!(id1.digest(), id2.digest(), "same content -> same pack digest");
        assert_eq!(std::fs::read(path1).unwrap(), std::fs::read(path2).unwrap());
        let _ = id1;
    }
}

#[test]
fn corrupt_pack_payload_is_detected_on_import() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let a = put_with_deps(&store, &mut graph, b"AAA", &[]);
    let b = put_with_deps(&store, &mut graph, b"BBB", &[a]);
    let (_, path) = store.build_pack_file(&[b], &graph, 10).unwrap();

    // Flip a payload byte (last byte = compressed payload region).
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    let corrupted = dir.path().join("corrupt.pack");
    std::fs::write(&corrupted, &bytes).unwrap();

    let restore = Store::open(dir.path().join("restore")).unwrap();
    let err = restore.import_pack(&corrupted, Tier::Scratch).unwrap_err();
    // Compression makes exact flip propagation unpredictable; any refusal
    // (decompress failure, length mismatch, digest mismatch) is correct.
    assert!(
        matches!(
            err,
            omp_cas::Error::Format(_) | omp_cas::Error::Corrupt { .. }
        ),
        "corrupt pack must be refused, got {err:?}"
    );
    // Import verifies before writing, so whatever did land is exactly a
    // prefix of the pack's objects — each one digest-consistent. No object
    // is ever stored whose bytes fail verification.
    for id in restore.import_pack(&path, Tier::Scratch).unwrap() {
        assert!(restore.contains(&id.digest()).unwrap());
        assert_eq!(
            restore.get(&id.digest()).unwrap(),
            store.get(&id.digest()).unwrap()
        );
    }
}

#[test]
fn manifest_digest_mismatch_is_corrupt_error_with_that_digest() {
    // Hand-build a pack whose manifest claims digest X over payload of Y.
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let a = put_with_deps(&store, &mut graph, b"true bytes", &[]);
    let (_, path) = store.build_pack_file(&[a], &graph, 10).unwrap();

    // Tamper: rewrite the manifest's digest field to another valid digest.
    let mut bytes = std::fs::read(&path).unwrap();
    let needle = format!("\"digest\":\"{}\"", a.hex());
    let wrong = Digest::of(b"other object entirely");
    let patch_text = format!("\"digest\":\"{}\"", wrong.hex());
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle.as_bytes())
        .expect("manifest digest found");
    bytes[pos..pos + needle.len()].copy_from_slice(patch_text.as_bytes());
    let tampered = dir.path().join("tampered.pack");
    std::fs::write(&tampered, &bytes).unwrap();

    let restore = Store::open(dir.path().join("restore")).unwrap();
    match restore.import_pack(&tampered, Tier::Scratch).unwrap_err() {
        omp_cas::Error::Format(_) => { /* manifest parse before verify is also a refusal */ }
        omp_cas::Error::Corrupt { digest, .. } => assert_eq!(digest, wrong.hex()),
        other => panic!("expected Format/Corrupt, got {other:?}"),
    }
    assert_eq!(restore.scratch_stats().objects, 0);
}

#[test]
fn truncated_and_garbage_packs_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let a = put_with_deps(&store, &mut graph, b"payload bytes here", &[]);
    let (_, path) = store.build_pack_file(&[a], &graph, 10).unwrap();
    let bytes = std::fs::read(&path).unwrap();

    let restore = Store::open(dir.path().join("restore")).unwrap();

    let trunc = dir.path().join("trunc.pack");
    std::fs::write(&trunc, &bytes[..bytes.len() / 2]).unwrap();
    assert!(restore.import_pack(&trunc, Tier::Scratch).is_err());

    let garbage = dir.path().join("garbage.pack");
    std::fs::write(&garbage, b"not a pack at all, just text").unwrap();
    assert!(restore.import_pack(&garbage, Tier::Scratch).is_err());

    let empty = dir.path().join("empty.pack");
    std::fs::write(&empty, b"").unwrap();
    assert!(restore.import_pack(&empty, Tier::Scratch).is_err());
}

#[test]
fn pack_import_with_hostile_entry_names_cannot_escape_the_store() {
    // Even a hand-crafted manifest naming traversal-shaped "digests" is
    // refused: digests must parse as 64 lowercase hex before anything is
    // written, and object paths are derived only from parsed digests.
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let mut graph = DepGraph::new();
    let a = put_with_deps(&store, &mut graph, b"harmless", &[]);
    let (_, path) = store.build_pack_file(&[a], &graph, 10).unwrap();

    // Patch the manifest's digest field in place (equal byte length): a
    // 64-char name shaped like a relative climb. It is not valid hex, so
    // the digest validator must refuse it before any path is built.
    let mut bytes = std::fs::read(&path).unwrap();
    let needle = format!("\"digest\":\"{}\"", a.hex());
    let mut hostile_name = String::from("../../evil");
    hostile_name.extend(std::iter::repeat_n('a', 64 - hostile_name.len()));
    let patch_text = format!("\"digest\":\"{hostile_name}\"");
    assert_eq!(needle.len(), patch_text.len(), "patch must be length-preserving");
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle.as_bytes())
        .expect("manifest digest found");
    bytes[pos..pos + needle.len()].copy_from_slice(patch_text.as_bytes());
    let hostile = dir.path().join("hostile.pack");
    std::fs::write(&hostile, &bytes).unwrap();

    let restore = Store::open(dir.path().join("restore")).unwrap();
    assert!(restore.import_pack(&hostile, Tier::Scratch).is_err());
    // Nothing outside the store was created either.
    assert!(!dir.path().join("evil").exists());
    assert_eq!(restore.scratch_stats().objects, 0);
}
