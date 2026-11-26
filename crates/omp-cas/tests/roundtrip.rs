//! Round-trip: arbitrary bytes and JSON edge payloads must come back
//! byte-identical, through both tiers and through a pack round-trip.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly


use omp_cas::{Digest, Store, Tier};

#[test]
fn binary_content_round_trips_exactly_through_both_tiers() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();

    // Arbitrary binary content: full 0..=255 sweep, embedded NULs,
    // non-UTF-8 sequences, zlib/gzip lookalikes, zeros.
    let bytes: Vec<u8> = (0..=255_u8)
        .chain([0_u8; 512])
        .chain([0xFF_u8; 512])
        .chain([0x1F, 0x8B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) // gzip magic
        .chain([0x28, 0xB5, 0x2F, 0xFD, 0x00]) // zstd magic
        .chain((0..64_u32).map(|i| (i * 7 % 251) as u8))
        .collect();

    let scratch_id = store.put_scratch(&bytes).unwrap();
    let durable_id = store.put_durable(&bytes).unwrap();
    assert_eq!(scratch_id.digest(), durable_id.digest());

    assert_eq!(store.get(&scratch_id.digest()).unwrap(), bytes);
    assert_eq!(store.get(&durable_id.digest()).unwrap(), bytes);
    assert_eq!(store.get(&scratch_id.digest()).unwrap(), bytes);
}

#[test]
fn empty_object_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let id = store.put_scratch(&[]).unwrap();
    assert_eq!(store.get(&id.digest()).unwrap(), Vec::<u8>::new());
    assert_eq!(id.size(), 0);
}

#[test]
fn false_zero_null_json_payload_bytes_round_trip_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();

    // Raw JSON bytes: falsy scalars everywhere, including inside arrays,
    // nested objects, and as the entire payload. The store is byte-oriented
    // and never parses these — the test pins that contract.
    let payloads: [&str; 6] = [
        "false",
        "0",
        "null",
        r#"{"a":false,"b":0,"c":null}"#,
        r#"[false,0,null,{"deep":{"deeper":[false,0,null,null,0]}, "z":"","empty":{}}]"#,
        r#"{"f":0.0,"g":0e0,"h":-0.0,"i":false,"j":null,"k":"false","l":"0","m":"null","n":{}}"#,
    ];
    for payload in payloads {
        let bytes = payload.as_bytes();
        let id = store.put_durable(bytes).unwrap();
        assert_eq!(
            store.get(&id.digest()).unwrap(),
            bytes,
            "falsy payload {payload} must round-trip byte-exactly"
        );
    }

    // And through the pack path (scratch put -> pack -> import -> get).
    let mut graph = omp_cas::DepGraph::new();
    let root = store.put_scratch(r#"{"a":false,"b":0,"c":null}"#.as_bytes()).unwrap();
    graph.declare(root.digest(), []);
    let (pack_id, path) = store.build_pack_file(&[root.digest()], &graph, 100).unwrap();
    let other = Store::open(dir.path().join("restored")).unwrap();
    other.import_pack(&path, Tier::Scratch).unwrap();
    assert_eq!(
        other.get(&root.digest()).unwrap(),
        r#"{"a":false,"b":0,"c":null}"#.as_bytes()
    );
    // Pack itself is a durable object; its digest must equal the stored one.
    let pack_bytes = std::fs::read(&path).unwrap();
    assert_eq!(pack_id.digest(), Digest::of(&pack_bytes));
}

#[test]
fn digest_parse_is_strict_and_rejects_path_shapes() {
    let valid = "a".repeat(64);
    assert!(Digest::parse(&valid).is_ok());
    for hostile in [
        "",
        "ABCDEF",                                   // uppercase
        "../../../etc/passwd",                      // traversal shape
        "/etc/passwd",                              // absolute
        "../../objects/x",                          // relative climb
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0", // 65 chars
        "0123456789abcdef",                         // 16 chars
        "g23456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", // 'g'
        "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef", // mixed case
    ] {
        assert!(
            Digest::parse(hostile).is_err(),
            "digest parse must reject {hostile:?}"
        );
    }
    // Round-trip hex form.
    let bytes = b"payload";
    assert_eq!(Digest::of(bytes).hex(), Digest::parse(&Digest::of(bytes).hex()).unwrap().hex());
}

#[test]
fn tier_paths_are_digest_derived_and_confined() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let id = store.put_durable(b"confined").unwrap();
    let path = store.tier_path(Tier::Durable, &id.digest());
    let path_abs = path
        .strip_prefix(store.root().canonicalize().unwrap().as_path())
        .expect("path must stay under the store root");
    assert!(path_abs.starts_with("durable/objects"));
    // Every component is hex-derived: no separators or dots smuggled in.
    for component in path_abs.components() {
        let s = component.as_os_str().to_string_lossy();
        assert!(
            s.chars().all(|c| c.is_ascii_hexdigit()) || s == "objects" || s == "durable",
            "unexpected component {s:?} in {path:?}"
        );
    }
}
