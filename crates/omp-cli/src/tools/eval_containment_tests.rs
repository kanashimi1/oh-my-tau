use super::eval_observation::{
    Cell, ObservationPack, ProvenanceRef, GATE_ENV_LOCK,
};
use super::{Ctx, EnvGuard};
use serde_json::{json, Value};

#[test]
fn internal_schema_omits_only_deserializable_defaults() {
     mut pack = ObservationPack {
        version: 1,
        cells: vec![Cell {
            cell_id: 7,
            step: 1,
            lang: "js".to_string(),
            keys: Vec::new(),
            summary: "summary".to_string(),
            head: "head".to_string(),
            tail: "tail".to_string(),
            base_tokens: 0,
            provenance: Vec::new(),
        }],
        next_step: 2,
    };
     value = serde_json::to_value(&pack).expect("serialize");
    assert!(value["cells"][0].get("base_tokens").is_none());
    assert!(value["cells"][0].get("provenance").is_none());
     decoded: ObservationPack = serde_json::from_value(value).expect("defaults decode");
    assert_eq!(decoded.cells[0].base_tokens, 0);
    assert!(decoded.cells[0].provenance.is_empty());

    pack.cells[0].base_tokens = 9;
    pack.cells[0].provenance.push(ProvenanceRef {
        step: 2,
        tool: Some("read".to_string()),
    });
     nondefault = serde_json::to_value(&pack).expect("nondefault");
    assert_eq!(nondefault["cells"][0]["base_tokens"], 9);
    assert_eq!(nondefault["cells"][0]["provenance"][0]["step"], 2);

     arbitrary = json!({"false": false, "zero": 0, "null": null, "empty": []});
    assert_eq!(
        super::structured::encode(&arbitrary),
        serde_json::to_string(&arbitrary).expect("arbitrary JSON")
    );
}

#[tokio::test]
async fn large_structured_result_spills_to_retrievable_lossless_table() {
     _guard = GATE_ENV_LOCK.lock().await;
     dir = tempfile::tempdir().expect("tempdir");
     session = dir.path().join("session.jsonl");
     _env = EnvGuard::set("PI_SESSION_FILE", Some(session.to_string_lossy().as_ref()));
     ctx = Ctx::default();
     result = super::execute(
        &json!({
            "code": "return Array.from({length:1500},(_,i)=>({id:i,ok:false,zero:0,nil:null,text:'a\\tb\\nc\\\\d'}));",
            "language": "js"
        }),
        &ctx,
    )
    .await
    .expect("large structured result");
     bounded: Value = serde_json::from_str(&result).expect("bounded envelope");
    assert_eq!(bounded["full_output"].as_str(), Some("artifact://0"));
    assert!(bounded["head"].as_str().is_some_and(|head| !head.is_empty()));
    assert!(bounded["tail"].as_str().is_some_and(|tail| !tail.is_empty()));
    assert!(bounded["omitted"]["bytes"].as_u64().is_some_and(|bytes| bytes > 0));

     path = omp_urls::resolve_to_cwd("artifact://0", &ctx.env.cwd).expect("artifact path");
     full = std::fs::read_to_string(path).expect("retrieve full output handle");
     table: Value = serde_json::from_str(&full).expect("full typed table");
    assert_eq!(table["$omt"], "omt.table.v1");
    assert_eq!(table["rows"].as_array().expect("rows").len(), 1500);
    assert_eq!(table["rows"][0][1], false);
    assert_eq!(table["rows"][0][2], 0);
    assert_eq!(table["rows"][0][3], Value::Null);
    assert_eq!(table["rows"][0][4], "a\tb\nc\\d");


#[tokio::test]
async fn display_uses_the_same_typed_structured_encoding() {
    let ctx = Ctx::default();
    let result = super::execute(
        &json!({
            "code": "display([{ok:false,zero:0,nil:null,text:'a\\tb\\nc\\\\d'}])",
            "language": "js"
        }),
        &ctx,
    )
    .await
    .expect("structured display");
    let table: Value = serde_json::from_str(&result).expect("display table");
    assert_eq!(table["$omt"], "omt.table.v1");
    assert_eq!(table["rows"][0][0], false);
    assert_eq!(table["rows"][0][1], 0);
    assert_eq!(table["rows"][0][2], Value::Null);
    assert_eq!(table["rows"][0][3], "a\tb\nc\\d");
}
}
