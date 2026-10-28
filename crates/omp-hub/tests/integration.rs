#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! Acceptance tests for the hub (`SPEC.md`): store durability, dedup,
//! per-recipient seq, priority order, replay after restart, deadline
//! expiry, schema-valid/invalid answers, claims (conflict/fencing/expiry),
//! and the client/server wire over a real UDS.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::time::Duration;

use omp_hub::{
    new_message_id, DeliverStream, Envelope, HubClient, HubDir, Kind, Priority, ServeExit, State,
    Store,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::task::JoinHandle;

fn env(sender: &str, recipient: &str, payload: &str) -> Envelope {
    Envelope {
        message_id: new_message_id(),
        sender: sender.into(),
        recipient: recipient.into(),
        kind: Kind::AGENT_MESSAGE.into(),
        payload: Some(payload.into()),
        ..Envelope::default()
    }
}

/// Start a hub server on a fresh tempdir; returns (dir, hub, task).
fn start_hub() -> (TempDir, HubDir, JoinHandle<()>) {
    let dir = TempDir::new().unwrap();
    let hub = HubDir::explicit(dir.path().to_path_buf(), None);
    let task = spawn_serve(hub.clone());
    (dir, hub, task)
}

/// Respawn on an existing hub dir (restart tests).
fn respawn_hub(hub: &HubDir) -> JoinHandle<()> {
    spawn_serve(hub.clone())
}

fn spawn_serve(hub: HubDir) -> JoinHandle<()> {
    tokio::spawn(async move {
        let _ = omp_hub::serve(&hub).await;
    })
}

async fn connect(hub: &HubDir) -> HubClient {
    for _ in 0..100 {
        if let Ok(c) = HubClient::connect_with(hub.clone(), false).await {
            return c;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("hub did not come up");
}

async fn recv(stream: &mut DeliverStream) -> Envelope {
    tokio::time::timeout(Duration::from_secs(5), stream.recv())
        .await
        .expect("deliver timeout")
        .expect("deliver stream closed")
}

#[tokio::test]
async fn store_seq_dedup_and_replay_after_restart() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("db1");
    let store = Store::open(&db).unwrap();

    let mut a1 = env("s1", "agent-x", "one");
    let out1 = store.insert(&mut a1, 1000).unwrap();
    assert_eq!(out1.seq, 1);
    assert!(!out1.dedup);
    assert_eq!(a1.state.as_deref(), Some(State::SENT));
    assert_eq!(
        a1.state_log.as_ref().unwrap()[State::SENT],
        Value::from(1000)
    );

    let mut a2 = env("s1", "agent-x", "two");
    assert_eq!(store.insert(&mut a2, 1001).unwrap().seq, 2);
    // Per-recipient seq: another recipient starts at 1.
    let mut b1 = env("s1", "agent-y", "other");
    assert_eq!(store.insert(&mut b1, 1002).unwrap().seq, 1);

    // Dedup on message_id: same id -> same seq, no new row.
    let mut dupe = env("s1", "agent-x", "changed payload");
    dupe.message_id = a1.message_id.clone();
    let out = store.insert(&mut dupe, 2000).unwrap();
    assert!(out.dedup);
    assert_eq!(out.seq, 1);
    assert_eq!(store.inbox("agent-x", 0, 10, false).unwrap().len(), 2);

    // Simulate a restart: drop the store, reopen the same db.
    drop(store);
    let store2 = Store::open(&db).unwrap();
    // Replay from last_seq=1: only the second message comes back.
    let replay = store2.inbox("agent-x", 1, 10, true).unwrap();
    assert_eq!(replay.len(), 1);
    assert_eq!(replay[0].envelope.payload.as_deref(), Some("two"));
    // Everything for the recipient, any state.
    assert_eq!(store2.inbox("agent-x", 0, 10, false).unwrap().len(), 2);
}

#[tokio::test]
async fn store_priority_order() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("db")).unwrap();
    let mut bulk = env("s", "p", "bulk");
    bulk.priority = Priority::BULK;
    store.insert(&mut bulk, 1).unwrap();
    let mut normal = env("s", "p", "normal");
    store.insert(&mut normal, 2).unwrap();
    let mut urgent = env("s", "p", "urgent");
    urgent.priority = Priority::URGENT;
    store.insert(&mut urgent, 3).unwrap();
    let rows = store.inbox("p", 0, 10, false).unwrap();
    let payloads: Vec<&str> = rows
        .iter()
        .map(|r| r.envelope.payload.as_deref().unwrap())
        .collect();
    assert_eq!(payloads, ["urgent", "normal", "bulk"]);
}

#[tokio::test]
async fn claims_conflict_fencing_and_expiry() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("db")).unwrap();
    let t0 = 10_000;

    // Grant to A with token 1; B conflicts and sees the holder + expiry.
    let c1 = store
        .claim("file:x.rs", "worker-a", 60, Some("edit"), t0)
        .unwrap();
    assert!(c1.granted);
    let tok1 = c1.fencing_token.unwrap();
    let c2 = store
        .claim("file:x.rs", "worker-b", 60, None, t0 + 1000)
        .unwrap();
    assert!(!c2.granted);
    assert_eq!(c2.holder.as_deref(), Some("worker-a"));
    assert!(c2.expires_ms.unwrap() > t0);

    // Re-claim by the same owner renews with a HIGHER token (monotonic).
    let c3 = store
        .claim("file:x.rs", "worker-a", 60, None, t0 + 2000)
        .unwrap();
    let tok3 = c3.fencing_token.unwrap();
    assert!(tok3 > tok1);

    // Stale token cannot release; the current token can.
    assert!(!store
        .release("file:x.rs", "worker-a", tok1, t0 + 3000)
        .unwrap());
    assert!(store
        .release("file:x.rs", "worker-a", tok3, t0 + 3000)
        .unwrap());

    // After release B gets it, with an even higher token.
    let c4 = store
        .claim("file:x.rs", "worker-b", 60, None, t0 + 4000)
        .unwrap();
    assert!(c4.granted);
    assert!(c4.fencing_token.unwrap() > tok3);

    // Expiry frees the resource without a release.
    let short = store
        .claim("tmp:lock", "worker-a", 1, None, t0 + 5000)
        .unwrap();
    assert!(short.granted);
    assert!(store.check("tmp:lock", t0 + 7000).unwrap().is_none());
    let takeover = store
        .claim("tmp:lock", "worker-b", 60, None, t0 + 7000)
        .unwrap();
    assert!(takeover.granted);

    // check() reports the holder for warnings while held.
    assert_eq!(
        store.check("file:x.rs", t0 + 4100).unwrap().unwrap().owner,
        "worker-b"
    );
}

#[tokio::test]
async fn wire_send_ack_inbox_and_dead_pane_fails_fast() {
    let (_guard, hub, task) = start_hub();
    let c = connect(&hub).await;
    let r = c.send(env("agent-a", "agent-b", "hello")).await.unwrap();
    assert_eq!(r.seq, 1);
    assert_eq!(r.state, State::SENT);

    // Dedup over the wire.
    let mut again = env("agent-a", "agent-b", "hello");
    again.message_id = r.message_id.clone();
    let r2 = c.send(again).await.unwrap();
    assert!(r2.dedup);
    assert_eq!(r2.seq, r.seq);

    let inbox = c.inbox("agent-b", None, None).await.unwrap();
    assert_eq!(inbox.len(), 1);
    c.ack(&r.message_id, State::READ).await.unwrap();
    // unread excludes read messages.
    let c2 = connect(&hub).await;
    let status = c2.status(Some("agent-b")).await.unwrap();
    assert_eq!(status["unread"].as_array().unwrap().len(), 0);

    // Dead pane fails fast with state failed.
    let dead = c
        .send(env("agent-a", "w4:pZZ", "nobody home"))
        .await
        .unwrap();
    assert_eq!(dead.state, State::FAILED);
    assert!(dead.error.unwrap().contains("dead pane"));
    task.abort();
}

#[tokio::test]
async fn wire_hello_replay_urgent_first_and_nothing_lost_on_restart() {
    let (dir, hub, task) = start_hub();
    let sender = connect(&hub).await;

    // Two messages to a recipient that never connected...
    let mut bulk = env("agent-a", "agent-b", "bulk-first");
    bulk.priority = Priority::BULK;
    let bulk_seq = sender.send(bulk).await.unwrap().seq;
    let mut urgent = env("agent-a", "agent-b", "urgent-second");
    urgent.priority = Priority::URGENT;
    sender.send(urgent).await.unwrap();

    // ...a live endpoint that will vanish mid-flight...
    let mut gone = HubClient::connect_with(hub.clone(), false).await.unwrap();
    let mut gone_stream = gone.take_deliver_stream().unwrap();
    gone.hello("agent-gone", 0).await.unwrap();
    let to_gone = sender
        .send(env("agent-a", "agent-gone", "for-gone"))
        .await
        .unwrap();
    let _ = recv(&mut gone_stream).await; // delivered, never acked
    assert!(to_gone.seq >= 1);
    drop(gone);

    // Kill the hub mid-flight and restart on the same db.
    task.abort();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let hub2 = HubDir::explicit(dir.path().to_path_buf(), None);
    let task2 = respawn_hub(&hub2);
    let mut c = connect(&hub2).await;
    let mut stream = c.take_deliver_stream().unwrap();
    let hello = c.hello("agent-b", 0).await.unwrap();
    assert_eq!(hello["replayed"].as_u64().unwrap(), 2);
    // Urgent first despite being sent second.
    let first = recv(&mut stream).await;
    assert_eq!(first.payload.as_deref(), Some("urgent-second"));
    let second = recv(&mut stream).await;
    assert_eq!(second.payload.as_deref(), Some("bulk-first"));
    assert_eq!(second.seq, Some(bulk_seq));

    // agent-gone reconnects from before its message -> replayed too.
    let mut c2 = connect(&hub2).await;
    let mut stream2 = c2.take_deliver_stream().unwrap();
    let h2 = c2.hello("agent-gone", 0).await.unwrap();
    assert_eq!(h2["replayed"].as_u64().unwrap(), 1);
    assert_eq!(
        recv(&mut stream2).await.payload.as_deref(),
        Some("for-gone")
    );
    task2.abort();
}

#[tokio::test]
async fn wire_answer_schema_valid_and_invalid() {
    let (_guard, hub, task) = start_hub();
    let mut asker = connect(&hub).await;
    let mut asker_stream = asker.take_deliver_stream().unwrap();
    asker.hello("agent-asker", 0).await.unwrap();
    let answerer = connect(&hub).await;

    let mut question = env("agent-asker", "agent-helper", "what is 2+2?");
    question.kind = Kind::AGENT_QUESTION.into();
    question.expected = Some("json".into());
    question.reply_schema = Some(json!({
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "integer"}},
    }));
    let q = answerer.send(question).await.unwrap();

    // The helper sees the question.
    let mut helper = connect(&hub).await;
    let mut helper_stream = helper.take_deliver_stream().unwrap();
    helper.hello("agent-helper", 0).await.unwrap();
    let got = recv(&mut helper_stream).await;
    assert_eq!(got.message_id, q.message_id);
    assert!(got.reply_schema.is_some());

    // Invalid answer: schema errors come back to the answerer, the
    // question stays unanswered.
    let mut bad = Envelope {
        message_id: new_message_id(),
        sender: "agent-helper".into(),
        kind: Kind::ANSWER.into(),
        payload: Some(r#"{"answer":"four"}"#.into()),
        ..Envelope::default()
    };
    bad.causation_id = Some(q.message_id.clone());
    let err = helper
        .answer(&q.message_id, bad)
        .await
        .expect_err("invalid answer must fail");
    assert!(err.to_string().contains("reply_schema"), "err: {err:#}");
    let st = asker.status_message(&q.message_id).await.unwrap();
    assert_ne!(st["state"].as_str().unwrap(), State::ANSWERED);

    // Valid answer: original goes `answered`, reply reaches the asker.
    let mut good = Envelope {
        message_id: new_message_id(),
        sender: "agent-helper".into(),
        kind: Kind::ANSWER.into(),
        payload: Some(r#"{"answer":4}"#.into()),
        ..Envelope::default()
    };
    good.causation_id = Some(q.message_id.clone());
    let out = helper.answer(&q.message_id, good).await.unwrap();
    assert_eq!(out["question"]["state"].as_str().unwrap(), State::ANSWERED);
    let reply = recv(&mut asker_stream).await;
    assert_eq!(reply.kind, Kind::ANSWER);
    assert_eq!(reply.causation_id.as_deref(), Some(q.message_id.as_str()));
    assert_eq!(reply.payload.as_deref(), Some(r#"{"answer":4}"#));
    let st = asker.status_message(&q.message_id).await.unwrap();
    assert_eq!(st["state"].as_str().unwrap(), State::ANSWERED);

    // A second answer is refused.
    let mut late = Envelope {
        message_id: new_message_id(),
        sender: "agent-helper".into(),
        kind: Kind::ANSWER.into(),
        payload: Some(r#"{"answer":5}"#.into()),
        ..Envelope::default()
    };
    late.causation_id = Some(q.message_id.clone());
    assert!(helper.answer(&q.message_id, late).await.is_err());
    task.abort();
}

#[tokio::test]
async fn wire_deadline_expiry_sends_sender_a_notice() {
    let (_guard, hub, task) = start_hub();
    let sender = connect(&hub).await;
    let mut late = env("agent-a", "agent-b", "reply please");
    late.expected = Some("ack".into());
    let short_deadline = omp_hub::now_ms() + 1500;
    late.deadline = Some(short_deadline);
    let sent = sender.send(late).await.unwrap();
    assert_eq!(sent.state, State::SENT);

    // Wait for the sweeper to expire it (1 s tick + margin).
    let mut expired_at = None;
    for _ in 0..40 {
        let st = sender.status_message(&sent.message_id).await.unwrap();
        if st["state"].as_str() == Some(State::EXPIRED) {
            expired_at = Some(());
            break;
        }
        assert!(
            omp_hub::now_ms() < short_deadline + 10_000,
            "expiry never happened"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    expired_at.expect("envelope must expire");

    // The sender got a delivery notice naming the expired message.
    let mut notices = 0;
    for env in sender.inbox("agent-a", None, None).await.unwrap() {
        if env.kind == Kind::DELIVERY_ACK {
            notices += 1;
            let payload: Value = serde_json::from_str(env.payload.as_deref().unwrap()).unwrap();
            assert_eq!(
                payload["notice"].as_str().unwrap(),
                "reply_deadline_expired"
            );
            assert_eq!(payload["message_id"].as_str().unwrap(), sent.message_id);
        }
    }
    assert_eq!(notices, 1);
    task.abort();
}

#[tokio::test]
async fn wire_claims_roundtrip() {
    let (_guard, hub, task) = start_hub();
    let c = connect(&hub).await;
    let c1 = c
        .claim("crates/omp-hub", "hub-worker", 60, Some("build"))
        .await
        .unwrap();
    assert!(c1.granted);
    let c2 = c
        .claim("crates/omp-hub", "task-worker", 60, None)
        .await
        .unwrap();
    assert!(!c2.granted);
    assert_eq!(c2.holder.as_deref(), Some("hub-worker"));
    let check = c.check("crates/omp-hub").await.unwrap();
    assert!(check["held"].as_bool().unwrap());
    c.release("crates/omp-hub", "hub-worker", c1.fencing_token.unwrap())
        .await
        .unwrap();
    let check = c.check("crates/omp-hub").await.unwrap();
    assert!(!check["held"].as_bool().unwrap());
    let claims = c.claims(Some("crates/"), None).await.unwrap();
    assert_eq!(claims["claims"].as_array().unwrap().len(), 0);
    task.abort();
}

fn spawn_serve_with(
    hub: &HubDir,
    idle_exit: Option<Duration>,
) -> JoinHandle<anyhow::Result<ServeExit>> {
    let hub = hub.clone();
    tokio::spawn(async move { omp_hub::serve_with(&hub, idle_exit).await })
}

async fn exit_of(task: JoinHandle<anyhow::Result<ServeExit>>) -> ServeExit {
    tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("hub did not exit")
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn idle_exit_never_under_a_registered_agent_then_after_the_limit() {
    let dir = TempDir::new().unwrap();
    let hub = HubDir::explicit(dir.path().to_path_buf(), None);
    let task = spawn_serve_with(&hub, Some(Duration::from_millis(600)));
    let c = connect(&hub).await;
    c.hello("idle-agent", 0).await.unwrap();
    // A registered agent keeps the hub up far past the idle limit.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(!task.is_finished(), "hub exited under a registered agent");
    assert!(hub.sock.exists());
    drop(c);
    assert_eq!(exit_of(task).await, ServeExit::Idle);
    assert!(!hub.sock.exists(), "idle exit leaves no stale socket");
}

#[tokio::test]
async fn orphaned_hub_exits_once_its_dir_is_gone_and_clients_left() {
    let home = TempDir::new().unwrap();
    let hub = HubDir::explicit(home.path().join(".omp/hub"), None);
    // Idle limit disabled: only the orphan rule can end it.
    let task = spawn_serve_with(&hub, None);
    let c = connect(&hub).await;
    c.hello("orphan-agent", 0).await.unwrap();
    std::fs::remove_dir_all(home.path().join(".omp")).unwrap();
    // Longer than one watchdog tick (2 s): still up while the agent is.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(!task.is_finished(), "hub exited under a registered agent");
    drop(c);
    assert_eq!(exit_of(task).await, ServeExit::Orphaned);
}
