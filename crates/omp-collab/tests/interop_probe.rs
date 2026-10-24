//! Throwaway probe: does `RelayHandle::send_binary` followed immediately by
//! `close()` deliver the queued frame? Documented: "queued frames still flush
//! to the live socket ahead of the close frame".

use std::time::Duration;

use futures::StreamExt;

use omp_collab::crypto::CollabCipher;
use omp_collab::relay_client::{connect_relay, CollabRole, RelayEvent};
use omp_collab::relay_server::start_local_relay;

#[tokio::test(flavor = "multi_thread")]
async fn queued_send_flushes_before_close() {
    let relay = start_local_relay(0).unwrap();
    let (host, mut host_events) = connect_relay(
        &relay.url,
        "probe-room-1",
        CollabRole::Host,
        &Default::default(),
    )
    .await
    .unwrap();
    while let Some(ev) = host_events.next().await {
        if matches!(ev, RelayEvent::Opened) {
            break;
        }
    }
    // Held so the guest connection stays open for the whole probe.
    let (_guest, mut guest_events) = connect_relay(
        &relay.url,
        "probe-room-1",
        CollabRole::Guest,
        &Default::default(),
    )
    .await
    .unwrap();
    let mut guest_opened = false;
    let cipher = CollabCipher::new(&[7u8; 32]);
    let sealed = cipher.seal(b"probe-payload");
    let mut envelope = vec![0, 0, 0, 0];
    envelope.extend_from_slice(&sealed);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut guest_frames: Vec<Vec<u8>> = Vec::new();
    let mut guest_closed: Option<String> = None;
    let mut sent = false;
    while tokio::time::Instant::now() < deadline {
        tokio::select! {
            ev = guest_events.next() => match ev {
                Some(RelayEvent::Opened) => guest_opened = true,
                Some(RelayEvent::Binary(data)) => guest_frames.push(data.to_vec()),
                Some(RelayEvent::Closed { code, reason, .. }) => {
                    guest_closed = Some(format!("{code}:{reason}"));
                    break;
                }
                _ => {}
            },
            _ = tokio::time::sleep(Duration::from_millis(50)) => {
                if guest_opened && !sent {
                    assert!(host.send_binary(envelope.clone()), "send queued");
                    host.close(); // immediately, no grace
                    sent = true;
                }
                if sent && guest_closed.is_some() {
                    break;
                }
            }
        }
    }
    println!("guest_frames={}", guest_frames.len());
    println!("guest_closed={guest_closed:?}");
    assert!(
        !guest_frames.is_empty(),
        "queued frame was LOST across immediate close() (documented flush did not happen)"
    );
}
