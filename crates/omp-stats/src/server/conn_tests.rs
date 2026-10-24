//! Real-TCP behaviour of the connection layer: Host/Origin guard, header-read
//! timeout, idle timeout (SSE exempt).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use super::{start_server, StartServerOptions};
use crate::testutil::TestHome;

fn start(options: StartServerOptions) -> (TestHome, super::StatsServerHandle) {
    let home = TestHome::new();
    let handle = start_server(0, "127.0.0.1", options).expect("start");
    (home, handle)
}

fn connect(port: u16) -> TcpStream {
    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("timeout");
    stream
}

fn status_of(raw: &str) -> u16 {
    raw.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn request(port: u16, headers: &str) -> String {
    let mut stream = connect(port);
    write!(
        stream,
        "GET /api/stats/models HTTP/1.1\r\n{headers}Connection: close\r\n\r\n"
    )
    .expect("send");
    let mut raw = String::new();
    let _ = stream.read_to_string(&mut raw);
    raw
}

/// Whether the server closed the connection (EOF) within the read timeout.
fn closed_by_server(stream: &mut TcpStream) -> bool {
    let mut buf = [0u8; 256];
    matches!(stream.read(&mut buf), Ok(0))
}

#[test]
fn foreign_host_and_origin_get_403_loopback_gets_through() {
    let (_home, handle) = start(StartServerOptions::default());
    let port = handle.port;
    assert_eq!(status_of(&request(port, "Host: evil.example\r\n")), 403);
    assert_eq!(
        status_of(&request(port, &format!("Host: evil.example:{port}\r\n"))),
        403
    );
    assert_eq!(
        status_of(&request(
            port,
            &format!("Host: localhost:{port}\r\nOrigin: http://evil.example\r\n")
        )),
        403
    );
    let ok = request(
        port,
        &format!("Host: 127.0.0.1:{port}\r\nOrigin: http://localhost:{port}\r\n"),
    );
    assert_eq!(status_of(&ok), 200, "{ok}");
    handle.stop();
}

#[test]
fn slow_header_connection_is_closed() {
    let (_home, handle) = start(StartServerOptions {
        header_read_timeout: Some(Duration::from_millis(300)),
        idle_timeout: Some(Duration::from_secs(60)),
    });
    let mut stream = connect(handle.port);
    write!(
        stream,
        "GET /api/stats/models HTTP/1.1\r\nHost: localhost:{}\r\n",
        handle.port
    )
    .expect("partial headers");
    let started = std::time::Instant::now();
    assert!(
        closed_by_server(&mut stream),
        "server should drop a stalled header read"
    );
    assert!(started.elapsed() < Duration::from_secs(4));
    handle.stop();
}

#[test]
fn idle_keepalive_connection_closes_but_sse_stays_open() {
    let (_home, handle) = start(StartServerOptions {
        header_read_timeout: Some(Duration::from_secs(30)),
        idle_timeout: Some(Duration::from_millis(400)),
    });
    let host = format!("Host: localhost:{}\r\n", handle.port);

    let mut idle = connect(handle.port);
    write!(idle, "GET /api/stats/models HTTP/1.1\r\n{host}\r\n").expect("send");
    let mut sse = connect(handle.port);
    write!(sse, "GET /api/events HTTP/1.1\r\n{host}\r\n").expect("send");
    sse.set_read_timeout(Some(Duration::from_millis(1500)))
        .expect("timeout");

    // The idle keep-alive connection answers once, then is closed by the server.
    let mut raw = Vec::new();
    let started = std::time::Instant::now();
    idle.read_to_end(&mut raw)
        .expect("server closes the idle connection (EOF), not a read timeout");
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(status_of(&String::from_utf8_lossy(&raw)), 200);

    // The SSE stream outlives the idle timeout: reads time out, never EOF.
    let mut buf = [0u8; 4096];
    let first = sse.read(&mut buf).expect("first sse frame");
    assert!(first > 0);
    std::thread::sleep(Duration::from_millis(900));
    loop {
        match sse.read(&mut buf) {
            Ok(0) => panic!("SSE connection was closed by the idle timeout"),
            Ok(_) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                break
            }
            Err(e) => panic!("sse read: {e}"),
        }
    }
    drop(sse);
    handle.stop();
}

/// Regression (review): a rejected `/api/events` (403 from the Host guard)
/// must not mark the connection as an SSE stream, which used to exempt it
/// from the idle timeout forever.
#[test]
fn rejected_events_request_is_still_idle_closed() {
    let (_home, handle) = start(StartServerOptions {
        header_read_timeout: Some(Duration::from_secs(30)),
        idle_timeout: Some(Duration::from_millis(400)),
    });
    let mut stream = connect(handle.port);
    write!(
        stream,
        "GET /api/events HTTP/1.1\r\nHost: evil.example\r\n\r\n"
    )
    .expect("send");
    let mut raw = Vec::new();
    let started = std::time::Instant::now();
    stream
        .read_to_end(&mut raw)
        .expect("server closes the rejected connection (EOF), not a read timeout");
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(status_of(&String::from_utf8_lossy(&raw)), 403);
    handle.stop();
}
