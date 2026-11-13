//! Minimal async HTTP client plumbing for providers: one streaming HTTP request through a shared
//! `reqwest` client (TS SDKs run `maxRetries: 0`). Aborts surface as [`ProviderError::Aborted`];
//! HTTP failures as [`ProviderError::Http`]; request-send failures as
//! [`ProviderError::Connection`].

use std::sync::OnceLock;

use tokio_util::sync::CancellationToken;

use crate::utils::stream_failure::{
    ConnectionErrorKind, ConnectionErrorProfile, ProviderConnectionError, ProviderError,
    ProviderHttpError,
};

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
static H2_ALPN_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

/// The HTTP/1.1 client every provider shares, pinned with `http1_only()` so that enabling the
/// reqwest `http2` feature (bedrock) cannot change the transport of any other provider.
// invariant: one-shot static client init; a TLS-backend failure is an
// environment defect caught by tests.
#[allow(clippy::expect_used)]
fn client() -> &'static reqwest::Client {
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .http1_only()
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .expect("reqwest client")
    })
}

/// The TLS-ALPN client for bedrock https endpoints: HTTP/2 preferred (ALPN-negotiated), like the TS
/// default transport; cleartext bedrock endpoints go through `providers/bedrock/h2.rs` instead.
// invariant: one-shot static client init; a TLS-backend failure is an
// environment defect caught by tests.
#[allow(clippy::expect_used)]
fn h2_alpn_client() -> &'static reqwest::Client {
    H2_ALPN_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .expect("reqwest h2 client")
    })
}

/// Open (and pool) a connection to `url`'s origin on the shared HTTP/1.1 client, so the next
/// provider request to that host skips DNS, TCP and TLS. A bodiless `HEAD` to the origin: any
/// status counts, failures are ignored (the real request reports them).
pub async fn prewarm(url: &str) {
    let Ok(mut origin) = reqwest::Url::parse(url) else {
        return;
    };
    if !matches!(origin.scheme(), "http" | "https") {
        return;
    }
    origin.set_path("/");
    origin.set_query(None);
    let _ = client()
        .head(origin)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;
}

/// The wire transport a request is issued with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// HTTP/1.1 (the default; all providers but bedrock https).
    Http1,
    /// HTTP/2 preferred over TLS ALPN (bedrock https endpoints).
    H2Alpn,
}

/// An opened HTTP response: status, headers, and the byte stream.
pub struct HttpResponse {
    pub status: u16,
    pub headers: std::collections::HashMap<String, String>,
    body: reqwest::Response,
    signal: Option<CancellationToken>,
    /// Trailing bytes of a UTF-8 sequence split across body chunks, held until the rest arrives.
    utf8_tail: Vec<u8>,
    /// The request's connection-error profile: body-read failures on the AWS http2 profile surface
    /// the TS bedrock transport's mid-stream texts.
    pub(crate) connection: ConnectionErrorProfile,
    /// Longest silence allowed between body chunks (None: no limit).
    idle_timeout: Option<std::time::Duration>,
    /// Longest wait for the first body chunk (None: no limit).
    first_chunk_timeout: Option<std::time::Duration>,
    seen_chunk: bool,
}

/// The TS default for both the inter-chunk idle watchdog and the first-event watchdog.
const DEFAULT_STREAM_TIMEOUT_MS: u64 = 300_000;

/// TS `normalizeIdleTimeoutMs`: an env value overrides the fallback; non-numeric keeps the
/// fallback; `<= 0` disables the watchdog (None).
fn normalize_timeout_ms(env: Option<String>, fallback: u64) -> Option<u64> {
    let value = match env {
        None => fallback,
        Some(raw) => match raw.trim().parse::<f64>() {
            Ok(v) if v.is_finite() => {
                if v <= 0.0 {
                    return None;
                }
                // Truncation of a finite positive millisecond count is the TS `Math.trunc`.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    v.trunc() as u64
                }
            }
            _ => fallback,
        },
    };
    (value > 0).then_some(value)
}

/// TS `getOpenAIStreamIdleTimeoutMs`/`getStreamFirstEventTimeoutMs`: the stream watchdogs a
/// request runs with. `compat_*` are the model's `streamIdleTimeoutMs`/`streamFirstEventTimeoutMs`
/// (0 disables); env `PI_OPENAI_STREAM_IDLE_TIMEOUT_MS`/`PI_STREAM_IDLE_TIMEOUT_MS` and
/// `PI_STREAM_FIRST_EVENT_TIMEOUT_MS` override them. The first-event wait never undershoots idle.
fn stream_timeouts(compat_idle: Option<u64>, compat_first: Option<u64>) -> (Option<u64>, Option<u64>) {
    let env = |k: &str| std::env::var(k).ok();
    let idle = normalize_timeout_ms(
        env("PI_OPENAI_STREAM_IDLE_TIMEOUT_MS").or_else(|| env("PI_STREAM_IDLE_TIMEOUT_MS")),
        compat_idle.unwrap_or(DEFAULT_STREAM_TIMEOUT_MS),
    );
    let first_fallback = compat_first.unwrap_or_else(|| {
        DEFAULT_STREAM_TIMEOUT_MS.max(idle.unwrap_or(0))
    });
    let first = normalize_timeout_ms(env("PI_STREAM_FIRST_EVENT_TIMEOUT_MS"), first_fallback);
    (idle, first)
}

fn stream_timeout_error(profile: &ConnectionErrorProfile, what: &str, ms: u64) -> ProviderError {
    ProviderError::Connection(ProviderConnectionError {
        kind: ConnectionErrorKind::Timeout,
        profile: profile.clone(),
        cause: format!("provider stream {what} after {ms}ms"),
    })
}

impl HttpResponse {
    /// One body chunk, raced against the abort signal and the stream watchdog (first-chunk
    /// deadline until a chunk arrived, then the inter-chunk idle deadline).
    async fn read_chunk(&mut self) -> Result<Option<bytes::Bytes>, ProviderError> {
        if self
            .signal
            .as_ref()
            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
        {
            return Err(ProviderError::Aborted);
        }
        let limit = if self.seen_chunk { self.idle_timeout } else { self.first_chunk_timeout };
        let signal = self.signal.clone();
        let next = self.body.chunk();
        let watchdog = async {
            match limit {
                Some(d) => tokio::time::sleep(d).await,
                None => std::future::pending::<()>().await,
            }
        };
        let cancelled = async {
            match &signal {
                Some(s) => s.cancelled().await,
                None => std::future::pending::<()>().await,
            }
        };
        let chunk = tokio::select! {
            () = cancelled => return Err(ProviderError::Aborted),
            () = watchdog => {
                let what = if self.seen_chunk { "went idle" } else { "sent no data" };
                #[allow(clippy::cast_possible_truncation)]
                let ms = limit.map_or(0, |d| d.as_millis() as u64);
                return Err(stream_timeout_error(&self.connection, what, ms));
            }
            result = next => result,
        };
        match chunk {
            Ok(Some(bytes)) => {
                self.seen_chunk = true;
                Ok(Some(bytes))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(self.body_error(&error)),
        }
    }

    /// Read the next text chunk from the body (None at end of stream). A UTF-8 sequence split
    /// across body chunks is held until it completes; a sequence still incomplete at end of
    /// stream yields one U+FFFD.
    pub async fn next_text(&mut self) -> Result<Option<String>, ProviderError> {
        loop {
            let chunk = self.read_chunk().await?;
            let text = match chunk {
                Some(bytes) => {
                    self.utf8_tail.extend_from_slice(&bytes);
                    match std::str::from_utf8(&self.utf8_tail) {
                        Ok(text) => {
                            let text = text.to_string();
                            self.utf8_tail.clear();
                            text
                        }
                        Err(error) if error.error_len().is_none() => {
                            let valid_up_to = error.valid_up_to();
                            let text =
                                String::from_utf8_lossy(&self.utf8_tail[..valid_up_to]).to_string();
                            self.utf8_tail.drain(..valid_up_to);
                            text
                        }
                        Err(_) => {
                            let text = String::from_utf8_lossy(&self.utf8_tail).to_string();
                            self.utf8_tail.clear();
                            text
                        }
                    }
                }
                None if self.utf8_tail.is_empty() => return Ok(None),
                None => {
                    let text = String::from_utf8_lossy(&self.utf8_tail).to_string();
                    self.utf8_tail.clear();
                    text
                }
            };
            if !text.is_empty() {
                return Ok(Some(text));
            }
        }
    }

    /// Read the next raw byte chunk from the body (None at end of stream).
    pub async fn next_bytes(&mut self) -> Result<Option<Vec<u8>>, ProviderError> {
        Ok(self.read_chunk().await?.map(|bytes| bytes.to_vec()))
    }

    /// Classify a body-read failure: the AWS bedrock http2 profile surfaces the TS transport's
    /// mid-stream texts (the AWS SDK appends its deserialization hint); every other provider keeps
    /// the generic body-read error.
    fn body_error(&self, error: &reqwest::Error) -> ProviderError {
        if let ConnectionErrorProfile::AwsHttp2 { .. } = self.connection {
            let failure = crate::utils_inner::h2_classify::classify_reqwest_error(error);
            return ProviderError::Connection(ProviderConnectionError {
                kind: ConnectionErrorKind::H2MidStream(failure),
                profile: self.connection.clone(),
                cause: error.to_string(),
            });
        }
        ProviderError::Http(ProviderHttpError {
            message: format!("Failed to read provider response body: {error}"),
            status: Some(self.status),
            body: None,
            headers: self.headers.clone(),
            request_id: None,
            sdk_name: None,
            retry_after_ms: None,
            provider_error_type: None,
        })
    }

    /// Read the entire body as text (for error responses and small payloads).
    pub async fn read_all_text(&mut self) -> Result<String, ProviderError> {
        let mut out = String::new();
        while let Some(chunk) = self.next_text().await? {
            out.push_str(&chunk);
        }
        Ok(out)
    }
}

pub struct RequestOptions {
    pub method: reqwest::Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
    pub signal: Option<CancellationToken>,
    pub timeout_ms: Option<u64>,
    /// The provider family's connection-error shape (fixed texts, names, and error codes the TS
    /// binary surfaces per SDK); the openai/anthropic `Sdk` default covers the Stainless-generated
    /// SDK family.
    pub connection: ConnectionErrorProfile,
    /// The wire transport (HTTP/1.1 by default; bedrock https requests use ALPN-negotiated HTTP/2).
    pub transport: Transport,
    /// The model's `streamIdleTimeoutMs` (None: the 300 s default; 0 disables). Env overrides.
    pub stream_idle_timeout_ms: Option<u64>,
    /// The model's `streamFirstEventTimeoutMs` (None: max(300 s, idle); 0 disables). Env overrides.
    pub stream_first_event_timeout_ms: Option<u64>,
}

impl RequestOptions {
    /// Request options with the openai/anthropic `Sdk` connection profile.
    pub fn new(method: reqwest::Method, url: String) -> Self {
        Self {
            method,
            url,
            headers: Vec::new(),
            body: None,
            signal: None,
            timeout_ms: None,
            connection: ConnectionErrorProfile::Sdk,
            transport: Transport::Http1,
            stream_idle_timeout_ms: None,
            stream_first_event_timeout_ms: None,
        }
    }
}

/// Issue a request and return the response with a streaming body. No retries: retry ownership lives
/// with the caller (agent layer), matching the TS `maxRetries: 0` client configuration.
pub async fn send(request: RequestOptions) -> Result<HttpResponse, ProviderError> {
    let signal = request.signal.clone();
    if signal
        .as_ref()
        .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
    {
        return Err(ProviderError::Aborted);
    }

    let client = match request.transport {
        Transport::Http1 => client(),
        Transport::H2Alpn => h2_alpn_client(),
    };
    let mut builder = client.request(request.method, &request.url);
    for (name, value) in &request.headers {
        builder = builder.header(name, value);
    }
    // A JSON request body carries `content-type: application/json` —
    // strict OpenAI-compatible frontends (self-hosted vLLM) validate the
    // media type and 400 a label-less body. A caller-supplied label
    // (bedrock's signed set, mistral's, a user's `model.headers`/
    // options override) always wins.
    if request.body.is_some()
        && !request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
    {
        builder = builder.header("content-type", "application/json");
    }
    if let Some(body) = request.body {
        builder = builder.body(body);
    }

    let (idle_ms, first_ms) = stream_timeouts(
        request.stream_idle_timeout_ms,
        request.stream_first_event_timeout_ms,
    );
    // Response headers must arrive within the caller's request timeout, else within the
    // first-event watchdog (a hung provider used to block the turn forever when a signal was set).
    let (head_ms, head_cause) = match request.timeout_ms {
        Some(ms) => (Some(ms), "request exceeded the {ms}ms timeout"),
        None => (first_ms, "provider sent no response headers after {ms}ms"),
    };
    let send_future = builder.send();
    let head_watchdog = async {
        match head_ms {
            Some(ms) => tokio::time::sleep(std::time::Duration::from_millis(ms)).await,
            None => std::future::pending::<()>().await,
        }
    };
    let cancelled = async {
        match &signal {
            Some(s) => s.cancelled().await,
            None => std::future::pending::<()>().await,
        }
    };
    let response = tokio::select! {
        () = cancelled => return Err(ProviderError::Aborted),
        () = head_watchdog => {
            let ms = head_ms.unwrap_or(0);
            return Err(ProviderError::Connection(ProviderConnectionError {
                kind: ConnectionErrorKind::Timeout,
                profile: request.connection.clone(),
                cause: head_cause.replace("{ms}", &ms.to_string()),
            }));
        }
        result = send_future => result,
    };

    // The TS SDKs surface request-send failures as their fixed connection error texts: the
    // openai/anthropic SDK family throws `APIConnectionError` ("Connection error.") /
    // `APIConnectionTimeoutError` ("Request timed out.") for every fetch failure; providers whose
    // SDK appends the raw cause (mistral) or surfaces undici's raw text (codex, bedrock, google:
    // "fetch failed") rewrite it at their catch site.
    let response = response.map_err(|error| {
        let kind = if error.is_timeout() {
            ConnectionErrorKind::Timeout
        } else if error.is_connect() {
            ConnectionErrorKind::Connect
        } else if matches!(request.connection, ConnectionErrorProfile::AwsHttp2 { .. }) {
            // Pre-response http2 failure on the bedrock https transport: the h2 failure detail (no
            // deserialization hint — no response yet).
            ConnectionErrorKind::H2Request(crate::utils_inner::h2_classify::classify_reqwest_error(
                &error,
            ))
        } else {
            // The peer closed or reset after the connection was established but before the response
            // arrived (only distinguishable from refused connects on the AWS handler surfaces).
            ConnectionErrorKind::Reset
        };
        ProviderError::Connection(ProviderConnectionError {
            kind,
            profile: request.connection.clone(),
            cause: error.to_string(),
        })
    })?;

    let status = response.status().as_u16();
    let mut headers = std::collections::HashMap::new();
    for (name, value) in response.headers() {
        if let Ok(value) = value.to_str() {
            headers.insert(name.as_str().to_ascii_lowercase(), value.to_string());
        }
    }

    Ok(HttpResponse {
        status,
        headers,
        body: response,
        signal,
        utf8_tail: Vec::new(),
        connection: request.connection,
        idle_timeout: idle_ms.map(std::time::Duration::from_millis),
        first_chunk_timeout: first_ms.map(std::time::Duration::from_millis),
        seen_chunk: false,
    })
}

/// JSON POST helper used by non-streaming calls (OAuth token refresh, catalogs).
#[allow(dead_code)] // token refresh/catalog fetches for upcoming providers
pub async fn post_json(
    url: &str,
    headers: Vec<(String, String)>,
    body: serde_json::Value,
    signal: Option<CancellationToken>,
) -> Result<(u16, serde_json::Value), ProviderError> {
    let mut response = send(RequestOptions {
        headers,
        body: Some(body.to_string()),
        signal,
        timeout_ms: Some(30_000),
        ..RequestOptions::new(reqwest::Method::POST, url.to_string())
    })
    .await?;
    let status = response.status;
    let text = response.read_all_text().await?;
    let parsed = if text.trim().is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_str(&text)
            .map_err(|error| ProviderError::Message(format!("Invalid JSON response: {error}")))?
    };
    Ok((status, parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve one plain-text body split across two writes. The second write is held until `hold`
    /// fires, so the chunk boundary lands where the caller put it.
    async fn serve_split_body(
        first: Vec<u8>,
        second: Vec<u8>,
        hold: tokio::sync::oneshot::Receiver<()>,
    ) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = String::new();
            let mut buffer = [0u8; 4096];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                if read == 0 {
                    return;
                }
                head.push_str(&String::from_utf8_lossy(&buffer[..read]));
                if head.contains("\r\n\r\n") {
                    break;
                }
            }
            let content_length = first.len() + second.len();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&first).await.unwrap();
            hold.await.unwrap();
            socket.write_all(&second).await.unwrap();
        });
        addr
    }

    /// A multi-byte character split across two body chunks decodes once, whole: the first chunk
    /// ends inside 你, and the server sends the rest only after that chunk was read.
    #[tokio::test]
    async fn next_text_rejoins_a_multi_byte_character_split_across_body_chunks() {
        let (hold_tx, hold_rx) = tokio::sync::oneshot::channel();
        let addr = serve_split_body(b"hi \xE4".to_vec(), b"\xBD\xA0!".to_vec(), hold_rx).await;
        let mut response = send(RequestOptions::new(
            reqwest::Method::GET,
            format!("http://{addr}/"),
        ))
        .await
        .unwrap();
        let mut text = String::new();
        if let Some(chunk) = response.next_text().await.unwrap() {
            text.push_str(&chunk);
            let _ = hold_tx.send(());
        }
        while let Some(chunk) = response.next_text().await.unwrap() {
            text.push_str(&chunk);
        }
        assert_eq!(text, "hi 你!");
    }

    /// A body that ends inside a multi-byte character yields the valid text plus exactly one
    /// U+FFFD at end of stream.
    #[tokio::test]
    async fn next_text_ends_a_torn_body_with_one_replacement_character() {
        let (hold_tx, hold_rx) = tokio::sync::oneshot::channel();
        let addr = serve_split_body(b"cut \xE4".to_vec(), b"\xBD".to_vec(), hold_rx).await;
        let mut response = send(RequestOptions::new(
            reqwest::Method::GET,
            format!("http://{addr}/"),
        ))
        .await
        .unwrap();
        let mut text = String::new();
        if let Some(chunk) = response.next_text().await.unwrap() {
            text.push_str(&chunk);
            let _ = hold_tx.send(());
        }
        while let Some(chunk) = response.next_text().await.unwrap() {
            text.push_str(&chunk);
        }
        assert_eq!(text, "cut \u{FFFD}");
    }

    /// A provider that stops sending mid-stream fails the read with a timeout instead of hanging
    /// the turn forever (the omt pane hang).
    #[tokio::test]
    async fn a_stalled_body_times_out_after_the_idle_limit() {
        let (_hold_tx, hold_rx) = tokio::sync::oneshot::channel();
        let addr = serve_split_body(b"data: 1".to_vec(), b"never".to_vec(), hold_rx).await;
        let mut response = send(RequestOptions {
            signal: Some(CancellationToken::new()),
            stream_idle_timeout_ms: Some(150),
            ..RequestOptions::new(reqwest::Method::GET, format!("http://{addr}/"))
        })
        .await
        .unwrap();
        assert_eq!(response.next_text().await.unwrap().as_deref(), Some("data: 1"));
        let started = std::time::Instant::now();
        let error = response.next_text().await.unwrap_err();
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        match error {
            ProviderError::Connection(e) => {
                assert_eq!(e.kind, ConnectionErrorKind::Timeout);
                assert!(e.cause.contains("went idle after 150ms"), "{}", e.cause);
            }
            other => panic!("expected a connection timeout, got {other:?}"),
        }
    }

    /// No response headers at all: the first-event watchdog fires even with an abort signal set.
    #[tokio::test]
    async fn missing_response_headers_time_out_even_with_a_signal() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        });
        let result = send(RequestOptions {
            signal: Some(CancellationToken::new()),
            stream_first_event_timeout_ms: Some(150),
            ..RequestOptions::new(reqwest::Method::GET, format!("http://{addr}/"))
        })
        .await;
        match result {
            Err(ProviderError::Connection(e)) => {
                assert_eq!(e.kind, ConnectionErrorKind::Timeout);
                assert!(e.cause.contains("no response headers after 150ms"), "{}", e.cause);
            }
            Err(other) => panic!("expected a connection timeout, got {other:?}"),
            Ok(_) => panic!("expected a timeout"),
        }
    }

    #[test]
    fn stream_timeout_env_semantics_match_ts() {
        assert_eq!(normalize_timeout_ms(None, 300_000), Some(300_000));
        assert_eq!(normalize_timeout_ms(Some("0".into()), 300_000), None);
        assert_eq!(normalize_timeout_ms(Some("-5".into()), 300_000), None);
        assert_eq!(normalize_timeout_ms(Some("abc".into()), 300_000), Some(300_000));
        assert_eq!(normalize_timeout_ms(Some("1500.9".into()), 300_000), Some(1500));
        assert_eq!(normalize_timeout_ms(None, 0), None);
    }
}
