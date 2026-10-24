//! Google Cloud Code Assist streaming provider (`google-gemini-cli` and `google-antigravity`):
//! `POST {base}/v1internal:streamGenerateContent?alt=sse` with the OAuth bearer in `Authorization`
//! and the request wrapped in the Cloud Code Assist envelope `{project, model, request: {contents,
//! systemInstruction, generationConfig, tools, toolConfig}}` (TS `buildRequest`). Port of the TS
//! provider `google-gemini-cli.ts`:
//!
//! - endpoints: the plain gemini-cli provider streams from the catalog `baseUrl` (default
//!   `cloudcode-pa.googleapis.com`); antigravity fails over `daily-cloudcode-pa` ->
//!   `daily-cloudcode-pa.sandbox` with a `lastGoodEndpoint` sticky preference;
//! - antigravity requests add the metadata envelope fields (`requestId`, `userAgent`,
//!   `requestType: "agent"`) and per-request `sessionId`/`labels`, the fixed per-wire-id
//!   `maxOutputTokens` cap, and Claude rows force `functionCallingConfig.mode: "VALIDATED"`;
//! - SSE chunks arrive wrapped as `{"response": {...GenerateContentResponse}}` with in-band
//!   `{"error": {...}}` failures and `promptFeedback` block reports;
//! - the credential is the structured OAuth key `{token, projectId, ...}` (TS `getOAuthApiKey`),
//!   parsed for both the bearer and the `project` the envelope reports.

use serde_json::{json, Map, Value};
use sha2::Digest as _;

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageEventWriter,
};
use crate::providers::google::GoogleOptions;
use crate::providers::google_shared::{
    antigravity_claude_tool_mode, convert_messages, convert_tools, map_tool_choice,
    API_GOOGLE_GEMINI_CLI,
};
use crate::providers::google_stream::GoogleStreamState;
use crate::providers::simple_options::build_base_options;
use crate::registry::Provider;
use crate::types::{
    done_reason, error_reason, AssistantMessage, Context, Model, SimpleStreamOptions, StopReason,
    StreamOptions, Usage,
};
use crate::utils_inner::diagnostics::now_ms;
use crate::utils_inner::http::{send, HttpResponse, RequestOptions};
use crate::utils_inner::http_inspector::{finalize_http_error, RawHttpRequestDump};
use crate::utils_inner::json_parse::parse_json_with_repair;
use crate::utils_inner::sanitize_unicode::sanitize_surrogates;
use crate::utils_inner::sse::SseDecoder;
use crate::utils_inner::stream_failure::{ConnectionErrorProfile, ProviderError};

/// Default Cloud Code Assist host (TS `DEFAULT_ENDPOINT`): the gemini-cli provider's endpoint.
pub const BASE_URL_STABLE: &str = "https://cloudcode-pa.googleapis.com";
/// Antigravity production host (`ANTIGRAVITY_DAILY_ENDPOINT`), first in the failover list.
pub const BASE_URL_PRODUCTION: &str = "https://daily-cloudcode-pa.googleapis.com";
/// Antigravity sandbox host (`ANTIGRAVITY_SANDBOX_ENDPOINT`), second in the failover list.
pub const BASE_URL_SANDBOX: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
/// Antigravity failover order (TS `ANTIGRAVITY_ENDPOINT_FALLBACKS`): `[daily, sandbox]`.
pub const ENDPOINTS: [&str; 2] = [BASE_URL_PRODUCTION, BASE_URL_SANDBOX];

const STREAM_FIRST_EVENT_TIMEOUT_MS: u64 = 300_000;
const FIRST_EVENT_TIMEOUT_ERROR: &str =
    "Cloud Code Assist stream timed out while waiting for the first event";
/// TS `MAX_RETRIES`: extra fetch attempts for the last endpoint in the failover list.
const MAX_HTTP_RETRIES: usize = 3;
const RETRY_BASE_DELAY_MS: u64 = 1_000;
/// TS `RATE_LIMIT_BUDGET_MS`: per-attempt backoff ceiling.
const RETRY_MAX_DELAY_MS: u64 = 5 * 60 * 1000;
/// TS `MAX_EMPTY_STREAM_RETRIES` (google-shared): replays of an empty stream per endpoint.
const MAX_EMPTY_STREAM_RETRIES: u32 = 2;
/// TS `EMPTY_STREAM_BASE_DELAY_MS` (google-shared).
const EMPTY_STREAM_BASE_DELAY_MS: u64 = 500;
const USER_AGENT: &str = "antigravity";
const CLAUDE_THINKING_BETA_HEADER: &str = "interleaved-thinking-2025-05-14";
/// TS `GOOGLE_GEMINI_REFRESH_SKEW_MS` / `ANTIGRAVITY_REFRESH_SKEW_MS`: an already-expired
/// credential is refused instead of POSTing a stale token.
const REFRESH_SKEW_MS: i64 = 60_000;
/// `google-antigravity-forced-tool.md`: Cloud Code Assist drops `toolConfig` on Antigravity's
/// Gemini routes, so a forced choice is restated in the transcript as a tool-only turn.
const FORCED_TOOL_DIRECTIVE: &str = "TOOL-ONLY TURN. This turn accepts a tool call and nothing else; a text reply here is discarded unread and you will be re-prompted. Emit the tool call now.";

/// TS `getAntigravityUserAgent` (`antigravity/hub/{version}` with the hub's aidev_client
/// metadata; version falls back to the pinned hub release).
fn antigravity_user_agent() -> String {
    let version = std::env::var("PI_AI_ANTIGRAVITY_VERSION").unwrap_or_else(|_| "2.19.1".into());
    let cl = std::env::var("PI_AI_ANTIGRAVITY_CL").unwrap_or_else(|_| "963137146".into());
    let os = std::env::var("PI_AI_ANTIGRAVITY_OS").unwrap_or_else(|_| "darwin".into());
    let arch = std::env::var("PI_AI_ANTIGRAVITY_ARCH").unwrap_or_else(|_| "arm64".into());
    format!("antigravity/hub/{version} (aidev_client; os_type={os}; arch={arch}; cl={cl})")
}

/// TS `getGeminiCliUserAgent` (`GeminiCLI/{version}/{model} (os; arch; terminal)`).
fn gemini_cli_user_agent(model_id: &str) -> String {
    let version = std::env::var("PI_AI_GEMINI_CLI_VERSION").unwrap_or_else(|_| "0.46.0".into());
    format!("GeminiCLI/{version}/{model_id} (linux; x64; terminal)")
}

/// Per-process sticky preference: the last endpoint that completed a stream (`lastGoodEndpoint`).
static LAST_GOOD_ENDPOINT: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

fn last_good_endpoint() -> Option<String> {
    LAST_GOOD_ENDPOINT
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn note_good_endpoint(endpoint: &str) {
    *LAST_GOOD_ENDPOINT
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(endpoint.to_string());
}

/// The wire model id the envelope reports (`options.requestModelId ?? model.requestModelId ??
/// model.id`).
fn wire_model_id(model: &Model) -> String {
    model
        .request_model_id
        .clone()
        .unwrap_or_else(|| model.id.clone())
}

/// Endpoints for one request (TS endpoint selection): the gemini-cli arm streams from the catalog
/// `baseUrl` (or the stable host); the antigravity arm fails over daily -> sandbox with the
/// `lastGoodEndpoint` sticky preference, unless the catalog pins a custom base.
fn endpoints_for(model: &Model) -> Vec<String> {
    let configured = model.base_url.trim().trim_end_matches('/').to_string();
    if model.provider != "google-antigravity" {
        return vec![if configured.is_empty() {
            BASE_URL_STABLE.to_string()
        } else {
            configured
        }];
    }
    if !configured.is_empty()
        && configured != BASE_URL_PRODUCTION
        && configured != BASE_URL_SANDBOX
    {
        return vec![configured];
    }
    let mut endpoints: Vec<String> = ENDPOINTS.iter().map(|e| (*e).to_string()).collect();
    if let Some(good) = last_good_endpoint() {
        if endpoints.contains(&good) {
            endpoints.retain(|e| *e != good);
            endpoints.insert(0, good);
        }
    }
    endpoints
}

/// TS `isTransientStatus`: retriable on the next endpoint / attempt.
fn is_transient_status(status: u16) -> bool {
    matches!(status, 408 | 429) || status >= 500
}

/// Whether a transport-level failure is retriable (no status: connection-classified).
fn error_is_transient(error: &ProviderError) -> bool {
    match error.status() {
        Some(status) => is_transient_status(status),
        None => matches!(error, ProviderError::Connection(_)),
    }
}

/// `Ips` in the TS provider: lazily-created mutable antigravity session state.
#[derive(Debug, Default)]
pub struct GeminiCliSessionState {
    pub agent_id: Option<String>,
    pub trajectory_id: Option<String>,
    pub session_id: Option<String>,
    pub step_index: Option<u64>,
    pub last_execution_id: Option<String>,
}

fn uuid_v4() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

const INT63_MASK: u64 = (1u64 << 63) - 1;
const ANTIGRAVITY_RANDOM_BOUND: u64 = 9_000_000_000_000_000_000;

/// TS `deriveSignedDecimalFromHash`: the first 8 SHA-256 bytes as a signed-decimal int63.
fn signed_decimal_from_hash(text: &str) -> String {
    let digest = sha2::Sha256::digest(text.as_bytes());
    let mut value: u64 = 0;
    for byte in &digest[..8] {
        value = (value << 8) | u64::from(*byte);
    }
    format!("-{}", value & INT63_MASK)
}

/// TS `randomBoundedInt63` + `randomSignedDecimalSessionId`.
fn random_signed_decimal_session_id() -> String {
    use rand::Rng;
    loop {
        let mut bytes = [0u8; 8];
        rand::thread_rng().fill(&mut bytes);
        let value = u64::from_be_bytes(bytes) & INT63_MASK;
        if value < ANTIGRAVITY_RANDOM_BOUND {
            return format!("-{value}");
        }
    }
}

/// TS `getFirstUserTextForAntigravitySession`: the first user message's text, when it has one.
fn first_user_text(context: &Context) -> Option<String> {
    context
        .messages
        .iter()
        .find_map(|message| match message {
            crate::types::Message::User(user) | crate::types::Message::Developer(user) => {
                match &user.content {
                    crate::types::UserMessageContent::Text(text) => Some(text.clone()),
                    crate::types::UserMessageContent::Blocks(blocks) => blocks.iter().find_map(
                        |block| match block {
                            crate::types::UserOrToolContent::Text(text) => {
                                Some(text.text.clone())
                            }
                            _ => None,
                        },
                    ),
                }
            }
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
}

/// TS `deriveAntigravitySessionId`: hash of the first user text, or a random signed decimal.
fn derive_session_id(context: &Context) -> String {
    match first_user_text(context) {
        Some(text) => signed_decimal_from_hash(&text),
        None => random_signed_decimal_session_id(),
    }
}

/// TS `AntigravityModelWireProfile` table: per-wire-id `labels.model_enum` token and the fixed
/// `generationConfig.maxOutputTokens` cap the backend enforces.
pub struct AntigravityWireProfile {
    pub model_enum: Option<&'static str>,
    pub max_output_tokens: u64,
}

/// TS `getAntigravityModelWireProfile`, keyed by the routed upstream wire id.
fn antigravity_wire_profile(wire_model_id: &str) -> Option<&'static AntigravityWireProfile> {
    const fn profile(
        model_enum: Option<&'static str>,
        max_output_tokens: u64,
    ) -> AntigravityWireProfile {
        AntigravityWireProfile {
            model_enum,
            max_output_tokens,
        }
    }
    static PROFILES: [(&str, AntigravityWireProfile); 7] = [
        (
            "gemini-3.5-flash-extra-low",
            profile(Some("MODEL_PLACEHOLDER_M187"), 65_536),
        ),
        (
            "gemini-3.5-flash-low",
            profile(Some("MODEL_PLACEHOLDER_M20"), 65_536),
        ),
        (
            "gemini-3-flash-agent",
            profile(Some("MODEL_PLACEHOLDER_M132"), 65_536),
        ),
        (
            "gemini-3.1-pro-low",
            profile(Some("MODEL_PLACEHOLDER_M36"), 65_535),
        ),
        (
            "gemini-pro-agent",
            profile(Some("MODEL_PLACEHOLDER_M16"), 65_535),
        ),
        // Claude on `daily-cloudcode-pa` rejects `maxOutputTokens > 64000` with a 400; the
        // model_enum label is untracked for these ids.
        ("claude-sonnet-4-6", profile(None, 64_000)),
        ("claude-opus-4-6-thinking", profile(None, 64_000)),
    ];
    PROFILES
        .iter()
        .find(|(id, _)| *id == wire_model_id)
        .map(|(_, profile)| profile)
}

/// The antigravity request envelope metadata (`buildAntigravityRequestEnvelope`): session id,
/// structured `requestId` (`agent/<agentId>/<ts>/<trajectoryId>/<step>`), and the labels the real
/// client tags each request with.
struct AntigravityMetadata {
    session_id: String,
    request_id: String,
    labels: Map<String, Value>,
}

fn antigravity_metadata(
    state: &mut GeminiCliSessionState,
    context: &Context,
    is_claude: bool,
    usage_label: Option<&str>,
    wire_model_id: &str,
) -> AntigravityMetadata {
    let agent_id = state.agent_id.get_or_insert_with(uuid_v4).clone();
    let trajectory_id = state
        .trajectory_id
        .get_or_insert_with(uuid_v4)
        .clone();
    let session_id = state
        .session_id
        .get_or_insert_with(|| derive_session_id(context))
        .clone();
    let step = state.step_index.unwrap_or(1) + 1;
    state.step_index = Some(step);
    let request_id = format!("agent/{agent_id}/{}/{trajectory_id}/{step}", now_ms());

    let mut labels = Map::new();
    if let Some(last) = &state.last_execution_id {
        labels.insert("last_execution_id".into(), json!(last));
    }
    labels.insert("last_step_index".into(), json!((step - 1).to_string()));
    if let Some(model_enum) = antigravity_wire_profile(wire_model_id).and_then(|p| p.model_enum) {
        labels.insert("model_enum".into(), json!(model_enum));
    }
    labels.insert("trajectory_id".into(), json!(trajectory_id));
    let used_claude = usage_label.unwrap_or(if is_claude { "true" } else { "false" });
    labels.insert("used_claude".into(), json!(used_claude));
    labels.insert("used_claude_conservative".into(), json!(used_claude));

    AntigravityMetadata {
        session_id,
        request_id,
        labels,
    }
}

/// TS `parseGeminiCliCredentials` (`_Kn`): the structured OAuth key carries the bearer (`token`)
/// and the Cloud project the envelope reports. Returns the token, the project id and the
/// normalized expiry.
fn parse_credentials(credential: &str) -> Result<(String, String, Option<i64>), ProviderError> {
    const INVALID: &str =
        "Invalid Google Cloud Code Assist credentials. Use /login to re-authenticate.";
    const MISSING: &str =
        "Missing token or projectId in Google Cloud credentials. Use /login to re-authenticate.";
    let parsed: Value = serde_json::from_str(credential)
        .map_err(|_| ProviderError::Message(INVALID.to_string()))?;
    let Some(fields) = parsed.as_object() else {
        return Err(ProviderError::Message(INVALID.to_string()));
    };
    let token = fields
        .get("token")
        .or_else(|| fields.get("access"))
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty());
    let project_id = fields
        .get("projectId")
        .or_else(|| fields.get("project_id"))
        .and_then(Value::as_str)
        .filter(|project| !project.is_empty());
    let (Some(token), Some(project_id)) = (token, project_id) else {
        return Err(ProviderError::Message(MISSING.to_string()));
    };
    // TS `normalizeExpiryMs`: seconds-epoch values scale to milliseconds.
    let expires_at = fields
        .get("expiresAt")
        .or_else(|| fields.get("expires"))
        .and_then(Value::as_f64)
        .filter(|expires| expires.is_finite() && *expires > 0.0)
        .map(|expires| {
            let millis = if expires < 10_000_000_000.0 {
                expires * 1000.0
            } else {
                expires
            };
            millis as i64
        });
    Ok((token.to_string(), project_id.to_string(), expires_at))
}

/// TS `systemInstruction` shape: an object with parts, tagged `role: "user"` on antigravity to
/// mirror the real client.
fn system_instruction(system_prompt: &str, antigravity: bool) -> Value {
    let mut instruction = Map::new();
    if antigravity {
        instruction.insert("role".into(), json!("user"));
    }
    instruction.insert(
        "parts".into(),
        json!([{ "text": sanitize_surrogates(system_prompt) }]),
    );
    Value::Object(instruction)
}

/// A loose `model.compat.<flag>` lookup for flags outside the typed compat structs.
fn compat_flag(model: &Model, key: &str) -> bool {
    crate::compat::ModelView::of(model)
        .compat
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Build the Cloud Code Assist request body (TS `buildRequest`). Both arms wrap the plain
/// GenerativeLanguage fields in the envelope `{project, model, request}`; the antigravity arm
/// adds `requestId`/`userAgent`/`requestType` plus the in-request `sessionId`/`labels`.
fn build_body(
    model: &Model,
    context: &Context,
    options: &GoogleOptions,
    project_id: &str,
    metadata: Option<&AntigravityMetadata>,
) -> Map<String, Value> {
    let antigravity = metadata.is_some();
    let claude_mode = antigravity_claude_tool_mode(model);
    let wire_model_id = wire_model_id(model);
    let mut contents = convert_messages(model, context);

    let mut generation_config = Map::new();
    if let Some(temperature) = options.base.temperature {
        generation_config.insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = options.base.max_tokens {
        generation_config.insert("maxOutputTokens".into(), json!(max_tokens));
    }
    if let Some(thinking) = &options.thinking {
        if thinking.enabled && model.reasoning {
            let mut thinking_config = Map::new();
            thinking_config.insert("includeThoughts".into(), json!(true));
            if let Some(level) = thinking.level {
                thinking_config.insert("thinkingLevel".into(), json!(level.as_str()));
            } else if let Some(budget) = thinking.budget_tokens {
                thinking_config.insert("thinkingBudget".into(), json!(budget));
            }
            generation_config.insert("thinkingConfig".into(), Value::Object(thinking_config));
        }
        // No `thinkingConfig` key when thinking is off: the CCA backend picks its server-side
        // default (TS only sends a suppress config for compat `thinking.suppressWhenOff`).
    }
    if antigravity {
        // The real client sends a fixed per-model output cap independent of the caller's budget.
        if let Some(profile) = antigravity_wire_profile(&wire_model_id) {
            generation_config.insert("maxOutputTokens".into(), json!(profile.max_output_tokens));
        }
    }

    let mut request = Map::new();
    request.insert("contents".into(), json!(contents));
    if let Some(system_prompt) = &context.system_prompt {
        if !system_prompt.trim().is_empty() {
            request.insert(
                "systemInstruction".into(),
                system_instruction(system_prompt, antigravity),
            );
        }
    }
    if !generation_config.is_empty() {
        request.insert("generationConfig".into(), Value::Object(generation_config));
    }

    let has_tools = context
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    if has_tools {
        // Cloud Code Assist takes the legacy `parameters` field on Claude routes
        // (`ccaLegacyParametersSchema`) and on every antigravity request (TS
        // `normalizeAntigravityTools` rewrites `parametersJsonSchema` -> `parameters`).
        let use_parameters = antigravity || compat_flag(model, "ccaLegacyParametersSchema");
        let tools = context.tools.as_deref().unwrap_or_default();
        if let Some(converted) = convert_tools(tools, use_parameters) {
            request.insert("tools".into(), json!(converted));
        }
        if let Some(tool_choice) = options.tool_choice {
            let mode = map_tool_choice(tool_choice.as_str());
            if mode != "AUTO" {
                request.insert(
                    "toolConfig".into(),
                    json!({ "functionCallingConfig": { "mode": mode } }),
                );
            }
            // Cloud Code Assist drops `toolConfig` on Antigravity's Gemini routes; the forced
            // choice is restated in the transcript as a tool-only turn.
            if antigravity && !claude_mode && mode == "ANY" {
                contents.push(json!({
                    "role": "user",
                    "parts": [{ "text": FORCED_TOOL_DIRECTIVE }],
                }));
                request.insert("contents".into(), json!(contents));
            }
        }
        // Antigravity's default tool mode is VALIDATED; an explicit choice above wins.
        if antigravity && !request.contains_key("toolConfig") {
            request.insert(
                "toolConfig".into(),
                json!({ "functionCallingConfig": { "mode": "VALIDATED" } }),
            );
        }
    }
    // Claude on Antigravity always forces VALIDATED, even with no tools declared.
    if antigravity && claude_mode {
        request.insert(
            "toolConfig".into(),
            json!({ "functionCallingConfig": { "mode": "VALIDATED" } }),
        );
    }

    let mut envelope = Map::new();
    envelope.insert("project".into(), json!(project_id));
    if let Some(metadata) = metadata {
        envelope.insert("requestId".into(), json!(metadata.request_id));
        request.insert("sessionId".into(), json!(metadata.session_id));
        request.insert("labels".into(), Value::Object(metadata.labels.clone()));
        envelope.insert("request".into(), Value::Object(request));
        envelope.insert("model".into(), json!(wire_model_id));
        envelope.insert("userAgent".into(), json!(USER_AGENT));
        envelope.insert("requestType".into(), json!("agent"));
    } else {
        envelope.insert("request".into(), Value::Object(request));
        envelope.insert("model".into(), json!(wire_model_id));
    }
    envelope
}

/// Request headers (TS `requestHeaders`): OAuth bearer, SSE accept, then the client identity —
/// the antigravity hub UA or the Gemini CLI UA + `Client-Metadata` — with config headers
/// (model + options) applied over the defaults.
fn build_headers(
    model: &Model,
    options: &GoogleOptions,
    access_token: &str,
    antigravity: bool,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = vec![
        ("Authorization".into(), format!("Bearer {access_token}")),
        ("Content-Type".into(), "application/json".into()),
        ("Accept".into(), "text/event-stream".into()),
    ];
    if antigravity {
        headers.push(("User-Agent".into(), antigravity_user_agent()));
    } else {
        headers.push(("User-Agent".into(), gemini_cli_user_agent(&model.id)));
        headers.push((
            "Client-Metadata".into(),
            "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI".into(),
        ));
    }
    if model.reasoning && compat_flag(model, "claudeThinkingBetaHeader") {
        headers.push(("anthropic-beta".into(), CLAUDE_THINKING_BETA_HEADER.into()));
    }
    let add = |headers: &mut Vec<(String, String)>, name: &str, value: &str| {
        headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
        headers.push((name.to_string(), value.to_string()));
    };
    for (name, value) in model.headers.iter().flatten() {
        add(&mut headers, name, value);
    }
    for (name, value) in options.base.headers.iter().flatten() {
        add(&mut headers, name, value);
    }
    headers
}

pub fn stream_google_gemini_cli(
    model: &Model,
    context: &Context,
    options: Option<&GoogleOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let model = model.clone();
    let context = context.clone();
    let (writer, reader) = create_assistant_message_event_stream();

    tokio::spawn(async move {
        let mut output = AssistantMessage {
            content: Vec::new(),
            api: API_GOOGLE_GEMINI_CLI.to_string(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            stop_reason_raw: None,
            error_status: None,
            error_message: None,
            timestamp: now_ms(),
            rest: Map::default(),
        };

        let result = run_stream(&model, &context, options.as_ref(), &mut output, &writer).await;
        match result {
            Ok(()) => {
                writer.push(AssistantMessageEvent::Done {
                    reason: done_reason(output.stop_reason),
                    message: output,
                });
                writer.end(None);
            }
            Err(error) => {
                output.stop_reason = if error == ProviderError::Aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message =
                    Some(crate::utils_inner::stream_failure::format_stream_failure_message(&error));
                crate::utils_inner::stream_failure::record_stream_failure(
                    (&model.provider, &model.id, &model.api),
                    &mut output,
                    &error,
                );
                writer.push(AssistantMessageEvent::Error {
                    reason: error_reason(output.stop_reason),
                    error: output.clone(),
                });
                writer.end(Some(output));
            }
        }
    });

    reader
}

/// One POST attempt against `url`.
async fn post(
    url: &str,
    headers: &[(String, String)],
    body: &str,
    options: &GoogleOptions,
) -> Result<HttpResponse, ProviderError> {
    send(RequestOptions {
        method: reqwest::Method::POST,
        url: url.to_string(),
        headers: headers.to_vec(),
        body: Some(body.to_string()),
        signal: options.base.signal.clone(),
        timeout_ms: options.base.timeout_ms,
        connection: ConnectionErrorProfile::RawFetch,
        transport: crate::utils_inner::http::Transport::Http1,
    })
    .await
}

fn backoff_delay_ms(attempt: usize) -> u64 {
    (RETRY_BASE_DELAY_MS * (1u64 << attempt.min(30))).min(RETRY_MAX_DELAY_MS)
}

fn cancelled(options: &GoogleOptions) -> bool {
    options
        .base
        .signal
        .as_ref()
        .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
}

fn reset_output(output: &mut AssistantMessage) {
    output.content.clear();
    output.usage = Usage::default();
    output.stop_reason = StopReason::Stop;
    output.stop_reason_raw = None;
    output.error_message = None;
    output.rest.remove("executionId");
}

// Long by design: mirrors the TS provider's failover loop.
#[allow(clippy::too_many_lines)]
async fn run_stream(
    model: &Model,
    context: &Context,
    options: Option<&GoogleOptions>,
    output: &mut AssistantMessage,
    writer: &AssistantMessageEventWriter,
) -> Result<(), ProviderError> {
    let options = options.cloned().unwrap_or_default();
    let Some(api_key) = options.base.api_key.clone().filter(|key| !key.is_empty()) else {
        return Err(ProviderError::Message(
            "Google Cloud Code Assist requires OAuth authentication. Use /login to authenticate."
                .to_string(),
        ));
    };
    let (access_token, project_id, expires_at) = parse_credentials(&api_key)?;
    if expires_at.is_some_and(|expires_at| expires_at <= (now_ms() as i64) + REFRESH_SKEW_MS) {
        return Err(ProviderError::Message(
            "OAuth token expired before request — please retry; the stored credential will be \
             refreshed on the next attempt."
                .to_string(),
        ));
    }

    if cancelled(&options) {
        return Err(ProviderError::Aborted);
    }

    let antigravity = model.provider == "google-antigravity";
    let is_claude = antigravity_claude_tool_mode(model);
    let mut state = GeminiCliSessionState::default();
    let headers = build_headers(model, &options, &access_token, antigravity);

    let endpoints = endpoints_for(model);
    let last = endpoints.len().saturating_sub(1);
    let mut got_output = false;
    let mut saw_finish = false;

    for (index, endpoint) in endpoints.iter().enumerate() {
        let is_last = index == last;
        reset_output(output);

        let usage_label = crate::compat::ModelView::of(model)
            .compat
            .get("antigravityUsageLabel")
            .and_then(Value::as_str)
            .map(std::string::ToString::to_string);
        let metadata = antigravity.then(|| {
            antigravity_metadata(
                &mut state,
                context,
                is_claude,
                usage_label.as_deref(),
                &wire_model_id(model),
            )
        });
        let body_value =
            Value::Object(build_body(model, context, &options, &project_id, metadata.as_ref()));
        let mut body_value = body_value;
        if let Some(on_payload) = &options.base.on_payload {
            if let Some(next) = on_payload(body_value.clone(), model) {
                body_value = next;
            }
        }
        let url = format!("{endpoint}/v1internal:streamGenerateContent?alt=sse");
        let body_text = serde_json::to_string(&body_value).unwrap_or_else(|_| "{}".to_string());
        let request_dump = RawHttpRequestDump {
            provider: model.provider.to_string(),
            api: model.api.to_string(),
            model: model.id.clone(),
            method: "POST".to_owned(),
            url: url.clone(),
            headers: headers
                .iter()
                .map(|(k, v)| {
                    if k.eq_ignore_ascii_case("authorization") {
                        (k.clone(), format!("Bearer <len={}>", v.len()))
                    } else {
                        (k.clone(), v.clone())
                    }
                })
                .collect(),
            body: body_text.clone(),
        };

        // TS `fetchWithRetry`: the last endpoint in the list retries transient failures
        // (`MAX_RETRIES + 1` attempts with exponential backoff); earlier endpoints fail over.
        let attempts = if is_last { MAX_HTTP_RETRIES + 1 } else { 1 };
        let mut response = post(&url, &headers, &body_text, &options).await;
        for attempt in 0..attempts.saturating_sub(1) {
            let retry = match &response {
                Ok(response) => is_transient_status(response.status),
                Err(error) => error_is_transient(error),
            };
            if !retry {
                break;
            }
            if cancelled(&options) {
                return Err(ProviderError::Aborted);
            }
            tokio::time::sleep(std::time::Duration::from_millis(backoff_delay_ms(attempt))).await;
            response = post(&url, &headers, &body_text, &options).await;
        }
        let mut response = match response {
            Ok(response) => response,
            Err(error) => {
                if !is_last && error_is_transient(&error) {
                    continue;
                }
                return Err(error);
            }
        };

        if let Some(on_response) = &options.base.on_response {
            on_response(
                crate::types::ProviderResponse {
                    status: response.status,
                    headers: response.headers.clone().into_iter().collect(),
                },
                model,
            );
        }

        if response.status >= 400 {
            let error_body = response.read_all_text().await.unwrap_or_default();
            if !is_last && is_transient_status(response.status) {
                continue;
            }
            let error = ProviderError::from_http_status_body(
                response.status,
                &error_body,
                response.headers.clone(),
            );
            return Err(finalize_http_error(error, false, Some(&request_dump)));
        }

        writer.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        // An empty STOP answer may be a transient upstream hiccup: replay the identical request
        // up to `MAX_EMPTY_STREAM_RETRIES` times (TS empty-stream loop). A thought-only STOP is a
        // complete provider response and is never replayed.
        let mut received_content = false;
        for empty_attempt in 0..=MAX_EMPTY_STREAM_RETRIES {
            let (had_output, finished) =
                read_sse(&mut response, model, output, writer, &options).await?;
            saw_finish = finished;
            if had_output || output.stop_reason != StopReason::Stop {
                received_content = had_output;
                break;
            }
            let thought_only = output
                .content
                .iter()
                .any(|block| matches!(block, crate::types::AssistantContent::Thinking(_)));
            if thought_only || empty_attempt == MAX_EMPTY_STREAM_RETRIES {
                break;
            }
            if cancelled(&options) {
                return Err(ProviderError::Aborted);
            }
            tokio::time::sleep(std::time::Duration::from_millis(
                EMPTY_STREAM_BASE_DELAY_MS * (1 << empty_attempt),
            ))
            .await;
            reset_output(output);
            response = post(&url, &headers, &body_text, &options).await?;
            if response.status >= 400 {
                let error_body = response.read_all_text().await.unwrap_or_default();
                let error = ProviderError::from_http_status_body(
                    response.status,
                    &error_body,
                    response.headers.clone(),
                );
                return Err(finalize_http_error(error, false, Some(&request_dump)));
            }
        }
        if let Some(execution_id) = output.rest.get("executionId").and_then(Value::as_str) {
            state.last_execution_id = Some(execution_id.to_string());
        }
        got_output |= received_content
            || output.content.iter().any(|block| match block {
                crate::types::AssistantContent::Text(text) => !text.text.trim().is_empty(),
                crate::types::AssistantContent::ToolCall(_) => true,
                _ => false,
            });

        if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
            return Err(ProviderError::StreamFailure(
                crate::utils_inner::stream_failure::stream_failure_from_stop_reason(
                    output.stop_reason_raw.as_deref(),
                    None,
                ),
            ));
        }
        if !got_output {
            let thought_only = output
                .content
                .iter()
                .any(|block| matches!(block, crate::types::AssistantContent::Thinking(_)));
            return Err(ProviderError::Message(if thought_only {
                "Cloud Code Assist API returned a thought-only response without final output"
                    .to_string()
            } else {
                "Cloud Code Assist API returned an empty response".to_string()
            }));
        }
        if cancelled(&options) {
            return Err(ProviderError::Aborted);
        }
        if !saw_finish {
            return Err(ProviderError::Message(
                "Cloud Code Assist stream ended without a finish reason (connection dropped or \
                 response truncated)"
                    .to_string(),
            ));
        }

        if antigravity {
            note_good_endpoint(endpoint);
        }
        break;
    }

    Ok(())
}

/// Read one SSE response to completion, feeding chunks to the shared Gemini stream state.
/// Returns whether the accumulated message carries user-visible output and whether any chunk
/// carried a `finishReason`.
async fn read_sse(
    response: &mut HttpResponse,
    model: &Model,
    output: &mut AssistantMessage,
    writer: &AssistantMessageEventWriter,
    options: &GoogleOptions,
) -> Result<(bool, bool), ProviderError> {
    let mut state = GoogleStreamState::new();
    let mut finished = false;
    let mut decoder = SseDecoder::new();
    let first_event_deadline = tokio::time::Instant::now()
        + std::time::Duration::from_millis(
            options
                .base
                .timeout_ms
                .unwrap_or(STREAM_FIRST_EVENT_TIMEOUT_MS),
        );
    let mut saw_event = false;
    loop {
        if !saw_event {
            let remaining =
                first_event_deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(ProviderError::Message(FIRST_EVENT_TIMEOUT_ERROR.to_string()));
            }
            match tokio::time::timeout(remaining, response.next_text()).await {
                Ok(chunk) => {
                    let Some(chunk) = chunk? else {
                        break;
                    };
                    handle_chunk_text(
                        &chunk,
                        &mut decoder,
                        &mut state,
                        model,
                        output,
                        writer,
                        &mut finished,
                    )?;
                    saw_event = true;
                }
                Err(_) => {
                    return Err(ProviderError::Message(FIRST_EVENT_TIMEOUT_ERROR.to_string()));
                }
            }
            continue;
        }
        let Some(chunk) = response.next_text().await? else {
            break;
        };
        handle_chunk_text(
            &chunk,
            &mut decoder,
            &mut state,
            model,
            output,
            writer,
            &mut finished,
        )?;
    }
    let tail = decoder.finish();
    let mut tail_decoder = SseDecoder::new();
    for sse in tail {
        handle_chunk_text(
            &sse.data,
            &mut tail_decoder,
            &mut state,
            model,
            output,
            writer,
            &mut finished,
        )?;
    }
    state.finish(output, writer);
    let had_output = output.content.iter().any(|block| match block {
        crate::types::AssistantContent::Text(text) => !text.text.trim().is_empty(),
        crate::types::AssistantContent::ToolCall(_) => true,
        _ => false,
    });
    Ok((had_output, finished))
}

fn handle_chunk_text(
    text: &str,
    decoder: &mut SseDecoder,
    state: &mut GoogleStreamState,
    model: &Model,
    output: &mut AssistantMessage,
    writer: &AssistantMessageEventWriter,
    finished: &mut bool,
) -> Result<(), ProviderError> {
    for sse in decoder.push_text(text) {
        if sse.data.trim().is_empty() || sse.data.trim() == "[DONE]" {
            continue;
        }
        let parsed = parse_json_with_repair(&sse.data).map_err(|error| {
            ProviderError::Message(format!(
                "Could not parse Gemini SSE chunk: {error}; data={:?}",
                &sse.data[..sse.data.len().min(200)]
            ))
        })?;
        // In-band stream failure (quota, internal error) delivered as a final JSON event.
        if let Some(error) = parsed.get("error").filter(|error| error.is_object()) {
            let detail = error
                .get("message")
                .or_else(|| error.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            let message = format!("Cloud Code Assist stream error: {detail}");
            return Err(match error.get("code").and_then(Value::as_u64) {
                // TS `GeminiCliApiError`: the in-band code is the failure's status; the
                // message stays verbatim (the body is not a provider error envelope).
                Some(code) if code >= 400 => ProviderError::Http(
                    crate::utils_inner::stream_failure::ProviderHttpError {
                        message,
                        status: Some(code.min(u16::MAX as u64) as u16),
                        body: None,
                        headers: std::collections::HashMap::new(),
                        request_id: None,
                        sdk_name: None,
                        retry_after_ms: None,
                        provider_error_type: error
                            .get("status")
                            .and_then(Value::as_str)
                            .map(std::string::ToString::to_string),
                    },
                ),
                _ => ProviderError::Message(message),
            });
        }
        // Cloud Code Assist wraps each GenerateContentResponse as `{"response": {...}}`; the
        // plain GenerativeLanguage shape is accepted unwrapped.
        let parsed = match parsed.get("response") {
            Some(inner) if inner.is_object() => inner.clone(),
            _ => parsed,
        };
        let candidates = parsed.get("candidates").and_then(Value::as_array);
        if candidates.is_none_or(Vec::is_empty) {
            if let Some(feedback) = parsed.get("promptFeedback") {
                if let Some(reason) = feedback.get("blockReason").and_then(Value::as_str) {
                    let detail = feedback
                        .get("blockReasonMessage")
                        .and_then(Value::as_str)
                        .map(|detail| format!(": {detail}"))
                        .unwrap_or_default();
                    return Err(ProviderError::Message(format!(
                        "Request blocked by Google ({reason}){detail}"
                    )));
                }
            }
        }
        if candidates.is_some_and(|c| {
            c.iter()
                .any(|x| x.get("finishReason").is_some_and(Value::is_string))
        }) {
            *finished = true;
        }
        if let Some(id) = parsed.get("responseId").and_then(Value::as_str) {
            output.rest.insert("executionId".into(), json!(id));
        }
        state
            .handle_chunk(&parsed, model, output, writer)
            .map_err(ProviderError::Message)?;
    }
    Ok(())
}

/// The thinking arm `stream_simple` picks for a model + reasoning level (TS parity with
/// `stream_simple_google`): level-based thinking for Gemini 3 models and Gemma 4, explicit
/// budget otherwise.
fn resolve_simple_thinking(
    model: &Model,
    options: Option<&SimpleStreamOptions>,
) -> crate::providers::google::GoogleThinking {
    use crate::models::clamp_thinking_level;
    use crate::providers::google::GoogleThinking;
    use crate::providers::google_shared::{
        budget_effort, get_google_thinking_budget, get_thinking_level, is_gemini3_flash_model,
        is_gemini3_pro_model, is_gemma4_model,
    };
    use crate::types::ModelThinkingLevel;

    let reasoning = options.and_then(|options| options.reasoning);
    if reasoning.is_none() || reasoning == Some(ModelThinkingLevel::Off) {
        return GoogleThinking {
            enabled: false,
            budget_tokens: None,
            level: None,
        };
    }

    let clamped = reasoning.map(|reasoning| clamp_thinking_level(model, reasoning));
    let effort = clamped.unwrap_or(ModelThinkingLevel::High);

    if is_gemini3_pro_model(&model.id)
        || is_gemini3_flash_model(&model.id)
        || is_gemma4_model(&model.id)
    {
        return GoogleThinking {
            enabled: true,
            budget_tokens: None,
            level: Some(get_thinking_level(effort, &model.id)),
        };
    }

    let budgets = options.and_then(|options| options.thinking_budgets.as_ref());
    let budget = get_google_thinking_budget(&model.id, budget_effort(effort), budgets);
    GoogleThinking {
        enabled: true,
        budget_tokens: Some(budget),
        level: None,
    }
}

/// `stream_simple` entry: the structured Cloud Code Assist credential (TS `getOAuthApiKey`)
/// threads through `options.api_key`; the env-key fallback only applies when no key was supplied.
pub fn stream_simple_google_gemini_cli(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let api_key = options.and_then(|options| {
        options
            .base
            .api_key
            .clone()
            .filter(|key| !key.is_empty())
            .or_else(|| crate::env_api_keys::get_env_api_key(&model.provider))
    });
    let base = build_base_options(model, options, api_key.as_deref());
    let stream_options = GoogleOptions {
        base,
        thinking: Some(resolve_simple_thinking(model, options)),
        tool_choice: None,
    };
    stream_google_gemini_cli(model, context, Some(&stream_options))
}

/// The `google-gemini-cli` API provider (Google Cloud Code Assist / antigravity).
pub struct GoogleGeminiCliProvider;

impl Provider for GoogleGeminiCliProvider {
    fn api(&self) -> &str {
        API_GOOGLE_GEMINI_CLI
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| GoogleOptions::from_base(base.clone()));
        stream_google_gemini_cli(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple_google_gemini_cli(model, context, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with(base_url: &str) -> Model {
        Model {
            id: "gemini-3.8-flash".into(),
            name: "gemini-3.8-flash".into(),
            api: API_GOOGLE_GEMINI_CLI.into(),
            provider: "google-antigravity".into(),
            base_url: base_url.into(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![pa_types::ai::ModelInput::Text],
            cost: pa_types::ai::ModelCost {
                input: pa_types::JsNumber(0.0),
                output: pa_types::JsNumber(0.0),
                cache_read: pa_types::JsNumber(0.0),
                cache_write: pa_types::JsNumber(0.0),
            },
            context_window: 1_000_000,
            max_tokens: 65_536,
            max_tokens_explicit: false,
            omit_max_output_tokens: None,
            request_model_id: None,
            featured: None,
            headers: None,
            compat: None,
        }
    }

    fn claude_model() -> Model {
        let mut model = model_with("https://daily-cloudcode-pa.googleapis.com");
        model.id = "claude-sonnet-4-5".into();
        // The antigravity catalog marks Claude ids with the claude tool-mode compat flags.
        model.compat = Some(pa_types::ai::ModelCompat {
            raw: serde_json::Map::from_iter([
                ("antigravityClaudeToolMode".to_string(), serde_json::json!(true)),
                ("antigravityUsageLabel".to_string(), serde_json::json!("true")),
                (
                    "ccaLegacyParametersSchema".to_string(),
                    serde_json::json!(true),
                ),
            ]),
        });
        model
    }

    #[test]
    fn endpoints_prefer_configured_base_url() {
        let model = model_with("http://127.0.0.1:8791");
        assert_eq!(
            endpoints_for(&model),
            vec!["http://127.0.0.1:8791".to_string()]
        );
    }

    #[test]
    fn endpoints_failover_order() {
        let model = model_with("https://daily-cloudcode-pa.googleapis.com");
        let endpoints = endpoints_for(&model);
        // TS `ANTIGRAVITY_ENDPOINT_FALLBACKS` parity: `[daily, sandbox]`.
        assert_eq!(endpoints.len(), 2);
        assert_eq!(endpoints[0], BASE_URL_PRODUCTION);
        assert_eq!(endpoints[1], BASE_URL_SANDBOX);
    }

    #[test]
    fn gemini_cli_uses_catalog_base_url() {
        let mut model = model_with("https://cloudcode-pa.googleapis.com");
        model.provider = "google-gemini-cli".into();
        // TS gemini-cli arm: the catalog baseUrl (no antigravity failover).
        assert_eq!(
            endpoints_for(&model),
            vec!["https://cloudcode-pa.googleapis.com".to_string()]
        );
    }

    #[test]
    fn gemini_cli_defaults_to_stable_host() {
        let mut model = model_with("");
        model.provider = "google-gemini-cli".into();
        assert_eq!(endpoints_for(&model), vec![BASE_URL_STABLE.to_string()]);
    }

    #[test]
    fn transient_statuses() {
        assert!(is_transient_status(408));
        assert!(is_transient_status(429));
        assert!(is_transient_status(500));
        assert!(is_transient_status(503));
        assert!(!is_transient_status(404));
        assert!(!is_transient_status(401));
        assert!(!is_transient_status(409));
    }

    #[test]
    fn metadata_carries_session_labels() {
        let mut state = GeminiCliSessionState::default();
        let context = Context::default();
        let meta = antigravity_metadata(&mut state, &context, false, None, "gemini-3.8-flash");
        // Signed-decimal session id derived from a random int63 (no user text yet).
        assert!(meta.session_id.starts_with('-'));
        assert!(meta.request_id.starts_with("agent/"));
        assert_eq!(
            meta.labels.get("used_claude").and_then(Value::as_str),
            Some("false")
        );
        assert_eq!(
            meta.labels
                .get("used_claude_conservative")
                .and_then(Value::as_str),
            Some("false")
        );
        let meta2 = antigravity_metadata(&mut state, &context, true, None, "gemini-3.8-flash");
        assert_eq!(meta2.session_id, meta.session_id);
        assert_eq!(
            meta2.labels.get("used_claude").and_then(Value::as_str),
            Some("true")
        );
        // TS step numbering: the first envelope carries step 2 / last_step_index 1.
        assert_eq!(
            meta.labels.get("last_step_index").and_then(Value::as_str),
            Some("1")
        );
        assert_eq!(
            meta2.labels.get("last_step_index").and_then(Value::as_str),
            Some("2")
        );
    }

    #[test]
    fn session_id_hashed_from_first_user_text() {
        let context = Context {
            messages: vec![crate::types::Message::User(crate::types::UserMessage {
                content: crate::types::UserMessageContent::Text("hello antigravity".into()),
                timestamp: 0,
                rest: Map::default(),
            })],
            ..Default::default()
        };
        // Deterministic: SHA-256 first 8 bytes as signed-decimal int63.
        assert_eq!(
            derive_session_id(&context),
            signed_decimal_from_hash("hello antigravity")
        );
    }

    #[test]
    fn wire_profiles_match_ts_table() {
        assert_eq!(
            antigravity_wire_profile("gemini-3.5-flash-extra-low").map(|p| p.max_output_tokens),
            Some(65_536)
        );
        assert_eq!(
            antigravity_wire_profile("claude-sonnet-4-6").map(|p| p.max_output_tokens),
            Some(64_000)
        );
        assert!(
            antigravity_wire_profile("gemini-3.5-flash-extra-low")
                .unwrap()
                .model_enum
                .is_some()
        );
        assert!(antigravity_wire_profile("unknown-model").is_none());
    }

    #[test]
    fn body_wraps_plain_request_in_envelope() {
        let mut model = model_with("https://cloudcode-pa.googleapis.com");
        model.provider = "google-gemini-cli".into();
        let context = Context {
            system_prompt: Some("You are helpful.".into()),
            tools: None,
            messages: vec![],
            ..Default::default()
        };
        let options = GoogleOptions {
            base: StreamOptions {
                max_tokens: Some(1024),
                ..Default::default()
            },
            thinking: None,
            tool_choice: None,
        };
        let body = build_body(&model, &context, &options, "proj-42", None);

        // TS gemini-cli arm: `{project, model, request}` — no antigravity metadata.
        assert_eq!(body.get("project").and_then(Value::as_str), Some("proj-42"));
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("gemini-3.8-flash")
        );
        assert!(body.get("userAgent").is_none());
        assert!(body.get("requestType").is_none());
        assert!(body.get("requestId").is_none());
        let request = body.get("request").and_then(Value::as_object).unwrap();
        // The GenerativeLanguage fields ride inside `request`, never at the envelope top level.
        assert!(request.get("contents").is_some());
        assert!(body.get("contents").is_none());
        assert!(request.get("sessionId").is_none());
        assert!(request.get("labels").is_none());
        let instruction = request
            .get("systemInstruction")
            .and_then(Value::as_object)
            .unwrap();
        // Non-antigravity system instruction carries no `role`.
        assert!(instruction.get("role").is_none());
        assert!(instruction.get("parts").is_some());
        let generation = request
            .get("generationConfig")
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(
            generation.get("maxOutputTokens").and_then(Value::as_u64),
            Some(1024)
        );
        assert!(request.get("toolConfig").is_none());
    }

    #[test]
    fn antigravity_body_carries_envelope_and_labels() {
        let mut model = model_with("https://daily-cloudcode-pa.googleapis.com");
        // `gemini-3-flash` collapses to the `gemini-3.5-flash-extra-low` wire row.
        model.id = "gemini-3-flash".into();
        model.request_model_id = Some("gemini-3.5-flash-extra-low".into());
        let context = Context {
            system_prompt: Some("You are helpful.".into()),
            tools: Some(vec![crate::types::Tool {
                name: "read".into(),
                description: "read".into(),
                parameters: json!({"type": "object"}),
            }]),
            messages: vec![],
            ..Default::default()
        };
        let options = GoogleOptions {
            base: StreamOptions {
                max_tokens: Some(1024),
                ..Default::default()
            },
            thinking: None,
            tool_choice: None,
        };
        let mut state = GeminiCliSessionState::default();
        let metadata = antigravity_metadata(
            &mut state,
            &context,
            false,
            None,
            "gemini-3.5-flash-extra-low",
        );
        let body = build_body(&model, &context, &options, "proj-42", Some(&metadata));

        assert_eq!(
            body.get("userAgent").and_then(Value::as_str),
            Some("antigravity")
        );
        assert_eq!(
            body.get("requestType").and_then(Value::as_str),
            Some("agent")
        );
        assert!(body.get("requestId").is_some_and(Value::is_string));
        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("gemini-3.5-flash-extra-low")
        );
        let request = body.get("request").and_then(Value::as_object).unwrap();
        assert_eq!(
            request.get("sessionId").and_then(Value::as_str),
            Some(metadata.session_id.as_str())
        );
        let labels = request.get("labels").and_then(Value::as_object).unwrap();
        assert_eq!(
            labels.get("used_claude").and_then(Value::as_str),
            Some("false")
        );
        // The wire row's model_enum telemetry label and fixed output cap.
        assert_eq!(
            labels.get("model_enum").and_then(Value::as_str),
            Some("MODEL_PLACEHOLDER_M187")
        );
        let generation = request
            .get("generationConfig")
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(
            generation.get("maxOutputTokens").and_then(Value::as_u64),
            Some(65_536)
        );
        // Antigravity tool mode defaults to VALIDATED; tools go out as legacy `parameters`.
        let tool_config = request
            .get("toolConfig")
            .and_then(Value::as_object)
            .and_then(|config| config.get("functionCallingConfig"))
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(
            tool_config.get("mode").and_then(Value::as_str),
            Some("VALIDATED")
        );
        let tools = request.get("tools").and_then(Value::as_array).unwrap();
        let declaration = tools[0]
            .get("functionDeclarations")
            .and_then(Value::as_array)
            .unwrap()[0]
            .as_object()
            .unwrap();
        assert!(declaration.contains_key("parameters"));
        assert!(!declaration.contains_key("parametersJsonSchema"));
        // Antigravity tags the system instruction with role "user".
        let instruction = request
            .get("systemInstruction")
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(
            instruction.get("role").and_then(Value::as_str),
            Some("user")
        );
    }

    #[test]
    fn claude_model_gets_validated_tool_mode() {
        let model = claude_model();
        let context = Context {
            system_prompt: None,
            tools: Some(vec![crate::types::Tool {
                name: "read".into(),
                description: "read".into(),
                parameters: json!({"type": "object"}),
            }]),
            messages: vec![],
            ..Default::default()
        };
        let options = GoogleOptions::default();
        let mut state = GeminiCliSessionState::default();
        let metadata = antigravity_metadata(
            &mut state,
            &context,
            true,
            Some("true"),
            "claude-sonnet-4-5",
        );
        let body = build_body(&model, &context, &options, "proj-42", Some(&metadata));
        let request = body.get("request").and_then(Value::as_object).unwrap();
        let tool_config = request
            .get("toolConfig")
            .and_then(Value::as_object)
            .and_then(|config| config.get("functionCallingConfig"))
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(
            tool_config.get("mode").and_then(Value::as_str),
            Some("VALIDATED")
        );
        let labels = request.get("labels").and_then(Value::as_object).unwrap();
        assert_eq!(
            labels.get("used_claude").and_then(Value::as_str),
            Some("true")
        );
    }

    #[test]
    fn claude_model_without_tools_still_validated() {
        let model = claude_model();
        let context = Context::default();
        let mut state = GeminiCliSessionState::default();
        let metadata = antigravity_metadata(&mut state, &context, true, None, "claude-sonnet-4-5");
        let body = build_body(
            &model,
            &context,
            &GoogleOptions::default(),
            "p",
            Some(&metadata),
        );
        let request = body.get("request").and_then(Value::as_object).unwrap();
        assert_eq!(
            request
                .get("toolConfig")
                .and_then(|t| t.pointer("/functionCallingConfig/mode"))
                .and_then(Value::as_str),
            Some("VALIDATED")
        );
    }

    #[test]
    fn forced_tool_choice_restates_directive_for_gemini_rows() {
        let model = model_with("https://daily-cloudcode-pa.googleapis.com");
        let context = Context {
            tools: Some(vec![crate::types::Tool {
                name: "bash".into(),
                description: "bash".into(),
                parameters: json!({"type": "object"}),
            }]),
            messages: vec![],
            ..Default::default()
        };
        let options = GoogleOptions {
            tool_choice: Some(crate::providers::google::GoogleToolChoice::Any),
            ..Default::default()
        };
        let mut state = GeminiCliSessionState::default();
        let metadata = antigravity_metadata(&mut state, &context, false, None, "gemini-3.8-flash");
        let body = build_body(&model, &context, &options, "p", Some(&metadata));
        let request = body.get("request").and_then(Value::as_object).unwrap();
        // The forced choice stays on the wire AND is restated as a tool-only user turn.
        assert_eq!(
            request
                .get("toolConfig")
                .and_then(|t| t.pointer("/functionCallingConfig/mode"))
                .and_then(Value::as_str),
            Some("ANY")
        );
        let contents = request.get("contents").and_then(Value::as_array).unwrap();
        let last = contents.last().unwrap();
        assert_eq!(last.get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(
            last.pointer("/parts/0/text").and_then(Value::as_str),
            Some(FORCED_TOOL_DIRECTIVE)
        );
    }

    #[test]
    fn credentials_parse_structured_key() {
        let key = r#"{"token":"ya29.a0","projectId":"my-project","expiresAt":4102444800000}"#;
        let (token, project, expires_at) = parse_credentials(key).unwrap();
        assert_eq!(token, "ya29.a0");
        assert_eq!(project, "my-project");
        assert_eq!(expires_at, Some(4_102_444_800_000));
    }

    #[test]
    fn credentials_reject_bare_token() {
        let error = parse_credentials("ya29.a0").unwrap_err();
        assert!(error.to_string().contains("Invalid Google Cloud Code Assist"));
        let error = parse_credentials(r#"{"token":"ya29.a0"}"#).unwrap_err();
        assert!(error.to_string().contains("Missing token or projectId"));
    }

    fn feed_chunks(chunks: &[String]) -> Result<AssistantMessage, ProviderError> {
        let model = model_with("https://daily-cloudcode-pa.googleapis.com");
        let (writer, reader) = create_assistant_message_event_stream();
        drop(reader);
        let mut output = AssistantMessage {
            content: Vec::new(),
            api: API_GOOGLE_GEMINI_CLI.to_string(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            stop_reason_raw: None,
            error_status: None,
            error_message: None,
            timestamp: 0,
            rest: Map::default(),
        };
        let mut state = GoogleStreamState::new();
        let mut decoder = SseDecoder::new();
        let mut finished = false;
        let result = (|| {
            for chunk in chunks {
                handle_chunk_text(
                    chunk,
                    &mut decoder,
                    &mut state,
                    &model,
                    &mut output,
                    &writer,
                    &mut finished,
                )?;
            }
            Ok(())
        })();
        state.finish(&output, &writer);
        result.map(|()| output)
    }

    #[test]
    fn sse_chunk_response_wrapper_and_finish() {
        let output = feed_chunks(&[
            "data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"PONG\"}]}}]}}\n\n"
                .to_string(),
            "data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1,\"totalTokenCount\":4},\"responseId\":\"exec-1\"}}\n\n"
                .to_string(),
        ])
        .unwrap();
        // The `response` wrapper is unwrapped per chunk before the shared Gemini processing.
        assert!(matches!(
            &output.content[..],
            [crate::types::AssistantContent::Text(text)] if text.text == "PONG"
        ));
        assert_eq!(output.stop_reason, StopReason::Stop);
        assert_eq!(output.usage.input, 3);
        assert_eq!(
            output.rest.get("executionId").and_then(Value::as_str),
            Some("exec-1")
        );
    }

    #[test]
    fn sse_chunk_tool_call_round_trips() {
        let output = feed_chunks(&[
            "data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"bash\",\"args\":{\"command\":\"echo hi\"}}}]}}]}}\n\n"
                .to_string(),
            "data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\"}]}}\n\n".to_string(),
        ])
        .unwrap();
        let [crate::types::AssistantContent::ToolCall(call)] = &output.content[..] else {
            panic!("expected one tool call, got {:?}", output.content);
        };
        assert_eq!(call.name, "bash");
        assert_eq!(
            call.arguments.get("command").and_then(Value::as_str),
            Some("echo hi")
        );
        // A trailing tool call upgrades the benign STOP finish to toolUse.
        assert_eq!(output.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn sse_in_band_error_maps_to_status_error() {
        let error =
            feed_chunks(&["data: {\"error\":{\"code\":429,\"message\":\"Quota exceeded\"}}\n\n"
                .to_string()])
            .unwrap_err();
        // The shared classifier keeps the status (429 → rate limit) and the in-band detail.
        assert_eq!(error.status(), Some(429));
        assert!(error.to_string().contains("Quota exceeded"));
    }

    #[test]
    fn sse_prompt_feedback_blocks() {
        let error = feed_chunks(&[
            "data: {\"response\":{\"promptFeedback\":{\"blockReason\":\"SAFETY\",\"blockReasonMessage\":\"unsafe\"}}}\n\n"
                .to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Request blocked by Google (SAFETY): unsafe"
        );
    }
}
