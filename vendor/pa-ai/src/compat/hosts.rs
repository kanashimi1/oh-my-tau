//! Known model-endpoint host classification (TS `hosts.ts`) and the Anthropic endpoint-shape
//! predicates (TS `compat/anthropic.ts`). Markers are case-insensitive substrings matched against
//! the base URL, never parsed hostnames, except where noted.

use regex::Regex;
use std::sync::OnceLock;

struct HostClass {
    name: &'static str,
    providers: &'static [&'static str],
    prefixes: &'static [&'static str],
    markers: &'static [&'static str],
}

const fn h(
    name: &'static str,
    providers: &'static [&'static str],
    prefixes: &'static [&'static str],
    markers: &'static [&'static str],
) -> HostClass {
    HostClass {
        name,
        providers,
        prefixes,
        markers,
    }
}

static KNOWN_HOSTS: &[HostClass] = &[
    h("openai", &["openai"], &[], &["api.openai.com"]),
    h(
        "openaiCodex",
        &[],
        &[],
        &["chatgpt.com/backend-api", "chat.openai.com/backend-api"],
    ),
    h(
        "azureOpenAI",
        &["azure"],
        &[],
        &[
            ".openai.azure.com",
            "azure.com/openai",
            "models.inference.ai.azure.com",
        ],
    ),
    h("openrouter", &["openrouter"], &[], &["openrouter.ai"]),
    h(
        "vercelAIGateway",
        &["vercel-ai-gateway"],
        &[],
        &["ai-gateway.vercel.sh"],
    ),
    h(
        "githubCopilot",
        &["github-copilot"],
        &[],
        &["githubcopilot.com", "copilot-api."],
    ),
    h("anthropic", &["anthropic"], &[], &["api.anthropic.com"]),
    h("deepseekDirect", &["deepseek"], &[], &["api.deepseek.com"]),
    h("deepseekFamily", &["deepseek"], &[], &["deepseek.com"]),
    h("cerebras", &["cerebras"], &[], &["cerebras.ai"]),
    h("zai", &["zai"], &[], &["api.z.ai"]),
    h("zhipu", &["zhipu-coding-plan"], &[], &["open.bigmodel.cn"]),
    h("kilo", &["kilo"], &[], &["api.kilo.ai"]),
    h(
        "alibabaDashscope",
        &["alibaba-coding-plan", "alibaba-token-plan"],
        &[],
        &["dashscope", "token-plan."],
    ),
    h("umans", &["umans"], &[], &["api.code.umans.ai"]),
    h(
        "xiaomi",
        &["xiaomi"],
        &["xiaomi-token-plan-"],
        &["xiaomimimo.com"],
    ),
    h("xai", &["xai", "xai-oauth"], &[], &["api.x.ai"]),
    h("mistral", &["mistral"], &[], &["mistral.ai"]),
    h("together", &["together"], &[], &["api.together.xyz"]),
    h("baseten", &["baseten"], &[], &["baseten.co"]),
    h("fireworks", &[], &[], &["fireworks.ai"]),
    h("groq", &["groq"], &[], &["api.groq.com"]),
    h(
        "minimax",
        &["minimax", "minimax-code", "minimax-code-cn"],
        &[],
        &["api.minimax.io", "api.minimaxi.com"],
    ),
    h("qwenPortal", &["qwen-portal"], &[], &["portal.qwen.ai"]),
    h("nvidia", &["nvidia"], &[], &["integrate.api.nvidia.com"]),
    h("venice", &["venice"], &[], &["api.venice.ai"]),
    h(
        "moonshotNative",
        &["moonshot", "kimi-code"],
        &[],
        &["api.moonshot.ai", "api.kimi.com"],
    ),
    h(
        "googleAistudio",
        &[],
        &[],
        &["generativelanguage.googleapis.com"],
    ),
    h(
        "opencode",
        &["opencode-go", "opencode-zen"],
        &[],
        &["opencode.ai"],
    ),
    h("zenmux", &["zenmux"], &[], &["zenmux.ai"]),
    h("chutes", &[], &[], &["chutes.ai"]),
];

fn class(host: &str) -> &'static HostClass {
    KNOWN_HOSTS
        .iter()
        .find(|c| c.name == host)
        .unwrap_or_else(|| panic!("unknown host class {host}"))
}

/// TS `includesAsciiCaseInsensitive`: bytewise compare with the `| 0x20` fold.
fn includes_ascii_ci(value: &str, lower_needle: &str) -> bool {
    let v = value.as_bytes();
    let n = lower_needle.as_bytes();
    if n.len() > v.len() {
        return false;
    }
    (0..=v.len() - n.len()).any(|s| n.iter().enumerate().all(|(o, &nb)| (v[s + o] | 0x20) == nb))
}

/// URL-only host check.
pub fn host_matches_url(base_url: &str, host: &str) -> bool {
    if base_url.is_empty() {
        return false;
    }
    class(host)
        .markers
        .iter()
        .any(|m| includes_ascii_ci(base_url, m))
}

/// Provider-or-URL host check.
pub fn model_matches_host(provider: &str, base_url: &str, host: &str) -> bool {
    let spec = class(host);
    spec.providers.contains(&provider)
        || spec.prefixes.iter().any(|p| provider.starts_with(p))
        || host_matches_url(base_url, host)
}

fn bedrock_public() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"^(?:(?P<runtime>bedrock-runtime(?:-fips)?)\.[a-z0-9-]+\.amazonaws\.com|bedrock-mantle\.(?:[a-z0-9-]+|\{region\})\.api\.aws)$")
            .expect("static regex")
    })
}

fn bedrock_privatelink() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"^vpce-[a-z0-9-]+\.(?:(?P<runtime>bedrock-runtime(?:-fips)?)|bedrock-mantle)\.[a-z0-9-]+\.vpce\.amazonaws\.com$")
            .expect("static regex")
    })
}

fn has_path_prefix(pathname: &str, prefix: &str) -> bool {
    pathname == prefix || pathname.starts_with(&format!("{prefix}/"))
}

/// Amazon Bedrock API route on a bedrock-runtime / bedrock-mantle endpoint (strict hostname).
pub fn is_bedrock_route_url(base_url: &str, route: &str) -> bool {
    if base_url.is_empty() {
        return false;
    }
    let Ok(url) = url::Url::parse(base_url) else {
        return false;
    };
    if url.scheme() != "https" {
        return false;
    }
    let Some(hostname) = url.host_str() else {
        return false;
    };
    let caps = bedrock_public()
        .captures(hostname)
        .or_else(|| bedrock_privatelink().captures(hostname));
    let Some(caps) = caps else { return false };
    if route == "anthropic" {
        return has_path_prefix(url.path(), "/anthropic");
    }
    let is_mantle = caps.name("runtime").is_none();
    has_path_prefix(url.path(), "/openai") || (is_mantle && has_path_prefix(url.path(), "/v1"))
}

/// Loopback / private-network / `.local` base URL (TS `hasLocalLoopbackBaseUrl`).
pub fn has_local_loopback_base_url(base_url: &str) -> bool {
    if base_url.is_empty() {
        return false;
    }
    let Ok(url) = url::Url::parse(base_url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.to_lowercase();
    if matches!(
        host.as_str(),
        "localhost" | "127.0.0.1" | "0.0.0.0" | "::1" | "[::1]"
    ) {
        return true;
    }
    if host.starts_with("10.") || host.starts_with("192.168.") || host.ends_with(".local") {
        return true;
    }
    if let Some(rest) = host.strip_prefix("172.") {
        let second: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if rest[second.len()..].starts_with('.') {
            if let Ok(n) = second.parse::<u32>() {
                return (16..=31).contains(&n);
            }
        }
    }
    false
}

// --- Anthropic endpoint-shape predicates ---

const OFFICIAL_ANTHROPIC_URL: &str = "https://api.anthropic.com";

pub fn is_official_anthropic_api_url(base_url: &str) -> bool {
    if base_url.is_empty() {
        return true;
    }
    let lower = base_url.to_lowercase();
    lower == OFFICIAL_ANTHROPIC_URL || lower.starts_with(&format!("{OFFICIAL_ANTHROPIC_URL}/"))
}

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("static regex"))
}

pub fn is_bedrock_anthropic_route(base_url: &str) -> bool {
    is_bedrock_route_url(base_url, "anthropic")
}

pub fn is_azure_anthropic_route(base_url: &str) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    re(
        &R,
        r"(?i)(?:^|//|\.)[a-z0-9-]+\.(?:inference|services)\.ai\.azure\.com",
    )
    .is_match(base_url)
}

pub fn is_anthropic_signing_proxy_url(base_url: &str) -> bool {
    static CF: OnceLock<Regex> = OnceLock::new();
    static VX: OnceLock<Regex> = OnceLock::new();
    static BR: OnceLock<Regex> = OnceLock::new();
    host_matches_url(base_url, "githubCopilot")
        || host_matches_url(base_url, "zenmux")
        || (!base_url.is_empty()
            && (re(&CF, r"(?i)gateway\.ai\.cloudflare\.com/.+/anthropic(?:/|$)")
                .is_match(base_url)
                || re(
                    &VX,
                    r"(?i)aiplatform\.googleapis\.com/.+/publishers/anthropic/",
                )
                .is_match(base_url)
                || re(
                    &BR,
                    r"(?i)(?:^|//|\.)bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com",
                )
                .is_match(base_url)))
        || is_bedrock_anthropic_route(base_url)
        || is_azure_anthropic_route(base_url)
}
