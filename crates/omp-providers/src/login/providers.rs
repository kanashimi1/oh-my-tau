//! The `/login` roster: every OAuth/API-key provider TS `getOAuthProviders()` lists, in TS order
//! (`rules/auth/_order.kdl`), with its display name.

/// `(provider id, display name)` in `/login` order.
pub const PROVIDERS: [(&str, &str); 80] = [
    ("openai-codex", "ChatGPT Plus/Pro (Codex Subscription)"),
    ("anthropic", "Anthropic (Claude Pro/Max)"),
    ("zai", "Z.AI (GLM Coding Plan)"),
    ("zai-coding-plan", "Z.AI (GLM Coding Plan · Sign in)"),
    ("kimi-code", "Kimi Code"),
    ("openrouter", "OpenRouter"),
    ("github-copilot", "GitHub Copilot"),
    ("cursor", "Cursor (Claude, GPT, etc.)"),
    ("devin", "Devin"),
    (
        "google-antigravity",
        "Antigravity (Gemini 3, Claude, GPT-OSS)",
    ),
    ("google-gemini-cli", "Google Cloud Code Assist (Gemini CLI)"),
    (
        "openai-codex-device",
        "ChatGPT Plus/Pro (Codex, headless/device)",
    ),
    ("xai", "xAI API"),
    ("xai-oauth", "xAI Grok OAuth (SuperGrok or X Premium+)"),
    ("gitlab-duo", "GitLab Duo Non-Agentic"),
    ("gitlab-duo-agent", "GitLab Duo Agent"),
    ("alibaba-coding-plan", "Alibaba Coding Plan"),
    ("alibaba-token-plan", "QwenCloud Token Plan"),
    ("aiand", "ai&"),
    ("abliteration", "Abliteration"),
    ("zhipu-coding-plan", "Zhipu Coding Plan (智谱)"),
    ("umans", "Umans AI Coding Plan"),
    ("qwen-portal", "Qwen Portal"),
    ("sakana", "Sakana AI"),
    ("minimax-code", "MiniMax Token Plan (International)"),
    ("minimax-code-cn", "MiniMax Token Plan (China)"),
    ("xiaomi", "Xiaomi MiMo"),
    ("xiaomi-token-plan-sgp", "Xiaomi Token Plan (Singapore)"),
    ("xiaomi-token-plan-ams", "Xiaomi Token Plan (Europe)"),
    ("xiaomi-token-plan-cn", "Xiaomi Token Plan (China)"),
    ("firepass", "Fire Pass (Fireworks subscription)"),
    ("cline-pass", "ClinePass"),
    ("factory-droid", "Factory Droid"),
    ("commandcode", "Command Code"),
    ("charm-hyper", "Charm Hyper"),
    ("deepseek", "DeepSeek"),
    ("stepfun", "StepFun"),
    ("muse-code", "Muse Code (Subscription)"),
    ("meta", "Meta Model API"),
    ("moonshot", "Moonshot (Kimi API)"),
    ("cerebras", "Cerebras"),
    ("baseten", "Baseten"),
    ("helmcode", "Helmcode"),
    ("fireworks", "Fireworks"),
    ("together", "Together"),
    ("nvidia", "NVIDIA"),
    ("novita", "Novita"),
    ("deepinfra", "DeepInfra"),
    ("huggingface", "Hugging Face Inference"),
    ("perplexity", "Perplexity (Pro/Max)"),
    ("qianfan", "Qianfan"),
    ("venice", "Venice"),
    ("siliconflow", "SiliconFlow"),
    ("siliconflow-cn", "SiliconFlow (China)"),
    ("synthetic", "Synthetic"),
    ("nanogpt", "NanoGPT"),
    ("wafer-serverless", "Wafer Serverless (pay-as-you-go)"),
    ("coreweave", "CoreWeave Serverless Inference"),
    ("vercel-ai-gateway", "Vercel AI Gateway"),
    ("cloudflare-ai-gateway", "Cloudflare AI Gateway"),
    ("litellm", "LiteLLM"),
    ("kilo", "Kilo Gateway"),
    ("zenmux", "ZenMux"),
    ("opencode-zen", "OpenCode Zen"),
    ("opencode-go", "OpenCode Go"),
    ("yolo-auto", "Yolo-Auto"),
    ("tavily", "Tavily"),
    ("kagi", "Kagi"),
    ("exa", "Exa"),
    ("parallel", "Parallel"),
    ("typesafe", "TypeSafe"),
    ("ollama", "Ollama (Local OpenAI-compatible)"),
    ("ollama-cloud", "Ollama Cloud"),
    ("lm-studio", "LM Studio (Local OpenAI-compatible)"),
    ("llama.cpp", "llama.cpp (Local OpenAI-compatible)"),
    ("vllm", "vLLM (Local OpenAI-compatible)"),
    ("gmi-cloud", "GMI Cloud"),
    ("stencil", "Stencil (invite only)"),
    ("singularityapi-dev", "SingularityAPI"),
    ("singularityapi-tech", "SingularityAPI Reserved Lanes"),
];

/// The provider id a credential is stored under (TS `storeCredentialsAs`).
#[must_use]
pub fn store_as(id: &str) -> &str {
    match id {
        "zai-coding-plan" => "zai",
        "openai-codex-device" => "openai-codex",
        other => other,
    }
}

/// Display name of a provider id.
#[must_use]
pub fn display_name(id: &str) -> Option<&'static str> {
    PROVIDERS.iter().find(|(p, _)| *p == id).map(|(_, n)| *n)
}
