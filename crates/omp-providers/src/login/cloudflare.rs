//! The Cloudflare AI Gateway login (TS `registry/oauth/cloudflare-ai-gateway.ts`
//! via the `custom` hook): three prompts — token, account ID, gateway ID —
//! serialized into the catalog's structured credential payload. No network.

use pa_ai::oauth::{OAuthLoginUi, OAuthPrompt};

use super::api_key::LOGIN_CANCELLED;

const AUTH_URL: &str = "https://developers.cloudflare.com/ai-gateway/configuration/authentication/";

/// The TS `serializeCloudflareAiGatewayCredential` payload (trimmed fields).
pub(crate) fn serialize_credential(token: &str, account_id: &str, gateway_id: &str) -> String {
    serde_json::json!({
        "token": token.trim(),
        "accountId": account_id.trim(),
        "gatewayId": gateway_id.trim(),
    })
    .to_string()
}

/// # Errors
/// Cancel ([`LOGIN_CANCELLED`]), a missing prompt surface, or an empty answer.
pub(crate) async fn login_cloudflare_ai_gateway(ui: &dyn OAuthLoginUi) -> Result<String, String> {
    ui.on_auth(
        AUTH_URL,
        Some("Create an AI Gateway token with Run permission, then copy it here."),
    );
    let token = prompt(
        ui,
        "Paste your Cloudflare AI Gateway token/API key",
        "cfut_...",
    )
    .await?;
    if token.trim().is_empty() {
        return Err("API key is required".to_string());
    }
    let account_id = prompt(
        ui,
        "Enter your Cloudflare account ID",
        "32-character account ID",
    )
    .await?;
    if account_id.trim().is_empty() {
        return Err("Cloudflare account ID is required".to_string());
    }
    let gateway_id = prompt(ui, "Enter your Cloudflare AI Gateway ID", "default").await?;
    if gateway_id.trim().is_empty() {
        return Err("Cloudflare AI Gateway ID is required".to_string());
    }
    Ok(serialize_credential(&token, &account_id, &gateway_id))
}

async fn prompt(ui: &dyn OAuthLoginUi, message: &str, placeholder: &str) -> Result<String, String> {
    let answer = ui
        .on_prompt(&OAuthPrompt {
            message: message.to_string(),
            placeholder: Some(placeholder.to_string()),
            allow_empty: false,
        })
        .await
        .ok_or_else(|| LOGIN_CANCELLED.to_string())?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    Ok(answer)
}
