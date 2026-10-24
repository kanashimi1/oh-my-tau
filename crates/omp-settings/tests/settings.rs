//! Behavior tests for the layered settings model.

use omp_settings::{lookup, Provenance, ServiceTier, SettingType, Settings, SETTINGS};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("omp-settings-test-{tag}-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn schema_covers_ts_registry() {
    // 526 register({...}) definitions extracted from TS v18.6.1, plus the 19
    // settings only the TS panel (`createSettingsHost`) lists, plus the omp-rs
    // `magicKeywords.ultracode` gate.
    assert_eq!(SETTINGS.len(), 546);
    assert!(lookup("magicKeywords.ultracode").is_some());
    assert!(lookup("modelRoles").is_some());
    assert!(lookup("cycleOrder").is_some());
    assert!(lookup("workspace.additionalDirectories").is_some());
    assert!(lookup("nope.unknown").is_none());
    let tier = lookup("tier.openai").unwrap();
    assert!(matches!(tier.ty, SettingType::Enum(vals) if vals.contains(&"flex")));
}

#[test]
fn defaults_come_from_ts() {
    let dir = tempdir("defaults");
    let settings = Settings::load_from(None, &dir);
    assert_eq!(
        settings.cycle_order(),
        vec![
            "smol".to_string(),
            "default".to_string(),
            "slow".to_string()
        ]
    );
    assert_eq!(
        settings.stream_server_url().as_deref(),
        Some("https://live.omp.sh")
    );
    assert_eq!(settings.tier("openai"), Some(ServiceTier::None));
    assert_eq!(settings.tier("subagent"), Some(ServiceTier::Inherit));
    assert!(settings.disabled_extensions().is_empty());
    assert!(settings.disabled_providers().is_empty());
    assert!(settings.workspace_additional_directories().is_empty());
    assert!(!settings.images_urls_enabled());
    assert_eq!(
        settings.images_urls_backends(),
        vec!["provider-files", "tailscale", "cloudflared", "litterbox"]
    );
    // defaultThinkingLevel TS default is "high".
    assert_eq!(
        settings.get_string("defaultThinkingLevel").as_deref(),
        Some("high")
    );
}

#[test]
fn project_overrides_global_and_unknown_keys_preserved() {
    let root = tempdir("layers");
    let global = root.join("config.yml");
    fs::write(
        &global,
        "
cycleOrder: [a, b]
modelRoles:
  default: big-model
  judge: judge-model
customUserKey:
  nested: 1
",
    )
    .unwrap();
    let project = root.join(".omp");
    fs::create_dir_all(&project).unwrap();
    fs::write(
        project.join("config.yml"),
        "
modelRoles:
  default: null
  smol: small-model
stream:
  serverUrl: http://localhost:9999
",
    )
    .unwrap();

    let settings = Settings::load_from(Some(&global), &project);
    assert_eq!(
        settings.cycle_order(),
        vec!["a".to_string(), "b".to_string()]
    );
    // null project role falls back to global; new project role merges in.
    let roles = settings.model_roles();
    assert_eq!(roles.get("default").map(String::as_str), Some("big-model"));
    assert_eq!(roles.get("judge").map(String::as_str), Some("judge-model"));
    assert_eq!(roles.get("smol").map(String::as_str), Some("small-model"));
    assert_eq!(
        settings.stream_server_url().as_deref(),
        Some("http://localhost:9999")
    );
    // Unknown keys are kept in the raw map, but only unknown paths are flagged.
    assert_eq!(
        settings.raw()["customUserKey"]["nested"],
        serde_json::json!(1)
    );
    assert_eq!(settings.unknown_keys(), vec!["customUserKey.nested"]);
}

#[test]
fn env_overrides_every_layer() {
    let dir = tempdir("env");
    // skillshare/client.ts documents OMP_AUTH_BROKER_TOKEN etc.; use a string env
    // setting and a boolean one with parseFlag semantics.
    std::env::set_var("OMP_AUTH_BROKER_URL", "https://broker.example");
    std::env::set_var("PI_AUTO_QA", "yes");
    let settings = Settings::load_from(None, &dir);
    assert_eq!(
        settings.get_string("auth.broker.url").as_deref(),
        Some("https://broker.example")
    );
    assert_eq!(settings.get_bool("dev.autoqa"), Some(true));
    std::env::remove_var("OMP_AUTH_BROKER_URL");
    std::env::remove_var("PI_AUTO_QA");
    let _ = Provenance::Default; // provenance API smoke
    assert!(settings.global_path().is_none());
}

#[test]
fn model_presets_and_tags_parse() {
    let root = tempdir("presets");
    let global = root.join("config.yml");
    fs::write(
        &global,
        "
modelPresets:
  focused:
    modelRoles:
      default: glm
    defaultThinkingLevel: high
modelTags:
  cheap:
    models: [glm, qwen]
",
    )
    .unwrap();
    let settings = Settings::load_from(Some(&global), &root.join(".omp"));
    let presets = settings.model_presets();
    let preset = presets.get("focused").expect("preset parsed");
    assert_eq!(
        preset.model_roles.get("default").and_then(|v| v.as_str()),
        Some("glm")
    );
    assert_eq!(preset.default_thinking_level.as_deref(), Some("high"));
    let tags: BTreeMap<_, _> = settings.model_tags();
    assert_eq!(
        tags["cheap"].models,
        vec!["glm".to_string(), "qwen".to_string()]
    );
}

#[test]
fn real_user_config_loads_without_unknown_key_warnings() {
    // Read-only smoke against the actual user config TS already accepts.
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME set");
    let config = home.join(".omp").join("agent").join("config.yml");
    if !config.is_file() {
        eprintln!("skipping: no user config at {}", config.display());
        return;
    }
    let settings = Settings::load_from(Some(&config), &home.join(".omp"));
    assert!(
        settings.unknown_keys().is_empty(),
        "unknown keys TS knows: {:?}",
        settings.unknown_keys()
    );
    // A few known values from the real file (structure only).
    assert!(settings.get("modelRoles").is_object());
    assert!(settings.model_roles().contains_key("default"));
}
