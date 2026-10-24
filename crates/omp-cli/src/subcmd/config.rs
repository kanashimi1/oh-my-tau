//! `omp-rs config` — read-only view of omp settings, mirroring the reading
//! half of TS `cli/config-cli.ts` (`list`, `get`, `path`). `set`/`reset` are
//! intentionally not ported here: writing config.yml belongs to the full
//! settings registry, and a subcommand must never surprise-modify user config.

use anyhow::Result;
use omp_settings::{SettingType, Settings, SETTINGS};
use serde_json::Value;

const REDACTED: &str = "********";

/// Entry point: `omp-rs config [list|get <key>|path] [--json]`.
///
/// # Errors
/// Never mutates; fails only on unexpected argument shapes.
pub fn run(args: &[String]) -> Result<i32> {
    let mut json = false;
    let mut positional: Vec<&str> = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            other if other.starts_with('-') => anyhow::bail!("unknown flag: {other}"),
            other => positional.push(other),
        }
    }
    match positional.first().copied().unwrap_or("list") {
        "list" => list(json),
        "get" => {
            let Some(key) = positional.get(1) else {
                anyhow::bail!("Usage: omt config get <key>\n\nRun 'omt config list' to see available keys");
            };
            get(key, json)?;
        }
        "path" => {
            println!("{}", omp_providers::AgentDir::default().0.display());
        }
        "set" | "reset" => {
            anyhow::bail!(
                "config {} is not supported by omt (read-only view)",
                positional[0]
            );
        }
        other => anyhow::bail!("Unknown config command: {other}\nValid commands: list, get, path"),
    }
    Ok(0)
}

fn list(json: bool) {
    let settings = Settings::load(&std::env::current_dir().unwrap_or_default());
    if json {
        let mut out = serde_json::Map::new();
        for def in SETTINGS {
            let value = settings.get(def.id);
            if def.credential && value.as_str().is_some_and(|s| !s.is_empty()) {
                out.insert(
                    def.id.to_string(),
                    Value::Object(serde_json::Map::from_iter([
                        ("redacted".to_string(), Value::Bool(true)),
                        (
                            "type".to_string(),
                            Value::String(def.ty.ts_name().to_string()),
                        ),
                    ])),
                );
            } else {
                out.insert(
                    def.id.to_string(),
                    Value::Object(serde_json::Map::from_iter([
                        ("value".to_string(), value),
                        (
                            "type".to_string(),
                            Value::String(def.ty.ts_name().to_string()),
                        ),
                    ])),
                );
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&Value::Object(out)).unwrap_or_default()
        );
        return;
    }

    println!("Settings:\n");
    let mut current_tab: Option<&str> = None;
    for def in SETTINGS {
        // The TS registry groups by UI tab; the generated table has no tab, so
        // group by the first path segment, "config"-ish keys first.
        let tab = def.id.split('.').next().unwrap_or(def.id);
        if current_tab != Some(tab) {
            println!("[{tab}]");
            current_tab = Some(tab);
        }
        let value = settings.get(def.id);
        let is_secret_string = value.as_str().is_some_and(|s| !s.is_empty());
        let rendered = if def.credential && is_secret_string {
            REDACTED.to_string()
        } else {
            format_value(&value)
        };
        println!("  {} = {} {}", def.id, rendered, type_display(def.ty));
    }
}

fn get(key: &str, json: bool) -> Result<()> {
    let Some(def) = omp_settings::lookup(key) else {
        anyhow::bail!("Unknown setting: {key}\n\nRun 'omt config list' to see available keys");
    };
    let settings = Settings::load(&std::env::current_dir().unwrap_or_default());
    let value = settings.get(def.id);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "key": def.id,
                "value": value,
                "type": def.ty.ts_name(),
            }))?
        );
    } else {
        println!("{}", format_value(&value));
    }
    Ok(())
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Null => "(not set)".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn type_display(ty: SettingType) -> String {
    match ty {
        SettingType::Enum(values) if !values.is_empty() => format!("({})", values.join("|")),
        other => format!("({})", other.ts_name()),
    }
}
