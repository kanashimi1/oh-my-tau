//! Layered settings store: TS defaults < global config.yml < project config < env vars.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::{lookup, SettingDef, SETTINGS};

/// One saved model preset (`modelPresets.<name>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPreset {
    /// Role assignments captured by `/modelpreset save`.
    #[serde(default)]
    pub model_roles: Map<String, Value>,
    /// `defaultThinkingLevel` at save time; absent when the preset leaves it alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_thinking_level: Option<String>,
}

/// One `modelTags.<name>` entry: a named group of model selectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelTagDef {
    #[serde(default)]
    pub models: Vec<String>,
}

/// Service-tier selection for one provider family (`tier.*` settings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceTier {
    /// Omit the tier header / standard processing.
    None,
    /// Inherit the main agent's live per-family tier (`tier.subagent`/`tier.advisor` only).
    Inherit,
    /// Explicit tier name (e.g. `auto`, `flex`, `priority`, `ultrafast`, `default`, `scale`).
    Named(&'static str),
}

impl ServiceTier {
    /// Resolve from the raw configured string, validating against the TS value list.
    #[must_use]
    pub fn parse(family: &str, raw: &str) -> Option<Self> {
        let id = match family {
            "openai" => "tier.openai",
            "anthropic" => "tier.anthropic",
            "google" => "tier.google",
            "subagent" => "tier.subagent",
            "advisor" => "tier.advisor",
            _ => return None,
        };
        let def = lookup(id)?;
        let crate::SettingType::Enum(values) = def.ty else {
            return None;
        };
        if !values.contains(&raw) {
            return None;
        }
        Some(match raw {
            "none" => Self::None,
            "inherit" => Self::Inherit,
            other => Self::Named(
                values
                    .iter()
                    .find(|v| **v == other)
                    .copied()
                    .unwrap_or("none"),
            ),
        })
    }
}

/// Which layer supplied a setting's effective value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    Default,
    Global,
    Project,
    Env,
}

/// A loaded settings scope. Construct with [`Settings::load`].
#[derive(Debug, Clone)]
pub struct Settings {
    /// Deep-merged view: defaults < global < project < env, unknown keys kept.
    merged: Map<String, Value>,
    /// Dotted paths present in no TS setting (group prefix, exact id, or record subtree).
    unknown: Vec<String>,
    /// Sources actually read, for diagnostics.
    global_path: Option<PathBuf>,
    project_path: Option<PathBuf>,
}

impl Settings {
    /// Load settings for `cwd`: global `~/.omp/agent/config.yml` (or the `OMP_PROFILE`
    /// profile's agent dir) merged under `<cwd>/.omp/config.yml` / `config.yaml`, then
    /// environment overrides applied per setting's `env:` name.
    ///
    /// Missing files are not an error; an unreadable/invalid file is skipped and noted
    /// in [`Settings::global_path`]/[`Settings::project_path`] being `None` for that layer.
    #[must_use]
    pub fn load(cwd: &Path) -> Self {
        let global = agent_config_path();
        let project_dir = cwd.join(".omp");
        Self::load_from(global.as_deref(), &project_dir)
    }

    /// Same layering with explicit paths (used by tests and embedders).
    pub fn load_from(global_yaml: Option<&Path>, project_dir: &Path) -> Self {
        // Layer 1: TS defaults. Compile-time constants: parsed once per process
        // (507 YAML snippets cost ~9ms, and tools load settings on every call).
        let mut merged = defaults_layer().clone();

        // Layer 2: global config.yml / config.yaml.
        let global_path = global_yaml.and_then(load_yaml_file);
        if let Some((path, value)) = &global_path {
            if let Some(obj) = value.as_object() {
                deep_merge(&mut merged, obj.clone());
            }
            let _ = path;
        }

        // Layer 3: project .omp/config.yml then config.yaml (later wins).
        let mut project_path = None;
        for name in ["config.yml", "config.yaml"] {
            let candidate = project_dir.join(name);
            if let Some((_, value)) = load_yaml_file(&candidate) {
                if let Some(obj) = value.as_object() {
                    deep_merge(&mut merged, obj.clone());
                }
                project_path = Some(candidate);
            }
        }

        // Layer 4: environment overrides.
        let mut applied_env = Map::new();
        for def in SETTINGS {
            if let Some(env_name) = def.env {
                if let Ok(raw) = std::env::var(env_name) {
                    if let Some(value) = parse_env_value(def, &raw) {
                        set_path(&mut applied_env, def.id, value);
                    }
                }
            }
        }
        deep_merge(&mut merged, applied_env);

        let unknown = collect_unknown(&merged);
        Self {
            merged,
            unknown,
            global_path: global_path.map(|(p, _)| p),
            project_path,
        }
    }

    /// Raw effective value at dotted `id` (merged layers; `Null` when unset).
    #[must_use]
    pub fn get(&self, id: &str) -> Value {
        get_path(&self.merged, id).cloned().unwrap_or(Value::Null)
    }

    /// Effective string value, or the TS default when unset/mistyped.
    #[must_use]
    pub fn get_string(&self, id: &str) -> Option<String> {
        self.get(id).as_str().map(ToOwned::to_owned)
    }

    /// Effective boolean value.
    #[must_use]
    pub fn get_bool(&self, id: &str) -> Option<bool> {
        self.get(id).as_bool()
    }

    /// Effective numeric value.
    #[must_use]
    pub fn get_number(&self, id: &str) -> Option<f64> {
        self.get(id).as_f64()
    }

    /// Effective string-array value.
    #[must_use]
    pub fn get_string_array(&self, id: &str) -> Vec<String> {
        self.get(id)
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Effective record value (`modelRoles`-style string maps).
    #[must_use]
    pub fn get_record(&self, id: &str) -> Map<String, Value> {
        self.get(id).as_object().cloned().unwrap_or_default()
    }

    /// Dotted paths in the merged config that no TS setting knows (sorted).
    #[must_use]
    pub fn unknown_keys(&self) -> &[String] {
        &self.unknown
    }

    /// Which layer effectively supplies `id` (env beats project beats global beats default).
    #[must_use]
    pub fn provenance(&self, id: &str) -> Option<Provenance> {
        lookup(id)?;
        // Env wins: re-check the variable directly.
        if let Some(def) = lookup(id) {
            if let Some(env_name) = def.env {
                if std::env::var(env_name)
                    .ok()
                    .and_then(|raw| parse_env_value(def, &raw))
                    .is_some()
                {
                    return Some(Provenance::Env);
                }
            }
        }
        let value = get_path(&self.merged, id);
        let has_default = lookup(id).and_then(|def| def.default_yaml).is_some();
        match (value, has_default) {
            (Some(_), _) => Some(Provenance::Project), // merged value present; layer detail below
            (None, true) => Some(Provenance::Default),
            (None, false) => None,
        }
    }

    /// Path of the global config.yml actually read, if any.
    #[must_use]
    pub fn global_path(&self) -> Option<&Path> {
        self.global_path.as_deref()
    }

    /// Path of the project config actually read, if any.
    #[must_use]
    pub fn project_path(&self) -> Option<&Path> {
        self.project_path.as_deref()
    }

    /// The full merged raw map (unknown keys included).
    #[must_use]
    pub fn raw(&self) -> &Map<String, Value> {
        &self.merged
    }
}

impl Settings {
    /// Named model presets (`modelPresets`, TS default `{}`).
    #[must_use]
    pub fn model_presets(&self) -> BTreeMap<String, ModelPreset> {
        self.get("modelPresets")
            .as_object()
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        serde_json::from_value::<ModelPreset>(v.clone())
                            .ok()
                            .map(|preset| (k.clone(), preset))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Settings {
    /// Role assignments (`modelRoles`, TS default `{}`); project layer may set a
    /// role to `null` to fall back to the global value (TS `projectLayerForMerge`).
    #[must_use]
    pub fn model_roles(&self) -> BTreeMap<String, String> {
        let mut roles = BTreeMap::new();
        if let Some(obj) = self.get("modelRoles").as_object() {
            for (role, selector) in obj {
                if let Some(selector) = selector.as_str() {
                    roles.insert(role.clone(), selector.to_owned());
                }
            }
        }
        roles
    }

    /// Named model tags (`modelTags`, TS default `{}`).
    #[must_use]
    pub fn model_tags(&self) -> BTreeMap<String, ModelTagDef> {
        self.get("modelTags")
            .as_object()
            .map(|obj| -> BTreeMap<String, ModelTagDef> {
                obj.iter()
                    .filter_map(|(k, v)| {
                        serde_json::from_value::<ModelTagDef>(v.clone())
                            .ok()
                            .map(|tag| (k.clone(), tag))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Model cycle order (TS default `["smol", "default", "slow"]`).
    #[must_use]
    pub fn cycle_order(&self) -> Vec<String> {
        self.get_string_array("cycleOrder")
    }

    /// Globally disabled extensions (`disabledExtensions`, TS default `[]`).
    #[must_use]
    pub fn disabled_extensions(&self) -> Vec<String> {
        self.get_string_array("disabledExtensions")
    }

    /// Disabled providers (`disabledProviders`, TS default `[]`).
    #[must_use]
    pub fn disabled_providers(&self) -> Vec<String> {
        self.get_string_array("disabledProviders")
    }

    /// Enabled-models allowlist (`enabledModels`, TS default `[]` = all).
    #[must_use]
    pub fn enabled_models(&self) -> Vec<String> {
        self.get_string_array("enabledModels")
    }

    /// Enabled-providers allowlist (`enabledProviders`, TS default `[]` = all).
    #[must_use]
    pub fn enabled_providers(&self) -> Vec<String> {
        self.get_string_array("enabledProviders")
    }

    /// Extra workspace roots beyond the project cwd
    /// (`workspace.additionalDirectories`, TS default `[]`).
    #[must_use]
    pub fn workspace_additional_directories(&self) -> Vec<String> {
        self.get_string_array("workspace.additionalDirectories")
    }

    /// Live-stream server URL (`stream.serverUrl`, TS default `https://live.omp.sh`).
    #[must_use]
    pub fn stream_server_url(&self) -> Option<String> {
        self.get_string("stream.serverUrl")
    }

    /// Extra redaction patterns for streamed rows (`stream.redactPatterns`, default `[]`).
    #[must_use]
    pub fn stream_redact_patterns(&self) -> Vec<String> {
        self.get_string_array("stream.redactPatterns")
    }

    /// Whether blob-broker image URLs are enabled (`images.urls.enabled`, default `false`).
    #[must_use]
    pub fn images_urls_enabled(&self) -> bool {
        self.get_bool("images.urls.enabled").unwrap_or(false)
    }

    /// Blob URL backend chain (`images.urls.backends`; TS default
    /// `["provider-files", "tailscale", "cloudflared", "litterbox"]`).
    #[must_use]
    pub fn images_urls_backends(&self) -> Vec<String> {
        self.get_string_array("images.urls.backends")
    }

    /// Per-family service tier (`tier.openai` etc., TS default `"none"`).
    /// Returns `None` for unknown families or invalid configured values.
    #[must_use]
    pub fn tier(&self, family: &str) -> Option<ServiceTier> {
        let id = match family {
            "openai" => "tier.openai",
            "anthropic" => "tier.anthropic",
            "google" => "tier.google",
            "subagent" => "tier.subagent",
            "advisor" => "tier.advisor",
            _ => return None,
        };
        self.get_string(id)
            .and_then(|raw| ServiceTier::parse(family, &raw))
    }
}

/// Persist `value` for the registered setting `id` into the global config.yml
/// (TS `cfg://` set → `saveConfigFile`), creating the agent dir when missing.
///
/// The existing file is read and updated in place (unknown keys preserved,
/// same guarantee as [`Settings::load`]); the value's shape must match the
/// setting's registry type. Returns the path written.
/// A JSON `null` value unsets the key (the default applies again).
///
/// # Errors
///
/// `InvalidInput` when `id` is not a registered setting or `value` does not
/// match its type; `Io` when the file cannot be written.
pub fn write_global(id: &str, value: &Value) -> std::io::Result<PathBuf> {
    let def = lookup(id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown setting: {id}"),
        )
    })?;
    let valid = match def.ty {
        crate::SettingType::Boolean => value.is_boolean(),
        crate::SettingType::Number => value.is_number(),
        crate::SettingType::String => value.is_string(),
        // An empty value list means TS derives the choices at runtime
        // (status-line presets, voices, devices): any string is accepted.
        crate::SettingType::Enum(values) => value
            .as_str()
            .is_some_and(|s| values.is_empty() || values.contains(&s)),
        crate::SettingType::Array => value.is_array(),
        crate::SettingType::Record => value.is_object(),
    };
    // `null` unsets the key so the default applies again (TS `settings.unset`).
    if !valid && !value.is_null() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "value for {id} does not match registry type {}",
                def.ty.ts_name()
            ),
        ));
    }

    let path = agent_config_path().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no agent config dir (HOME unset)",
        )
    })?;
    let mut merged = load_yaml_file(&path)
        .and_then(|(_, value)| value.as_object().cloned())
        .unwrap_or_default();
    if value.is_null() {
        remove_path(&mut merged, id);
    } else {
        set_path(&mut merged, id, value.clone());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let yaml = serde_yaml_ng::to_string(&Value::Object(merged)).map_err(std::io::Error::other)?;
    std::fs::write(&path, yaml)?;
    Ok(path)
}

/// Resolve the agent config dir: `OMP_PROFILE` profile agent dir when the profile
/// exists, else `~/.omp/agent` (TS `getAgentDir`).
fn agent_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let root = home.join(".omp");
    if let Ok(profile) = std::env::var("OMP_PROFILE") {
        let profile_dir = root.join("profiles").join(profile).join("agent");
        if profile_dir.is_dir() {
            return Some(profile_dir);
        }
    }
    Some(root.join("agent"))
}

fn agent_config_path() -> Option<PathBuf> {
    let agent = agent_dir()?;
    for name in ["config.yml", "config.yaml"] {
        let candidate = agent.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    Some(agent.join("config.yml"))
}

/// The TS-defaults layer: every `default_yaml` parsed and nested by id.
fn defaults_layer() -> &'static Map<String, Value> {
    static DEFAULTS: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    DEFAULTS.get_or_init(|| {
        let mut merged = Map::new();
        for def in SETTINGS {
            if let Some(raw) = def.default_yaml {
                if let Ok(value) = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(raw) {
                    let value = yaml_to_json(value);
                    if !value.is_null() {
                        set_path(&mut merged, def.id, value);
                    }
                }
            }
        }
        merged
    })
}

fn load_yaml_file(path: &Path) -> Option<(PathBuf, Value)> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).ok()?;
    Some((path.to_path_buf(), yaml_to_json(value)))
}

/// Convert a YAML value to JSON, dropping non-string mapping keys (TS keeps only
/// string-keyed records in config.yml).
fn yaml_to_json(value: serde_yaml_ng::Value) -> Value {
    match value {
        serde_yaml_ng::Value::Null => Value::Null,
        serde_yaml_ng::Value::Bool(b) => Value::Bool(b),
        serde_yaml_ng::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i)
            } else if let Some(f) = n.as_f64() {
                Value::from(f)
            } else {
                Value::Null
            }
        }
        serde_yaml_ng::Value::String(s) => Value::String(s),
        serde_yaml_ng::Value::Sequence(items) => {
            Value::Array(items.into_iter().map(yaml_to_json).collect())
        }
        serde_yaml_ng::Value::Mapping(map) => {
            let mut obj = Map::new();
            for (k, v) in map {
                if let serde_yaml_ng::Value::String(key) = k {
                    obj.insert(key, yaml_to_json(v));
                }
            }
            Value::Object(obj)
        }
        serde_yaml_ng::Value::Tagged(tagged) => yaml_to_json(tagged.value),
    }
}

/// Deep merge `over` into `base` (objects merge recursively; `Null` deletes the key,
/// mirroring TS project-layer `null` semantics for records such as `modelRoles`).
fn deep_merge(base: &mut Map<String, Value>, over: Map<String, Value>) {
    for (key, value) in over {
        match (base.get_mut(&key), value) {
            (Some(Value::Object(base_obj)), Value::Object(over_obj)) => {
                deep_merge(base_obj, over_obj);
            }
            (_, Value::Null) => {
                // `null` in an upper layer means "unset": the lower layer's value
                // stands (TS `projectLayerForMerge` role fallback).
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

fn set_path(base: &mut Map<String, Value>, id: &str, value: Value) {
    let mut segments = id.split('.').peekable();
    let mut current = base;
    while let Some(segment) = segments.next() {
        if segments.peek().is_none() {
            current.insert(segment.to_owned(), value);
            return;
        }
        let entry = current
            .entry(segment.to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        if !entry.is_object() {
            *entry = Value::Object(Map::new());
        }
        current = entry.as_object_mut().expect("object just ensured");
    }
}

/// Remove the dotted `id` from the map, pruning parents left empty.
fn remove_path(base: &mut Map<String, Value>, id: &str) {
    fn walk(map: &mut Map<String, Value>, segments: &[&str]) {
        let Some((head, rest)) = segments.split_first() else {
            return;
        };
        if rest.is_empty() {
            map.remove(*head);
            return;
        }
        if let Some(Value::Object(child)) = map.get_mut(*head) {
            walk(child, rest);
            if child.is_empty() {
                map.remove(*head);
            }
        }
    }
    let segments: Vec<&str> = id.split('.').collect();
    walk(base, &segments);
}

fn get_path<'a>(base: &'a Map<String, Value>, id: &str) -> Option<&'a Value> {
    let mut map = base;
    let last = id.rsplit('.').next().unwrap_or(id);
    let depth = id.matches('.').count();
    for (idx, segment) in id.split('.').enumerate() {
        let value = map.get(segment)?;
        if segment == last && idx == depth {
            return Some(value);
        }
        map = value.as_object()?;
    }
    None
}

/// Registry `env:` parsing (mirrors TS `parseFlag`/type parsers):
/// booleans accept 1/y/true/yes/on (any case) as true, other non-blank text as false,
/// empty text is unset; numbers must parse; enums must be listed; strings must be non-blank.
fn parse_env_value(def: &SettingDef, raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    match def.ty {
        crate::SettingType::Boolean => {
            if trimmed.is_empty() {
                None
            } else {
                Some(Value::Bool(matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "1" | "y" | "true" | "yes" | "on"
                )))
            }
        }
        crate::SettingType::Number => trimmed.parse::<f64>().ok().map(Value::from),
        crate::SettingType::String | crate::SettingType::Array | crate::SettingType::Record => {
            if trimmed.is_empty() {
                None
            } else {
                Some(Value::String(trimmed.to_owned()))
            }
        }
        crate::SettingType::Enum(values) => {
            if values.is_empty() {
                // Runtime-derived values: accept any non-blank text like a string.
                if trimmed.is_empty() {
                    None
                } else {
                    Some(Value::String(trimmed.to_owned()))
                }
            } else if values.contains(&trimmed) {
                Some(Value::String(trimmed.to_owned()))
            } else {
                None
            }
        }
    }
}

/// Dotted paths in `merged` that no registered setting explains: not an exact id,
/// not a group prefix of one, and not inside a `record`-typed subtree.
fn collect_unknown(merged: &Map<String, Value>) -> Vec<String> {
    let mut unknown = Vec::new();
    let mut stack: Vec<(String, &Map<String, Value>)> = vec![(String::new(), merged)];
    while let Some((prefix, map)) = stack.pop() {
        for (key, value) in map {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            if is_known_path(&path) {
                if value.is_object() && !is_record_prefix(&path) {
                    stack.push((path, value.as_object().expect("checked is_object")));
                }
            } else if value.as_object().is_some_and(|obj| !obj.is_empty()) {
                // Unknown container: report its unknown leaves rather than the group.
                stack.push((path, value.as_object().expect("checked is_object")));
            } else {
                unknown.push(path);
            }
        }
    }
    unknown.sort();
    unknown
}

/// `path` lies strictly under the dotted group `prefix` (`prefix.` + more).
fn is_dotted_child(path: &str, prefix: &str) -> bool {
    path.strip_prefix(prefix)
        .is_some_and(|rest| rest.starts_with('.'))
}

/// Setting ids plus their dotted group prefixes, and the record-typed ids;
/// built once (`collect_unknown` probes every merged path on each load).
struct KnownPaths {
    ids_and_groups: std::collections::HashSet<&'static str>,
    records: Vec<&'static str>,
}

fn known_paths() -> &'static KnownPaths {
    static KNOWN: std::sync::OnceLock<KnownPaths> = std::sync::OnceLock::new();
    KNOWN.get_or_init(|| {
        let mut ids_and_groups = std::collections::HashSet::new();
        let mut records = Vec::new();
        for def in SETTINGS {
            ids_and_groups.insert(def.id);
            for (dot, _) in def.id.match_indices('.') {
                ids_and_groups.insert(&def.id[..dot]);
            }
            if matches!(def.ty, crate::SettingType::Record) {
                records.push(def.id);
            }
        }
        KnownPaths {
            ids_and_groups,
            records,
        }
    })
}

fn is_known_path(path: &str) -> bool {
    // Exact id, group prefix of an id, or leaf inside a record subtree.
    let known = known_paths();
    known.ids_and_groups.contains(path) || known.records.iter().any(|id| is_dotted_child(path, id))
}

fn is_record_prefix(path: &str) -> bool {
    known_paths()
        .records
        .iter()
        .any(|id| *id == path || is_dotted_child(path, id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::sync::LazyLock;

    /// HOME is process-global; serialize the env-mutating tests.
    static HOME_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct TempHome(PathBuf);
    impl TempHome {
        fn new() -> Self {
            let base =
                std::env::temp_dir().join(format!("omp-settings-test-{}", std::process::id()));
            std::fs::create_dir_all(&base).unwrap();
            std::env::set_var("HOME", &base);
            std::env::remove_var("OMP_PROFILE");
            Self(base)
        }
    }
    impl Drop for TempHome {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn write_global_round_trips_through_load() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let old_profile = std::env::var_os("OMP_PROFILE");
        {
            let _home = TempHome::new();
            let path = write_global("colorBlindMode", &Value::Bool(true)).unwrap();
            assert!(path.ends_with(".omp/agent/config.yml"), "{path:?}");
            // Unknown pre-existing keys survive the write.
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(text.contains("colorBlindMode: true"), "{text}");

            let loaded = Settings::load(Path::new("/nonexistent-project"));
            assert_eq!(loaded.get_bool("colorBlindMode"), Some(true));
        }
        if let Some(h) = old_home {
            std::env::set_var("HOME", h);
        }
        if let Some(p) = old_profile {
            std::env::set_var("OMP_PROFILE", p);
        }
    }

    #[test]
    fn write_global_rejects_bad_values() {
        let _guard = HOME_LOCK.lock();
        assert!(write_global("no.such.setting", &Value::Bool(true)).is_err());
        assert!(write_global("colorBlindMode", &Value::String("x".into())).is_err());
        assert!(write_global("theme.dark", &Value::Bool(true)).is_err());
    }
}
