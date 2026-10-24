//! Startup prepaint cache — port of `pi-tui/src/status-line/startup.ts` §6
//! (omp-rs half): persist the chrome-relevant session state at exit and
//! reload it at launch so the first frame (before the runtime connects and
//! replays session state) already shows the real model, cwd and session name
//! instead of blanks.
//!
//! The file lives at `~/.omp/agent/cache/tui-startup.json` (single-slot, one
//! per user — the last exited pane wins, matching TS `InteractiveMode`'s
//! persisted startup data). Written on graceful teardown, read once at
//! [`StartupCache::load`].

use std::path::PathBuf;

use serde_json::Value;

/// Chrome state persisted across launches.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StartupCache {
    /// Last active model (`provider/id`).
    pub model: Option<String>,
    /// Working directory of the last session.
    pub cwd: Option<PathBuf>,
    /// Session name at exit (herdr-pane-names source of truth for the first
    /// title/banner frame).
    pub session_name: Option<String>,
    /// Thinking level name, if any.
    pub thinking_level: Option<String>,
}

impl StartupCache {
    /// Cache file path (`$HOME/.omp/agent/cache/tui-startup.json`); `None`
    /// without a home dir.
    #[must_use]
    pub fn path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join(".omp")
                .join("agent")
                .join("cache")
                .join("tui-startup.json")
        })
    }

    /// Load the cache; missing/corrupt file → default (never fails launch).
    #[must_use]
    pub fn load() -> Self {
        let Some(path) = Self::path() else {
            return Self::default();
        };
        let Ok(raw) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        let Ok(v) = serde_json::from_str::<Value>(&raw) else {
            return Self::default();
        };
        let str_field = |key: &str| {
            v.get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|s| !s.is_empty())
        };
        Self {
            model: str_field("model"),
            cwd: str_field("cwd").map(PathBuf::from),
            session_name: str_field("sessionName"),
            thinking_level: str_field("thinkingLevel"),
        }
    }

    /// Persist atomically (temp file + rename); failures are silent — the
    /// cache is an optimization, never a requirement (TS fire-and-forget).
    pub fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        let v = serde_json::json!({
            "model": self.model,
            "cwd": self.cwd.as_ref().map(|p| p.display().to_string()),
            "sessionName": self.session_name,
            "thinkingLevel": self.thinking_level,
        });
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, v.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }

    /// Drop the cache (e.g. explicit reset); failures silent.
    pub fn clear() {
        if let Some(path) = Self::path() {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::HOME_LOCK;

    fn with_home(tag: &str, f: impl FnOnce()) {
        let dir = std::env::temp_dir().join(format!("omp-tui-startup-{tag}"));
        let _guard = HOME_LOCK.lock();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".omp/agent/cache")).unwrap();
        let prev = std::env::var_os("HOME");
        std::env::set_var("HOME", &dir);
        f();
        match prev {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_then_load_roundtrip() {
        with_home("rt", || {
            let cache = StartupCache {
                model: Some("zai/glm-5.3-flash".into()),
                cwd: Some("/home/user/omrs".into()),
                session_name: Some("conv".into()),
                thinking_level: Some("medium".into()),
            };
            cache.save();
            let loaded = StartupCache::load();
            assert_eq!(loaded, cache);
        });
    }

    #[test]
    fn missing_file_loads_default() {
        with_home("miss", || {
            assert_eq!(StartupCache::load(), StartupCache::default());
        });
    }

    #[test]
    fn corrupt_file_loads_default() {
        with_home("bad", || {
            let path = StartupCache::path().unwrap();
            std::fs::write(path, "{not json").unwrap();
            assert_eq!(StartupCache::load(), StartupCache::default());
        });
    }
}
