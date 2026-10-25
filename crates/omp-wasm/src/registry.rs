//! WASM plugin discovery.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::host::WasmModule;

/// A discovered, successfully loaded plugin.
#[derive(Debug)]
pub struct PluginEntry {
    /// The loaded module (compiled + manifest parsed).
    pub module: Arc<WasmModule>,
}

/// The result of a discovery pass.
#[derive(Debug, Default)]
pub struct Discovery {
    /// Plugins that loaded successfully.
    pub plugins: Vec<PluginEntry>,
    /// Plugins that failed to load: `(path, error)`.
    pub failures: Vec<(PathBuf, String)>,
}

/// Scan `~/.omp/agent/plugins/*.wasm` and `<cwd>/.omp/plugins/*.wasm` and
/// load every `.wasm` file. Broken plugins are skipped (logged) and recorded
/// in [`Discovery::failures`]; they are never fatal.
pub fn discover(cwd: &Path) -> Discovery {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".omp/agent/plugins"));
    }
    dirs.push(cwd.join(".omp/plugins"));

    let mut discovery = Discovery::default();
    let mut seen = std::collections::HashSet::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "wasm"))
            .collect();
        paths.sort();
        for path in paths {
            if !seen.insert(path.clone()) {
                continue;
            }
            match WasmModule::load(&path) {
                Ok(module) => discovery.plugins.push(PluginEntry {
                    module: Arc::new(module),
                }),
                Err(error) => {
                    tracing::warn!(
                        "omt: skipping broken WASM plugin {}: {error:#}",
                        path.display()
                    );
                    discovery.failures.push((path, format!("{error:#}")));
                }
            }
        }
    }
    discovery
}
