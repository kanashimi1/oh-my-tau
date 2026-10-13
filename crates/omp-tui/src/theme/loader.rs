//! Theme file loading — the port of `pi-tui/src/theme/loader.ts`.
//!
//! Planned: read a theme JSON/YAML file (omp user themes under
//! `~/.omp/agent/themes/`), validate against the TS schema
//! (`theme/schema.ts` + `schema-validation.ts`), and merge onto
//! [`Theme::default`]. Until implemented, [`load`] accepts only the built-in
//! names and returns the default theme.

use thiserror::Error;

use super::Theme;

/// Why a theme file could not be loaded.
#[derive(Debug, Error)]
pub enum ThemeError {
    /// The file could not be read.
    #[error("theme file unreadable: {0}")]
    Io(String),
    /// The file parsed but failed schema validation.
    #[error("theme file invalid: {0}")]
    Invalid(String),
}

/// Load a theme by name or path. MVP: only `"default"` is supported.
///
/// # Errors
///
/// [`ThemeError::Invalid`] for any name other than `"default"` until the
/// file loader lands.
pub fn load(name_or_path: &str) -> Result<Theme, ThemeError> {
    if name_or_path == "default" {
        return Ok(Theme::default());
    }
    Err(ThemeError::Invalid(format!(
        "theme files not supported yet: {name_or_path}"
    )))
}
