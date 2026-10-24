//! Where each language's tree-sitter grammar comes from.
//!
//! Common languages link their grammar into the addon (`super::parsers`). The
//! rest are compiled to `.wasm` by
//! [stencil-hq/wasm-grammars](https://github.com/stencil-hq/wasm-grammars) and
//! published as `<file>.zst` assets of its
//! [`RELEASE`](super::wasm_grammars::RELEASE), which `bun run gen:grammars`
//! pins in [`super::wasm_grammars`]. The host
//! downloads a module into the directory set with [`set_grammar_dir`] the first
//! time it needs the language; [`WasmGrammar::load`] then reads it and keeps
//! the language for the life of the process.
//!
//! Only lexing runs in wasm: tree-sitter copies the parse tables into native
//! memory, so the resulting trees are ordinary native trees. A parser needs a
//! [`WasmStore`] to run a wasm grammar; [`parse`] keeps one per thread, and
//! every parse in this crate goes through it.
//!
//! `LanguageExt::get_ts_language` and `LanguageExt::ast_grep` panic for a wasm
//! grammar that is not installed, like ast-grep's own parse failures. Entry
//! points check [`GrammarSource::load`] first and treat
//! [`GrammarError::NotInstalled`] as "no tree" or report it.

use std::{
    error::Error,
    fmt,
    path::PathBuf,
    sync::{OnceLock, PoisonError, RwLock},
};

use anyhow::Result;
use tree_sitter::{Language, Parser, Tree};

/// Where a language's grammar comes from.
#[derive(Clone, Copy, Debug)]
pub enum GrammarSource {
    /// Linked into the addon.
    Native(fn() -> Language),
    /// Downloaded on demand (see the module docs).
    Wasm(&'static WasmGrammar),
}

impl GrammarSource {
    /// The grammar's language.
    ///
    /// # Errors
    /// Only for [`Self::Wasm`]; see [`WasmGrammar::load`].
    pub fn load(self) -> Result<Language, GrammarError> {
        match self {
            Self::Native(language) => Ok(language()),
            Self::Wasm(grammar) => grammar.load(),
        }
    }
}

/// Languages whose grammar comes from a [`GrammarSource`].
pub trait LanguageGrammar {
    fn grammar(&self) -> GrammarSource;
}

/// A grammar published as a WebAssembly module (see the module docs).
#[derive(Debug)]
pub struct WasmGrammar {
    /// `SupportLang::canonical_name` of the language.
    pub language: &'static str,
    /// Name the module exports its language under (`tree_sitter_<symbol>`).
    pub symbol: &'static str,
    /// File name in the grammar directory; the release asset is `<file>.zst`.
    pub file: &'static str,
    /// Lowercase hex SHA-256 of the module.
    pub sha256: &'static str,
    /// Byte size of the module.
    pub size: u64,
    loaded: OnceLock<Language>,
}

/// Why a wasm grammar is not available.
#[derive(Debug)]
pub enum GrammarError {
    /// No grammar directory is set, or the module is not in it yet.
    NotInstalled { language: &'static str },
    /// The module could not be read or is not a tree-sitter grammar.
    Invalid {
        language: &'static str,
        reason: String,
    },
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled { language } => write!(f, "the {language} grammar is not installed"),
            Self::Invalid { language, reason } => {
                write!(f, "the {language} grammar is invalid: {reason}")
            }
        }
    }
}

impl Error for GrammarError {}

static GRAMMAR_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Sets the directory wasm grammars are read from (`<dir>/<file>`).
pub fn set_grammar_dir(dir: PathBuf) {
    *GRAMMAR_DIR.write().unwrap_or_else(PoisonError::into_inner) = Some(dir);
}

fn module_path(file: &str) -> Option<PathBuf> {
    let dir = GRAMMAR_DIR.read().unwrap_or_else(PoisonError::into_inner);
    Some(dir.as_ref()?.join(file)).filter(|path| path.is_file())
}

impl WasmGrammar {
    /// Describes a published module; the generated `wasm_grammars` table is
    /// built from these.
    pub const fn new(
        language: &'static str,
        symbol: &'static str,
        file: &'static str,
        sha256: &'static str,
        size: u64,
    ) -> Self {
        Self {
            language,
            symbol,
            file,
            sha256,
            size,
            loaded: OnceLock::new(),
        }
    }

    /// Whether [`Self::load`] can succeed without a download.
    pub fn is_installed(&self) -> bool {
        self.loaded.get().is_some() || module_path(self.file).is_some()
    }

    /// The grammar's language, loading the module on first use.
    ///
    /// # Errors
    /// [`GrammarError::NotInstalled`] when no grammar directory is set or the
    /// module is not in it; [`GrammarError::Invalid`] when the module cannot be
    /// read or loaded.
    pub fn load(&self) -> Result<Language, GrammarError> {
        if let Some(language) = self.loaded.get() {
            return Ok(language.clone());
        }
        // This build links no wasm runtime: a module on disk cannot be loaded.
        match module_path(self.file) {
            Some(_) => Err(GrammarError::Invalid {
                language: self.language,
                reason: "wasm grammars are not supported in this build".to_owned(),
            }),
            None => Err(GrammarError::NotInstalled {
                language: self.language,
            }),
        }
    }

    /// Uses `language` for this grammar instead of a downloaded module; tests
    /// register the natively compiled grammar crates this way. Returns `false`
    /// when the grammar was already loaded.
    pub fn register(&self, language: Language) -> bool {
        self.loaded.set(language).is_ok()
    }
}

/// Parses `source` as `language`, on this thread's wasm-capable parser when
/// `language` is a wasm grammar.
///
/// # Errors
/// When the parser rejects `language` (incompatible ABI) or the thread's wasm
/// store cannot be created.
pub fn parse(language: &Language, source: &str) -> Result<Option<Tree>> {
    let mut parser = Parser::new();
    parser.set_language(language)?;
    Ok(parser.parse(source, None))
}
