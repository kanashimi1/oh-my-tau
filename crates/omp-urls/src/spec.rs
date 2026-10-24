//! Scheme declarations and resource payloads (TS `internal-urls/types.ts`).

/// How a scheme's resources exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeBacking {
    File,
    Virtual,
    Remote,
    Device,
}

/// Read-selector grammar supported after the URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeSelectors {
    /// `:N`, `:N-M`, `:N+M`, comma chains.
    Lines,
    /// No selectors.
    None,
    /// Selector text is part of the path.
    Opaque,
}

/// Mutation class of a writable scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteScope {
    Workspace,
    Sandbox,
    Coordination,
    Device,
}

/// How writes to a writable scheme are performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteVia {
    /// The provider's `write` handles it.
    Handler,
    /// The scheme is file-backed; the file pipeline writes it.
    File,
}

/// Write policy for a writable scheme.
#[derive(Debug, Clone)]
pub struct WritePolicy {
    pub via: WriteVia,
    pub scope: WriteScope,
}

/// Declared facts about a scheme.
#[derive(Debug, Clone)]
pub struct SchemeSpec {
    pub backing: SchemeBacking,
    pub selectors: SchemeSelectors,
    pub immutable: bool,
    /// Accept the single-slash alias `scheme:/x` and normalize to `scheme://x`.
    pub single_slash_alias: bool,
    /// Sync-locatable for renderers.
    pub linkable: bool,
    /// The authority is a port-bearing host, not a path segment.
    pub port_authority: bool,
    pub write: Option<WritePolicy>,
}

impl SchemeSpec {
    /// Read-only file-backed scheme.
    pub const FILE_READONLY: Self = Self {
        backing: SchemeBacking::File,
        selectors: SchemeSelectors::Lines,
        immutable: false,
        single_slash_alias: false,
        linkable: false,
        port_authority: false,
        write: None,
    };
}

/// Raw resource payload returned by providers.
#[derive(Debug, Clone)]
pub struct InternalResource {
    /// Text content (or a directory listing / synthesized note).
    pub content: String,
    /// The URL names a directory.
    pub is_directory: bool,
    /// Per-resource override of the scheme default.
    pub immutable: Option<bool>,
    /// Absolute local path when the resource is file-backed.
    pub path: Option<String>,
}

impl InternalResource {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_directory: false,
            immutable: None,
            path: None,
        }
    }

    pub fn directory(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_directory: true,
            immutable: None,
            path: None,
        }
    }

    /// Apply the scheme default when the provider did not set one (TS router.resolve).
    pub fn with_scheme_default(mut self, spec: &SchemeSpec) -> Self {
        self.immutable.get_or_insert(spec.immutable);
        self
    }
}

/// A single autocomplete candidate for `scheme://<query>`.
#[derive(Debug, Clone)]
pub struct UrlCompletion {
    pub label: String,
    pub detail: Option<String>,
}
