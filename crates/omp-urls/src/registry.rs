//! Process-global scheme registry (TS `internal-urls/router.ts`).

use std::path::PathBuf;

use crate::parse::{extract_uri_scheme, parse_internal_url, InternalUrl};
use crate::selectors::{split_selectors, split_url_scheme};
use crate::spec::{InternalResource, SchemeSpec, WriteVia};
use crate::UrlCtx;
use crate::UrlProvider;

/// A routed read/search target.
#[derive(Debug)]
pub enum UrlTarget {
    /// A file-backed scheme whose URL located a local path; read through the filesystem pipeline.
    File {
        url: InternalUrl,
        spec: SchemeSpec,
        path: PathBuf,
        sel: Option<String>,
    },
    /// Anything else; resolve through [`Registry::resolve`].
    Resource {
        url: InternalUrl,
        spec: SchemeSpec,
        sel: Option<String>,
    },
}

struct Registered {
    scheme: &'static str,
    provider: Box<dyn UrlProvider>,
}

/// Scheme registry; tools route internal URLs through its spec-driven API.
pub struct Registry {
    handlers: Vec<Registered>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

/// Error message for an unknown scheme (TS router `#route`).
fn unknown_protocol(scheme: &str, handlers: &[Registered]) -> anyhow::Error {
    let available = handlers
        .iter()
        .map(|r| format!("{}://", r.scheme))
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::anyhow!(
        "Unknown protocol: {scheme}://\nSupported: {}",
        if available.is_empty() {
            "none"
        } else {
            &available
        }
    )
}

impl Registry {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Install (or replace) the provider for its scheme, checking write-policy
    /// consistency (TS `router.register`).
    ///
    /// # Errors
    /// The provider's write policy disagrees with its implemented hooks.
    pub fn register(&mut self, provider: impl UrlProvider + 'static) -> anyhow::Result<()> {
        let spec = provider.spec().clone();
        match spec.write.as_ref().map(|w| w.via) {
            Some(WriteVia::Handler) if !provider.impl_write() => {
                anyhow::bail!(
                    "{}:// spec.write.via is \"handler\" but the provider lacks write()",
                    provider.scheme()
                );
            }
            None if provider.impl_write() => {
                anyhow::bail!(
                    "{}:// provider has write() but spec.write is absent",
                    provider.scheme()
                );
            }
            _ => {}
        }
        if spec.write.as_ref().is_some_and(|w| w.via == WriteVia::File) {
            anyhow::ensure!(
                matches!(spec.backing, crate::spec::SchemeBacking::File),
                "{}:// spec.write.via \"file\" requires backing \"file\"",
                provider.scheme()
            );
            anyhow::ensure!(
                provider.impl_locate(),
                "{}:// spec.write.via \"file\" requires a locate hook",
                provider.scheme()
            );
            anyhow::ensure!(
                !spec.immutable,
                "{}:// spec.write.via \"file\" contradicts spec.immutable",
                provider.scheme()
            );
        }
        let scheme = provider.scheme();
        self.handlers.retain(|r| r.scheme != scheme);
        self.handlers.push(Registered {
            scheme,
            provider: Box::new(provider),
        });
        Ok(())
    }

    /// Registered provider for `scheme` (case-insensitive).
    pub fn get(&self, scheme: &str) -> Option<&dyn UrlProvider> {
        let lower = scheme.to_lowercase();
        self.handlers
            .iter()
            .find(|r| r.scheme == lower)
            .map(|r| r.provider.as_ref())
    }

    /// Declared spec of a registered scheme (case-insensitive).
    pub fn spec(&self, scheme: &str) -> Option<&SchemeSpec> {
        self.get(scheme).map(UrlProvider::spec)
    }

    /// Schemes and specs in registration order.
    pub fn schemes(&self) -> impl Iterator<Item = (&str, &SchemeSpec)> {
        self.handlers.iter().map(|r| (r.scheme, r.provider.spec()))
    }

    /// Prompt documentation lines in registration order.
    pub fn describe(&self) -> Vec<String> {
        self.handlers
            .iter()
            .filter_map(|r| r.provider.prompt_doc())
            .collect()
    }

    /// Rewrite a single-slash alias (`local:/x`) or a filesystem path with a
    /// URL glued on (`/home/me/repo/local://x`) to `scheme://x` (TS `normalize`).
    pub fn normalize(&self, input: &str) -> String {
        // `/home/me/repo/local://x.md`: a filesystem path in front of a URL.
        // The first eligible scheme wins.
        let is_url_prefix = {
            let b = input.as_bytes();
            b.len() > 2
                && b[0].is_ascii_alphabetic()
                && !(b[1] == b':' && matches!(b[2], b'/' | b'\\'))
                && input[1..].find(':').is_some_and(|i| {
                    input[1..1 + i]
                        .chars()
                        .all(|c: char| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
                })
        };
        if !is_url_prefix {
            let bytes = input.as_bytes();
            for i in 0..bytes.len().saturating_sub(3) {
                if (i == 0 || matches!(bytes[i - 1], b'/' | b'\\'))
                    && bytes[i].is_ascii_alphabetic()
                {
                    let mut j = i + 1;
                    while j < bytes.len()
                        && (bytes[j].is_ascii_alphanumeric()
                            || matches!(bytes[j], b'+' | b'-' | b'.'))
                    {
                        j += 1;
                    }
                    if input[j..].starts_with("://")
                        && j + 3 < bytes.len()
                        && bytes[j + 3] != b'/'
                        && self
                            .spec(&input[i..j])
                            .is_some_and(|s| s.single_slash_alias)
                    {
                        return input[i..].to_string();
                    }
                }
            }
        }
        // `local:/x` single-slash alias.
        if let Some((scheme, rest)) = input.split_once(":/") {
            if !rest.starts_with('/')
                && scheme.chars().enumerate().all(|(i, c)| {
                    c.is_ascii_alphabetic() && i == 0
                        || (c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')) && i > 0
                })
                && self.spec(scheme).is_some_and(|s| s.single_slash_alias)
            {
                return format!("{scheme}://{rest}");
            }
        }
        input.to_string()
    }

    /// Whether `input` (after [`Self::normalize`]) is a hierarchical `scheme://`
    /// URL of a registered scheme.
    pub fn can_handle(&self, input: &str) -> bool {
        self.registered(input).is_some()
    }

    /// Normalize, then peel a trailing read selector. Non-URLs pass through.
    pub fn split(&self, input: &str) -> (String, Option<String>) {
        let normalized = self.normalize(input);
        split_selectors(
            &normalized,
            |scheme| self.spec(scheme).map(|s| s.selectors),
            |scheme| self.spec(scheme).is_some_and(|s| s.port_authority),
        )
    }

    /// Route a read/search target: file-backed located paths read through the
    /// filesystem pipeline, everything else resolves as a resource.
    ///
    /// # Errors
    /// Provider locate/parse failures.
    pub fn target(&self, input: &str, ctx: &UrlCtx) -> anyhow::Result<Option<UrlTarget>> {
        let (bare, sel) = self.split(input);
        let Some(scheme) = extract_uri_scheme(&bare) else {
            return Ok(None);
        };
        let Some(provider) = self.get(&scheme) else {
            return Ok(None);
        };
        if !self.can_handle(&bare) {
            return Ok(None);
        }
        let url = parse_internal_url(&bare)?;
        let spec = provider.spec().clone();
        if matches!(spec.backing, crate::spec::SchemeBacking::File) {
            if let Some(path) = provider.locate(&url, ctx) {
                return Ok(Some(UrlTarget::File {
                    url,
                    spec,
                    path,
                    sel,
                }));
            }
        }
        Ok(Some(UrlTarget::Resource { url, spec, sel }))
    }

    /// Resolve an internal URL through its registered provider, applying the
    /// scheme-default `immutable`.
    ///
    /// # Errors
    /// Unknown protocol or provider resolve failure.
    pub fn resolve(&self, input: &str, ctx: &UrlCtx) -> anyhow::Result<InternalResource> {
        let normalized = self.normalize(input);
        let url = parse_internal_url(&normalized).map_err(|_| {
            unknown_protocol(
                extract_uri_scheme(&normalized).as_deref().unwrap_or(""),
                &self.handlers,
            )
        })?;
        let Some(provider) = self.get(&url.scheme) else {
            anyhow::bail!(unknown_protocol(&url.scheme, &self.handlers));
        };
        provider
            .read(&url, ctx)
            .map(|r| r.with_scheme_default(provider.spec()))
    }

    /// Write an internal URL through its registered provider.
    ///
    /// # Errors
    /// Unknown protocol, read-only scheme, or provider write failure.
    pub fn write(&self, input: &str, content: &str, ctx: &UrlCtx) -> anyhow::Result<()> {
        let normalized = self.normalize(input);
        let url = parse_internal_url(&normalized).map_err(|_| {
            unknown_protocol(
                extract_uri_scheme(&normalized).as_deref().unwrap_or(""),
                &self.handlers,
            )
        })?;
        let Some(provider) = self.get(&url.scheme) else {
            anyhow::bail!(unknown_protocol(&url.scheme, &self.handlers));
        };
        let Some(policy) = provider.spec().write.clone() else {
            anyhow::bail!(
                "{}:// URLs are read-only for write; use the protocol-specific tool for mutations.",
                url.scheme
            );
        };
        match policy.via {
            WriteVia::Handler => provider.write(&url, content, ctx),
            WriteVia::File => {
                let Some(path) = provider.locate_create(&url, ctx) else {
                    anyhow::bail!(
                        "Cannot write {}: no local file backs {}",
                        url.raw_href,
                        url.raw_href
                    );
                };
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, content)?;
                Ok(())
            }
        }
    }
}

impl Registry {
    /// Normalized hierarchical URL + provider of a registered scheme.
    fn registered(&self, input: &str) -> Option<(String, &dyn UrlProvider)> {
        let normalized = self.normalize(input);
        let (scheme, _rest) = split_url_scheme(&normalized)?;
        let provider = self.get(&scheme)?;
        Some((normalized, provider))
    }
}
