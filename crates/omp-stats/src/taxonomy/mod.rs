//! Model-identity taxonomy: classification, suffix collapse, and revisions.
//!
//! Port of `catalog/src/compat/{taxonomy,revision}.ts`. Rule data is the
//! embedded `data/rules.json` (see [`rules`]).

mod classify;
mod collapse;
mod revision;
mod rules;

#[cfg(test)]
mod tests;

pub use classify::{classify_model, ClassifyOptions, ModelIdentity};
pub use collapse::{
    billing_variant_plain, collapse_variant_id, recovers_canonical_params, strip_effort_lane,
    strip_thinking_variant_suffix, CollapsedVariant,
};
pub use revision::{
    compare_revision, format_revision, parse_revision, parse_revision_prefix, Revision,
};
pub use rules::Effort;
