//! Port of the TS model-policy compat engine (`packages/catalog/src/compat/*`): the single
//! resolver that turns a model spec `(provider, id, api, baseUrl, …)` plus sparse overrides into
//! the complete resolved compat record, thinking metadata and request policy every request
//! builder reads. Rule data (`data/compat/rules.json`, `axes.json`) is generated from the TS tree
//! by `runs/omp-rust/compat-oracle/gen-data.ts`; behaviour is checked against TS by the oracle
//! test (`oracle.rs`).

pub mod cascade;
pub mod hosts;
pub mod model;
pub mod resolve;
pub mod revision;
pub mod taxonomy;
pub mod view;

#[cfg(test)]
mod oracle;

use std::sync::OnceLock;

use serde::Deserialize;

pub use model::{compat_overrides, materialize, model_from_catalog_row, resolve_for_model};
pub use resolve::{resolve_model_policy, ModelSpec, ResolveError, ResolvedModelPolicy};
pub use view::ModelView;

/// The compiled rule tree (the parts the resolver reads).
#[derive(Debug, Deserialize)]
pub struct Rules {
    pub taxonomy: taxonomy::Taxonomy,
    pub cascade: cascade::Cascade,
}

/// Axis applicability tables (TS `AXES` wire/request keys and `API_COMPAT_RECORDS`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Axes {
    pub wire: std::collections::HashMap<String, Vec<String>>,
    pub request: Vec<String>,
    pub api_records: std::collections::HashMap<String, Vec<String>>,
}

pub fn rules() -> &'static Rules {
    static RULES: OnceLock<Rules> = OnceLock::new();
    RULES.get_or_init(|| {
        serde_json::from_str(include_str!("../../data/compat/rules.json"))
            .expect("compat rules.json parses")
    })
}

pub fn axes() -> &'static Axes {
    static AXES: OnceLock<Axes> = OnceLock::new();
    AXES.get_or_init(|| {
        serde_json::from_str(include_str!("../../data/compat/axes.json"))
            .expect("compat axes.json parses")
    })
}
