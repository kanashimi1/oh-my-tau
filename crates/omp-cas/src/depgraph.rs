//! Dependency graphs over CAS objects and the transitive pack builder.
//!
//! A "dependency" of an object is another digest the object refers to. The
//! store does not parse content — callers describe edges via
//! [`DepGraph::declare`], and this module closes over them iteratively.
//!
//! Cycles are legal input (content graphs can be cyclic): traversal is
//! BFS with a visited set, so a cycle terminates instead of recursing
//! forever. No recursive descent anywhere in this module.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::digest::Digest;
use crate::error::Result;

/// Caller-declared dependency edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DepGraph {
    edges: BTreeMap<Digest, BTreeSet<Digest>>,
}

impl DepGraph {
    /// New empty graph.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
        }
    }

    /// Declare `from` depends on each `to`.
    pub fn declare(&mut self, from: Digest, to: impl IntoIterator<Item = Digest>) {
        let slot = self.edges.entry(from).or_default();
        slot.extend(to);
    }

    /// Roots plus everything transitively reachable, BFS order from each
    /// root in sorted order. Cycle-safe: visited-set, iterative queue.
    ///
    /// Capacity is bounded by [`crate::error::Error::Limit`] once the graph
    /// exceeds `max_nodes` — the bound is a contract, not a hope.
    ///
    /// # Errors
    /// [`crate::error::Error::Limit`] when the reachable set exceeds `max_nodes`.
    pub fn transitive(&self, roots: &[Digest], max_nodes: usize) -> Result<Vec<Digest>> {
        let mut seen: BTreeSet<Digest> = BTreeSet::new();
        let mut queue: VecDeque<Digest> = VecDeque::new();
        for root in roots {
            if !seen.contains(root) {
                seen.insert(*root);
                queue.push_back(*root);
            }
        }
        while let Some(node) = queue.pop_front() {
            if seen.len() > max_nodes {
                return Err(crate::error::Error::Limit("dependency graph exceeds max_nodes"));
            }
            if let Some(deps) = self.edges.get(&node) {
                for dep in deps {
                    if seen.insert(*dep) {
                        // Bound check happens when the node is dequeued, so
                        // the set can overshoot by the last frontier only.
                        queue.push_back(*dep);
                    }
                }
            }
        }
        Ok(seen.into_iter().collect())
    }

    /// Self-edge and mutual cycles remain valid reachability input.
    #[must_use]
    pub fn has_edge(&self, from: Digest, to: Digest) -> bool {
        self.edges.get(&from).is_some_and(|s| s.contains(&to))
    }
}
