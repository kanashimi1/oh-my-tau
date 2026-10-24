//! Wire-only dedup of repeated tool results (cost lever `COST-CACHE-HYGIENE-20261014`).
//!
//! Agent loops re-carry every past tool result through every subsequent
//! request: measured on the bench mock, ~65% of billed input bytes were
//! occurrences of content the context already contained (repeated reads,
//! repeated grep/bash outputs). This pass collapses exact duplicates — same
//! full text content — into a one-line stub that points at the first
//! occurrence, so the model can still resolve the reference.
//!
//! Properties that make this cache- and resume-safe:
//! - Wire-only: the session file keeps full fidelity; this runs inside
//!   `convert_to_llm`, so only the API request is affected.
//! - Deterministic: a pure function of the message list, applied in order —
//!   the first occurrence always stays full, every later identical result
//!   becomes a stub, so prefixes stay append-only and byte-stable.
//! - Conservative: results with images, errors, or small bodies (<
//!   [`MIN_DEDUP_BYTES`]) are never touched, and only whole-result exact
//!   matches dedup (no fuzzy heuristics the model cannot verify).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pa_agent::types::{AgentMessage, Message, TextContent, ToolResultContent};

/// Results smaller than this keep their body verbatim: the stub plus the
/// surrounding notice would cost about as much as the content itself.
const MIN_DEDUP_BYTES: usize = 256;

fn content_hash(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

fn text_of(content: &[ToolResultContent]) -> Option<String> {
    if content.is_empty()
        || content
            .iter()
            .any(|c| matches!(c, ToolResultContent::Image(_)))
    {
        return None;
    }
    let text: String = content
        .iter()
        .filter_map(|c| match c {
            ToolResultContent::Text(t) => Some(t.text.as_str()),
            ToolResultContent::Image(_) => None,
        })
        .collect();
    Some(text)
}

fn stub_for(orig_call_id: &str, bytes: usize) -> ToolResultContent {
    ToolResultContent::Text(TextContent {
        text: format!(
            "<duplicate: identical to the tool result of call {orig_call_id} ({bytes} bytes) — full content shown above>"
        ),
        text_signature: None,
    })
}

/// Collapse exact-duplicate tool results into pointer stubs.
///
/// Pure and order-deterministic: rebuilding from the same history yields the
/// same stubs at the same positions, so provider prefix caches keep hitting.
#[must_use]
pub fn dedup_tool_results(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let mut first_seen: std::collections::HashMap<u64, (String, usize)> =
        std::collections::HashMap::new();

    messages
        .into_iter()
        .map(|m| {
            let AgentMessage::Standard(Message::ToolResult(tr)) = m else {
                return m;
            };
            let Some(text) = text_of(&tr.content) else {
                return AgentMessage::Standard(Message::ToolResult(tr));
            };
            // Conservative, per the module docs: error results carry the
            // diagnostic the model needs to recover; never collapse them.
            if tr.is_error {
                return AgentMessage::Standard(Message::ToolResult(tr));
            }
            if text.len() < MIN_DEDUP_BYTES {
                return AgentMessage::Standard(Message::ToolResult(tr));
            }
            let hash = content_hash(&text);
            if let Some((orig_call_id, bytes)) = first_seen.get(&hash) {
                let mut tr = tr;
                tr.content = vec![stub_for(orig_call_id, *bytes)];
                return AgentMessage::Standard(Message::ToolResult(tr));
            }
            first_seen.insert(hash, (tr.tool_call_id.clone(), text.len()));
            AgentMessage::Standard(Message::ToolResult(tr))
        })
        .collect()
}

/// Whether the wire pass runs this process: explicit opt-in, or default-on
/// when both stdin and stderr are not TTYs (rpc/print/CI), overridable with
/// `OMP_RS_DEDUP_TOOL_RESULTS=0`/`=false`.
#[must_use]
pub fn dedup_active() -> bool {
    match std::env::var("OMP_RS_DEDUP_TOOL_RESULTS") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => {
            use std::io::IsTerminal;
            !std::io::stdin().is_terminal() && !std::io::stderr().is_terminal()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::types::{
        AgentMessage, Message, TextContent, ToolResultContent, ToolResultMessage,
    };

    fn tool_result(call_id: &str, text: &str) -> AgentMessage {
        AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
            tool_call_id: call_id.to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultContent::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        }))
    }

    fn body_of(m: &AgentMessage) -> String {
        let AgentMessage::Standard(Message::ToolResult(tr)) = m else {
            panic!("expected tool result");
        };
        text_of(&tr.content).unwrap()
    }

    #[test]
    fn error_results_never_stubbed() {
        let big = "x".repeat(1000);
        let mut err = tool_result("call_e", &big);
        let AgentMessage::Standard(Message::ToolResult(tr)) = &mut err else {
            panic!("expected tool result");
        };
        tr.is_error = true;
        let msgs = vec![
            err,
            tool_result("call_f", &big),
            tool_result("call_g", &big),
        ];
        let out = dedup_tool_results(msgs);
        assert_eq!(body_of(&out[0]), big, "error result keeps its body");
        assert_eq!(
            body_of(&out[1]),
            big,
            "first non-error occurrence stays full"
        );
        assert!(
            body_of(&out[2]).contains("call_f"),
            "later duplicate stubs to the non-error first"
        );
        assert!(body_of(&out[2]).contains("1000 bytes"));
    }

    #[test]
    fn second_identical_result_becomes_stub() {
        let big = "x".repeat(1000);
        let msgs = vec![
            tool_result("call_a", &big),
            tool_result("call_b", &big),
            tool_result("call_c", &big),
        ];
        let out = dedup_tool_results(msgs);
        assert_eq!(body_of(&out[0]), big, "first occurrence stays full");
        assert!(body_of(&out[1]).contains("call_a"));
        assert!(body_of(&out[2]).contains("call_a"));
        assert!(body_of(&out[1]).contains("1000 bytes"));
    }

    #[test]
    fn unique_results_untouched() {
        let msgs = vec![
            tool_result("call_a", &"a".repeat(1000)),
            tool_result("call_b", &"b".repeat(1000)),
        ];
        let out = dedup_tool_results(msgs);
        assert_eq!(body_of(&out[0]), "a".repeat(1000));
        assert_eq!(body_of(&out[1]), "b".repeat(1000));
    }

    #[test]
    fn small_results_never_stubbed() {
        let small = "short output";
        let msgs = vec![tool_result("call_a", small), tool_result("call_b", small)];
        let out = dedup_tool_results(msgs);
        assert_eq!(body_of(&out[1]), small);
    }

    #[test]
    fn deterministic_across_rebuilds() {
        let big = "y".repeat(500);
        let build = || {
            dedup_tool_results(vec![
                tool_result("call_a", &big),
                tool_result("call_b", "other-other-other"),
                tool_result("call_c", &big),
            ])
        };
        let a = build();
        let b = build();
        assert_eq!(a, b, "same history -> same wire bytes (prefix-cache safe)");
    }

    #[test]
    fn wire_of_turn_n_is_a_prefix_of_turn_n_plus_one() {
        // Cache contract: appending a message may only append wire bytes —
        // never change what an earlier request already sent.
        let big = "z".repeat(600);
        let history = vec![
            tool_result("call_a", &big),
            tool_result("call_b", "small"),
            tool_result("call_c", &big),
            tool_result("call_d", &"w".repeat(600)),
            tool_result("call_e", &big),
        ];
        for n in 1..history.len() {
            let shorter = dedup_tool_results(history[..n].to_vec());
            let longer = dedup_tool_results(history[..=n].to_vec());
            assert_eq!(
                &longer[..n],
                &shorter[..],
                "turn {n}: earlier wire bytes changed"
            );
        }
    }
}
