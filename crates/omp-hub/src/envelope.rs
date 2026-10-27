//! The one envelope type for every channel (`SPEC.md`), plus uuid-v7
//! generation and the pane-facing 4-line render.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default reply deadline: 30 minutes (`SPEC.md` item 3).
pub const DEFAULT_REPLY_DEADLINE_MS: u64 = 30 * 60 * 1000;
/// Envelope kind for replies (`agent.answer`).
pub const KIND_ANSWER: &str = "agent.answer";
/// Synthetic sender used by the hub itself for control/delivery notices.
pub const SENDER_HUB: &str = "hub";
/// Tool hint embedded in the pane wake line for notes.
pub const PANE_WAKE_TOOL: &str = "omt hub inbox";

/// Priority: 0 urgent (cancel/steer/question), 1 normal, 2 bulk (progress).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Priority(pub u8);

impl Priority {
    pub const URGENT: Self = Self(0);
    pub const NORMAL: Self = Self(1);
    pub const BULK: Self = Self(2);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Well-known envelope kinds (`SPEC.md`). Stored as the literal string so
/// unknown kinds from newer senders round-trip instead of failing to parse.
pub struct Kind;

impl Kind {
    pub const TASK_ASSIGN: &str = "task.assign";
    pub const TASK_ACCEPTED: &str = "task.accepted";
    pub const TASK_PROGRESS: &str = "task.progress";
    pub const TASK_COMPLETED: &str = "task.completed";
    pub const TASK_FAILED: &str = "task.failed";
    pub const TASK_CANCEL: &str = "task.cancel";
    pub const AGENT_STEER: &str = "agent.steer";
    pub const AGENT_MESSAGE: &str = "agent.message";
    pub const AGENT_QUESTION: &str = "agent.question";
    pub const ANSWER: &str = KIND_ANSWER;
    pub const RESOURCE_CLAIM: &str = "resource.claim";
    pub const RESOURCE_RELEASE: &str = "resource.release";
    pub const DELIVERY_ACK: &str = "delivery.ack";
}

/// Hub-side envelope state: `sent -> delivered -> read -> answered`,
/// or terminal `expired` / `failed` / `cancelled`.
pub struct State;

impl State {
    pub const SENT: &str = "sent";
    pub const DELIVERED: &str = "delivered";
    pub const READ: &str = "read";
    pub const ANSWERED: &str = "answered";
    pub const EXPIRED: &str = "expired";
    pub const FAILED: &str = "failed";
    pub const CANCELLED: &str = "cancelled";

    /// States in which the envelope still owes the recipient a look.
    pub const UNREAD: [&str; 2] = [Self::SENT, Self::DELIVERED];
}

/// Large bodies go by reference, never inline (`SPEC.md`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadRef {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

/// One type for every channel (`SPEC.md`). `seq`, `state`, `state_log` and
/// `reply_message_id` are hub-assigned and absent on the wire from senders.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    /// uuid v7, sender-generated; the dedup key.
    pub message_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Thread id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    /// Id of the message this one answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causation_id: Option<String>,
    pub sender: String,
    pub recipient: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub priority: Priority,
    /// Free-text expectation (legacy `[EXPECTED]`); reply-expected when set,
    /// non-empty and not `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// JSON Schema a reply is validated against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_schema: Option<Value>,
    /// Unix ms; defaulted to +30 min for reply-expected envelopes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<PayloadRef>,
    // Hub-assigned:
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// `state -> unix ms` transition log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_log: Option<serde_json::Map<String, Value>>,
    /// Message that answered this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_message_id: Option<String>,
}

fn default_kind() -> String {
    Kind::AGENT_MESSAGE.to_string()
}

impl Envelope {
    /// Reply-expected: `expected` non-empty/`none`, or a `reply_schema` set
    /// (`SPEC.md` item 3).
    #[must_use]
    pub fn is_reply_expected(&self) -> bool {
        self.reply_schema.is_some()
            || self
                .expected
                .as_deref()
                .is_some_and(|e| !e.is_empty() && e != "none")
    }

    /// Deadline: explicit, or now + [`DEFAULT_REPLY_DEADLINE_MS`] when
    /// reply-expected.
    #[must_use]
    pub fn effective_deadline(&self, now: u64) -> Option<u64> {
        if let Some(d) = self.deadline {
            return Some(d);
        }
        self.is_reply_expected()
            .then_some(now + DEFAULT_REPLY_DEADLINE_MS)
    }
}

/// Build the envelope answering `original`: `causation_id` points at the
/// original message, `correlation_id` continues the thread.
#[must_use]
pub fn answer_envelope(
    original: &Envelope,
    sender: &str,
    message_id: String,
    payload: String,
) -> Envelope {
    Envelope {
        message_id,
        run_id: original.run_id.clone(),
        parent_id: original.parent_id.clone(),
        correlation_id: Some(
            original
                .correlation_id
                .clone()
                .unwrap_or_else(|| original.message_id.clone()),
        ),
        causation_id: Some(original.message_id.clone()),
        sender: sender.to_string(),
        recipient: original.sender.clone(),
        kind: KIND_ANSWER.to_string(),
        priority: Priority::URGENT,
        payload: Some(payload),
        ..Envelope::default()
    }
}

/// Strip the `agent://` prefix; pane ids keep their `:` (`w4:p8T`).
#[must_use]
pub fn normalize_address(a: &str) -> String {
    a.strip_prefix("agent://").unwrap_or(a).to_string()
}

/// Is this address a herdr pane id (`w4:p8T` shape)?
#[must_use]
pub fn looks_like_pane(a: &str) -> bool {
    let Some((ws, pane)) = a.split_once(':') else {
        return false;
    };
    let ws_ok = ws.len() > 1 && ws.starts_with('w') && ws[1..].chars().all(|c| c.is_ascii_digit());
    let pane_ok = pane.len() > 1
        && pane.starts_with('p')
        && pane[1..].chars().all(|c| c.is_ascii_alphanumeric());
    ws_ok && pane_ok
}

/// The existing 4-line head (`~/bin/fleet-envelope`) + body, so current
/// agents understand pane deliveries unchanged.
#[must_use]
pub fn pane_text(env: &Envelope) -> String {
    let pointer = env.payload_ref.as_ref().map_or("none", |r| r.path.as_str());
    let expected = env.expected.as_deref().unwrap_or("none");
    let mut text = format!(
        "[FROM] {}\n[ID] {}\n[POINTER] {}\n[EXPECTED] {}",
        env.sender, env.message_id, pointer, expected
    );
    if let Some(p) = &env.payload {
        text.push('\n');
        text.push_str(p);
    }
    text
}

/// New uuid v7 string (RFC 9562): 48-bit unix-ms + 74 random bits, no deps.
///
/// # Panics
/// Never; falls back to a constant random source if `/dev/urandom` is gone.
#[must_use]
pub fn new_message_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    let mut b = [0u8; 16];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut b))
        .is_err()
    {
        // Last-resort entropy: address-space layout + time nanos.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let x =
            u64::try_from(nanos).unwrap_or(0) ^ (u64::from(std::process::id()) << 32) ^ (ms << 16);
        b = x.to_le_bytes().repeat(2).try_into().unwrap_or(b);
    }
    b[0..6].copy_from_slice(&ms.to_be_bytes()[2..8]);
    b[6] = (b[6] & 0x0f) | 0x70; // version 7
    b[8] = (b[8] & 0x3f) | 0x80; // variant 10
    let hex = |v: u8| -> [u8; 2] {
        const H: &[u8; 16] = b"0123456789abcdef";
        [H[(v >> 4) as usize], H[(v & 0xf) as usize]]
    };
    let mut out = String::with_capacity(36);
    for (i, byte) in b.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            out.push('-');
        }
        let [hi, lo] = hex(*byte);
        out.push(char::from(hi));
        out.push(char::from(lo));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v7_shape_and_monotonic_ms() {
        let a = new_message_id();
        let b = new_message_id();
        assert_eq!(a.len(), 36);
        assert_eq!(a.chars().filter(|c| *c == '-').count(), 4);
        assert_eq!(&a[14..15], "7");
        assert!(matches!(a.chars().nth(19), Some('8' | '9' | 'a' | 'b')));
        assert_ne!(a, b);
        // Same millisecond prefix (same ms almost always).
        let pa = u128::from_str_radix(&a[0..13].replace('-', ""), 16).unwrap();
        let pb = u128::from_str_radix(&b[0..13].replace('-', ""), 16).unwrap();
        let diff = pa.abs_diff(pb);
        assert!(
            diff < 10_000,
            "v7 timestamps should be within seconds: {diff}"
        );
    }

    #[test]
    fn normalize_and_pane_shape() {
        assert_eq!(normalize_address("agent://w4:p8T"), "w4:p8T");
        assert_eq!(normalize_address("w4:p8T"), "w4:p8T");
        assert!(looks_like_pane("w4:p8T"));
        assert!(looks_like_pane("w12:p99"));
        assert!(!looks_like_pane("agent-Alpha"));
        assert!(!looks_like_pane("w4:"));
        assert!(!looks_like_pane("hub"));
    }

    #[test]
    fn reply_expected_rules() {
        let mut e = Envelope {
            message_id: "x".into(),
            sender: "a".into(),
            recipient: "b".into(),
            ..Envelope::default()
        };
        assert!(!e.is_reply_expected());
        e.expected = Some("none".into());
        assert!(!e.is_reply_expected());
        e.expected = Some(String::new());
        assert!(!e.is_reply_expected());
        e.expected = Some("ack".into());
        assert!(e.is_reply_expected());
        e.expected = None;
        e.reply_schema = Some(serde_json::json!({"type":"object"}));
        assert!(e.is_reply_expected());
        let now = 1_000_000;
        assert_eq!(
            e.effective_deadline(now),
            Some(now + DEFAULT_REPLY_DEADLINE_MS)
        );
        e.deadline = Some(42);
        assert_eq!(e.effective_deadline(now), Some(42));
    }

    #[test]
    fn pane_text_uses_four_line_head() {
        let e = Envelope {
            message_id: "m1".into(),
            sender: "w4:p8T".into(),
            recipient: "w4:p8W".into(),
            expected: Some("ack".into()),
            payload: Some("body line".into()),
            ..Envelope::default()
        };
        let text = pane_text(&e);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "[FROM] w4:p8T");
        assert_eq!(lines[1], "[ID] m1");
        assert_eq!(lines[2], "[POINTER] none");
        assert_eq!(lines[3], "[EXPECTED] ack");
        assert_eq!(lines[4], "body line");
    }
}
