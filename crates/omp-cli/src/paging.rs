//! `get_messages_page`: omp `rpc-messages.ts` (opaque base64url cursor over a
//! stable message snapshot, byte-budgeted pages).

use serde_json::{json, Value};

const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 256;
const MAX_PAGE_BYTES: usize = 768 * 1024;
const MAX_CURSOR_CHARS: usize = 2048;

pub const STALE_ERROR: &str = "RPC message cursor is stale";

/// A paging failure: message plus the machine-readable `code`, when any.
#[derive(Debug, PartialEq, Eq)]
pub struct PageError {
    pub message: String,
    pub code: Option<&'static str>,
}

impl PageError {
    fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
        }
    }
}

/// Identity of the message list a cursor was minted for.
pub struct Snapshot<'a> {
    pub session_id: &'a str,
    pub leaf_id: Option<&'a str>,
    pub message_count: usize,
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64url_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..=chunk.len() {
            out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

fn b64url_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in text.bytes() {
        let v = ALPHABET.iter().position(|a| *a == c)?;
        acc = (acc << 6) | u32::try_from(v).ok()?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(u8::try_from((acc >> bits) & 0xff).ok()?);
        }
    }
    // Canonical form only: leftover bits must be zero and the length valid.
    (text.len() % 4 != 1 && acc & ((1 << bits) - 1) == 0).then_some(out)
}

fn encode_cursor(s: &Snapshot<'_>, offset: usize) -> String {
    let payload = json!({
        "version": 1,
        "sessionId": s.session_id,
        "leafId": s.leaf_id,
        "messageCount": s.message_count,
        "offset": offset,
    });
    b64url_encode(payload.to_string().as_bytes())
}

fn decode_cursor(cursor: &str) -> Result<(String, Option<String>, usize, usize), PageError> {
    let invalid = || PageError::plain("Invalid RPC message cursor");
    if cursor.is_empty() || cursor.len() > MAX_CURSOR_CHARS {
        return Err(invalid());
    }
    let bytes = b64url_decode(cursor).ok_or_else(invalid)?;
    if b64url_encode(&bytes) != cursor {
        return Err(invalid());
    }
    let v: Value = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    let o = v.as_object().ok_or_else(invalid)?;
    let session_id = o
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && s.len() <= 256);
    let leaf = match o.get("leafId") {
        Some(Value::Null) => Some(None),
        Some(Value::String(s)) if !s.is_empty() && s.len() <= 256 => Some(Some(s.clone())),
        _ => None,
    };
    let count = o
        .get("messageCount")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok());
    let offset = o
        .get("offset")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok());
    match (
        o.get("version").and_then(Value::as_u64),
        session_id,
        leaf,
        count,
        offset,
    ) {
        (Some(1), Some(sid), Some(leaf), Some(count), Some(offset)) if offset <= count => {
            Ok((sid.to_string(), leaf, count, offset))
        }
        _ => Err(invalid()),
    }
}

/// Page one stable snapshot of `messages` without crossing the frame budget.
///
/// # Errors
/// Invalid limit or cursor, or a cursor minted for another snapshot.
pub fn page(
    messages: &[Value],
    snapshot: &Snapshot<'_>,
    cursor: Option<&str>,
    limit: Option<usize>,
) -> Result<Value, PageError> {
    if snapshot.message_count != messages.len() {
        return Err(PageError::plain(
            "RPC message snapshot does not match current messages",
        ));
    }
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if !(1..=MAX_LIMIT).contains(&limit) {
        return Err(PageError::plain(format!(
            "RPC message page limit must be between 1 and {MAX_LIMIT}"
        )));
    }
    let mut offset = 0;
    if let Some(c) = cursor {
        let (sid, leaf, count, off) = decode_cursor(c)?;
        if sid != snapshot.session_id
            || leaf.as_deref() != snapshot.leaf_id
            || count != snapshot.message_count
        {
            return Err(PageError {
                message: STALE_ERROR.into(),
                code: Some("stale_cursor"),
            });
        }
        offset = off;
    }
    let mut page: Vec<Value> = Vec::new();
    let mut page_bytes = 2usize;
    while offset + page.len() < messages.len() && page.len() < limit {
        let message = &messages[offset + page.len()];
        let bytes = message.to_string().len() + usize::from(!page.is_empty());
        if !page.is_empty() && page_bytes + bytes > MAX_PAGE_BYTES {
            break;
        }
        page.push(message.clone());
        page_bytes += bytes;
    }
    let next = offset + page.len();
    let mut out = json!({"messages": page, "totalMessages": messages.len()});
    if next < messages.len() {
        out["nextCursor"] = Value::String(encode_cursor(snapshot, next));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| json!({"role": "user", "content": format!("m{i}")}))
            .collect()
    }

    #[test]
    fn pages_through_all_messages_with_cursor() {
        let m = msgs(5);
        let snap = Snapshot {
            session_id: "s1",
            leaf_id: Some("leaf"),
            message_count: 5,
        };
        let first = page(&m, &snap, None, Some(2)).unwrap();
        assert_eq!(first["messages"].as_array().unwrap().len(), 2);
        assert_eq!(first["totalMessages"], 5);
        let cursor = first["nextCursor"].as_str().unwrap().to_string();
        let second = page(&m, &snap, Some(&cursor), Some(2)).unwrap();
        assert_eq!(second["messages"][0]["content"], "m2");
        let cursor = second["nextCursor"].as_str().unwrap().to_string();
        let last = page(&m, &snap, Some(&cursor), Some(2)).unwrap();
        assert_eq!(last["messages"].as_array().unwrap().len(), 1);
        assert!(last.get("nextCursor").is_none());
    }

    #[test]
    fn stale_and_invalid_cursors_are_rejected() {
        let m = msgs(3);
        let snap = Snapshot {
            session_id: "s1",
            leaf_id: None,
            message_count: 3,
        };
        let cursor = page(&m, &snap, None, Some(1)).unwrap()["nextCursor"]
            .as_str()
            .unwrap()
            .to_string();
        let other = Snapshot {
            session_id: "s2",
            leaf_id: None,
            message_count: 3,
        };
        let err = page(&m, &other, Some(&cursor), None).unwrap_err();
        assert_eq!(
            (err.message.as_str(), err.code),
            (STALE_ERROR, Some("stale_cursor"))
        );
        assert_eq!(
            page(&m, &snap, Some("!!"), None).unwrap_err().message,
            "Invalid RPC message cursor"
        );
        assert_eq!(
            page(&m, &snap, None, Some(0)).unwrap_err().message,
            "RPC message page limit must be between 1 and 256"
        );
    }

    #[test]
    fn base64url_round_trips() {
        for s in ["", "a", "ab", "abc", "abcd", "{\"version\":1}"] {
            assert_eq!(
                b64url_decode(&b64url_encode(s.as_bytes())).unwrap(),
                s.as_bytes()
            );
        }
    }
}
