//! Collab live-session wire protocol primitives.
//!
//! Hub topology: the host is authoritative, guests never peer. All session
//! payloads travel sealed (see [`crate::crypto`]); the relay only sees the
//! plaintext envelope `[4B uint32 BE peerId][sealed payload]` plus TEXT JSON
//! control messages that carry no session data.
//!
//! Port of the wire half of
//! `vendor-src/oh-my-pi/packages/coding-agent/src/collab/protocol.ts` and the
//! browser mirror `vendor-src/oh-my-pi/packages/collab-web/src/lib/link.ts`
//! (byte-identical wire formats; TS line refs cite the coding-agent copy
//! unless marked `link.ts`). The byte-layout constants re-export the values
//! pinned by `vendor-src/oh-my-pi/packages/wire/src/index.ts:411-426`.

use rand::RngCore;

/// Default public relay; bare `<roomId>.<key>` links resolve against it
/// (wire/src/index.ts:426).
pub const DEFAULT_RELAY_URL: &str = "wss://my.omp.sh";

/// Plaintext envelope prefix: `[4B uint32 BE peerId][sealed payload]`
/// (wire/src/index.ts:411).
pub const ENVELOPE_HEADER_LENGTH: usize = 4;

/// Random room id size; rendered as base64url (wire/src/index.ts:413).
pub const ROOM_ID_BYTES: usize = 16;

/// AES-256-GCM room key; the seal key for every collab frame
/// (wire/src/index.ts:416).
pub const ROOM_KEY_BYTES: usize = 32;

/// Random write token appended to the room key in full links
/// (`base64url(key || token)`); view links carry the bare key. Possession
/// proves prompt/abort/agent-cmd capability to the host
/// (wire/src/index.ts:423).
pub const WRITE_TOKEN_BYTES: usize = 16;

/// Plaintext envelope, split at the [`ENVELOPE_HEADER_LENGTH`] boundary
/// (protocol.ts:164-168 — TS borrows a subarray; the port copies the payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Envelope {
    /// Host→relay: 0 broadcasts to all guests, N targets guest N.
    /// Guest→relay: always 0; the relay rewrites it to the sender's id.
    pub peer_id: u32,
    /// Sealed payload bytes (opaque here; opened by [`crate::crypto`]).
    pub payload: Vec<u8>,
}

/// Prefix `sealed` with the big-endian peer id (protocol.ts:157-162,
/// `setUint32(…, false)`).
#[must_use]
pub fn pack_envelope(peer_id: u32, sealed: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ENVELOPE_HEADER_LENGTH + sealed.len());
    out.extend_from_slice(&peer_id.to_be_bytes());
    out.extend_from_slice(sealed);
    out
}

/// Split an envelope; `None` below the 4-byte header floor (protocol.ts:164).
#[must_use]
pub fn unpack_envelope(data: &[u8]) -> Option<Envelope> {
    let header: [u8; ENVELOPE_HEADER_LENGTH] =
        data.get(..ENVELOPE_HEADER_LENGTH)?.try_into().ok()?;
    Some(Envelope {
        peer_id: u32::from_be_bytes(header),
        payload: data[ENVELOPE_HEADER_LENGTH..].to_vec(),
    })
}

/// Rewrite the peer id in place without copying the payload (protocol.ts:171).
///
/// # Panics
/// When `data` is shorter than [`ENVELOPE_HEADER_LENGTH`], mirroring the TS
/// `DataView.setUint32` `RangeError` on the same input.
pub fn rewrite_envelope_peer(data: &mut [u8], peer_id: u32) {
    data[..ENVELOPE_HEADER_LENGTH].copy_from_slice(&peer_id.to_be_bytes());
}

/// 16 random bytes as base64url — a 22-char id over the alphabet
/// `[A-Za-z0-9_-]` (protocol.ts:188-192 via `Buffer.toString("base64url")`,
/// link.ts:80-84 via the btoa mirror; both produce the canonical unpadded
/// alphabet).
#[must_use]
pub fn generate_room_id() -> String {
    let mut bytes = [0_u8; ROOM_ID_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    encode_base64_url(&bytes)
}

/// base64url (RFC 4648 §5, unpadded), canonical emit — TS
/// `Buffer.toString("base64url")` (protocol.ts:191) and the btoa mirror
/// (link.ts:33-37, `+→-`, `/→_`, strip `=`).
#[must_use]
pub fn encode_base64_url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(chunk.get(1).copied().unwrap_or(0));
        let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let word = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(word >> 18 & 0x3f) as usize] as char);
        out.push(ALPHABET[(word >> 12 & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(word >> 6 & 0x3f) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(word & 0x3f) as usize] as char);
        }
    }
    out
}

/// Inverse of [`encode_base64_url`] with the TS acceptance rules: the charset
/// gate first (`B64URL_RE = /^[A-Za-z0-9_-]+$/`, link.ts:26/40 — so the empty
/// string is rejected like the `+`-quantifier regex) and then a *forgiving*
/// decode, i.e. dirty trailing bits are tolerated rather than checked, as in
/// `atob` (link.ts:39-52) and Node's `Buffer.from(…, "base64url")`
/// (protocol.ts:332). The strict `base64` crate would reject non-canonical
/// trailing bits, so the decoder is table-driven here instead.
#[must_use]
pub fn decode_base64_url(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() {
        return None; // B64URL_RE requires one or more chars
    }
    if text.len() % 4 == 1 {
        return None; // forgiving-base64: impossible length
    }
    let mut out = Vec::with_capacity(text.len() * 6 / 8);
    let mut acc = 0_u32;
    let mut bits = 0_u32;
    for &byte in text.as_bytes() {
        let value = base64url_value(byte)?;
        acc = (acc << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            // Exactly the top byte of `acc`: `acc` holds `bits + 8` bits here,
            // so the shifted value always fits (truncation-free by design).
            #[allow(clippy::cast_possible_truncation)]
            out.push((acc >> bits) as u8);
        }
    }
    // Remaining `bits` (0/2/4) are discarded without a zero check — atob and
    // Node Buffer both tolerate non-canonical trailing bits.
    Some(out)
}

fn base64url_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TS oracle (bun, oracle-t1.mts): `packEnvelope(0xdeadbeef, ct)`.
    const CT: &[u8] = &[
        0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
        0xa5, 0x74, 0x65, 0x73, 0x74, 0x2d, 0x70, 0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64,
    ];

    #[test]
    fn pack_envelope_big_endian_golden() {
        let packed = pack_envelope(0xdead_beef, CT);
        let mut expected = vec![0xde, 0xad, 0xbe, 0xef];
        expected.extend_from_slice(CT);
        assert_eq!(packed, expected);
    }

    #[test]
    fn pack_envelope_peer_1_is_00000001() {
        // TS oracle: packEnvelope(1, []) → "00000001" (big-endian, not LE).
        assert_eq!(pack_envelope(1, &[]), vec![0, 0, 0, 1]);
    }

    #[test]
    fn unpack_envelope_round_trip() {
        // TS oracle: unpackEnvelope(packEnvelope(0xdeadbeef, ct)) → peerId
        // 3735928559, payload == ct.
        let env = unpack_envelope(&pack_envelope(0xdead_beef, CT)).expect("well-formed");
        assert_eq!(env.peer_id, 0xdead_beef);
        assert_eq!(env.payload, CT);
    }

    #[test]
    fn unpack_envelope_below_header_is_none() {
        // TS oracle: unpackEnvelope(3 bytes) → null.
        assert_eq!(unpack_envelope(&CT[..3]), None);
        assert_eq!(unpack_envelope(&[]), None);
    }

    #[test]
    fn unpack_envelope_header_only_yields_empty_payload() {
        let env = unpack_envelope(&[0, 0, 0, 7]).expect("well-formed");
        assert_eq!(env.peer_id, 7);
        assert!(env.payload.is_empty());
    }

    #[test]
    fn rewrite_envelope_peer_in_place_golden() {
        // TS oracle: rewriteEnvelopePeer(ct-copy, 7) → "00000007…" prefix,
        // payload untouched.
        let mut data = pack_envelope(0xdead_beef, CT);
        rewrite_envelope_peer(&mut data, 7);
        let mut expected = vec![0, 0, 0, 7];
        expected.extend_from_slice(CT);
        assert_eq!(data, expected);
    }

    #[test]
    fn base64url_encode_goldens() {
        // TS oracle: encodeBase64Url / Buffer.toString("base64url") on the
        // 32-byte key and the 48-byte key||token.
        let key: Vec<u8> = (0..=31_u8).collect();
        assert_eq!(
            encode_base64_url(&key),
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
        );
        let token: Vec<u8> = (0xf0..=0xff_u8).collect();
        let mut secret = key;
        secret.extend_from_slice(&token);
        assert_eq!(
            encode_base64_url(&secret),
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh_w8fLz9PX29_j5-vv8_f7_"
        );
        // TS oracle: encodeBase64Url([]) → "" (btoa of nothing).
        assert_eq!(encode_base64_url(&[]), "");
    }

    #[test]
    fn base64url_decode_round_trip_and_rejections() {
        let key: Vec<u8> = (0..=31_u8).collect();
        let text = encode_base64_url(&key);
        // TS oracle: decodeBase64Url(keyText) → key bytes.
        assert_eq!(decode_base64_url(&text), Some(key));
        // TS oracle: "AB+C" → null (charset gate, link.ts:40).
        assert_eq!(decode_base64_url("AB+C"), None);
        assert_eq!(decode_base64_url("AB=C"), None);
        // TS oracle: "A" → null (forgiving base64 rejects length % 4 == 1).
        assert_eq!(decode_base64_url("A"), None);
        // TS quirk: "" is rejected by the B64URL_RE `+` quantifier before any
        // decoding (link.ts:26,40).
        assert_eq!(decode_base64_url(""), None);
    }

    #[test]
    fn base64url_decode_is_forgiving_on_trailing_bits() {
        // TS oracle: atob-padded forgiving decode accepts non-canonical
        // trailing bits: "AB" → 0x00, "ABB" → 0x0010 (the strict `base64`
        // crate would reject "ABB" as InvalidLastSymbol — hence this
        // hand-rolled decoder).
        assert_eq!(decode_base64_url("AB"), Some(vec![0x00]));
        assert_eq!(decode_base64_url("ABB"), Some(vec![0x00, 0x10]));
    }

    #[test]
    fn generate_room_id_shape() {
        // TS shape: 16 bytes → 22 base64url chars (protocol.ts:188-192).
        for _ in 0..1_000 {
            let id = generate_room_id();
            assert_eq!(id.len(), 22);
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
                "non-base64url char in {id}"
            );
        }
        assert_ne!(generate_room_id(), generate_room_id());
    }
}
