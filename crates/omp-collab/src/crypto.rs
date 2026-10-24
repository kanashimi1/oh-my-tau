//! AES-256-GCM sealing for collab frames.
//!
//! The room key lives only in the link fragment; the relay sees opaque bytes.
//! Sealed layout: `[12B IV][ciphertext+tag]`.
//!
//! Port of `vendor-src/oh-my-pi/packages/coding-agent/src/collab/crypto.ts`.
//! The TS side uses `WebCrypto` `AES-GCM` with default parameters (128-bit
//! tag, no additional data — crypto.ts:10-11,42); the Rust primitive is
//! `ring::aead::AES_256_GCM`, whose output is bit-identical for the same
//! key/IV/plaintext (asserted byte-for-byte against `crypto.subtle` oracle
//! vectors below).

use rand::RngCore;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};

use crate::protocol::{ROOM_KEY_BYTES, WRITE_TOKEN_BYTES};

/// GCM IV/nonce size (crypto.ts:11).
pub const IV_LENGTH: usize = 12;

/// Fresh random room key (crypto.ts:15-19).
#[must_use]
pub fn generate_room_key() -> [u8; ROOM_KEY_BYTES] {
    let mut key = [0_u8; ROOM_KEY_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut key);
    key
}

/// Fresh random write token (crypto.ts:21-25).
#[must_use]
pub fn generate_write_token() -> [u8; WRITE_TOKEN_BYTES] {
    let mut token = [0_u8; WRITE_TOKEN_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut token);
    token
}

/// Sealing cipher for one collab room (crypto.ts:27-58, `importRoomKey` +
/// `sealSerialized`/`open` fused over the raw 32-byte key).
pub struct CollabCipher {
    key: LessSafeKey,
}

impl CollabCipher {
    /// Import a raw room key.
    ///
    /// # Panics
    /// When `key` is not [`ROOM_KEY_BYTES`] bytes, mirroring the TS
    /// `importRoomKey` throw (crypto.ts:28-30).
    #[must_use]
    pub fn new(key: &[u8]) -> Self {
        assert!(
            key.len() == ROOM_KEY_BYTES,
            "Room key must be {ROOM_KEY_BYTES} bytes, got {}",
            key.len()
        );
        let unbound = UnboundKey::new(&AES_256_GCM, key).expect("32-byte AES-256 key");
        // `LessSafeKey` because the nonce is chosen here (random IV) instead
        // of by a ring `NonceSequence` — the TS code does the same.
        Self {
            key: LessSafeKey::new(unbound),
        }
    }

    /// Seal a frame: random 12-byte IV, then
    /// `[IV][ciphertext][16B tag]` (crypto.ts:38-47).
    #[must_use]
    pub fn seal(&self, plaintext: &[u8]) -> Vec<u8> {
        let mut iv = [0_u8; IV_LENGTH];
        rand::rngs::OsRng.fill_bytes(&mut iv);
        self.seal_with_iv(&iv, plaintext)
    }

    /// Inverse of [`CollabCipher::seal`]; `None` on auth failure or
    /// malformed input, mirroring the TS throws (crypto.ts:50-58).
    #[must_use]
    pub fn open(&self, sealed: &[u8]) -> Option<Vec<u8>> {
        if sealed.len() <= IV_LENGTH {
            return None; // "Sealed frame too short"
        }
        let iv = Nonce::try_assume_unique_for_key(&sealed[..IV_LENGTH]).ok()?;
        let mut in_out = sealed[IV_LENGTH..].to_vec();
        let plaintext = self.key.open_in_place(iv, Aad::empty(), &mut in_out).ok()?;
        Some(plaintext.to_vec())
    }

    /// Test hook: seal with a caller-chosen IV so golden vectors against
    /// `WebCrypto` (`crypto.subtle.encrypt` with a fixed iv) stay deterministic.
    #[must_use]
    fn seal_with_iv(&self, iv: &[u8; IV_LENGTH], plaintext: &[u8]) -> Vec<u8> {
        let nonce = Nonce::assume_unique_for_key(*iv);
        let mut in_out = plaintext.to_vec();
        self.key
            .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
            .expect("AES-GCM seal cannot fail for these inputs");
        let mut out = Vec::with_capacity(IV_LENGTH + in_out.len());
        out.extend_from_slice(iv);
        out.extend_from_slice(&in_out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];
    const FIXED_IV: [u8; 12] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
    ];
    const PLAINTEXT: &[u8] = br#"{"t":"welcome","proto":3,"entryCount":2}"#;

    /// TS oracle: `crypto.subtle.encrypt` of PLAINTEXT under the room key
    /// with the fixed IV, prefixed with the IV (bun, oracle-t1.mts).
    const FIXED_SEALED_HEX: &str = "00112233445566778899aabb a1a4849175453c7b0403af711faeae1f4970dc217d484184a6ad739ee5c267f817f5861f9f6197247f587ab5e661c51477f4bf8ba1487461";

    /// TS oracle: empty plaintext under the same key/IV.
    const EMPTY_PT_SEALED_HEX: &str = "00112233445566778899aabb af8a386d1f7213dafc342ba9606b5297";

    /// TS oracle: one-byte plaintext `[0x41]` under the same key/IV.
    const ONE_PT_SEALED_HEX: &str = "00112233445566778899aabb 9bd09e0f319d3d46caa8036b60ed66638a";

    /// TS oracle: `sealSerialized(key, PLAINTEXT)` — random IV, golden for
    /// `open` (the layout `[12B IV][ct+tag]` is what `open` validates).
    const TS_SEALED_HEX: &str = "8a9bad3dc2f2739e343cbf5d1141b8ec9372a9353b314a7ded581f1eb95d54f08ef9fb5db06c2b32c0f5001e7a134c72832bb4d7caab184f04b3c5a6064500d5ed60e5bb";

    #[must_use]
    fn hex(text: &str) -> Vec<u8> {
        text.split_whitespace()
            .flat_map(|part| {
                (0..part.len())
                    .step_by(2)
                    .map(move |i| u8::from_str_radix(&part[i..i + 2], 16).expect("hex digit"))
            })
            .collect()
    }

    #[test]
    fn seal_matches_webcrypto_byte_for_byte() {
        let cipher = CollabCipher::new(&KEY);
        assert_eq!(
            cipher.seal_with_iv(&FIXED_IV, PLAINTEXT),
            hex(FIXED_SEALED_HEX)
        );
        // Padding-free edges: 0-byte and 1-byte plaintexts (12B IV + 16/17B).
        assert_eq!(
            cipher.seal_with_iv(&FIXED_IV, b""),
            hex(EMPTY_PT_SEALED_HEX)
        );
        assert_eq!(cipher.seal_with_iv(&FIXED_IV, b"A"), hex(ONE_PT_SEALED_HEX));
    }

    #[test]
    fn open_ts_sealed_frame() {
        // TS oracle: open(sealSerialized(key, PLAINTEXT)) → the frame JSON.
        let cipher = CollabCipher::new(&KEY);
        assert_eq!(cipher.open(&hex(TS_SEALED_HEX)), Some(PLAINTEXT.to_vec()));
        // And the WebCrypto fixed-IV blob opens to the same plaintext.
        assert_eq!(
            cipher.open(&hex(FIXED_SEALED_HEX)),
            Some(PLAINTEXT.to_vec())
        );
    }

    #[test]
    fn seal_open_round_trip() {
        let cipher = CollabCipher::new(&KEY);
        let sealed = cipher.seal(PLAINTEXT);
        assert_eq!(sealed.len(), IV_LENGTH + PLAINTEXT.len() + 16); // GCM tag
        assert_eq!(cipher.open(&sealed), Some(PLAINTEXT.to_vec()));
        // Random IV: two seals of the same plaintext differ.
        assert_ne!(cipher.seal(PLAINTEXT), cipher.seal(PLAINTEXT));
    }

    #[test]
    fn open_rejects_wrong_key() {
        // Sealed under KEY, opened under a different key → auth failure → None
        // (TS: subtle.decrypt throws).
        let wrong: [u8; 32] = core::array::from_fn(|i| u8::try_from(i).expect("32 < 256") ^ 0xff);
        let sealed = CollabCipher::new(&KEY).seal(PLAINTEXT);
        assert_eq!(CollabCipher::new(&wrong).open(&sealed), None);
        // Tampered ciphertext byte → auth failure → None.
        let mut tampered = sealed.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        assert_eq!(CollabCipher::new(&KEY).open(&tampered), None);
        // Truncated tag → auth failure → None.
        assert_eq!(
            CollabCipher::new(&KEY).open(&sealed[..sealed.len() - 1]),
            None
        );
    }

    #[test]
    fn open_rejects_too_short_input() {
        // TS oracle: data.byteLength <= 12 throws "Sealed frame too short"
        // (crypto.ts:51-53).
        let cipher = CollabCipher::new(&KEY);
        assert_eq!(cipher.open(&[]), None);
        assert_eq!(cipher.open(&[0_u8; 12]), None);
        assert_eq!(cipher.open(&[0_u8; 13]), None); // 1-byte "ciphertext", no tag
    }

    #[test]
    #[should_panic(expected = "Room key must be 32 bytes, got 16")]
    fn new_rejects_wrong_key_length_like_ts_import() {
        // TS oracle: importRoomKey(16 bytes) throws
        // "Room key must be 32 bytes, got 16" (crypto.ts:28-30).
        let _ = CollabCipher::new(&KEY[..16]); // must_use: the assert fires first
    }

    #[test]
    fn key_and_token_generators_have_ts_lengths() {
        // crypto.ts:15-25 — key 32 bytes, token 16 bytes, random.
        let key = generate_room_key();
        let token = generate_write_token();
        assert_eq!(key.len(), 32);
        assert_eq!(token.len(), 16);
        assert_ne!(key, generate_room_key());
        assert_ne!(token, generate_write_token());
    }
}
