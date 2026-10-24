//! Port of the wyhash **final4** variant Bun uses for `Bun.hash`
//! (oven-sh/bun `src/wyhash/lib.rs`, `Wyhash` stateless one-shot path).
//! The checkpoint stored in `file_offsets.parser_state` must equal
//! `Bun.hash(bytes).toString(16)` byte-for-byte, so this is a faithful port,
//! not the canonical wyhash v3/v4.2 (which produce different outputs).

const SECRET: [u64; 4] = [
    0xa076_1d64_78bd_642f,
    0xe703_7ed1_a0b4_28db,
    0x8ebc_6af0_9c88_c6e3,
    0x5899_65cc_7537_4cc3,
];

#[inline(always)]
fn mum(a: &mut u64, b: &mut u64) {
    let x = (*a as u128).wrapping_mul(*b as u128);
    *a = x as u64;
    *b = (x >> 64) as u64;
}

#[inline(always)]
fn mix(a: u64, b: u64) -> u64 {
    let mut a = a;
    let mut b = b;
    mum(&mut a, &mut b);
    a ^ b
}

#[inline(always)]
fn read4(data: &[u8]) -> u64 {
    u64::from(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
}

#[inline(always)]
fn read8(data: &[u8]) -> u64 {
    u64::from_le_bytes(data[..8].try_into().expect("read8 needs 8 bytes"))
}

/// One-shot `Bun.hash(bytes, seed)` equivalent.
#[must_use]
pub fn wyhash_seed(seed: u64, input: &[u8]) -> u64 {
    let s0 = seed ^ mix(seed ^ SECRET[0], SECRET[1]);
    let mut a: u64 = 0;
    let mut b: u64 = 0;
    let mut state = [s0; 3];

    if input.len() <= 16 {
        // small_key
        if input.len() >= 4 {
            let end = input.len() - 4;
            let quarter = (input.len() >> 3) << 2;
            a = (read4(input) << 32) | read4(&input[quarter..]);
            b = (read4(&input[end..]) << 32) | read4(&input[end - quarter..]);
        } else if !input.is_empty() {
            a = (u64::from(input[0]) << 16)
                | (u64::from(input[input.len() >> 1]) << 8)
                | u64::from(input[input.len() - 1]);
        }
    } else {
        let mut i = 0usize;
        if input.len() >= 48 {
            let [mut s0, mut s1, mut s2] = state;
            let (k1, k2, k3) = (SECRET[1], SECRET[2], SECRET[3]);
            let bound = input.len() - 48;
            while i < bound {
                let m0 = ((read8(&input[i..]) ^ k1) as u128)
                    .wrapping_mul((read8(&input[i + 8..]) ^ s0) as u128);
                s0 = (m0 as u64) ^ ((m0 >> 64) as u64);
                let m1 = ((read8(&input[i + 16..]) ^ k2) as u128)
                    .wrapping_mul((read8(&input[i + 24..]) ^ s1) as u128);
                s1 = (m1 as u64) ^ ((m1 >> 64) as u64);
                let m2 = ((read8(&input[i + 32..]) ^ k3) as u128)
                    .wrapping_mul((read8(&input[i + 40..]) ^ s2) as u128);
                s2 = (m2 as u64) ^ ((m2 >> 64) as u64);
                i += 48;
            }
            state = [s0, s1, s2];
            // final0
            state[0] ^= state[1] ^ state[2];
        }
        // final1(input, i)
        let tail = &input[i..];
        let mut j = 0usize;
        while j + 16 < tail.len() {
            state[0] = mix(
                read8(&tail[j..]) ^ SECRET[1],
                read8(&tail[j + 8..]) ^ state[0],
            );
            j += 16;
        }
        a = read8(&input[input.len() - 16..]);
        b = read8(&input[input.len() - 8..]);
    }

    // final2
    a ^= SECRET[1];
    b ^= state[0];
    mum(&mut a, &mut b);
    mix(a ^ SECRET[0] ^ (input.len() as u64), b ^ SECRET[1])
}

/// `Bun.hash(bytes)` — seed 0.
#[must_use]
pub fn wyhash(bytes: &[u8]) -> u64 {
    wyhash_seed(0, bytes)
}

/// `Bun.hash(x).toString(16)` — lowercase hex, no padding (as JS `toString(16)`).
#[must_use]
pub fn wyhash_hex(bytes: &[u8]) -> String {
    format!("{:x}", wyhash(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors captured with `bun -e 'console.log(Bun.hash(s).toString(16))'`
    /// (bun ships the wyhash final4 variant).
    #[test]
    fn matches_bun_vectors() {
        for (s, hex) in [
            ("", "409638ee2bde459"),
            ("a", "28d2053309d28531"),
            ("ab", "bc9ce12eaf0083ec"),
            ("abc", "2a4f1d7cb516c72"),
            ("abcd", "48dfe2b09ab52113"),
            ("hello", "e24bbd9f93f532d"),
            ("hello world", "668d5e431c3b2573"),
            ("hello world foo bar baz quux q", "f21ff30d03923879"),
        ] {
            assert_eq!(wyhash_hex(s.as_bytes()), hex, "input {s:?}");
        }
    }

    #[test]
    fn matches_bun_large_buffer() {
        // 48-byte-round path verified against Bun.hash(new Uint8Array(1000).fill(120)).
        let buf = vec![120u8; 1000];
        assert_eq!(wyhash_hex(&buf), "b7fe88ea75c1f19a");
    }
}
