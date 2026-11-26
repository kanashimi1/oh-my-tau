//! Transitive dependency packs: container format and build/apply codecs.
//!
//! Wire format (little-endian, fixed offsets — no parser recursion, one
//! bounded pass):
//!
//! ```text
//! magic "OMPCASP1" (8B)
//! u32 header_len, then header_len bytes of JSON:
//!   { "objects": [ {"digest","size","offset","plen"} ... ] }
//! payload: zstd-compressed concatenation of object bytes in manifest order
//! ```
//!
//! Header length arrives as a plain `u32`, but import caps it at
//! [`MAX_HEADER_BYTES`] and total size at [`MAX_TOTAL_BYTES`] before any
//! allocation, so a hostile length can't turn into a giant buffer. Object
//! bounds are checked against the actual payload length with `checked_add`
//! before any indexing; digests are strictly parsed and re-verified when
//! the objects are written out.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::digest::Digest;
use crate::error::{io_err, Error, Result};
use crate::object::ObjectId;

/// Pack container magic.
const MAGIC: &[u8; 8] = b"OMPCASP1";
/// Compression level for pack payloads.
const ZSTD_LEVEL: i32 = 3;
/// Hard cap on the manifest JSON length (bounds the `u32` header field).
pub(crate) const MAX_HEADER_BYTES: u32 = 16 * 1024 * 1024;
/// Hard cap on total pack size and on declared content accepted by import.
pub(crate) const MAX_TOTAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// One manifest row: an object's digest and where its bytes live in the
/// decompressed payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackEntry {
    pub digest: String,
    pub size: u64,
    pub offset: u64,
    pub plen: u64,
}

/// Pack manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackManifest {
    pub objects: Vec<PackEntry>,
}

/// Label for sink write errors (packs are written to caller-owned sinks,
/// so there is no real path to attach).
pub(crate) fn sink_label() -> PathBuf {
    PathBuf::from("<pack-sink>")
}

/// Build a pack over `objects` (digest -> bytes) and write it to `writer`.
///
/// Manifest order follows the `BTreeMap` key order, so pack bytes are a
/// deterministic function of content: same objects, same pack bytes, same
/// pack digest.
///
/// Returns total bytes written to `writer`.
///
/// # Errors
/// Manifest serialization, compression cap, or sink write failure.
pub fn build_pack(objects: &std::collections::BTreeMap<Digest, Vec<u8>>, writer: &mut impl Write) -> Result<u64> {
    let mut raw = Vec::with_capacity(objects.values().map(Vec::len).sum::<usize>());
    let mut entries = Vec::with_capacity(objects.len());
    for (digest, bytes) in objects {
        entries.push(PackEntry {
            digest: digest.hex(),
            size: bytes.len() as u64,
            offset: raw.len() as u64,
            plen: bytes.len() as u64,
        });
        raw.extend_from_slice(bytes);
    }
    // One compression pass over the concatenation lets cross-object matches
    // find each other (small objects compress together instead of paying
    // per-object frame overhead).
    let payload = zstd::stream::encode_all(&raw[..], ZSTD_LEVEL)
        .map_err(|_| Error::Format("zstd compress failed"))?;

    let manifest = PackManifest { objects: entries };
    let header = serde_json::to_vec(&manifest)
        .map_err(|_| Error::Format("manifest serialize failed"))?;
    if header.len() > MAX_HEADER_BYTES as usize {
        return Err(Error::Limit("pack manifest exceeds header cap"));
    }

    let header_len = u32::try_from(header.len()).map_err(|_| Error::Limit("pack header exceeds cap"))?;
    writer.write_all(MAGIC).map_err(|e| io_err(&sink_label(), e))?;
    writer
        .write_all(&header_len.to_le_bytes())
        .map_err(|e| io_err(&sink_label(), e))?;
    writer.write_all(&header).map_err(|e| io_err(&sink_label(), e))?;
    writer.write_all(&payload).map_err(|e| io_err(&sink_label(), e))?;
    Ok((8 + 4 + header.len() + payload.len()) as u64)
}

/// Parse a pack into (manifest, decompressed payload). All bounds are
/// checked before any allocation or indexing.
pub(crate) fn parse_pack(bytes: &[u8]) -> Result<(PackManifest, Vec<u8>)> {
    if bytes.len() < 12 {
        return Err(Error::Format("pack shorter than fixed header"));
    }
    if &bytes[..8] != MAGIC {
        return Err(Error::Format("bad pack magic"));
    }
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| Error::Format("header length read"))?);
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::Limit("pack header exceeds cap"));
    }
    let header_end = usize::try_from(header_len)
        .ok()
        .and_then(|len| len.checked_add(12))
        .filter(|end| *end <= bytes.len())
        .ok_or(Error::Format("pack header overruns file"))?;
    let manifest: PackManifest = serde_json::from_slice(&bytes[12..header_end])
        .map_err(|_| Error::Format("manifest json invalid"))?;
    let payload = &bytes[header_end..];
    if payload.len() as u64 > MAX_TOTAL_BYTES {
        return Err(Error::Limit("pack payload exceeds cap"));
    }

    // Decompression-bomb bound: the manifest's declared content total is
    // the ceiling for the decompressed size, and the decoder output must
    // match it exactly.
    let declared_total = manifest.objects.iter().try_fold(0_u64, |total, entry| {
        total
            .checked_add(entry.size)
            .ok_or(Error::Limit("pack declared content size overflow"))
    })?;
    if declared_total > MAX_TOTAL_BYTES {
        return Err(Error::Limit("pack declares more content than cap"));
    }
    let decoder = zstd::stream::read::Decoder::new(payload)
        .map_err(|_| Error::Format("zstd decoder init failed"))?;
    let mut raw = Vec::new();
    decoder
        .take(declared_total + 1)
        .read_to_end(&mut raw)
        .map_err(|_| Error::Format("zstd decompress failed"))?;
    if raw.len() as u64 != declared_total {
        return Err(Error::Format("decompressed payload != manifest total"));
    }

    // Row-by-row bounds check before any caller indexes by these offsets.
    for entry in &manifest.objects {
        let valid = usize::try_from(entry.offset)
            .ok()
            .and_then(|off| off.checked_add(usize::try_from(entry.plen).unwrap_or(usize::MAX)))
            .is_some_and(|end| end <= raw.len() && entry.plen == entry.size)
            && Digest::parse(&entry.digest).is_ok();
        if !valid {
            return Err(Error::Format("manifest entry out of bounds or invalid"));
        }
    }
    Ok((manifest, raw))
}

/// One verified object handed to the apply callback.
pub struct VerifiedObject<'a> {
    /// Digest-verified id (digest re-hashed from payload bytes).
    pub id: ObjectId,
    /// The verified bytes.
    pub bytes: std::borrow::Cow<'a, [u8]>,
}

/// Parse and apply a pack from `bytes`: every object is digest-verified as
/// it is handed to `insert` (payload/manifest mismatch is
/// [`Error::Corrupt`], never silently accepted). Returns the verified ids
/// in manifest order.
///
/// # Errors
/// [`Error::Format`] on malformed containers, [`Error::Corrupt`] on digest
/// mismatch, or whatever `insert` returns.
pub fn apply_pack(
    bytes: &[u8],
    mut insert: impl FnMut(VerifiedObject) -> Result<()>,
) -> Result<Vec<ObjectId>> {
    let (manifest, raw) = parse_pack(bytes)?;
    // Verify the complete pack before invoking a stateful importer: a corrupt
    // later object must not leave earlier objects committed.
    let mut verified = Vec::with_capacity(manifest.objects.len());
    for entry in manifest.objects {
        let digest =
            Digest::parse(&entry.digest).map_err(|_| Error::Format("manifest entry digest invalid"))?;
        let start = usize::try_from(entry.offset).map_err(|_| Error::Format("entry offset too large"))?;
        let size = usize::try_from(entry.size).map_err(|_| Error::Format("entry size too large"))?;
        let end = start
            .checked_add(size)
            .ok_or(Error::Format("entry range overflows"))?;
        let object_bytes = raw
            .get(start..end)
            .ok_or(Error::Format("entry range overruns payload"))?;
        let id = ObjectId::of(object_bytes);
        if id.digest() != digest {
            return Err(Error::Corrupt {
                digest: digest.hex(),
                reason: "pack payload does not match manifest digest".into(),
            });
        }
        verified.push((id, start, end));
    }
    let mut ids = Vec::with_capacity(verified.len());
    for (id, start, end) in verified {
        insert(VerifiedObject {
            id,
            bytes: std::borrow::Cow::Borrowed(&raw[start..end]),
        })?;
        ids.push(id);
    }
    Ok(ids)
}

/// Read a pack file and apply it.
///
/// # Errors
/// Read failure or [`apply_pack`] errors.
pub fn apply_pack_file(
    path: &Path,
    insert: impl FnMut(VerifiedObject) -> Result<()>,
) -> Result<Vec<ObjectId>> {
    let bytes = std::fs::read(path).map_err(|e| io_err(path, e))?;
    apply_pack(&bytes, insert)
}
