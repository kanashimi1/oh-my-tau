//! Borrowing JSON tree for provider request bodies.
//!
//! Provider conversions used to build one `serde_json::Value` per message and then serialize the
//! whole tree: every message text, image and tool argument was copied into the tree first, so a
//! large context held the history twice (tree + body) at the peak of every request. [`Wire`] keeps
//! the same shape and the same object semantics as `serde_json::Map` under `preserve_order`
//! (insertion order; inserting an existing key replaces its value in place), but borrows strings
//! and JSON subtrees from the source messages. Serializing a [`Wire`] writes exactly the bytes the
//! equivalent `Value` tree writes, and [`Wire::to_value`] rebuilds that tree when a payload hook
//! needs it.

use std::borrow::Cow;
use std::fmt;
use std::io;

use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::{Map, Value};

/// One JSON value of a provider request, borrowing from the request's source messages.
#[derive(Debug, Clone)]
pub(crate) enum Wire<'a> {
    Null,
    Bool(bool),
    Str(Cow<'a, str>),
    /// One JSON string written from consecutive pieces (joins, data URLs) without concatenating
    /// them into a fresh buffer first.
    Concat(Vec<Cow<'a, str>>),
    /// A JSON string holding the compact JSON text of a borrowed object (`Value::to_string()` of
    /// it, TS `JSON.stringify(arguments)`), written without an intermediate `String`.
    JsonText(&'a Map<String, Value>),
    /// A borrowed JSON subtree.
    Ref(&'a Value),
    /// A borrowed JSON object.
    #[cfg(test)]
    RefMap(&'a Map<String, Value>),
    /// An owned JSON value (small scalars and option objects).
    Owned(Value),
    /// Owned compact JSON text spliced into the output verbatim (the per-session wire-fragment
    /// cache reuses the serialized tools/messages prefix byte-for-byte instead of re-walking the
    /// tree). `serde_json::to_value` parses it back into a tree, so payload hooks keep working.
    Raw(std::sync::Arc<serde_json::value::RawValue>),
    Array(Vec<Wire<'a>>),
    Object(WireObject<'a>),
}

/// An ordered JSON object: `serde_json::Map` with `preserve_order` semantics.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct WireObject<'a> {
    entries: Vec<(Cow<'a, str>, Wire<'a>)>,
}

impl<'a> WireObject<'a> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Insert `value` under `key`; an existing key keeps its position and takes the new value
    /// (`IndexMap::insert`, which backs `serde_json::Map` under `preserve_order`).
    pub(crate) fn insert(&mut self, key: impl Into<Cow<'a, str>>, value: impl Into<Wire<'a>>) {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.get_mut(&key) {
            *slot = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Wire<'a>> {
        self.entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut Wire<'a>> {
        self.entries
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub(crate) fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Remove `key`, keeping the order of the remaining entries.
    pub(crate) fn remove(&mut self, key: &str) {
        self.entries.retain(|(name, _)| name != key);
    }

    /// Build an object from literal entries, in order (`json!({ .. })`).
    pub(crate) fn from_entries<const N: usize>(entries: [(&'a str, Wire<'a>); N]) -> Self {
        let mut object = Self::new();
        for (key, value) in entries {
            object.insert(key, value);
        }
        object
    }
}

/// `json!({ .. })` for [`Wire`]: an object from literal entries, in order.
pub(crate) fn obj<'a, const N: usize>(entries: [(&'a str, Wire<'a>); N]) -> Wire<'a> {
    Wire::Object(WireObject::from_entries(entries))
}

impl PartialEq for Wire<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Wire::Null, Wire::Null) => true,
            (Wire::Bool(a), Wire::Bool(b)) => a == b,
            (Wire::Str(a), Wire::Str(b)) => a == b,
            (Wire::Concat(a), Wire::Concat(b)) => a == b,
            (Wire::JsonText(a), Wire::JsonText(b)) => a == b,
            (Wire::Ref(a), Wire::Ref(b)) => a == b,
            #[cfg(test)]
            (Wire::RefMap(a), Wire::RefMap(b)) => a == b,
            (Wire::Owned(a), Wire::Owned(b)) => a == b,
            (Wire::Raw(a), Wire::Raw(b)) => a.get() == b.get(),
            (Wire::Array(a), Wire::Array(b)) => a == b,
            (Wire::Object(a), Wire::Object(b)) => a == b,
            _ => false,
        }
    }
}

impl<'a> Wire<'a> {
    /// A borrowed string.
    pub(crate) fn str(text: &'a str) -> Self {
        Wire::Str(Cow::Borrowed(text))
    }

    /// `pieces.join(separator)` as one JSON string, without building the joined `String`.
    pub(crate) fn join<I>(pieces: I, separator: &'a str) -> Self
    where
        I: IntoIterator<Item = Cow<'a, str>>,
    {
        let mut parts = Vec::new();
        for (index, piece) in pieces.into_iter().enumerate() {
            if index > 0 {
                parts.push(Cow::Borrowed(separator));
            }
            parts.push(piece);
        }
        Wire::Concat(parts)
    }

    pub(crate) fn is_null(&self) -> bool {
        matches!(
            self,
            Wire::Null | Wire::Owned(Value::Null) | Wire::Ref(Value::Null)
        )
    }

    /// The string, when this is a single-piece JSON string.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Wire::Str(text) => Some(&**text),
            Wire::Owned(Value::String(text)) | Wire::Ref(Value::String(text)) => {
                Some(text.as_str())
            }
            _ => None,
        }
    }

    /// Whether this JSON string is empty; `None` when it is not a string.
    pub(crate) fn string_is_empty(&self) -> Option<bool> {
        match self {
            Wire::Str(text) => Some(text.is_empty()),
            Wire::Concat(pieces) => Some(pieces.iter().all(|piece| piece.is_empty())),
            // The JSON text of an object is at least `{}`.
            Wire::JsonText(_) => Some(false),
            Wire::Owned(Value::String(text)) | Wire::Ref(Value::String(text)) => {
                Some(text.is_empty())
            }
            _ => None,
        }
    }

    /// The array length; `None` when this is not an array.
    pub(crate) fn array_len(&self) -> Option<usize> {
        match self {
            Wire::Array(items) => Some(items.len()),
            Wire::Owned(Value::Array(items)) | Wire::Ref(Value::Array(items)) => Some(items.len()),
            _ => None,
        }
    }

    pub(crate) fn as_array_mut(&mut self) -> Option<&mut Vec<Wire<'a>>> {
        match self {
            Wire::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn as_object(&self) -> Option<&WireObject<'a>> {
        match self {
            Wire::Object(object) => Some(object),
            _ => None,
        }
    }

    pub(crate) fn as_object_mut(&mut self) -> Option<&mut WireObject<'a>> {
        match self {
            Wire::Object(object) => Some(object),
            _ => None,
        }
    }

    /// `value.get(key)` on an object; `None` for other kinds.
    pub(crate) fn get(&self, key: &str) -> Option<&Wire<'a>> {
        self.as_object().and_then(|object| object.get(key))
    }

    /// The equivalent `serde_json::Value` tree (for payload hooks and tests).
    pub(crate) fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("a wire tree has string keys and JSON numbers")
    }
}

impl<'a> From<&'a str> for Wire<'a> {
    fn from(text: &'a str) -> Self {
        Wire::str(text)
    }
}

impl<'a> From<&'a String> for Wire<'a> {
    fn from(text: &'a String) -> Self {
        Wire::str(text)
    }
}

impl From<String> for Wire<'_> {
    fn from(text: String) -> Self {
        Wire::Str(Cow::Owned(text))
    }
}

impl<'a> From<Cow<'a, str>> for Wire<'a> {
    fn from(text: Cow<'a, str>) -> Self {
        Wire::Str(text)
    }
}

impl From<bool> for Wire<'_> {
    fn from(value: bool) -> Self {
        Wire::Bool(value)
    }
}

impl From<Value> for Wire<'_> {
    fn from(value: Value) -> Self {
        Wire::Owned(value)
    }
}

impl<'a> From<&'a Value> for Wire<'a> {
    fn from(value: &'a Value) -> Self {
        Wire::Ref(value)
    }
}

impl<'a> From<Vec<Wire<'a>>> for Wire<'a> {
    fn from(items: Vec<Wire<'a>>) -> Self {
        Wire::Array(items)
    }
}

impl<'a> From<WireObject<'a>> for Wire<'a> {
    fn from(object: WireObject<'a>) -> Self {
        Wire::Object(object)
    }
}

impl Serialize for WireObject<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (key, value) in &self.entries {
            map.serialize_entry(&**key, value)?;
        }
        map.end()
    }
}

impl Serialize for Wire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Wire::Null => serializer.serialize_unit(),
            Wire::Bool(value) => serializer.serialize_bool(*value),
            Wire::Str(text) => serializer.serialize_str(text),
            Wire::Concat(pieces) => serializer.collect_str(&Pieces(pieces)),
            Wire::JsonText(map) => serializer.collect_str(&JsonText(map)),
            Wire::Ref(value) => value.serialize(serializer),
            #[cfg(test)]
            Wire::RefMap(map) => map.serialize(serializer),
            Wire::Owned(value) => value.serialize(serializer),
            Wire::Raw(raw) => raw.serialize(serializer),
            Wire::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Wire::Object(object) => object.serialize(serializer),
        }
    }
}

/// Writes string pieces back to back (the JSON serializer escapes each chunk as it arrives).
struct Pieces<'p, 'a>(&'p [Cow<'a, str>]);

impl fmt::Display for Pieces<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for piece in self.0 {
            f.write_str(piece)?;
        }
        Ok(())
    }
}

/// Writes the compact JSON text of an object (`Value::Object(map).to_string()`).
struct JsonText<'p>(&'p Map<String, Value>);

impl fmt::Display for JsonText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut writer = FmtWriter {
            out: f,
            pending: Vec::new(),
        };
        serde_json::to_writer(&mut writer, self.0).map_err(|_| fmt::Error)?;
        if writer.pending.is_empty() {
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}

/// `io::Write` onto a `fmt::Formatter`: UTF-8 bytes pass through as `str`; a multi-byte
/// sequence split across two writes is held until it completes.
struct FmtWriter<'f, 'g> {
    out: &'f mut fmt::Formatter<'g>,
    pending: Vec<u8>,
}

impl FmtWriter<'_, '_> {
    fn emit(&mut self, bytes: &[u8]) -> io::Result<()> {
        match std::str::from_utf8(bytes) {
            Ok(text) => self.out.write_str(text).map_err(io::Error::other),
            Err(error) => {
                if error.error_len().is_some() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, error));
                }
                let (valid, rest) = bytes.split_at(error.valid_up_to());
                let valid = std::str::from_utf8(valid).map_err(io::Error::other)?;
                self.out.write_str(valid).map_err(io::Error::other)?;
                self.pending.extend_from_slice(rest);
                Ok(())
            }
        }
    }
}

impl io::Write for FmtWriter<'_, '_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.pending.is_empty() {
            self.emit(buf)?;
        } else {
            let mut joined = std::mem::take(&mut self.pending);
            joined.extend_from_slice(buf);
            self.emit(&joined)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Counts the bytes a serialization writes.
struct ByteCount(usize);

impl io::Write for ByteCount {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Compact JSON text of `value` (`serde_json::to_string`), written into a buffer allocated at its
/// exact final size: a sizing pass first, so the multi-megabyte request body never goes through
/// the doubling-growth reallocations (each of which holds the old and the new buffer at once).
pub(crate) fn to_json_string<T: Serialize + ?Sized>(value: &T) -> String {
    let mut count = ByteCount(0);
    serde_json::to_writer(&mut count, value).expect("request JSON serializes");
    let mut out = Vec::with_capacity(count.0);
    serde_json::to_writer(&mut out, value).expect("request JSON serializes");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

/// The request body for `payload`. Without a payload hook it is serialized straight from the
/// borrowing tree; with one, the hook receives the equivalent `Value` tree (and may replace it),
/// exactly as before the wire tree existed.
pub(crate) fn request_body(
    payload: &Wire<'_>,
    on_payload: Option<&crate::types::OnPayloadHook>,
    model: &crate::types::Model,
) -> String {
    match on_payload {
        None => to_json_string(payload),
        Some(hook) => {
            let value = payload.to_value();
            match hook(value.clone(), model) {
                Some(next) => to_json_string(&next),
                None => to_json_string(&value),
            }
        }
    }
}

/// [`request_body`] for an owned payload.
pub(crate) fn request_body_value(
    payload: Value,
    on_payload: Option<&crate::types::OnPayloadHook>,
    model: &crate::types::Model,
) -> String {
    match on_payload {
        None => payload.to_string(),
        Some(hook) => hook(payload.clone(), model).unwrap_or(payload).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn insert_replaces_in_place_like_preserve_order_map() {
        let mut wire = WireObject::new();
        let mut map = Map::new();
        for (key, value) in [
            ("role", "a"),
            ("content", "b"),
            ("x", "c"),
            ("content", "d"),
        ] {
            wire.insert(key, value);
            map.insert(key.to_string(), json!(value));
        }
        assert_eq!(
            to_json_string(&wire),
            Value::Object(map.clone()).to_string()
        );
        assert_eq!(Wire::Object(wire).to_value(), Value::Object(map));
    }

    #[test]
    fn pieces_and_json_text_escape_like_one_string() {
        let args = json!({"path": "a\"b\\c\n", "n": 1.5, "u": "\u{1F648}\u{7}", "nested": {"k": [1, null, true]}});
        let Value::Object(args_map) = &args else {
            unreachable!()
        };
        let pieces = Wire::join(
            [
                Cow::from("x\"y"),
                Cow::from("\u{e9}\t"),
                Cow::from(String::new()),
            ],
            "\n\n",
        );
        let wire = obj([
            ("text", pieces),
            ("arguments", Wire::JsonText(args_map)),
            ("input", Wire::RefMap(args_map)),
            (
                "url",
                Wire::Concat(vec![
                    "data:".into(),
                    "image/png".into(),
                    ";base64,".into(),
                    "QUJD".into(),
                ]),
            ),
        ]);
        let expected = json!({
            "text": "x\"y\n\n\u{e9}\t\n\n",
            "arguments": args.to_string(),
            "input": args.clone(),
            "url": "data:image/png;base64,QUJD",
        });
        assert_eq!(to_json_string(&wire), expected.to_string());
        assert_eq!(wire.to_value(), expected);
    }

    #[test]
    fn json_text_survives_split_multibyte_writes() {
        // Drive the writer directly with a 4-byte sequence split across writes.
        struct Probe;
        impl fmt::Display for Probe {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut writer = FmtWriter {
                    out: f,
                    pending: Vec::new(),
                };
                let bytes = "a\u{1F648}b".as_bytes();
                for chunk in bytes.chunks(2) {
                    io::Write::write_all(&mut writer, chunk).map_err(|_| fmt::Error)?;
                }
                assert!(writer.pending.is_empty());
                Ok(())
            }
        }
        assert_eq!(Probe.to_string(), "a\u{1F648}b");
    }
}
