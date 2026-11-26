//! Write-once escaped wire arena for serialized projected data.
//!
//! Projected interfaces are serialized into a single contiguous
//! buffer with correct JSON string escaping. The arena is *bounded*
//! (a hard byte cap fixed at construction; a write that would exceed
//! it fails and leaves the buffer unchanged) and *write-once*: once
//! [`WireArena::freeze`] is called the buffer is immutable — there is
//! no append-after-freeze and no interior free, so a frozen arena can
//! be shared and re-read without lifetime hazards.
//!
//! Arbitrary JSON primitives are preserved losslessly:
//! `null`/`false`/`0` round-trip verbatim, integers keep full
//! `i64`/`u64` range, and floats use `serde_json`'s shortest
//! round-trip formatting.

use serde_json::Value;

/// Why a wire write was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    /// The arena was frozen; writes are permanently closed.
    Frozen,
    /// The write would exceed the arena's byte cap. The buffer is
    /// left exactly as it was before the attempted write.
    Overflow {
        /// Bytes the write would have needed.
        needed: usize,
        /// Bytes still available.
        remaining: usize,
    },
}

/// A bounded, append-then-freeze JSON wire buffer.
#[derive(Debug)]
pub struct WireArena {
    buf: String,
    cap: usize,
    frozen: bool,
}

impl WireArena {
    /// An empty arena that accepts at most `cap` bytes.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            buf: String::new(),
            cap,
            frozen: false,
        }
    }

    /// Whether the arena is frozen (writes permanently closed).
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Freeze the arena: all further writes fail with
    /// [`WireError::Frozen`]. Idempotent.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// The serialized bytes written so far.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.buf
    }

    /// Bytes written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Bytes still writable before the cap.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.cap.saturating_sub(self.buf.len())
    }

    /// Append pre-escaped structural text verbatim. Callers are
    /// responsible for it being valid JSON fragments; use
    /// [`WireArena::write_value`] for arbitrary data.
    ///
    /// # Errors
    /// [`WireError::Frozen`] after freeze; [`WireError::Overflow`] if
    /// the cap would be exceeded (buffer left unchanged).
    pub fn write_raw(&mut self, s: &str) -> Result<(), WireError> {
        if self.frozen {
            return Err(WireError::Frozen);
        }
        if s.len() > self.remaining() {
            return Err(WireError::Overflow {
                needed: s.len(),
                remaining: self.remaining(),
            });
        }
        self.buf.push_str(s);
        Ok(())
    }

    /// Append a JSON string literal (quotes included) with correct
    /// escaping: `"` and `\` are escaped, the short escapes
    /// `\n \r \t \u{8} \u{c}` are used where they exist, remaining
    /// control characters below U+0020 become `\u00XX`, and all other
    /// text (including non-ASCII) passes through as UTF-8.
    ///
    /// # Errors
    /// As [`WireArena::write_raw`]; on overflow the buffer is left
    /// unchanged (the write is transactional).
    pub fn write_escaped_str(&mut self, s: &str) -> Result<(), WireError> {
        if self.frozen {
            return Err(WireError::Frozen);
        }
        let start = self.buf.len();
        self.buf.push('"');
        for ch in s.chars() {
            match ch {
                '"' => self.buf.push_str("\\\""),
                '\\' => self.buf.push_str("\\\\"),
                '\n' => self.buf.push_str("\\n"),
                '\r' => self.buf.push_str("\\r"),
                '\t' => self.buf.push_str("\\t"),
                '\u{8}' => self.buf.push_str("\\b"),
                '\u{c}' => self.buf.push_str("\\f"),
                c if (c as u32) < 0x20 => {
                    let esc = format!("\\u{:04x}", c as u32);
                    self.buf.push_str(&esc);
                }
                c => self.buf.push(c),
            }
        }
        self.buf.push('"');
        if self.buf.len() > self.cap {
            let needed = self.buf.len() - start;
            let remaining = self.cap.saturating_sub(start);
            self.buf.truncate(start);
            return Err(WireError::Overflow { needed, remaining });
        }
        Ok(())
    }

    /// Append an arbitrary JSON value losslessly and transactionally:
    /// on any error the buffer is left exactly as before the call.
    ///
    /// # Errors
    /// As [`WireArena::write_raw`].
    pub fn write_value(&mut self, v: &Value) -> Result<(), WireError> {
        let start = self.buf.len();
        let r = self.write_value_inner(v);
        if r.is_err() {
            self.buf.truncate(start);
        }
        r
    }

    fn write_value_inner(&mut self, v: &Value) -> Result<(), WireError> {
        match v {
            Value::Null => self.write_raw("null"),
            Value::Bool(true) => self.write_raw("true"),
            Value::Bool(false) => self.write_raw("false"),
            Value::Number(n) => {
                let mut s = String::new();
                let _ = std::fmt::write(&mut s, format_args!("{n}"));
                self.write_raw(&s)
            }
            Value::String(s) => self.write_escaped_str(s),
            Value::Array(items) => {
                self.write_raw("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        self.write_raw(",")?;
                    }
                    self.write_value_inner(item)?;
                }
                self.write_raw("]")
            }
            Value::Object(map) => {
                self.write_raw("{")?;
                for (i, (k, val)) in map.iter().enumerate() {
                    if i > 0 {
                        self.write_raw(",")?;
                    }
                    self.write_escaped_str(k)?;
                    self.write_raw(":")?;
                    self.write_value_inner(val)?;
                }
                self.write_raw("}")
            }
        }
    }
}
