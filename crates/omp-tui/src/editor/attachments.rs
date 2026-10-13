//! Attachments — the port of `prompt/composer-attachments.ts` and the
//! `pendingImages` / `pendingTexts` / `pendingImageLinks` arrays on the
//! custom editor (`custom-editor.ts:447-455,710-780`).
//!
//! Draft attachments are represented in the buffer as atomic markers
//! (`[Image #N, WxH]`, `[Video #N]`, `[Text #N, name]`) plus a parallel
//! [`Attachments`] table with the real payloads. On submit the markers
//! expand back to concrete references. Numbers are never recycled: deleted
//! chips keep their slot (`pendingImages` retains records, TS :710).

/// One pending draft attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    /// 1-based marker number (never reused).
    pub id: usize,
    /// What kind of payload this is.
    pub kind: AttachmentKind,
}

/// Payload variants for a draft attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentKind {
    /// An image file: path + dimensions when known.
    Image {
        /// Absolute path or data/remote URL.
        source: String,
        /// Pixel width when known.
        width: Option<u32>,
        /// Pixel height when known.
        height: Option<u32>,
    },
    /// A video file path.
    Video {
        /// Absolute path.
        source: String,
    },
    /// A text attachment (e.g. large paste attached as a file).
    Text {
        /// Suggested file name.
        name: String,
        /// Full content.
        content: String,
    },
}

/// The parallel attachment table for one editor draft.
#[derive(Debug, Clone, Default)]
pub struct Attachments {
    next_id: usize,
    items: Vec<Attachment>,
}

impl Attachments {
    /// An empty table; numbering starts at 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            items: Vec::new(),
        }
    }

    /// Add an image attachment; returns its marker number + label.
    pub fn add_image(
        &mut self,
        source: &str,
        width: Option<u32>,
        height: Option<u32>,
    ) -> (usize, String) {
        let id = self.alloc(AttachmentKind::Image {
            source: source.to_owned(),
            width,
            height,
        });
        let label = match (width, height) {
            (Some(w), Some(h)) => format!("[Image #{id}, {w}x{h}]"),
            _ => format!("[Image #{id}]"),
        };
        (id, label)
    }

    /// Add a video attachment; returns its marker number + label.
    pub fn add_video(&mut self, source: &str) -> (usize, String) {
        let id = self.alloc(AttachmentKind::Video {
            source: source.to_owned(),
        });
        (id, format!("[Video #{id}]"))
    }

    /// Add a text attachment; returns its marker number + label.
    pub fn add_text(&mut self, name: &str, content: &str) -> (usize, String) {
        let id = self.alloc(AttachmentKind::Text {
            name: name.to_owned(),
            content: content.to_owned(),
        });
        (id, format!("[Text #{id}, {name}]"))
    }

    fn alloc(&mut self, kind: AttachmentKind) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.items.push(Attachment { id, kind });
        id
    }

    /// Remove an attachment by id (the marker text is removed from the
    /// buffer by the editor). The numbering of remaining items is
    /// unchanged.
    pub fn remove(&mut self, id: usize) {
        self.items.retain(|a| a.id != id);
    }

    /// Number of pending attachments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether nothing is attached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// All pending attachments (chips in render order).
    #[must_use]
    pub fn items(&self) -> &[Attachment] {
        &self.items
    }

    /// Expand every attachment marker in `text` to its source reference
    /// (`@path` for images/videos, the raw content for text attachments).
    #[must_use]
    pub fn expand(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for att in &self.items {
            let replacement = match &att.kind {
                AttachmentKind::Image { source, .. } | AttachmentKind::Video { source } => {
                    source.clone()
                }
                AttachmentKind::Text { content, .. } => content.clone(),
            };
            // Markers may carry a `, WxH` / `, name` suffix; match the
            // longest present form.
            let prefixes: &[String] = &if let AttachmentKind::Text { name, .. } = &att.kind {
                vec![format!("[Text #{}, {name}]", att.id)]
            } else {
                let base = if matches!(att.kind, AttachmentKind::Image { .. }) {
                    "[Image #"
                } else {
                    "[Video #"
                };
                let stem = format!("{base}{id}", id = att.id);
                vec![format!("{stem}]"), format!("{stem}, ")]
            };
            for prefix in prefixes {
                let Some(start) = out.find(prefix.as_str()) else {
                    continue;
                };
                let end = if prefix.ends_with(", ") {
                    match out[start + prefix.len()..].find(']') {
                        Some(rel) => start + prefix.len() + rel + 1,
                        None => continue,
                    }
                } else {
                    start + prefix.len()
                };
                out.replace_range(start..end, &replacement);
                break;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_marker_with_dimensions() {
        let mut atts = Attachments::new();
        let (id, label) = atts.add_image("/tmp/a.png", Some(640), Some(480));
        assert_eq!(id, 1);
        assert_eq!(label, "[Image #1, 640x480]");
        let expanded = atts.expand(&format!("look {label} here"));
        assert_eq!(expanded, "look /tmp/a.png here");
    }

    #[test]
    fn numbering_never_recycled() {
        let mut atts = Attachments::new();
        let (a, _) = atts.add_image("/a", None, None);
        let (b, _) = atts.add_video("/b.mp4");
        atts.remove(a);
        let (c, label) = atts.add_text("notes.txt", "data");
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(label, "[Text #3, notes.txt]");
        assert_eq!(atts.len(), 2);
    }

    #[test]
    fn text_attachment_expands_to_content() {
        let mut atts = Attachments::new();
        let (_, label) = atts.add_text("log.txt", "log line");
        assert_eq!(atts.expand(&label), "log line");
    }

    #[test]
    fn video_expands_to_source() {
        let mut atts = Attachments::new();
        let (_, label) = atts.add_video("/v.mp4");
        assert_eq!(atts.expand(&format!("see {label}")), "see /v.mp4");
    }
}
