//! Separator glyphs per style — the port of `status-line/separators.ts`.

use super::presets::SeparatorStyle;

/// Resolved separator glyphs for one style (unicode preset values;
/// `separators.ts:4-36`). `left` separates pieces inside the left group,
/// `right` inside the right group.
#[derive(Debug, Clone, Copy)]
pub struct Separators {
    /// Between pieces of the left group.
    pub left: &'static str,
    /// Between pieces of the right group.
    pub right: &'static str,
    /// End cap closing the left group.
    pub cap_left: &'static str,
    /// End cap opening the right group.
    pub cap_right: &'static str,
    /// Whether caps are painted with the group bg recolored to fg.
    pub use_bg_as_fg: bool,
}

impl Separators {
    /// Resolve a style to glyphs. Plain layouts use `{left:"·", right:"·"}`
    /// regardless of the setting (§5.1); ask for that with
    /// [`Separators::plain`].
    #[must_use]
    pub fn for_style(style: SeparatorStyle) -> Self {
        match style {
            SeparatorStyle::Powerline => Self {
                left: "▕",
                right: "▕",
                cap_left: "◀",
                cap_right: "▶",
                use_bg_as_fg: true,
            },
            SeparatorStyle::PowerlineThin | SeparatorStyle::Ascii => Self {
                left: ">",
                right: "<",
                cap_left: "◀",
                cap_right: "▶",
                use_bg_as_fg: true,
            },
            SeparatorStyle::Slash => Self {
                left: " / ",
                right: " / ",
                cap_left: "",
                cap_right: "",
                use_bg_as_fg: false,
            },
            SeparatorStyle::Pipe => Self {
                left: "│",
                right: "│",
                cap_left: "",
                cap_right: "",
                use_bg_as_fg: false,
            },
            SeparatorStyle::Block => Self {
                left: "▌",
                right: "▌",
                cap_left: "",
                cap_right: "",
                use_bg_as_fg: false,
            },
            SeparatorStyle::None => Self {
                left: " ",
                right: " ",
                cap_left: "",
                cap_right: "",
                use_bg_as_fg: false,
            },
        }
    }

    /// The dot separators used by the plain layouts (§5.1); used once
    /// the plain layouts land.
    #[must_use]
    #[allow(dead_code)]
    pub const fn plain() -> Self {
        Self {
            left: "·",
            right: "·",
            cap_left: "",
            cap_right: "",
            use_bg_as_fg: false,
        }
    }

    /// `true` when the separator includes its own surrounding padding
    /// (slash style carries spaces; the others don't).
    #[must_use]
    #[allow(dead_code)]
    pub fn is_padded(self) -> bool {
        // Only the slash form embeds spaces; block/pipe/thin abut the text.
        self.left == " / "
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powerline_uses_caps_and_bg_as_fg() {
        let s = Separators::for_style(SeparatorStyle::Powerline);
        assert_eq!(s.left, "▕");
        assert_eq!(s.cap_right, "▶");
        assert!(s.use_bg_as_fg);
    }

    #[test]
    fn plain_layout_uses_dots() {
        let s = Separators::plain();
        assert_eq!(s.left, "·");
        assert!(!s.use_bg_as_fg);
    }
}
