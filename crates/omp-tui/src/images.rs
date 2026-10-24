//! Inline terminal images — port of `pi-tui/src/kitty-graphics.ts`, the
//! `ImageBudget` core of `components/image.ts` and `terminal-capabilities.ts`
//! protocol detection (§8 and §10 of `tui-spec/core.md`).
//!
//! Scope (P0): protocol detection, kitty Unicode-placeholder encoding
//! (`U=1` + U+10EEEE with row/column diacritics), tmux DCS passthrough
//! wrapping, the destructive-reset/delete encodings, and the per-surface
//! image budget that keeps at most `cap` recent images live and queues the
//! demoted ids for `d=I` purges. The full commit-watermark / placement-epoch
//! machinery belongs to the inline renderer (`ShellRender`) and is intentionally
//! not ported here.

/// A detected terminal image protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageProtocol {
    /// No inline images; render placeholders.
    #[default]
    None,
    /// Kitty graphics protocol.
    Kitty,
    /// Sixel.
    Sixel,
    /// iTerm2 inline images.
    Iterm2,
}

/// Default count of inline images kept as live graphics before older ones
/// fall back to text (TS `DEFAULT_MAX_INLINE_IMAGES`).
pub const DEFAULT_MAX_INLINE_IMAGES: usize = 8;

/// Kitty Unicode placeholder base character (U+10EEEE, Plane 16 PUA).
pub const KITTY_PLACEHOLDER: char = '\u{10EEEE}';

/// Largest row/column index expressible with the diacritic table.
pub const KITTY_PLACEHOLDER_MAX_CELLS: usize = ROWCOLUMN_DIACRITICS.len();

/// Row/column diacritics (Unicode combining class 230, no decomposition);
/// index `i` → codepoint. Derived from kitty `gen/rowcolumn-diacritics.txt`
/// (Unicode 6.0.0 NSM set); 297 entries.
#[rustfmt::skip]
const ROWCOLUMN_DIACRITICS: [u32; 297] = [
    0x305, 0x30d, 0x30e, 0x310, 0x312, 0x33d, 0x33e, 0x33f, 0x346, 0x34a, 0x34b, 0x34c, 0x350, 0x351, 0x352, 0x357,
    0x35b, 0x363, 0x364, 0x365, 0x366, 0x367, 0x368, 0x369, 0x36a, 0x36b, 0x36c, 0x36d, 0x36e, 0x36f, 0x483, 0x484,
    0x485, 0x486, 0x487, 0x592, 0x593, 0x594, 0x595, 0x597, 0x598, 0x599, 0x59c, 0x59d, 0x59e, 0x59f, 0x5a0, 0x5a1,
    0x5a8, 0x5a9, 0x5ab, 0x5ac, 0x5af, 0x5c4, 0x610, 0x611, 0x612, 0x613, 0x614, 0x615, 0x616, 0x617, 0x657, 0x658,
    0x659, 0x65a, 0x65b, 0x65d, 0x65e, 0x6d6, 0x6d7, 0x6d8, 0x6d9, 0x6da, 0x6db, 0x6dc, 0x6df, 0x6e0, 0x6e1, 0x6e2,
    0x6e4, 0x6e7, 0x6e8, 0x6eb, 0x6ec, 0x730, 0x732, 0x733, 0x735, 0x736, 0x73a, 0x73d, 0x73f, 0x740, 0x741, 0x743,
    0x745, 0x747, 0x749, 0x74a, 0x7eb, 0x7ec, 0x7ed, 0x7ee, 0x7ef, 0x7f0, 0x7f1, 0x7f3, 0x816, 0x817, 0x818, 0x819,
    0x81b, 0x81c, 0x81d, 0x81e, 0x81f, 0x820, 0x821, 0x822, 0x823, 0x825, 0x826, 0x827, 0x829, 0x82a, 0x82b, 0x82c,
    0x82d, 0x951, 0x953, 0x954, 0xf82, 0xf83, 0xf86, 0xf87, 0x135d, 0x135e, 0x135f, 0x17dd, 0x193a, 0x1a17, 0x1a75,
    0x1a76, 0x1a77, 0x1a78, 0x1a79, 0x1a7a, 0x1a7b, 0x1a7c, 0x1b6b, 0x1b6d, 0x1b6e, 0x1b6f, 0x1b70, 0x1b71, 0x1b72,
    0x1b73, 0x1cd0, 0x1cd1, 0x1cd2, 0x1cda, 0x1cdb, 0x1ce0, 0x1dc0, 0x1dc1, 0x1dc3, 0x1dc4, 0x1dc5, 0x1dc6, 0x1dc7,
    0x1dc8, 0x1dc9, 0x1dcb, 0x1dcc, 0x1dd1, 0x1dd2, 0x1dd3, 0x1dd4, 0x1dd5, 0x1dd6, 0x1dd7, 0x1dd8, 0x1dd9, 0x1dda,
    0x1ddb, 0x1ddc, 0x1ddd, 0x1dde, 0x1ddf, 0x1de0, 0x1de1, 0x1de2, 0x1de3, 0x1de4, 0x1de5, 0x1de6, 0x1dfe, 0x20d0,
    0x20d1, 0x20d4, 0x20d5, 0x20d6, 0x20d7, 0x20db, 0x20dc, 0x20e1, 0x20e7, 0x20e9, 0x20f0, 0x2cef, 0x2cf0, 0x2cf1,
    0x2de0, 0x2de1, 0x2de2, 0x2de3, 0x2de4, 0x2de5, 0x2de6, 0x2de7, 0x2de8, 0x2de9, 0x2dea, 0x2deb, 0x2dec, 0x2ded,
    0x2dee, 0x2def, 0x2df0, 0x2df1, 0x2df2, 0x2df3, 0x2df4, 0x2df5, 0x2df6, 0x2df7, 0x2df8, 0x2df9, 0x2dfa, 0x2dfb,
    0x2dfc, 0x2dfd, 0x2dfe, 0x2dff, 0xa66f, 0xa67c, 0xa67d, 0xa6f0, 0xa6f1, 0xa8e0, 0xa8e1, 0xa8e2, 0xa8e3, 0xa8e4,
    0xa8e5, 0xa8e6, 0xa8e7, 0xa8e8, 0xa8e9, 0xa8ea, 0xa8eb, 0xa8ec, 0xa8ed, 0xa8ee, 0xa8ef, 0xa8f0, 0xa8f1, 0xaab0,
    0xaab2, 0xaab3, 0xaab7, 0xaab8, 0xaabe, 0xaabf, 0xaac1, 0xfe20, 0xfe21, 0xfe22, 0xfe23, 0xfe24, 0xfe25, 0xfe26,
    0x10a0f, 0x10a38, 0x1d185, 0x1d186, 0x1d187, 0x1d188, 0x1d189, 0x1d1aa, 0x1d1ab, 0x1d1ac, 0x1d1ad, 0x1d242,
    0x1d243, 0x1d244,
];

fn diacritic(index: usize) -> char {
    char::from_u32(ROWCOLUMN_DIACRITICS[index]).unwrap_or(KITTY_PLACEHOLDER)
}

/// Detect the image protocol from environment variables the way
/// `terminal-capabilities.ts` does (`KITTY_WINDOW_ID`, `TERM` containing
/// `kitty`, sixel hints, `TERM_PROGRAM`).
#[must_use]
pub fn detect() -> ImageProtocol {
    let term = std::env::var("TERM").unwrap_or_default();
    if std::env::var_os("KITTY_WINDOW_ID").is_some() || term.contains("kitty") {
        return ImageProtocol::Kitty;
    }
    if term.contains("sixel") {
        return ImageProtocol::Sixel;
    }
    if std::env::var("TERM_PROGRAM").as_deref() == Ok("iTerm.app") {
        return ImageProtocol::Iterm2;
    }
    ImageProtocol::None
}

/// Whether we are inside a Herdr pane (`HERDR_ENV=1` canonical; identity vars
/// survive env-sanitizing launchers). Do NOT use client-side vars like
/// `HERDR_SOCKET_PATH` here — they can be set outside a pane.
#[must_use]
pub fn is_inside_herdr() -> bool {
    std::env::var("HERDR_ENV").as_deref() == Ok("1")
        || std::env::var_os("HERDR_PANE_ID").is_some()
        || std::env::var_os("HERDR_TAB_ID").is_some()
        || std::env::var_os("HERDR_WORKSPACE_ID").is_some()
}

/// Whether we are inside a terminal multiplexer (tmux/screen/zellij).
#[must_use]
pub fn is_inside_terminal_multiplexer() -> bool {
    std::env::var_os("TMUX").is_some()
        || std::env::var_os("STY").is_some()
        || std::env::var_os("ZELLIJ").is_some()
}

fn env_flag_truthy(name: &str) -> Option<bool> {
    match std::env::var(name)
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        .as_str()
    {
        "1" | "true" | "on" | "yes" | "y" => Some(true),
        "0" | "false" | "off" | "no" | "n" => Some(false),
        _ => None,
    }
}

/// Whether the detected terminal renders kitty Unicode placeholders
/// (`U=1` + U+10EEEE with row/column diacritics). Env overrides win:
/// `PI_NO_KITTY_PLACEHOLDERS=1` and `PI_KITTY_PLACEHOLDERS=0` are hard opt-outs,
/// `PI_KITTY_PLACEHOLDERS=1` opts in anywhere, `PI_FORCE_IMAGE_PROTOCOL=kitty`
/// under a multiplexer matches `timg -pk`. Automatic Herdr fallback stays off
/// (its pane marker does not prove the kitty renderer is enabled); kitty,
/// ghostty, monstar, otty and rio advertise placeholder support.
#[must_use]
pub fn detect_kitty_unicode_placeholders_support(terminal_id: &str) -> bool {
    if env_flag_truthy("PI_NO_KITTY_PLACEHOLDERS") == Some(true) {
        return false;
    }
    if let Some(force) = env_flag_truthy("PI_KITTY_PLACEHOLDERS") {
        return force;
    }
    if is_inside_terminal_multiplexer()
        && std::env::var("PI_FORCE_IMAGE_PROTOCOL").as_deref() == Ok("kitty")
    {
        return true;
    }
    if is_inside_herdr() {
        return false;
    }
    matches!(
        terminal_id,
        "kitty" | "ghostty" | "monstar" | "otty" | "rio"
    )
}

/// Whether a `columns`×`rows` placeholder grid fits the diacritic table.
#[must_use]
pub fn kitty_placeholders_fit(columns: usize, rows: usize) -> bool {
    columns >= 1
        && rows >= 1
        && columns <= KITTY_PLACEHOLDER_MAX_CELLS
        && rows <= KITTY_PLACEHOLDER_MAX_CELLS
}

/// Wrap a graphics control sequence for tmux passthrough: inside tmux the APC
/// is re-wrapped as a DCS `tmux;` passthrough with the ESC doubled, so it
/// reaches the outer terminal verbatim.
#[must_use]
pub fn wrap_tmux_passthrough(sequence: &str) -> String {
    if is_inside_terminal_multiplexer() && std::env::var_os("TMUX").is_some() {
        format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b"))
    } else {
        sequence.to_owned()
    }
}

/// Options for the kitty placeholder encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KittyPlacement {
    /// Kitty image id (from [`ImageBudget`]).
    pub image_id: u32,
    /// Placement id; a stable id replaces the placement in place.
    pub placement_id: Option<u32>,
    /// Placement box in cells.
    pub columns: usize,
    /// Placement box in cells.
    pub rows: usize,
}

/// Virtual placement APC (`a=p,U=1`): placeholder cells carrying image id
/// `i` display the transmitted image scaled to the `c`×`r` cell box.
#[must_use]
pub fn encode_kitty_virtual_placement(p: KittyPlacement) -> String {
    use std::fmt::Write as _;
    let mut params = format!("a=p,U=1,q=2,i={}", p.image_id);
    if let Some(pid) = p.placement_id.filter(|&id| id != 0) {
        let _ = write!(params, ",p={pid}");
    }
    let _ = write!(params, ",c={},r={}", p.columns, p.rows);
    wrap_tmux_passthrough(&format!("\x1b_G{params}\x1b\\"))
}

/// Build the placeholder cell grid, one string per row: image id in the
/// foreground color, placement id in the underline color, every cell naming
/// its explicit row+column diacritic (robust to slicing). Returns `rows` rows.
#[must_use]
pub fn encode_kitty_placeholder_grid(p: KittyPlacement) -> Vec<String> {
    let id = p.image_id;
    let lead = format!(
        "\x1b[38;2;{};{};{}m{}",
        (id >> 16) & 0xff,
        (id >> 8) & 0xff,
        id & 0xff,
        match p.placement_id.filter(|&v| v != 0) {
            Some(pid) => format!(
                "\x1b[58:2::{}:{}:{}m",
                (pid >> 16) & 0xff,
                (pid >> 8) & 0xff,
                pid & 0xff
            ),
            None => String::new(),
        }
    );
    let reset = "\x1b[39;59m";
    (0..p.rows)
        .map(|r| {
            let row_diacritic = diacritic(r);
            let mut row = lead.clone();
            for c in 0..p.columns {
                row.push(KITTY_PLACEHOLDER);
                row.push(row_diacritic);
                row.push(diacritic(c));
            }
            row.push_str(reset);
            row
        })
        .collect()
}

/// Full placeholder render: the virtual-placement APC prefixes row 0, and
/// every row carries placeholder cells. Returns exactly `rows` lines (no
/// cursor moves).
#[must_use]
pub fn render_kitty_placeholder_lines(p: KittyPlacement) -> Vec<String> {
    let mut grid = encode_kitty_placeholder_grid(p);
    if let Some(first) = grid.first_mut() {
        first.insert_str(0, &encode_kitty_virtual_placement(p));
    }
    grid
}

/// Delete one image's data and all its placements (`a=d,d=I,i=<id>`).
#[must_use]
pub fn encode_kitty_delete(image_id: u32) -> String {
    wrap_tmux_passthrough(&format!("\x1b_Ga=d,d=I,i={image_id},q=1\x1b\\"))
}

/// Delete everything the terminal store holds (`a=d,d=A`) — the
/// destructive-reset companion (emit BEFORE `CSI 2J`/`3J`, which are graphics
/// clears that would otherwise reclaim freshly transmitted data).
#[must_use]
pub fn encode_kitty_delete_all() -> String {
    wrap_tmux_passthrough("\x1b_Ga=d,d=A,q=1\x1b\\")
}

/// Bounds how many inline images render as live terminal graphics at once
/// (P0 port of TS `ImageBudget`, commit-watermark machinery excluded).
///
/// Kitty keeps every transmitted image in a per-screen store and re-draws
/// placements as content scrolls, so unbounded inline images pile up store
/// memory and leave ghosts in scrollback. The budget assigns stable ids to
/// logical images (keyed, so repaints replace the same placement), keeps the
/// most recent `cap` images observed in a display pass live, demotes older
/// ones to their text fallback, and queues the demoted ids for explicit
/// `d=I` deletes ([`ImageBudget::take_purge_ids`], emit before the frame).
///
/// `cap == 0` disables budgeting: every image stays a live graphic.
#[derive(Debug)]
pub struct ImageBudget {
    cap: usize,
    next_id: u32,
    key_to_id: std::collections::HashMap<String, u32>,
    pass_order: Vec<(u32, String)>,
    purge_ids: Vec<u32>,
    live: std::collections::HashSet<u32>,
}

impl Default for ImageBudget {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_INLINE_IMAGES)
    }
}

impl ImageBudget {
    /// New budget with the given per-message live-image cap.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            next_id: 1,
            key_to_id: std::collections::HashMap::new(),
            pass_order: Vec::new(),
            purge_ids: Vec::new(),
            live: std::collections::HashSet::new(),
        }
    }

    /// Whether budgeting is active (`cap > 0`).
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.cap > 0
    }

    /// Stable graphics id for a logical image; a non-empty `key` maps to the
    /// same id across re-creations so repaints replace the placement.
    pub fn allocate(&mut self, key: Option<&str>) -> u32 {
        if let Some(key) = key.filter(|k| !k.is_empty()) {
            *self.key_to_id.entry(key.to_owned()).or_insert_with(|| {
                let id = self.next_id;
                self.next_id = self.next_id.wrapping_add(1).max(1);
                id
            })
        } else {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1).max(1);
            id
        }
    }

    /// Record one image's display position within the in-flight render pass.
    pub fn observe(&mut self, image_id: u32, key: Option<&str>) {
        let key = key.unwrap_or_default().to_owned();
        if !self.pass_order.iter().any(|(id, _)| *id == image_id) {
            self.pass_order.push((image_id, key));
        }
    }

    /// End the pass: ids beyond `cap` are demoted (dropped from `live`),
    /// and any previously-live id no longer shown as a graphic is queued for
    /// an explicit delete. Returns the ids the caller must render as text.
    pub fn end_pass(&mut self) -> Vec<u32> {
        let demoted_start = self.cap;
        let mut suppressed = Vec::new();
        let mut shown = std::collections::HashSet::new();
        for (index, (id, key)) in self.pass_order.iter().enumerate() {
            if self.enabled() && index >= demoted_start {
                suppressed.push(*id);
                self.key_to_id.remove(key);
            } else {
                shown.insert(*id);
            }
        }
        for id in &self.live {
            if !shown.contains(id) {
                self.purge_ids.push(*id);
            }
        }
        self.live = shown;
        self.pass_order.clear();
        suppressed
    }

    /// Image ids to delete from the terminal this frame; clears the queue.
    pub fn take_purge_ids(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.purge_ids)
    }

    /// Everything the store is believed to hold; clears all tracking. Emit
    /// the ids as explicit deletes before a destructive reset.
    pub fn take_all_transmitted_ids(&mut self) -> Vec<u32> {
        self.purge_ids.clear();
        self.pass_order.clear();
        self.live.clear();
        let ids: Vec<u32> = self.key_to_id.values().copied().collect();
        self.key_to_id.clear();
        ids
    }
}

/// Text fallback for an unsupported or budget-demoted image: a bordered
/// placeholder box naming the file and pixel size (TS `imageFallback`,
/// simplified for P0 — no box-drawing art, non-plain so blank-edge trimming
/// never collapses an image-only block).
#[must_use]
pub fn image_fallback_lines(
    filename: &str,
    mime: &str,
    width_px: u32,
    height_px: u32,
) -> Vec<String> {
    let label = format!("[image] {filename} · {mime} · {width_px}×{height_px}");
    let inner = label.chars().count();
    vec![
        format!("╭{:─<width$}╮", "", width = inner),
        format!("│{label}│"),
        format!("╰{:─<width$}╯", "", width = inner),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_grid_shape_and_diacritics() {
        let p = KittyPlacement {
            image_id: 0x010203,
            placement_id: Some(2),
            columns: 3,
            rows: 2,
        };
        let grid = encode_kitty_placeholder_grid(p);
        assert_eq!(grid.len(), 2);
        for row in &grid {
            assert_eq!(row.matches(KITTY_PLACEHOLDER).count(), 3);
        }
        // Foreground carries the image id bytes.
        assert!(grid[0].starts_with("\x1b[38;2;1;2;3m\x1b[58:2::0:0:2m"));
        // Rows name their own row+column diacritics.
        assert!(grid[0].contains("\u{10EEEE}\u{305}\u{305}"));
        assert!(grid[1].contains("\u{10EEEE}\u{30d}\u{305}"));
    }

    #[test]
    fn placement_apc_prefixes_row_zero() {
        let p = KittyPlacement {
            image_id: 7,
            placement_id: None,
            columns: 2,
            rows: 2,
        };
        let lines = render_kitty_placeholder_lines(p);
        assert!(lines[0].starts_with("\x1b_Ga=p,U=1,q=2,i=7,c=2,r=2\x1b"));
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn delete_encodings() {
        assert!(encode_kitty_delete(9).contains("a=d,d=I,i=9"));
        assert!(encode_kitty_delete_all().contains("d=A"));
    }

    #[test]
    fn placeholders_fit_bound() {
        assert!(kitty_placeholders_fit(1, KITTY_PLACEHOLDER_MAX_CELLS));
        assert!(!kitty_placeholders_fit(0, 1));
        assert!(!kitty_placeholders_fit(KITTY_PLACEHOLDER_MAX_CELLS + 1, 1));
    }

    #[test]
    fn budget_demotes_beyond_cap_and_purges_evicted() {
        let mut b = ImageBudget::new(2);
        let ids: Vec<u32> = (0..3).map(|_| b.allocate(None)).collect();
        for id in &ids {
            b.observe(*id, None);
        }
        let suppressed = b.end_pass();
        assert_eq!(suppressed, vec![ids[2]]);
        assert_eq!(b.take_purge_ids(), Vec::<u32>::new());
        // Next pass shows only the first image: previously-live ids[1] is purged.
        b.observe(ids[0], None);
        let suppressed = b.end_pass();
        assert!(suppressed.is_empty());
        assert_eq!(b.take_purge_ids(), vec![ids[1]]);
    }

    #[test]
    fn budget_stable_ids_for_keys() {
        let mut b = ImageBudget::default();
        assert_eq!(b.allocate(Some("shot.png")), b.allocate(Some("shot.png")));
        assert_ne!(b.allocate(Some("a")), b.allocate(Some("b")));
    }

    #[test]
    fn fallback_lines_framed() {
        let lines = image_fallback_lines("x.png", "image/png", 10, 20);
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("x.png · image/png · 10×20"));
        assert_eq!(lines[1].chars().count(), lines[0].chars().count());
    }
}
