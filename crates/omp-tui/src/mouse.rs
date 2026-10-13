//! SGR mouse report parsing and routing — port of `pi-tui/src/mouse.ts`
//! (§7 of `tui-spec/core.md`).
//!
//! Mouse tracking is enabled while a fullscreen overlay holds the alternate
//! screen (`\x1b[?1000h\x1b[?1003h\x1b[?1006h`), or — opt-in via `tui.mouse`
//! config — on the normal buffer when no overlay is visible. Consumers
//! hit-test against their own rendered frame: the frame paints from screen
//! row 0, hence [`SgrMouseEvent::row`]/[`SgrMouseEvent::col`] are 0-based for
//! direct indexing into rendered lines.
//!
//! Raw-string parsing ([`parse_sgr_mouse`]) is the byte-accurate TS port; the
//! [`Event`]-based helpers adapt it to the crossterm-driven input loop used by
//! [`crate::controller`].

use crossterm::event::{Event, MouseButton, MouseEvent, MouseEventKind};

/// A decoded SGR mouse report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SgrMouseEvent {
    /// Raw button code (bit 32 = motion, bit 64 = wheel, low bits = button).
    pub button: u8,
    /// 0-based column of the event.
    pub col: u16,
    /// 0-based row of the event.
    pub row: u16,
    /// True for a release report (`m` suffix).
    pub release: bool,
    /// Vertical wheel direction: -1 up, 1 down, `None` when not a vertical
    /// wheel event. Horizontal wheel reports (buttons 66/67) are not a
    /// direction: treating them as one scrolled selectors up and back down at
    /// the end of a gesture.
    pub wheel: Option<i8>,
    /// True when the pointer moved (hover or drag) rather than clicked.
    pub motion: bool,
    /// True for a left-button press (not motion, not release, not wheel).
    pub left_click: bool,
}

/// Decode an SGR mouse report, or `None` when `data` is not one. Hot keypress
/// paths should pre-check `data.starts_with("\x1b[<")`.
#[must_use]
pub fn parse_sgr_mouse(data: &str) -> Option<SgrMouseEvent> {
    parse_sgr_mouse_bytes(data.as_bytes())
}

/// Byte-slice variant of [`parse_sgr_mouse`] for callers holding raw chunks.
#[must_use]
pub fn parse_sgr_mouse_bytes(data: &[u8]) -> Option<SgrMouseEvent> {
    let rest = data.strip_prefix(b"\x1b[<")?;
    let (button, i) = parse_u16(rest)?;
    let rest = rest.get(i..)?.strip_prefix(b";")?;
    let (col, i) = parse_u16(rest)?;
    let rest = rest.get(i..)?.strip_prefix(b";")?;
    let (row, i) = parse_u16(rest)?;
    let suffix = rest.get(i..)?;
    let release = match suffix {
        b"M" => false,
        b"m" => true,
        _ => return None,
    };
    let button = u8::try_from(button).ok()?;
    // Vertical wheel = bit 64 set with bit 1 clear; direction from bit 0
    // (1 = down). Horizontal wheel (66/67, bit 2 set) is excluded.
    let wheel = if button & 64 != 0 && button & 2 == 0 {
        Some(if button & 1 != 0 { 1 } else { -1 })
    } else {
        None
    };
    // Low two button bits zero ⇔ plain left button (clippy::manual bits form).
    let motion = button & 32 != 0 && wheel.is_none();
    let left_click = !release && wheel.is_none() && !motion && button.trailing_zeros() >= 2;
    Some(SgrMouseEvent {
        button,
        col: col.saturating_sub(1),
        row: row.saturating_sub(1),
        release,
        wheel,
        motion,
        left_click,
    })
}

fn parse_u16(bytes: &[u8]) -> Option<(u16, usize)> {
    let mut v: u32 = 0;
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        v = v.checked_mul(10)?.checked_add(u32::from(bytes[i] - b'0'))?;
        i += 1;
    }
    if i == 0 {
        None
    } else {
        Some((u16::try_from(v).ok()?, i))
    }
}

/// Decode a crossterm mouse event into the shared [`SgrMouseEvent`] shape so
/// component code needs only one event type. `None` for non-mouse events.
#[must_use]
pub fn sgr_mouse_of(event: &Event) -> Option<SgrMouseEvent> {
    let (column, row, kind) = match event {
        Event::Mouse(MouseEvent {
            column, row, kind, ..
        }) => (*column, *row, *kind),
        _ => return None,
    };
    let (button, release, motion, wheel) = match kind {
        MouseEventKind::Down(MouseButton::Left) => (0u8, false, false, None),
        MouseEventKind::Down(MouseButton::Right) => (2, false, false, None),
        MouseEventKind::Down(MouseButton::Middle) => (1, false, false, None),
        MouseEventKind::Up(MouseButton::Left) => (0, true, false, None),
        MouseEventKind::Up(MouseButton::Right) => (2, true, false, None),
        MouseEventKind::Up(MouseButton::Middle) => (1, true, false, None),
        MouseEventKind::Drag(MouseButton::Left) => (32, false, true, None),
        MouseEventKind::Drag(MouseButton::Right) => (34, false, true, None),
        MouseEventKind::Drag(MouseButton::Middle) => (33, false, true, None),
        MouseEventKind::Moved => (35, false, true, None),
        MouseEventKind::ScrollDown => (64, false, false, Some(1)),
        MouseEventKind::ScrollUp => (64, false, false, Some(-1)),
        MouseEventKind::ScrollLeft => (66, false, false, None),
        MouseEventKind::ScrollRight => (67, false, false, None),
    };
    // crossterm reports 0-based cell coordinates already, matching the
    // frame-local indexing contract (TS subtracts 1 from the 1-based SGR).
    Some(SgrMouseEvent {
        button,
        col: column,
        row,
        release,
        wheel,
        motion,
        left_click: !release && wheel.is_none() && !motion && button.trailing_zeros() >= 2,
    })
}

/// Handler invoked with a decoded SGR event; returning `false` reports
/// unhandled so callers can fall through to other input handling.
pub type SgrMouseHandler<'a> = dyn FnMut(&SgrMouseEvent) -> bool + 'a;

/// Structural view of a select-list-like target for mouse routing (TS
/// `SelectListMouseTarget`). Declared here rather than importing a component
/// to keep this core module free of component-to-core cycles.
pub trait SelectListMouseTarget {
    /// Scroll the list by one item in the wheel direction.
    fn handle_wheel(&mut self, delta: i8);
    /// Index of the item rendered at frame-local `line`, if any.
    fn hit_test(&self, line: usize) -> Option<usize>;
    /// Update the hover highlight; `None` clears it.
    fn set_hover_index(&mut self, index: Option<usize>);
    /// Activate the item at `index`.
    fn click_item(&mut self, index: usize);
}

/// Route a decoded mouse event against a select-list-like target at the given
/// 0-based frame-local `line`. Centralizes the wheel/hit-test/hover/click
/// pattern. Returns `true` when the event was consumed.
pub fn route_select_list_mouse(
    target: &mut dyn SelectListMouseTarget,
    event: &SgrMouseEvent,
    line: usize,
) -> bool {
    if let Some(delta) = event.wheel {
        target.handle_wheel(delta);
        return true;
    }
    let index = target.hit_test(line);
    if event.motion {
        target.set_hover_index(index);
        return true;
    }
    if event.left_click {
        if let Some(index) = index {
            target.click_item(index);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sgr_click_release_wheel_motion() {
        let click = parse_sgr_mouse("\x1b[<0;13;5M").unwrap();
        assert!(click.left_click);
        assert!(!click.release);
        assert_eq!((click.col, click.row), (12, 4));
        let release = parse_sgr_mouse("\x1b[<0;13;5m").unwrap();
        assert!(release.release);
        assert!(!release.left_click);
        let wheel_up = parse_sgr_mouse("\x1b[<64;1;1M").unwrap();
        assert_eq!(wheel_up.wheel, Some(-1));
        let wheel_down = parse_sgr_mouse("\x1b[<65;1;1M").unwrap();
        assert_eq!(wheel_down.wheel, Some(1));
        // Horizontal wheel (66/67) is not a vertical direction.
        assert_eq!(parse_sgr_mouse("\x1b[<66;1;1M").unwrap().wheel, None);
        let motion = parse_sgr_mouse("\x1b[<32;3;3M").unwrap();
        assert!(motion.motion);
        assert!(!motion.left_click);
        assert!(parse_sgr_mouse("\x1b[<0;13").is_none());
        assert!(parse_sgr_mouse("hello").is_none());
    }

    struct FakeList {
        items: Vec<&'static str>,
        hover: Option<usize>,
        clicked: Option<usize>,
        scrolled: i32,
    }

    impl SelectListMouseTarget for FakeList {
        fn handle_wheel(&mut self, delta: i8) {
            self.scrolled += i32::from(delta);
        }
        fn hit_test(&self, line: usize) -> Option<usize> {
            if line < self.items.len() {
                Some(line)
            } else {
                None
            }
        }
        fn set_hover_index(&mut self, index: Option<usize>) {
            self.hover = index;
        }
        fn click_item(&mut self, index: usize) {
            self.clicked = Some(index);
        }
    }

    #[test]
    fn routes_wheel_hover_click() {
        let mut list = FakeList {
            items: vec!["a", "b"],
            hover: None,
            clicked: None,
            scrolled: 0,
        };
        let up = parse_sgr_mouse("\x1b[<64;1;1M").unwrap();
        assert!(route_select_list_mouse(&mut list, &up, 0));
        assert_eq!(list.scrolled, -1);
        let move_ = parse_sgr_mouse("\x1b[<35;1;2M").unwrap();
        assert!(route_select_list_mouse(&mut list, &move_, 1));
        assert_eq!(list.hover, Some(1));
        let click = parse_sgr_mouse("\x1b[<0;1;2M").unwrap();
        assert!(route_select_list_mouse(&mut list, &click, 1));
        assert_eq!(list.clicked, Some(1));
        // Click on empty space is not consumed.
        assert!(!route_select_list_mouse(&mut list, &click, 9));
    }

    #[test]
    fn adapts_crossterm_events() {
        let ev = Event::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 3,
            row: 7,
            modifiers: crossterm::event::KeyModifiers::empty(),
        });
        let m = sgr_mouse_of(&ev).unwrap();
        assert_eq!(m.wheel, Some(-1));
        assert_eq!((m.col, m.row), (3, 7));
        assert!(sgr_mouse_of(&Event::Resize(1, 1)).is_none());
    }
}
