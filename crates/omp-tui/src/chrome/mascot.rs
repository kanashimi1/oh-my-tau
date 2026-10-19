//! Omi's one-line kaomoji moods for the clean working indicator.

/// What Omi is doing (working indicator kaomoji).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    /// Model is thinking / streaming.
    Thinking,
    /// A tool is running.
    Working,
    /// Turn finished.
    Happy,
    /// Turn failed.
    Sad,
}

/// One-line face for `mood`; `tick` animates (blink / glance).
#[must_use]
pub fn face(mood: Mood, tick: u64) -> &'static str {
    match mood {
        Mood::Thinking => ["(•_• )", "( •_•)", "(•_• )", "(-_- )"][(tick % 4) as usize],
        Mood::Working => ["(•ᴗ•)~", "(•ᴗ•)~", "(-ᴗ-)~"][(tick % 3) as usize],
        Mood::Happy => "(◕ᴗ◕)✧",
        Mood::Sad => "(╥_╥)",
    }
}
