//! Pure event state machine for the global dictation button.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event { Press, Release }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action { Start, DiscardAndReinjectClick, FinishAndTranscribe }

#[derive(Default)]
pub struct ButtonState { pressed_at_ms: Option<u64> }

impl ButtonState {
    /// Ignore injected events and duplicate transitions. Timestamps must come
    /// from a monotonic clock, in milliseconds.
    pub fn event(&mut self, event: Event, timestamp_ms: u64, injected: bool, hold_ms: u64) -> Option<Action> {
        if injected { return None; }
        match event {
            Event::Press if self.pressed_at_ms.is_none() => {
                self.pressed_at_ms = Some(timestamp_ms);
                Some(Action::Start)
            }
            Event::Release => self.pressed_at_ms.take().map(|start| {
                if timestamp_ms.saturating_sub(start) < hold_ms {
                    Action::DiscardAndReinjectClick
                } else {
                    Action::FinishAndTranscribe
                }
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn quick_tap() { let mut s = ButtonState::default(); assert_eq!(s.event(Event::Press, 0, false, 250), Some(Action::Start)); assert_eq!(s.event(Event::Release, 249, false, 250), Some(Action::DiscardAndReinjectClick)); }
    #[test] fn exact_threshold() { let mut s = ButtonState::default(); s.event(Event::Press, 10, false, 250); assert_eq!(s.event(Event::Release, 260, false, 250), Some(Action::FinishAndTranscribe)); }
    #[test] fn long_hold() { let mut s = ButtonState::default(); s.event(Event::Press, 10, false, 250); assert_eq!(s.event(Event::Release, 2000, false, 250), Some(Action::FinishAndTranscribe)); }
    #[test] fn double_tap() { let mut s = ButtonState::default(); for t in [0, 100] { assert_eq!(s.event(Event::Press, t, false, 250), Some(Action::Start)); assert_eq!(s.event(Event::Release, t+20, false, 250), Some(Action::DiscardAndReinjectClick)); } }
    #[test] fn release_without_press() { assert_eq!(ButtonState::default().event(Event::Release, 1, false, 250), None); }
    #[test] fn injected_events_are_ignored() { let mut s = ButtonState::default(); assert_eq!(s.event(Event::Press, 0, true, 250), None); assert_eq!(s.event(Event::Release, 1, false, 250), None); s.event(Event::Press, 10, false, 250); assert_eq!(s.event(Event::Release, 11, true, 250), None); assert_eq!(s.event(Event::Release, 260, false, 250), Some(Action::FinishAndTranscribe)); }
}
