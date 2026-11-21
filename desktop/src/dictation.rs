//! Pure event state machine for the global dictation button.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Press,
    Release,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    DiscardAndReinjectClick,
    FinishAndTranscribe,
}

#[derive(Default)]
pub struct ButtonState {
    pressed_at_ms: Option<u64>,
}

impl ButtonState {
    /// Ignore injected events and duplicate transitions. Timestamps must come
    /// from a monotonic clock, in milliseconds.
    pub fn event(
        &mut self,
        event: Event,
        timestamp_ms: u64,
        injected: bool,
        hold_ms: u64,
    ) -> Option<Action> {
        if injected {
            return None;
        }
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

/// Two-pass preview: only the common whole-word prefix becomes stable.
pub(crate) fn stable_preview(previous: &str, current: &str) -> (String, String) {
    let old: Vec<&str> = previous.split_whitespace().collect();
    let new: Vec<&str> = current.split_whitespace().collect();
    let common = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    (new[..common].join(" "), new[common..].join(" "))
}

/// Allocation-free linear resampler for the first input channel.
pub(crate) struct LinearResampler {
    ratio: f64,
    previous: Option<f32>,
    next_position: f64,
}

impl LinearResampler {
    pub(crate) fn new(source_rate: u32, target_rate: u32) -> Self {
        let ratio = source_rate as f64 / target_rate as f64;
        Self {
            ratio,
            previous: None,
            next_position: ratio,
        }
    }

    pub(crate) fn push(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &sample in input {
            let Some(previous) = self.previous.replace(sample) else {
                continue;
            };
            while self.next_position <= 1.0 {
                output.push(previous + (sample - previous) * self.next_position as f32);
                self.next_position += self.ratio;
            }
            self.next_position -= 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quick_tap() {
        let mut s = ButtonState::default();
        assert_eq!(s.event(Event::Press, 0, false, 250), Some(Action::Start));
        assert_eq!(
            s.event(Event::Release, 249, false, 250),
            Some(Action::DiscardAndReinjectClick)
        );
    }
    #[test]
    fn exact_threshold() {
        let mut s = ButtonState::default();
        s.event(Event::Press, 10, false, 250);
        assert_eq!(
            s.event(Event::Release, 260, false, 250),
            Some(Action::FinishAndTranscribe)
        );
    }
    #[test]
    fn long_hold() {
        let mut s = ButtonState::default();
        s.event(Event::Press, 10, false, 250);
        assert_eq!(
            s.event(Event::Release, 2000, false, 250),
            Some(Action::FinishAndTranscribe)
        );
    }
    #[test]
    fn double_tap() {
        let mut s = ButtonState::default();
        for t in [0, 100] {
            assert_eq!(s.event(Event::Press, t, false, 250), Some(Action::Start));
            assert_eq!(
                s.event(Event::Release, t + 20, false, 250),
                Some(Action::DiscardAndReinjectClick)
            );
        }
    }
    #[test]
    fn release_without_press() {
        assert_eq!(
            ButtonState::default().event(Event::Release, 1, false, 250),
            None
        );
    }
    #[test]
    fn injected_events_are_ignored() {
        let mut s = ButtonState::default();
        assert_eq!(s.event(Event::Press, 0, true, 250), None);
        assert_eq!(s.event(Event::Release, 1, false, 250), None);
        s.event(Event::Press, 10, false, 250);
        assert_eq!(s.event(Event::Release, 11, true, 250), None);
        assert_eq!(
            s.event(Event::Release, 260, false, 250),
            Some(Action::FinishAndTranscribe)
        );
    }

    #[test]
    fn preview_stabilizes_only_common_words() {
        assert_eq!(
            stable_preview("раз два три", "раз два четыре пять"),
            ("раз два".into(), "четыре пять".into())
        );
        assert_eq!(
            stable_preview("раз два", "совсем другой текст"),
            (String::new(), "совсем другой текст".into())
        );
    }

    #[test]
    fn resampler_produces_expected_rate() {
        let input = vec![0.25; 48_000];
        let mut output = Vec::new();
        LinearResampler::new(48_000, 16_000).push(&input, &mut output);
        assert!((output.len() as isize - 16_000).abs() <= 1);
        assert!(output.iter().all(|&sample| (sample - 0.25).abs() < 1e-6));
    }
}
