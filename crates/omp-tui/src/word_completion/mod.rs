//! Ghost-text word completion for the composer — the port of the TS
//! `spelling.autocomplete` feature (`pi-tui/src/prompt/word-completion.ts`).
//!
//! While a prose word is typed at the end of a line, [`WordCompletion`] asks
//! the engine ([`host::WordHost`], the vendored `pi-predict` n-gram engine
//! learning from the prompt history) for the rest of the word; the editor paints
//! it dim after the cursor. Tab accepts it plus a space, Right accepts it as
//! typed, typing past it reports a rejection.
//!
//! Unlike the TS client the engine runs in-process, so a request is a cache
//! miss answered synchronously (a few microseconds) rather than a single-flight
//! daemon round trip; while the engine is still loading nothing is shown and
//! nothing is cached.

pub mod host;
pub mod query;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use host::{Answer, WordHost};
pub use query::{word_completion_query, WordQuery};

/// Answers kept per editor before the cache is dropped (TS `CACHE_LIMIT`).
const CACHE_LIMIT: usize = 256;

#[derive(Debug, Default, Clone)]
struct State {
    cache: HashMap<String, Option<String>>,
    /// Last non-empty suggestion shown, for projection while typing through it.
    last_shown: Option<(String, String)>,
}

/// Ghost-text provider over one engine host.
#[derive(Debug, Clone)]
pub struct WordCompletion {
    host: Arc<WordHost>,
    state: RefCell<State>,
}

impl WordCompletion {
    #[must_use]
    pub fn new(host: Arc<WordHost>) -> Self {
        Self {
            host,
            state: RefCell::new(State::default()),
        }
    }

    /// The engine host behind this provider.
    #[must_use]
    pub fn host(&self) -> &Arc<WordHost> {
        &self.host
    }

    /// Ghost-text suffix for the prose word ending at the cursor, or `None`.
    /// Only meaningful at the end of a line (the caller checks).
    pub fn ghost(&self, lines: &[String], line: usize, col: usize) -> Option<String> {
        let query = word_completion_query(lines, line, col)?;
        let key = query.key();
        let mut state = self.state.borrow_mut();
        let cached = state.cache.get(&key).cloned();
        let answer = match cached {
            Some(answer) => Some(answer),
            None => match self.host.complete(&query.before, &query.prefix) {
                Answer::Pending => None,
                Answer::Ready(suffix) => {
                    if state.cache.len() >= CACHE_LIMIT {
                        state.cache.clear();
                    }
                    state.cache.insert(key, suffix.clone());
                    Some(suffix)
                }
            },
        };
        // Typing through a shown ghost without Tab can answer nothing (the word
        // fell under the show threshold); keep that ghost until the engine
        // offers something else.
        let suffix = answer.flatten().or_else(|| project(&state, &query));
        if let Some(suffix) = &suffix {
            state.last_shown = Some((query.before.clone(), format!("{}{suffix}", query.prefix)));
        }
        suffix
    }

    /// Forward accept (Tab/→) or typed-past feedback for the suggestion shown
    /// at the cursor.
    pub fn feedback(
        &self,
        lines: &[String],
        line: usize,
        col: usize,
        suggestion: &str,
        accepted: bool,
    ) {
        let Some(query) = word_completion_query(lines, line, col) else {
            return;
        };
        self.host
            .feedback(&query.before, &query.prefix, suggestion, accepted);
        if !accepted {
            self.state.borrow_mut().last_shown = None;
        }
    }

    /// Learn a submitted prompt.
    pub fn observe(&self, prompt: &str) {
        self.host.observe(prompt);
    }
}

/// The previous suggestion's remainder while its word still extends what was
/// typed (`#project`).
fn project(state: &State, query: &WordQuery) -> Option<String> {
    let (before, word) = state.last_shown.as_ref()?;
    if *before != query.before || word.chars().count() <= query.prefix.chars().count() {
        return None;
    }
    let (lower_word, lower_prefix) = (word.to_lowercase(), query.prefix.to_lowercase());
    if !lower_word.starts_with(&lower_prefix) {
        return None;
    }
    // Slice the original word by the prefix's character count: case folding
    // can change byte lengths.
    let skip = word
        .char_indices()
        .nth(query.prefix.chars().count())
        .map_or(word.len(), |(at, _)| at);
    Some(word[skip..].to_owned())
}

#[cfg(test)]
pub(crate) mod test_support {
    use pi_predict::{Predictor, Query, Suggestion};

    /// Engine answering a fixed word list by prefix, recording feedback.
    pub struct Words(
        pub Vec<&'static str>,
        pub std::sync::Arc<parking_lot::Mutex<Vec<String>>>,
    );

    impl Predictor for Words {
        fn complete(&mut self, query: &Query<'_>) -> Option<Suggestion> {
            let prefix = query.prefix.to_lowercase();
            let word = self
                .0
                .iter()
                .find(|w| w.starts_with(&prefix) && w.len() > prefix.len())?;
            Some(Suggestion {
                suffix: word[prefix.len()..].to_owned(),
                confidence: 1.0,
            })
        }
        fn observe(&mut self, prompt: &str) {
            self.1.lock().push(format!("observe:{prompt}"));
        }
        fn feedback(&mut self, query: &Query<'_>, suggestion: &str, accepted: bool) {
            self.1
                .lock()
                .push(format!("feedback:{}:{suggestion}:{accepted}", query.prefix));
        }
        fn persist(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    pub fn host(
        words: Vec<&'static str>,
    ) -> (
        std::sync::Arc<super::host::WordHost>,
        std::sync::Arc<parking_lot::Mutex<Vec<String>>>,
    ) {
        let log = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
        (
            super::host::WordHost::with_engine(Box::new(Words(words, log.clone()))),
            log,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::host;
    use super::*;

    fn lines(text: &str) -> Vec<String> {
        text.split('\n').map(str::to_owned).collect()
    }

    fn ghost(wc: &WordCompletion, text: &str) -> Option<String> {
        let l = lines(text);
        let last = l.len() - 1;
        wc.ghost(&l, last, l[last].len())
    }

    #[test]
    fn suggests_the_rest_of_the_word() {
        let (h, _) = host(vec!["refactor"]);
        let wc = WordCompletion::new(h);
        assert_eq!(ghost(&wc, "please refac").as_deref(), Some("tor"));
        assert_eq!(ghost(&wc, "please REFAC").as_deref(), Some("tor"));
        assert_eq!(ghost(&wc, "please xyz"), None);
        assert_eq!(ghost(&wc, "please "), None, "no word at the cursor");
    }

    #[test]
    fn code_is_never_completed() {
        let (h, _) = host(vec!["refactor"]);
        let wc = WordCompletion::new(h);
        assert_eq!(ghost(&wc, "open src/refac"), None);
        let l = vec!["see `refac".to_owned(), "x` text".to_owned()];
        assert_eq!(
            wc.ghost(&l, 0, l[0].len()),
            None,
            "inside an inline code span"
        );
    }

    #[test]
    fn ghost_is_projected_while_typing_through_it() {
        // The engine stops offering the word once it is mostly typed; the
        // shown ghost stays until the engine says something else.
        let (h, _) = host(vec!["refactor"]);
        let wc = WordCompletion::new(h);
        assert_eq!(ghost(&wc, "do refa").as_deref(), Some("ctor"));
        assert_eq!(ghost(&wc, "do refact").as_deref(), Some("or"));
        // "refactor" fully typed: the engine has nothing to add, nothing projects.
        assert_eq!(ghost(&wc, "do refactor"), None);
        // A diverging word drops the projection.
        assert_eq!(ghost(&wc, "do refx"), None);
    }

    #[test]
    fn projection_survives_an_engine_that_answers_nothing() {
        let (h, _) = host(vec!["refactor"]);
        let wc = WordCompletion::new(h);
        assert_eq!(ghost(&wc, "do refa").as_deref(), Some("ctor"));
        wc.state.borrow_mut().cache.insert(
            word_completion_query(&lines("do refac"), 0, 8)
                .unwrap()
                .key(),
            None,
        );
        assert_eq!(ghost(&wc, "do refac").as_deref(), Some("tor"));
    }

    #[test]
    fn feedback_reaches_the_engine_and_rejection_clears_the_projection() {
        let (h, log) = host(vec!["refactor"]);
        let wc = WordCompletion::new(h);
        assert_eq!(ghost(&wc, "do refa").as_deref(), Some("ctor"));
        let l = lines("do refa");
        wc.feedback(&l, 0, 7, "ctor", true);
        wc.feedback(&l, 0, 7, "ctor", false);
        assert_eq!(
            *log.lock(),
            ["feedback:refa:ctor:true", "feedback:refa:ctor:false"]
        );
        assert!(wc.state.borrow().last_shown.is_none());
    }
}

#[cfg(test)]
mod editor_tests {
    use super::test_support::host;
    use crate::editor::Editor;
    use crate::keys::{KeyCode, KeyEvent, KeyModifiers};

    fn editor(
        words: Vec<&'static str>,
    ) -> (Editor, std::sync::Arc<parking_lot::Mutex<Vec<String>>>) {
        let (h, log) = host(words);
        let mut e = Editor::new();
        e.set_word_completion(Some(h));
        (e, log)
    }

    fn type_str(e: &mut Editor, s: &str) {
        for c in s.chars() {
            let _ = e.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    fn right(e: &mut Editor) {
        let _ = e.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    }

    #[test]
    fn ghost_only_at_line_end_and_when_enabled() {
        let (mut e, _) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        assert_eq!(e.word_ghost().as_deref(), Some("tor"));
        let _ = e.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(e.word_ghost(), None, "cursor off the end of the line");
        e.set_word_completion(None);
        assert_eq!(e.word_ghost(), None, "off disables ghost text");
    }

    #[test]
    fn right_accepts_without_a_space() {
        let (mut e, log) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        right(&mut e);
        assert_eq!(e.text(), "do refactor");
        assert_eq!(*log.lock(), ["feedback:refac:tor:true"]);
        // Nothing to accept now: Right is plain movement.
        right(&mut e);
        assert_eq!(e.text(), "do refactor");
    }

    #[test]
    fn tab_accepts_with_a_provisional_space() {
        let (mut e, _) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        assert!(e.accept_word_completion(true));
        assert_eq!(e.text(), "do refactor ");
        // A typed space is swallowed ...
        type_str(&mut e, " ");
        assert_eq!(e.text(), "do refactor ");
        // ... closing punctuation replaces it ...
        let (mut e2, _) = editor(vec!["refactor"]);
        type_str(&mut e2, "do refac");
        assert!(e2.accept_word_completion(true));
        type_str(&mut e2, ".");
        assert_eq!(e2.text(), "do refactor.");
        // ... and a letter keeps it.
        let (mut e3, _) = editor(vec!["refactor"]);
        type_str(&mut e3, "do refac");
        assert!(e3.accept_word_completion(true));
        type_str(&mut e3, "it");
        assert_eq!(e3.text(), "do refactor it");
    }

    #[test]
    fn newline_drops_the_provisional_space() {
        let (mut e, _) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        assert!(e.accept_word_completion(true));
        e.insert_newline();
        assert_eq!(e.text(), "do refactor\n");
    }

    #[test]
    fn typing_past_the_ghost_reports_a_rejection() {
        let (mut e, log) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        type_str(&mut e, "t"); // still on the ghost: no feedback
        assert!(log.lock().is_empty());
        type_str(&mut e, "x"); // diverges from "or"
        assert_eq!(*log.lock(), ["feedback:refact:or:false"]);
    }

    #[test]
    fn accept_is_one_undo_step() {
        let (mut e, _) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        assert!(e.accept_word_completion(true));
        e.apply_undo();
        assert_eq!(e.text(), "do refac");
    }

    #[test]
    fn render_paints_dim_ghost_after_the_cursor_cell() {
        use crate::component::Component;
        use crate::theme::Theme;
        use ratatui::{backend::TestBackend, layout::Rect, Terminal};
        let (mut e, _) = editor(vec!["refactor"]);
        type_str(&mut e, "do refac");
        let mut term = Terminal::new(TestBackend::new(40, 3)).unwrap();
        let theme = Theme::default();
        term.draw(|f| e.render(f, Rect::new(0, 0, 40, 3), &theme))
            .unwrap();
        let row: String = (0..40)
            .map(|x| term.backend().buffer()[(x, 1)].symbol().to_owned())
            .collect();
        // Band gutter, text, then the ghost suffix starting on the cursor cell.
        assert!(row.contains("do refactor"), "{row:?}");
        let cell = &term.backend().buffer()[(
            u16::try_from(row.chars().position(|c| c == 't').unwrap()).unwrap(),
            1,
        )];
        assert!(cell.modifier.contains(ratatui::style::Modifier::DIM));
    }
}
