//! Query extraction and the prose gate for ghost-text word completion — the
//! port of `wordCompletionQuery` (`pi-tui/src/prompt/word-completion.ts`) over
//! `isProseWord` / `ProseSource` (`prose-gate.ts`).
//!
//! Columns are **byte** offsets (the [`crate::editor`] cursor convention); the
//! TS length caps count UTF-16 units and so do the ones here.

use pi_predict::prose::{is_js_space, is_prose_word, is_word_char, mask_non_prose, utf16_len};

/// Context sent with each query: the editor text before the word, capped to
/// its last [`BEFORE_LIMIT`] UTF-16 units.
pub const BEFORE_LIMIT: usize = 2_000;
/// Buffers above this many UTF-16 units get no prose assistance at all.
const MAX_PROSE_BUFFER_LENGTH: usize = 20_000;
/// Lines above this many UTF-16 units get no prose assistance at all.
const MAX_PROSE_LINE_LENGTH: usize = 1_000;

/// The prose word ending at the cursor and the editor text before it: what an
/// engine is asked to complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordQuery {
    /// Editor text before the word (recent tail, may span lines).
    pub before: String,
    /// Letters of the word typed so far (`[\p{L}\p{M}']+`).
    pub prefix: String,
}

impl WordQuery {
    /// Cache key: prefix and context.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}\u{0}{}", self.prefix, self.before)
    }
}

/// Editor text before byte offset `start` of `lines[cursor_line]`, truncated to
/// its last [`BEFORE_LIMIT`] UTF-16 units (never splitting a surrogate pair).
fn text_before(lines: &[String], cursor_line: usize, start: usize) -> String {
    let mut text = lines[cursor_line][..start].to_owned();
    let mut units = utf16_len(&text);
    for line in (0..cursor_line).rev() {
        if units >= BEFORE_LIMIT {
            break;
        }
        text = format!("{}\n{text}", lines[line]);
        units = utf16_len(&text);
    }
    if units <= BEFORE_LIMIT {
        return text;
    }
    let mut kept = 0usize;
    let mut from = text.len();
    for (at, c) in text.char_indices().rev() {
        kept += c.len_utf16();
        if kept > BEFORE_LIMIT {
            break;
        }
        from = at;
    }
    text[from..].to_owned()
}

/// The query for the prose word ending at the cursor, or `None` where no ghost
/// text applies (mid-word, no word, code, paths, commands).
#[must_use]
pub fn word_completion_query(
    lines: &[String],
    cursor_line: usize,
    cursor_col: usize,
) -> Option<WordQuery> {
    let line = lines.get(cursor_line)?;
    if cursor_col > line.len() || !line.is_char_boundary(cursor_col) {
        return None;
    }
    if line[cursor_col..].chars().next().is_some_and(is_word_char) {
        return None;
    }
    let head = &line[..cursor_col];
    // Single letters are asked too (`figure o|ut`); engines own their minimum.
    let start = head
        .char_indices()
        .rev()
        .take_while(|&(_, c)| is_word_char(c))
        .last()
        .map(|(at, _)| at)?;
    if !is_prose_word(line, &mask_non_prose(line), start, cursor_col) {
        return None;
    }
    if !is_prose_in_buffer(lines, cursor_line, start, cursor_col) {
        return None;
    }
    Some(WordQuery {
        before: text_before(lines, cursor_line, start),
        prefix: head[start..].to_owned(),
    })
}

/// `ProseSource.isProse`: the word is prose in the context of the whole buffer
/// (fenced code spans lines) and the buffer/line caps hold.
fn is_prose_in_buffer(lines: &[String], cursor_line: usize, start: usize, end: usize) -> bool {
    let units: usize = lines.iter().map(|l| utf16_len(l)).sum::<usize>() + lines.len() - 1;
    if units > MAX_PROSE_BUFFER_LENGTH || utf16_len(&lines[cursor_line]) > MAX_PROSE_LINE_LENGTH {
        return false;
    }
    let text = lines.join("\n");
    let mask = mask_non_prose(&text);
    let offset: usize = lines[..cursor_line].iter().map(|l| l.len() + 1).sum();
    !mask[offset + start..offset + end]
        .trim_matches(is_js_space)
        .is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(text: &str) -> Option<WordQuery> {
        let lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
        let line = lines.len() - 1;
        let col = lines[line].len();
        word_completion_query(&lines, line, col)
    }

    #[test]
    fn extracts_prefix_and_context() {
        let query = q("please refactor the par").expect("prose word");
        assert_eq!(query.prefix, "par");
        assert_eq!(query.before, "please refactor the ");
    }

    #[test]
    fn single_letters_and_apostrophes_are_asked() {
        assert_eq!(q("figure o").unwrap().prefix, "o");
        assert_eq!(q("it don't").unwrap().prefix, "don't");
    }

    #[test]
    fn no_word_or_mid_word_means_no_query() {
        assert_eq!(q("hello "), None);
        assert_eq!(q(""), None);
        let lines = vec!["hello world".to_owned()];
        assert_eq!(
            word_completion_query(&lines, 0, 8),
            None,
            "cursor inside a word"
        );
    }

    #[test]
    fn code_paths_and_commands_are_not_prose() {
        assert_eq!(q("open src/ma"), None, "path token");
        assert_eq!(q("call fooBa"), None, "camelCase token");
        assert_eq!(q("snake_ca"), None, "identifier");
        assert_eq!(q("v2ab"), None, "digits");
        assert_eq!(q("/mod"), None, "slash command line");
        assert_eq!(q("-> sta"), None, "queue shorthand");
        assert_eq!(q("```\nlet fo"), None, "fenced code");
        // An unmatched backtick is literal text; a matched span is code even
        // when it closes on a later line.
        assert!(q("see `inl").is_some(), "unmatched backtick");
        let lines = vec!["see `inl".to_owned(), "more` text".to_owned()];
        assert_eq!(
            word_completion_query(&lines, 0, 8),
            None,
            "inline code span"
        );
    }

    #[test]
    fn context_spans_lines_and_is_capped() {
        let query = q("first line\nsecond wo").unwrap();
        assert_eq!(query.before, "first line\nsecond ");
        let long = format!("{}\nword", "x ".repeat(3_000));
        let query = q(&long).unwrap();
        assert!(utf16_len(&query.before) <= BEFORE_LIMIT);
        assert!(query.before.ends_with("x \n"));
    }

    #[test]
    fn oversized_buffers_get_nothing() {
        let big = format!("{}\nwor", "a b\n".repeat(6_000));
        assert_eq!(q(&big), None);
        let wide = format!("{} wor", "word ".repeat(250));
        assert_eq!(q(&wide), None, "line over 1000 units");
    }
}
