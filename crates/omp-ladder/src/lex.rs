//! Tokenisation built on `rustc_lexer` (the lexer extracted from the
//! rustc frontend). Comments — including nested block comments —
//! string literals, raw strings, and char literals are each emitted as
//! a single token, so consumers never need to reason about delimiters
//! inside literal or comment text.
//!
//! `rustc_lexer` caps token length at `u8::MAX` bytes and splits
//! longer tokens into adjacent fragments of the same kind; [`lex`]
//! re-merges those fragments so every token's span covers the full
//! source text of the construct.

use ra_ap_rustc_lexer as rustc_lexer;
use serde::{Deserialize, Serialize};

/// Half-open byte span `[start, end)` into the source string.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Length of the span in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the span is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Token kind. Punctuation is kept at rustc granularity (one variant
/// per source character) so `::`, `->`, `..=` etc. are *not* fused —
/// multi-character operators are reassembled by consumers from
/// adjacent tokens, which keeps spans exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tk {
    Ident,
    RawIdent,
    Lifetime,
    Int,
    Float,
    Char,
    Str,
    StrRaw,
    /// Doc comment (`///`, `//!`, `/** */`, `/*! */`).
    DocComment,
    /// Non-doc comment (line or nested block).
    Comment,
    Whitespace,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    Semi,
    Colon,
    Comma,
    Dot,
    Pound,
    Exclaim,
    Question,
    Dollar,
    Eq,
    Lt,
    Gt,
    Minus,
    And,
    Or,
    Plus,
    Star,
    Slash,
    Caret,
    Percent,
    At,
    Unknown,
}

impl Tk {
    /// Trivia that never carries structure: whitespace and plain
    /// comments. Doc comments are *not* trivia here — they attach to
    /// the item that follows them.
    #[must_use]
    pub fn is_blank(self) -> bool {
        matches!(self, Tk::Whitespace | Tk::Comment)
    }

    /// Whether adjacent same-kind tokens of this kind may be merged
    /// (the `rustc_lexer` long-token split). Never true for
    /// punctuation, where adjacency is meaningful (`>>`, `::`, `..`).
    fn mergeable(self) -> bool {
        matches!(
            self,
            Tk::Ident
                | Tk::RawIdent
                | Tk::Lifetime
                | Tk::Int
                | Tk::Float
                | Tk::Char
                | Tk::Str
                | Tk::StrRaw
                | Tk::DocComment
                | Tk::Comment
                | Tk::Whitespace
        )
    }
}

/// A single token with its byte span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Token {
    pub kind: Tk,
    pub span: Span,
}

impl Token {
    /// The token's source text.
    #[must_use]
    pub fn text<'s>(&self, src: &'s str) -> &'s str {
        src.get(self.span.start..self.span.end).unwrap_or("")
    }
}

fn map_kind(k: rustc_lexer::TokenKind) -> Tk {
    use rustc_lexer::{LiteralKind as L, TokenKind as R};
    match k {
        R::Ident | R::InvalidIdent | R::UnknownPrefix | R::ForcedKeywordIdent => Tk::Ident,
        R::RawIdent => Tk::RawIdent,
        R::Lifetime { .. } | R::RawLifetime | R::UnknownPrefixLifetime => Tk::Lifetime,
        R::Literal { kind, .. } => match kind {
            L::Int { .. } => Tk::Int,
            L::Float { .. } => Tk::Float,
            L::Char { .. } | L::Byte { .. } => Tk::Char,
            L::Str { .. } | L::ByteStr { .. } | L::CStr { .. } => Tk::Str,
            L::RawStr { .. } | L::RawByteStr { .. } | L::RawCStr { .. } => Tk::StrRaw,
        },
        R::LineComment { doc_style } | R::BlockComment { doc_style, .. } => {
            if doc_style.is_some() {
                Tk::DocComment
            } else {
                Tk::Comment
            }
        }
        // `---...---` file frontmatter: treat as inert comment text.
        R::Frontmatter { .. } => Tk::Comment,
        R::Whitespace => Tk::Whitespace,
        R::OpenParen => Tk::OpenParen,
        R::CloseParen => Tk::CloseParen,
        R::OpenBrace => Tk::OpenBrace,
        R::CloseBrace => Tk::CloseBrace,
        R::OpenBracket => Tk::OpenBracket,
        R::CloseBracket => Tk::CloseBracket,
        R::Semi => Tk::Semi,
        R::Colon => Tk::Colon,
        R::Comma => Tk::Comma,
        R::Dot => Tk::Dot,
        R::Pound => Tk::Pound,
        R::Bang => Tk::Exclaim,
        R::Question => Tk::Question,
        R::Dollar => Tk::Dollar,
        R::Eq => Tk::Eq,
        R::Lt => Tk::Lt,
        R::Gt => Tk::Gt,
        R::Minus => Tk::Minus,
        R::And => Tk::And,
        R::Or => Tk::Or,
        R::Plus => Tk::Plus,
        R::Star => Tk::Star,
        R::Slash => Tk::Slash,
        R::Caret => Tk::Caret,
        R::Percent => Tk::Percent,
        R::At => Tk::At,
        R::Tilde | R::GuardedStrPrefix | R::Unknown | R::Eof => Tk::Unknown,
    }
}

/// Lex a whole source file into a token vector whose spans tile
/// `[0, src.len())` exactly: concatenating every token's text
/// reproduces the input byte-for-byte.
#[must_use]
pub fn lex(src: &str) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let mut pos = 0usize;
    let iter = rustc_lexer::tokenize(src, rustc_lexer::FrontmatterAllowed::Yes);
    for raw in iter {
        let len = usize::try_from(raw.len).unwrap_or(usize::MAX);
        let end = pos + len;
        let kind = map_kind(raw.kind);
        if let Some(last) = out.last_mut() {
            // Re-merge fragments of a token longer than 255 bytes.
            if last.kind == kind && kind.mergeable() && last.span.end == pos {
                last.span.end = end;
                pos = end;
                continue;
            }
        }
        out.push(Token {
            kind,
            span: Span { start: pos, end },
        });
        pos = end;
    }
    out
}
