//! Concrete-syntax top-level pass over the token stream.
//!
//! Produces a flat list of top-level nodes with exact byte spans.
//! Items' brace bodies are recorded as opaque spans; macro invocation
//! and `macro_rules!` bodies are *never* descended into, so tokens
//! inside them (including item-looking tokens) cannot leak into the
//! semantic projection. Delimiter balancing is done on tokens, so
//! braces inside strings, raw strings, chars and comments are inert.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::cfg::{parse_meta, Meta};
use crate::hash::{hash_str, ContentHash};
use crate::lex::{lex, Span, Tk, Token};

/// Kind of a top-level node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopKind {
    Fn,
    Struct,
    Enum,
    Union,
    Impl,
    Trait,
    Mod,
    Use,
    TypeAlias,
    Const,
    Static,
    ExternCrate,
    ExternBlock,
    /// `macro_rules! name { ... }` or `macro name(...) { ... }`.
    MacroDef,
    /// `path::to::macro!(...)` / `{...}` / `[...]` — opaque.
    MacroCall,
    /// A `#![...]` inner attribute at file scope.
    InnerAttr,
    /// Anything unrecognised; exactly one token is consumed so the
    /// pass always makes progress.
    Unknown,
}

/// Item visibility as written syntactically. `pub(crate)`-style
/// restrictions are recorded as [`Visibility::PubRestricted`] without
/// resolving the restriction path (syntax-derived, no name
/// resolution); only [`Visibility::Pub`] is an unconditional public
/// interface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Visibility {
    /// No `pub` prefix: visible only in the defining module.
    #[default]
    Inherited,
    /// Unrestricted `pub`.
    Pub,
    /// `pub(...)`, `crate fn`, etc.: restricted visibility.
    PubRestricted,
}

/// One parsed attribute (`#[...]` or `#![...]`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attr {
    pub span: Span,
    pub is_inner: bool,
    /// Parsed meta (`cfg(...)`, `derive(...)`, ...), when it has a
    /// well-formed meta shape. Evaluation against a cfg context is a
    /// semantic-layer concern and happens in [`crate::item`].
    pub meta: Option<Meta>,
}

/// A top-level node. `span` covers leading doc comments and outer
/// attributes through the end of the item (including the trailing
/// `;` where one terminates the item).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopNode {
    pub kind: TopKind,
    pub span: Span,
    /// Brace-delimited body including the braces, for braced items
    /// and macro bodies.
    pub body_span: Option<Span>,
    /// Item name where one exists syntactically (`impl` has none).
    pub name: Option<String>,
    pub attrs: Vec<Attr>,
    /// Full path of a macro invocation, e.g. `foo::bar`.
    pub macro_path: Option<String>,
    /// Syntactic visibility of the item.
    pub visibility: Visibility,
    /// The parser's angle-bracket heuristic hit an ambiguous
    /// signature; the span is best-effort and the item must be
    /// treated as opaque (fail closed), never as a trusted
    /// item/body span.
    #[serde(default)]
    pub ambiguous: bool,
}

/// The parsed concrete syntax of one source file.
#[derive(Clone, Debug)]
pub struct Cst {
    pub source_hash: ContentHash,
    pub source_len: usize,
    /// Shared source text, kept so later phases can slice exact spans
    /// without re-reading the file.
    pub source: Arc<str>,
    /// Token stream; spans tile `[0, source_len)` exactly.
    pub tokens: Vec<Token>,
    pub nodes: Vec<TopNode>,
}

impl Cst {
    /// Source text of a span.
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        self.source.get(span.start..span.end).unwrap_or("")
    }
}

/// Parse the top level of a Rust source file.
#[must_use]
pub fn parse_top_level(src: &str) -> Cst {
    let tokens = lex(src);
    let mut p = Parser {
        src,
        tokens,
        pos: 0,
    };
    let mut nodes = Vec::new();
    while let Some(n) = p.next_node() {
        nodes.push(n);
    }
    Cst {
        source_hash: hash_str(src),
        source_len: src.len(),
        source: Arc::from(src),
        tokens: p.tokens,
        nodes,
    }
}

struct Parser<'a> {
    src: &'a str,
    tokens: Vec<Token>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.peek()?;
        self.pos += 1;
        Some(t)
    }

    /// Skip whitespace and plain comments. Doc comments are leading
    /// item trivia and are *not* skipped here.
    fn skip_blanks(&mut self) {
        while self.peek().is_some_and(|t| t.kind.is_blank()) {
            self.pos += 1;
        }
    }

    /// Skip blanks and doc comments (for scanning item interiors).
    fn skip_trivia(&mut self) {
        while self
            .peek()
            .is_some_and(|t| t.kind.is_blank() || t.kind == Tk::DocComment)
        {
            self.pos += 1;
        }
    }

    fn text_of(&self, t: &Token) -> &'a str {
        self.src.get(t.span.start..t.span.end).unwrap_or("")
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Consume a balanced delimiter group starting at the current
    /// open delimiter. Returns the span including both delimiters.
    /// On EOF or mismatched input, consumes to EOF and returns what
    /// it has (never loops forever).
    fn consume_balanced(&mut self) -> Option<Span> {
        let first = self.peek()?;
        let open = first.kind;
        if !matches!(open, Tk::OpenParen | Tk::OpenBrace | Tk::OpenBracket) {
            return None;
        }
        let mut stack: Vec<Tk> = vec![open];
        self.bump();
        while let Some(t) = self.bump() {
            match t.kind {
                Tk::OpenParen | Tk::OpenBrace | Tk::OpenBracket => stack.push(t.kind),
                Tk::CloseParen | Tk::CloseBrace | Tk::CloseBracket => {
                    stack.pop();
                    if stack.is_empty() {
                        return Some(Span {
                            start: first.span.start,
                            end: t.span.end,
                        });
                    }
                }
                _ => {}
            }
        }
        Some(Span {
            start: first.span.start,
            end: self.src.len(),
        })
    }

    /// Consume tokens until a `;` at delimiter depth 0 (inclusive).
    /// Stops without consuming on a stray `}` at depth 0.
    fn consume_until_semi(&mut self) {
        let mut depth = 0usize;
        while let Some(t) = self.peek() {
            match t.kind {
                Tk::OpenParen | Tk::OpenBrace | Tk::OpenBracket => depth += 1,
                Tk::CloseParen | Tk::CloseBracket => depth = depth.saturating_sub(1),
                Tk::CloseBrace => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                }
                Tk::Semi if depth == 0 => {
                    self.bump();
                    return;
                }
                _ => {}
            }
            self.bump();
        }
    }

    /// Scan an item header until its body brace (consumed, returned)
    /// or a terminating `;`. Angle brackets at depth 0 are tracked
    /// heuristically so const-generic braces (`Foo<{ N }>`) are not
    /// mistaken for the item body. The third return value is an
    /// *ambiguity* flag: when the heuristic cannot be trusted (a `;`
    /// or stray `}` at depth 0 while angle brackets are still open,
    /// or EOF with open angles — e.g. a comparison `<` in a
    /// signature), the caller must fail closed and treat the item as
    /// opaque rather than trust the span.
    fn consume_braced(&mut self) -> (usize, Option<Span>, bool) {
        let mut depth = 0usize;
        let mut angle = 0usize;
        while let Some(t) = self.peek() {
            match t.kind {
                Tk::OpenBrace if depth == 0 && angle == 0 => {
                    let start = t.span.start;
                    let body = self.consume_balanced();
                    let end = body.map_or(start, |b| b.end);
                    return (end, body, false);
                }
                Tk::Semi if depth == 0 => {
                    let ambiguous = angle > 0;
                    self.bump();
                    return (t.span.end, None, ambiguous);
                }
                Tk::CloseBrace if depth == 0 => return (t.span.start, None, angle > 0),
                Tk::OpenParen | Tk::OpenBrace | Tk::OpenBracket => depth += 1,
                Tk::CloseParen | Tk::CloseBracket | Tk::CloseBrace => {
                    depth = depth.saturating_sub(1);
                }
                Tk::Lt if depth == 0 => angle += 1,
                Tk::Gt if depth == 0 && angle > 0 => angle -= 1,
                _ => {}
            }
            self.bump();
        }
        (self.src.len(), None, angle > 0)
    }

    /// Parse one `#[...]` / `#![...]` attribute starting at `#`.
    fn parse_attribute(&mut self) -> Option<Attr> {
        let hash = self.bump()?; // '#'
        let mut is_inner = false;
        self.skip_blanks();
        if self.peek().is_some_and(|t| t.kind == Tk::Exclaim) {
            is_inner = true;
            self.bump();
            self.skip_blanks();
        }
        if !self.peek().is_some_and(|t| t.kind == Tk::OpenBracket) {
            let end = self
                .tokens
                .get(self.pos.saturating_sub(1))
                .map_or(hash.span.end, |t| t.span.end);
            return Some(Attr {
                span: Span {
                    start: hash.span.start,
                    end,
                },
                is_inner,
                meta: None,
            });
        }
        let inner_start = self.pos + 1;
        let group = self.consume_balanced()?;
        // Tokens strictly inside the brackets (drop the closing `]`).
        let inner_end = self.pos.saturating_sub(1);
        let inner_toks: Vec<Token> = self
            .tokens
            .get(inner_start..inner_end)
            .unwrap_or(&[])
            .to_vec();
        let meta = parse_meta(self.src, &inner_toks);
        Some(Attr {
            span: Span {
                start: hash.span.start,
                end: group.end,
            },
            is_inner,
            meta,
        })
    }

    fn next_name(&mut self) -> Option<String> {
        self.skip_trivia();
        let t = self.peek()?;
        if matches!(t.kind, Tk::Ident | Tk::RawIdent) {
            let text = self.text_of(&t).trim_start_matches("r#").to_owned();
            self.bump();
            Some(text)
        } else {
            None
        }
    }

    /// Try to parse a macro invocation path (`a::b::c`) followed by
    /// `!` and a delimiter group. On failure the cursor is restored.
    fn parse_macro_call(&mut self, start: usize, attrs: Vec<Attr>) -> Option<TopNode> {
        let save = self.pos;
        let mut first_ident_start = None;
        let (path_start, path_end) = loop {
            self.skip_trivia();
            let t = self.peek()?;
            if !matches!(t.kind, Tk::Ident | Tk::RawIdent) {
                self.pos = save;
                return None;
            }
            let first = *first_ident_start.get_or_insert(t.span.start);
            self.bump();
            self.skip_trivia();
            // `::` continues the path.
            if self.peek().is_some_and(|t| t.kind == Tk::Colon) {
                let save2 = self.pos;
                self.bump();
                self.skip_trivia();
                if self.peek().is_some_and(|t| t.kind == Tk::Colon) {
                    self.bump();
                    continue;
                }
                self.pos = save2;
            }
            break (first, t.span.end);
        };
        self.skip_trivia();
        if !self.peek().is_some_and(|t| t.kind == Tk::Exclaim) {
            self.pos = save;
            return None;
        }
        self.bump(); // '!'
        self.skip_trivia();
        let delim = self.peek()?;
        let paren_or_bracket = matches!(delim.kind, Tk::OpenParen | Tk::OpenBracket);
        let body = self.consume_balanced();
        let mut end = body.map_or(path_end, |b| b.end);
        if paren_or_bracket {
            self.skip_trivia();
            if let Some(t) = self.peek() {
                if t.kind == Tk::Semi {
                    end = t.span.end;
                    self.bump();
                }
            }
        }
        let path = self.src.get(path_start..path_end).map(str::to_owned);
        Some(TopNode {
            kind: TopKind::MacroCall,
            span: Span { start, end },
            body_span: body,
            name: None,
            attrs,
            macro_path: path,
            visibility: Visibility::Inherited,
            ambiguous: false,
        })
    }

    fn next_node(&mut self) -> Option<TopNode> {
        self.skip_blanks();
        if self.at_eof() {
            return None;
        }
        let start = self.peek()?.span.start;

        // Leading doc comments and outer attributes.
        let mut attrs: Vec<Attr> = Vec::new();
        loop {
            self.skip_blanks();
            let Some(t) = self.peek() else { break };
            match t.kind {
                Tk::DocComment => {
                    // Inner doc comments (`//!`, `/*! */`) document the
                    // enclosing module: they are inner attributes, not
                    // item-leading trivia.
                    let text = self.text_of(&t);
                    if text.starts_with("//!") || text.starts_with("/*!") {
                        if attrs.is_empty() {
                            self.bump();
                            return Some(TopNode {
                                kind: TopKind::InnerAttr,
                                span: Span {
                                    start,
                                    end: t.span.end,
                                },
                                body_span: None,
                                name: None,
                                attrs: Vec::new(),
                                macro_path: None,
                                visibility: Visibility::Inherited,
                                ambiguous: false,
                            });
                        }
                        break;
                    }
                    self.bump();
                }
                Tk::Pound => {
                    let attr = self.parse_attribute()?;
                    if attr.is_inner && attrs.is_empty() {
                        return Some(TopNode {
                            kind: TopKind::InnerAttr,
                            span: Span {
                                start,
                                end: attr.span.end,
                            },
                            body_span: None,
                            name: None,
                            attrs: vec![attr],
                            macro_path: None,
                            visibility: Visibility::Inherited,
                            ambiguous: false,
                        });
                    }
                    attrs.push(attr);
                }
                _ => break,
            }
        }

        Some(self.parse_item(start, attrs))
    }

    /// Convenience constructor for a node with no body, name, or
    /// macro path.
    fn plain_node(kind: TopKind, start: usize, end: usize, attrs: Vec<Attr>) -> TopNode {
        TopNode {
            kind,
            span: Span { start, end },
            body_span: None,
            name: None,
            attrs,
            macro_path: None,
            visibility: Visibility::Inherited,
            ambiguous: false,
        }
    }

    /// Parse one item (or macro invocation) after its leading doc
    /// comments and attributes have been collected.
    fn parse_item(&mut self, start: usize, attrs: Vec<Attr>) -> TopNode {
        self.skip_trivia();
        let Some(first) = self.peek() else {
            // Dangling attributes at EOF: keep them visible.
            let end = attrs.last().map_or(start, |a| a.span.end);
            return Self::plain_node(TopKind::Unknown, start, end, attrs);
        };

        if !matches!(first.kind, Tk::Ident | Tk::RawIdent) {
            let end = first.span.end;
            self.bump();
            return Self::plain_node(TopKind::Unknown, start, end, attrs);
        }

        // Visibility / modifier prefixes.
        let mut vis = Visibility::Inherited;
        loop {
            self.skip_trivia();
            let Some(t) = self.peek() else { break };
            if t.kind != Tk::Ident {
                break;
            }
            let text = self.text_of(&t);
            let is_prefix = match text {
                "pub" | "unsafe" | "async" | "auto" | "default" => true,
                "crate" => {
                    // `crate fn` is a visibility; `crate::m!()` is a path.
                    let save = self.pos;
                    self.bump();
                    self.skip_trivia();
                    let is_path = self.peek().is_some_and(|t| t.kind == Tk::Colon);
                    self.pos = save;
                    !is_path
                }
                "const" => {
                    // `const fn` / `const async fn` etc. vs `const X:`.
                    let save = self.pos;
                    self.bump();
                    self.skip_trivia();
                    let next = self.peek().map(|t| self.text_of(&t).to_owned());
                    self.pos = save;
                    matches!(next.as_deref(), Some("fn" | "async" | "unsafe" | "extern"))
                }
                _ => false,
            };
            if !is_prefix {
                break;
            }
            self.bump();
            if text == "pub" {
                self.skip_trivia();
                if self.peek().is_some_and(|t| t.kind == Tk::OpenParen) {
                    self.consume_balanced();
                    vis = Visibility::PubRestricted;
                } else {
                    vis = Visibility::Pub;
                }
            } else if text == "crate" {
                // `crate fn` / `crate struct` etc.: restricted visibility.
                vis = Visibility::PubRestricted;
            }
        }

        self.skip_trivia();
        let Some(kw) = self.peek() else {
            let end = attrs.last().map_or(start, |a| a.span.end);
            return Self::plain_node(TopKind::Unknown, start, end, attrs);
        };
        let mut node = self.dispatch_keyword(start, attrs, kw);
        node.visibility = vis;
        node
    }

    /// Dispatch on the item's leading keyword and consume the rest of
    /// the item.
    // One flat match over item keywords: splitting it would scatter
    // the keyword table without reducing complexity.
    #[allow(clippy::too_many_lines)]
    fn dispatch_keyword(&mut self, start: usize, attrs: Vec<Attr>, kw: Token) -> TopNode {
        let kw_text = self.text_of(&kw).to_owned();

        let braced = |p: &mut Self| -> (usize, Option<Span>, bool) { p.consume_braced() };
        let semi = |p: &mut Self| -> usize {
            p.consume_until_semi();
            p.tokens
                .get(p.pos.saturating_sub(1))
                .map_or(start, |t| t.span.end)
        };

        let (kind, name, end, body, ambiguous) = match kw_text.as_str() {
            "fn" => {
                self.bump();
                let name = self.next_name();
                let (end, body, amb) = braced(self);
                (TopKind::Fn, name, end, body, amb)
            }
            "struct" => {
                self.bump();
                let name = self.next_name();
                let (end, body, amb) = braced(self);
                (TopKind::Struct, name, end, body, amb)
            }
            "union" => {
                self.bump();
                let name = self.next_name();
                let (end, body, amb) = braced(self);
                (TopKind::Union, name, end, body, amb)
            }
            "enum" => {
                self.bump();
                let name = self.next_name();
                let (end, body, amb) = braced(self);
                (TopKind::Enum, name, end, body, amb)
            }
            "impl" => {
                self.bump();
                let (end, body, amb) = braced(self);
                (TopKind::Impl, None, end, body, amb)
            }
            "trait" => {
                self.bump();
                let name = self.next_name();
                let (end, body, amb) = braced(self);
                (TopKind::Trait, name, end, body, amb)
            }
            "mod" => {
                self.bump();
                let name = self.next_name();
                let (end, body, amb) = braced(self);
                (TopKind::Mod, name, end, body, amb)
            }
            "use" => {
                self.bump();
                let end = semi(self);
                (TopKind::Use, None, end, None, false)
            }
            "type" => {
                self.bump();
                let name = self.next_name();
                let end = semi(self);
                (TopKind::TypeAlias, name, end, None, false)
            }
            "static" => {
                self.bump();
                self.skip_trivia();
                // `static mut NAME`.
                if self
                    .peek()
                    .is_some_and(|t| t.kind == Tk::Ident && self.text_of(&t) == "mut")
                {
                    self.bump();
                }
                let name = self.next_name();
                let end = semi(self);
                (TopKind::Static, name, end, None, false)
            }
            "const" => {
                self.bump();
                let name = self.next_name();
                let end = semi(self);
                (TopKind::Const, name, end, None, false)
            }
            "extern" => self.parse_extern(start),
            "macro_rules" | "macro" => {
                let (k, n, e, b) = self.parse_macro_def(start, kw_text == "macro_rules");
                (k, n, e, b, false)
            }
            _ => {
                if let Some(node) = self.parse_macro_call(start, attrs.clone()) {
                    return node;
                }
                // Unrecognised keyword-ish token: consume it alone.
                let end = kw.span.end;
                self.bump();
                (TopKind::Unknown, None, end, None, false)
            }
        };

        TopNode {
            kind,
            span: Span { start, end },
            body_span: body,
            name,
            attrs,
            macro_path: None,
            visibility: Visibility::Inherited,
            ambiguous,
        }
    }
    /// `extern crate ...;`, `extern "C" { ... }`, or `extern "C" fn`.
    fn parse_extern(
        &mut self,
        start: usize,
    ) -> (TopKind, Option<String>, usize, Option<Span>, bool) {
        self.bump(); // `extern`
        self.skip_trivia();
        let next_is_crate = self
            .peek()
            .is_some_and(|t| t.kind == Tk::Ident && self.text_of(&t) == "crate");
        if next_is_crate {
            self.bump();
            let name = self.next_name();
            self.consume_until_semi();
            let end = self
                .tokens
                .get(self.pos.saturating_sub(1))
                .map_or(start, |t| t.span.end);
            return (TopKind::ExternCrate, name, end, None, false);
        }
        // Optional ABI string.
        if self
            .peek()
            .is_some_and(|t| matches!(t.kind, Tk::Str | Tk::StrRaw))
        {
            self.bump();
            self.skip_trivia();
        }
        if self.peek().is_some_and(|t| t.kind == Tk::OpenBrace) {
            let start_b = self.peek().map_or(start, |t| t.span.start);
            let body = self.consume_balanced();
            let end = body.map_or(start_b, |b| b.end);
            return (TopKind::ExternBlock, None, end, body, false);
        }
        // `extern "C" fn` — treat as a function item.
        if self
            .peek()
            .is_some_and(|t| t.kind == Tk::Ident && self.text_of(&t) == "fn")
        {
            self.bump();
        }
        let name = self.next_name();
        let (end, body, amb) = self.consume_braced();
        (TopKind::Fn, name, end, body, amb)
    }

    /// `macro_rules! name { ... }` or `macro name(...) { ... }`. The
    /// body is consumed as one opaque balanced group.
    fn parse_macro_def(&mut self, start: usize, is_rules: bool) -> (TopKind, Option<String>, usize, Option<Span>) {
        self.bump(); // `macro_rules` / `macro`
        let name;
        if is_rules {
            self.skip_trivia();
            if self.peek().is_some_and(|t| t.kind == Tk::Exclaim) {
                self.bump();
            }
            name = self.next_name();
        } else {
            name = self.next_name();
            self.skip_trivia();
            if self.peek().is_some_and(|t| t.kind == Tk::OpenParen) {
                self.consume_balanced();
            }
        }
        self.skip_trivia();
        let body = if self.peek().is_some_and(|t| {
            matches!(t.kind, Tk::OpenBrace | Tk::OpenParen | Tk::OpenBracket)
        }) {
            self.consume_balanced()
        } else {
            None
        };
        let end = body.map_or_else(
            || {
                self.consume_until_semi();
                self.tokens
                    .get(self.pos.saturating_sub(1))
                    .map_or(start, |t| t.span.end)
            },
            |b| b.end,
        );
        (TopKind::MacroDef, name, end, body)
    }
}
