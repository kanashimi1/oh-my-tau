//! `cfg(...)` / `cfg_attr(...)` attribute parsing and evaluation.
//!
//! Expressions are parsed from the token stream of an attribute's
//! meta (the part after `#[cfg` / `#[cfg_attr`), never by splitting
//! text, so string values containing commas or parens stay intact.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::lex::{Tk, Token};

/// A `cfg` expression.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CfgExpr {
    /// A bare flag atom, e.g. `unix`, `test`, `debug_assertions`.
    Atom(String),
    /// A key/value predicate, e.g. `feature = "serde"`, `target_os = "linux"`.
    KeyValue(String, String),
    /// `all(...)` — conjunction.
    All(Vec<CfgExpr>),
    /// `any(...)` — disjunction.
    Any(Vec<CfgExpr>),
    /// `not(...)` — negation of a single expression.
    Not(Box<CfgExpr>),
}

/// The cfg environment an evaluation runs against.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CfgContext {
    /// Bare flags, e.g. `test`, `unix`, `debug_assertions`.
    pub flags: BTreeSet<String>,
    /// Key/value predicates, e.g. `("feature", "serde")`.
    pub kv: BTreeMap<String, String>,
}

impl CfgContext {
    /// A context with no flags set: everything conditional is disabled.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
}

impl CfgExpr {
    /// Evaluate against a cfg context.
    #[must_use]
    pub fn eval(&self, ctx: &CfgContext) -> bool {
        match self {
            CfgExpr::Atom(name) => ctx.flags.contains(name),
            CfgExpr::KeyValue(k, v) => ctx.kv.get(k).is_some_and(|got| got == v),
            CfgExpr::All(parts) => parts.iter().all(|p| p.eval(ctx)),
            CfgExpr::Any(parts) => parts.iter().any(|p| p.eval(ctx)),
            CfgExpr::Not(inner) => !inner.eval(ctx),
        }
    }
}

/// A parsed meta item: `path`, `path = "value"`, or `path(args...)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Meta {
    Path(String),
    KeyValue(String, String),
    List(String, Vec<Meta>),
}

struct MetaParser<'a> {
    src: &'a str,
    toks: &'a [Token],
    /// Indices into `toks` of the significant (non-blank) tokens.
    sig: Vec<usize>,
    pos: usize,
}

impl<'a> MetaParser<'a> {
    fn new(src: &'a str, toks: &'a [Token]) -> Self {
        let sig = toks
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.kind.is_blank() && t.kind != Tk::DocComment)
            .map(|(i, _)| i)
            .collect();
        Self {
            src,
            toks,
            sig,
            pos: 0,
        }
    }

    fn peek(&self) -> Option<Tk> {
        self.sig.get(self.pos).map(|&i| self.toks[i].kind)
    }

    fn bump(&mut self) -> Option<Token> {
        let i = *self.sig.get(self.pos)?;
        self.pos += 1;
        Some(self.toks[i])
    }

    fn text(&self, t: &Token) -> String {
        let raw = t.text(self.src);
        match t.kind {
            Tk::Ident | Tk::RawIdent => raw.trim_start_matches("r#").to_owned(),
            _ => unquote(raw),
        }
    }

    /// Parse a comma-separated meta list until `CloseParen` or EOF.
    fn parse_list(&mut self) -> Vec<Meta> {
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None | Some(Tk::CloseParen) => break,
                Some(Tk::Comma) => {
                    self.bump();
                }
                _ => {
                    if let Some(m) = self.parse_meta() {
                        out.push(m);
                    } else {
                        // Unparseable fragment: skip one token to
                        // guarantee progress.
                        self.bump();
                    }
                }
            }
        }
        out
    }

    fn parse_meta(&mut self) -> Option<Meta> {
        let name_tok = self.bump()?;
        if name_tok.kind != Tk::Ident {
            return None;
        }
        let name = self.text(&name_tok);
        match self.peek() {
            Some(Tk::Eq) => {
                self.bump();
                let val = self.bump()?;
                if !matches!(val.kind, Tk::Str | Tk::StrRaw | Tk::Int) {
                    return Some(Meta::Path(name));
                }
                let v = self.text(&val);
                Some(Meta::KeyValue(name, v))
            }
            Some(Tk::OpenParen) => {
                self.bump();
                let args = self.parse_list();
                // Consume the matching close paren if present.
                if self.peek() == Some(Tk::CloseParen) {
                    self.bump();
                }
                Some(Meta::List(name, args))
            }
            _ => Some(Meta::Path(name)),
        }
    }
}

/// Parse the meta of one attribute given the tokens *inside* the
/// attribute's brackets, e.g. for `#[cfg(all(unix, feature = "x"))]`
/// the tokens of `cfg(all(unix, feature = "x"))`.
#[must_use]
pub fn parse_meta(src: &str, toks: &[Token]) -> Option<Meta> {
    let mut p = MetaParser::new(src, toks);
    let m = p.parse_meta()?;
    Some(m)
}

/// Convert a `cfg(...)` meta into a [`CfgExpr`]. Returns `None` for
/// any other meta shape.
#[must_use]
pub fn cfg_from_meta(meta: &Meta) -> Option<CfgExpr> {
    let Meta::List(name, args) = meta else {
        return None;
    };
    if name != "cfg" || args.len() != 1 {
        return None;
    }
    expr_from_meta(&args[0])
}

fn expr_from_meta(m: &Meta) -> Option<CfgExpr> {
    match m {
        Meta::Path(name) => Some(CfgExpr::Atom(name.clone())),
        Meta::KeyValue(k, v) => Some(CfgExpr::KeyValue(k.clone(), v.clone())),
        Meta::List(name, args) => match name.as_str() {
            "all" => Some(CfgExpr::All(
                args.iter().filter_map(expr_from_meta).collect(),
            )),
            "any" => Some(CfgExpr::Any(
                args.iter().filter_map(expr_from_meta).collect(),
            )),
            "not" if args.len() == 1 => Some(CfgExpr::Not(Box::new(expr_from_meta(&args[0])?))),
            _ => None,
        },
    }
}

/// Effective cfg contribution of a `cfg_attr(cond, meta, ...)` meta:
/// when `cond` evaluates true, every nested `cfg(...)` meta among the
/// remaining arguments applies (conjoined); non-cfg metas are ignored.
/// Returns `None` when the meta is not a well-formed `cfg_attr`.
#[must_use]
pub fn cfg_attr_from_meta(meta: &Meta, ctx: &CfgContext) -> Option<Option<CfgExpr>> {
    let Meta::List(name, args) = meta else {
        return None;
    };
    if name != "cfg_attr" || args.is_empty() {
        return None;
    }
    let cond = expr_from_meta(&args[0])?;
    if !cond.eval(ctx) {
        return Some(None);
    }
    let mut acc: Option<CfgExpr> = None;
    for m in &args[1..] {
        if let Some(e) = cfg_from_meta(m) {
            acc = Some(match acc {
                None => e,
                Some(prev) => CfgExpr::All(vec![prev, e]),
            });
        }
    }
    Some(acc)
}

/// Extract the cfg verdict of one attribute token list (the tokens
/// inside `#[ ... ]`). Returns:
/// - `None` — attribute is not cfg-related;
/// - `Some(None)` — cfg-related, no constraint (e.g. a false
///   `cfg_attr` condition);
/// - `Some(Some(expr))` — an effective `cfg` expression to conjoin.
#[must_use]
pub fn attr_cfg(src: &str, toks: &[Token], ctx: &CfgContext) -> Option<Option<CfgExpr>> {
    let meta = parse_meta(src, toks)?;
    if let Some(e) = cfg_from_meta(&meta) {
        return Some(Some(e));
    }
    cfg_attr_from_meta(&meta, ctx)
}

/// Strip the delimiters of a string literal token (`"..."`,
/// `r"..."`, `r#"..."#`, ...). Non-string text is returned unchanged.
#[must_use]
pub fn unquote(text: &str) -> String {
    if let Some(rest) = text.strip_prefix('r') {
        let hashes = rest.bytes().take_while(|b| *b == b'#').count();
        let body = &rest[hashes..];
        let trimmed = body
            .strip_prefix('"')
            .and_then(|b| b.strip_suffix(&"\"".repeat(hashes + 1)))
            .unwrap_or(body);
        return trimmed.to_owned();
    }
    text.strip_prefix('"')
        .and_then(|b| b.strip_suffix('"'))
        .unwrap_or(text)
        .to_owned()
}
