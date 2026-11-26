//! Lexer round-trip and top-level span correctness around comments,
//! strings, raw strings, and nested delimiters.

use omp_ladder::{lex, parse_top_level, Tk, TopKind};

const NASTY: &str = r###"
//! inner doc with { braces }
/// outer doc with "quotes" and } unbalanced
#[derive(Debug)]
pub struct S {
    s: &'static str, // comment with } brace
}

/* block /* nested */ still comment } */
fn tricky() {
    let a = "string with } and /* not a comment */";
    let b = r#"raw "string" with } brace and \" escape-looking"#;
    let c = r##"raw with "# inside"##;
    let d = '}';
    let e = b'}';
    let f = '\\';
    macro_like! { fn not_an_item() {} }
}

const RAW: &str = r#"const-ish text with ; semi"#;

macro_rules! gen {
    ($name:ident) => {
        fn $name() { let x = "}"; }
    };
}

gen!(made_by_macro);
"###;

#[test]
fn token_spans_tile_source_exactly() {
    let toks = lex(NASTY);
    let mut pos = 0usize;
    for t in &toks {
        assert_eq!(t.span.start, pos, "gap or overlap at {pos}");
        assert!(t.span.end > t.span.start, "empty token at {pos}");
        pos = t.span.end;
    }
    assert_eq!(pos, NASTY.len());
    let rebuilt: String = toks.iter().map(|t| t.text(NASTY)).collect();
    assert_eq!(rebuilt, NASTY);
}

#[test]
fn strings_and_comments_are_single_tokens() {
    let toks = lex(NASTY);
    // The raw string with an embedded `"#` must be one token.
    let raws: Vec<_> = toks.iter().filter(|t| t.kind == Tk::StrRaw).collect();
    assert_eq!(raws.len(), 3);
    assert!(raws.iter().any(|t| t.text(NASTY).contains("raw with \"# inside")));
    // Nested block comment is a single comment token.
    let nested = toks
        .iter()
        .find(|t| t.kind == Tk::Comment && t.text(NASTY).contains("nested"));
    assert!(nested.is_some(), "nested block comment must be one token");
}

#[test]
fn top_level_items_have_correct_kinds_and_spans() {
    let cst = parse_top_level(NASTY);
    let kinds: Vec<TopKind> = cst.nodes.iter().map(|n| n.kind).collect();
    assert_eq!(
        kinds,
        vec![
            TopKind::InnerAttr,
            TopKind::Struct,
            TopKind::Fn,
            TopKind::Const,
            TopKind::MacroDef,
            TopKind::MacroCall,
        ]
    );
    // Item-looking tokens inside macro bodies never become nodes.
    assert!(!cst.nodes.iter().any(|n| n.name.as_deref() == Some("not_an_item")));
    assert!(!cst.nodes.iter().any(|n| n.name.as_deref() == Some("made_by_macro")));

    let s = &cst.nodes[1];
    assert_eq!(s.name.as_deref(), Some("S"));
    let text = cst.text(s.span);
    assert!(text.starts_with("/// outer doc"), "doc comment attaches: {text}");
    assert!(text.contains("#[derive(Debug)]"));
    assert!(text.ends_with('}'));
    assert!(s.body_span.is_some());

    let f = &cst.nodes[2];
    assert_eq!(f.name.as_deref(), Some("tricky"));
    let body = f.body_span.map(|b| cst.text(b)).unwrap_or_default();
    assert!(body.contains("r##\"raw with \"# inside\"##"));
    assert!(body.contains("macro_like!"));

    let konst = &cst.nodes[3];
    assert_eq!(konst.name.as_deref(), Some("RAW"));
    assert!(cst.text(konst.span).ends_with(';'));

    let def = &cst.nodes[4];
    assert_eq!(def.name.as_deref(), Some("gen"));
    assert_eq!(def.kind, TopKind::MacroDef);
    // The whole macro_rules body is one opaque span.
    let dbody = def.body_span.map(|b| cst.text(b)).unwrap_or_default();
    assert!(dbody.contains("fn $name()"));

    let call = &cst.nodes[5];
    assert_eq!(call.kind, TopKind::MacroCall);
    assert_eq!(call.macro_path.as_deref(), Some("gen"));
    assert_eq!(cst.text(call.span), "gen!(made_by_macro);");
}

#[test]
fn nested_delimiters_and_macro_path_calls() {
    let src = "foo::bar::baz!({ [ ( { } ) ] });\nfn after() {}\n";
    let cst = parse_top_level(src);
    assert_eq!(cst.nodes.len(), 2);
    assert_eq!(cst.nodes[0].kind, TopKind::MacroCall);
    assert_eq!(cst.nodes[0].macro_path.as_deref(), Some("foo::bar::baz"));
    assert_eq!(cst.text(cst.nodes[0].span), "foo::bar::baz!({ [ ( { } ) ] });");
    assert_eq!(cst.nodes[1].kind, TopKind::Fn);
    assert_eq!(cst.nodes[1].name.as_deref(), Some("after"));
}

#[test]
fn every_item_kind_parses() {
    let src = r#"
use std::collections::HashMap;
pub use foo::{bar, baz};
type Alias = Vec<u8>;
pub(crate) const C: usize = { 1 + 2 };
static mut S: i32 = 0;
extern crate core;
unsafe extern "C" { fn c_fn(); }
pub trait T { fn m(&self); }
impl T for u8 { fn m(&self) {} }
pub enum E { A, B { x: u8 } }
union U { a: u32 }
mod inner { fn hidden() {} }
pub async unsafe fn f<const N: usize>() -> [u8; { N }] { [0; N] }
"#;
    let cst = parse_top_level(src);
    let kinds: Vec<TopKind> = cst.nodes.iter().map(|n| n.kind).collect();
    assert_eq!(
        kinds,
        vec![
            TopKind::Use,
            TopKind::Use,
            TopKind::TypeAlias,
            TopKind::Const,
            TopKind::Static,
            TopKind::ExternCrate,
            TopKind::ExternBlock,
            TopKind::Trait,
            TopKind::Impl,
            TopKind::Enum,
            TopKind::Union,
            TopKind::Mod,
            TopKind::Fn,
        ]
    );
    // `fn hidden` inside `mod inner` is not a top-level node.
    assert!(!cst.nodes.iter().any(|n| n.name.as_deref() == Some("hidden")));
    // Const-generic brace in the header is not the fn body; the real
    // body contains `[0; N]`.
    let f = cst.nodes.last().map(|n| cst.text(n.span)).unwrap_or_default();
    assert!(f.ends_with("[0; N] }"), "fn body span wrong: {f}");
    // Spans are ordered and non-overlapping.
    for w in cst.nodes.windows(2) {
        assert!(w[0].span.end <= w[1].span.start, "overlap: {:?} {:?}", w[0].span, w[1].span);
    }
}

#[test]
fn unterminated_input_never_hangs() {
    let cst = parse_top_level("fn f() { let s = \"unterminated");
    assert_eq!(cst.nodes.len(), 1);
    assert_eq!(cst.nodes[0].kind, TopKind::Fn);
    let cst = parse_top_level("macro_rules! m { ($x:ident) => {");
    assert_eq!(cst.nodes[0].kind, TopKind::MacroDef);
}
