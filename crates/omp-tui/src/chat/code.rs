//! Syntax highlighting for fenced code — the port of `pi-tui/src/lang-from-path.ts`
//! plus the engine behind `theme.highlightCode` / `createHighlightStream`
//! (consumed by markdown fences and `render/code-cell.ts` blocks).
//!
//! Engine choice: tree-sitter through the vendored `pi-ast` grammars, not
//! `syntect`. `pi-ast` already links accurate tree-sitter grammars for the
//! languages chat fences actually contain (rust, ts/tsx, js, python, bash, go,
//! c, cpp, java, json, yaml, toml, markdown, css, html, diff, regex) and can
//! load the remaining ones as WebAssembly when installed; syntect would add a
//! second, heavyweight regex-syntax stack whose Sublime scopes would need a
//! hand-rolled token mapping anyway. Node kinds are classified into the omp
//! theme categories (`syntaxComment` … `syntaxPunctuation`); unknown
//! languages degrade to unstyled text exactly like TS `highlightCode`
//! returning `null`.
//!
//! Streaming: [`HighlightStream`] re-parses the buffered prefix on each
//! completed line (tree-sitter handles a chat-sized fence in well under a
//! millisecond) and emits the highlighted tail; [`HighlightStream::finish`]
//! produces the final authoritative rendering of the whole fence.

use pi_ast::language::grammar::{self, LanguageGrammar};
use pi_ast::SupportLang;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

use crate::theme::Theme;

/// Resolved syntax category for a source range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cat {
    Plain,
    Comment,
    Keyword,
    Function,
    Variable,
    Str,
    Number,
    Type,
    Operator,
    Punct,
    Inserted,
    Deleted,
}

impl Cat {
    /// Theme token for the category (names match omp's `theme-schema.json`).
    fn token(self) -> &'static str {
        match self {
            Self::Plain => "text",
            Self::Comment => "syntaxComment",
            Self::Keyword => "syntaxKeyword",
            Self::Function => "syntaxFunction",
            Self::Variable => "syntaxVariable",
            Self::Str => "syntaxString",
            Self::Number => "syntaxNumber",
            Self::Type => "syntaxType",
            Self::Operator => "syntaxOperator",
            Self::Punct => "syntaxPunctuation",
            Self::Inserted => "toolDiffAdded",
            Self::Deleted => "toolDiffRemoved",
        }
    }
}

/// Per-highlight color table resolved once from the theme.
#[derive(Clone, Copy, Debug)]
struct Styles {
    colors: [Color; 12],
}

impl Styles {
    fn new(theme: &Theme) -> Self {
        let c = |cat: Cat| theme.color(cat.token());
        Self {
            colors: [
                c(Cat::Plain),
                c(Cat::Comment),
                c(Cat::Keyword),
                c(Cat::Function),
                c(Cat::Variable),
                c(Cat::Str),
                c(Cat::Number),
                c(Cat::Type),
                c(Cat::Operator),
                c(Cat::Punct),
                c(Cat::Inserted),
                c(Cat::Deleted),
            ],
        }
    }

    fn style(self, cat: Cat) -> Style {
        if cat == Cat::Plain {
            return Style::default();
        }
        Style::default().fg(self.colors[cat as usize])
    }
}

/// Highlight `code` written in `lang` (fence id or file language id) into
/// themed lines. Unknown/unsupported languages return unstyled text.
#[must_use]
pub fn highlight(code: &str, lang: Option<&str>, theme: &Theme) -> Text<'static> {
    highlight_with(code, lang, Styles::new(theme))
}

/// Incremental highlighter for one streaming fence (`createHighlightStream`).
///
/// `push` receives newline-terminated complete lines and returns the
/// highlighted text of the newly completed lines, or [`None`] when the line
/// produced no output yet. Because a later line can close a token opened
/// earlier (an unterminated string, block comment), already-emitted lines may
/// be re-colored once `finish` renders the authoritative final form; the
/// inline live region re-renders every frame, so the transient state is never
/// committed to scrollback.
#[derive(Debug)]
pub struct HighlightStream {
    lang: Option<&'static str>,
    styles: Styles,
    buf: String,
    done: usize,
}

impl HighlightStream {
    #[must_use]
    pub fn new(lang: Option<&str>, theme: &Theme) -> Self {
        Self {
            lang: lang.and_then(lang_from_fence),
            styles: Styles::new(theme),
            buf: String::new(),
            done: 0,
        }
    }

    /// Append one newline-terminated line; returns the highlighted new lines.
    pub fn push(&mut self, line: &str) -> Option<Text<'static>> {
        self.buf.push_str(line);
        self.take_completed()
    }

    /// Final rendering of any remaining tail; the authoritative result.
    #[must_use]
    pub fn finish(&mut self) -> Text<'static> {
        self.take_completed().unwrap_or_else(|| Text::default())
    }

    fn take_completed(&mut self) -> Option<Text<'static>> {
        let full = highlight_with(&self.buf, self.lang, self.styles);
        let lines = full.lines.len();
        if lines <= self.done {
            return None;
        }
        let fresh: Vec<Line<'static>> = full.lines[self.done..].to_vec();
        self.done = lines;
        Some(Text::from(fresh))
    }
}

/// [`highlight`] with a pre-resolved style table (avoids per-push lookups).
fn highlight_with(code: &str, lang: Option<&str>, styles: Styles) -> Text<'static> {
    let spans = match parse_tree(code, lang) {
        Some(tree) => collect_spans(&tree, code),
        None => Vec::new(),
    };
    to_text(code, &spans, styles)
}

// ── Language detection (`lang-from-path.ts`) ────────────────────────────

/// Extension/fence token → highlight language id. Port of `EXTENSION_LANG`
/// (highlight ids only; the LSP half lives in [`detect_language_id`]).
static EXT_LANG: &[(&str, &str)] = &[
    ("ts", "typescript"),
    ("cts", "typescript"),
    ("mts", "typescript"),
    ("tsx", "tsx"),
    ("js", "javascript"),
    ("jsx", "javascript"),
    ("mjs", "javascript"),
    ("cjs", "javascript"),
    ("rs", "rust"),
    ("go", "go"),
    ("c", "c"),
    ("h", "c"),
    ("cpp", "cpp"),
    ("cc", "cpp"),
    ("cxx", "cpp"),
    ("hh", "cpp"),
    ("hpp", "cpp"),
    ("hxx", "cpp"),
    ("cu", "cpp"),
    ("cuh", "cpp"),
    ("ino", "cpp"),
    ("zig", "zig"),
    ("py", "python"),
    ("pyi", "python"),
    ("rb", "ruby"),
    ("rbw", "ruby"),
    ("gemspec", "ruby"),
    ("lua", "lua"),
    ("sh", "bash"),
    ("bash", "bash"),
    ("zsh", "bash"),
    ("ksh", "bash"),
    ("bats", "bash"),
    ("tmux", "bash"),
    ("cgi", "bash"),
    ("fcgi", "bash"),
    ("command", "bash"),
    ("tool", "bash"),
    ("fish", "fish"),
    ("pl", "perl"),
    ("pm", "perl"),
    ("perl", "perl"),
    ("php", "php"),
    ("java", "java"),
    ("kt", "kotlin"),
    ("ktm", "kotlin"),
    ("kts", "kotlin"),
    ("scala", "scala"),
    ("sc", "scala"),
    ("sbt", "scala"),
    ("groovy", "groovy"),
    ("clj", "clojure"),
    ("cljc", "clojure"),
    ("cljs", "clojure"),
    ("edn", "clojure"),
    ("el", "emacs-lisp"),
    ("cs", "csharp"),
    ("fs", "fsharp"),
    ("vb", "vb"),
    ("html", "html"),
    ("htm", "html"),
    ("xhtml", "html"),
    ("css", "css"),
    ("scss", "scss"),
    ("sass", "sass"),
    ("less", "less"),
    ("vue", "vue"),
    ("svelte", "svelte"),
    ("astro", "astro"),
    ("json", "json"),
    ("jsonc", "jsonc"),
    ("yaml", "yaml"),
    ("yml", "yaml"),
    ("toml", "toml"),
    ("xml", "xml"),
    ("xsl", "xml"),
    ("xslt", "xml"),
    ("svg", "xml"),
    ("plist", "xml"),
    ("ini", "ini"),
    ("md", "markdown"),
    ("markdown", "markdown"),
    ("mdx", "markdown"),
    ("mdc", "markdown"),
    ("mkd", "markdown"),
    ("mdown", "markdown"),
    ("rst", "restructuredtext"),
    ("adoc", "asciidoc"),
    ("tex", "latex"),
    ("sql", "sql"),
    ("graphql", "graphql"),
    ("gql", "graphql"),
    ("proto", "protobuf"),
    ("dockerfile", "dockerfile"),
    ("containerfile", "dockerfile"),
    ("tf", "hcl"),
    ("hcl", "hcl"),
    ("tfvars", "hcl"),
    ("nix", "nix"),
    ("ex", "elixir"),
    ("exs", "elixir"),
    ("erl", "erlang"),
    ("hrl", "erlang"),
    ("hs", "haskell"),
    ("ml", "ocaml"),
    ("mli", "ocaml"),
    ("swift", "swift"),
    ("r", "r"),
    ("jl", "julia"),
    ("dart", "dart"),
    ("elm", "elm"),
    ("v", "verilog"),
    ("nim", "nim"),
    ("cr", "crystal"),
    ("d", "d"),
    ("pas", "pascal"),
    ("pp", "pascal"),
    ("lisp", "lisp"),
    ("lsp", "lisp"),
    ("rkt", "racket"),
    ("scm", "scheme"),
    ("ps1", "powershell"),
    ("psm1", "powershell"),
    ("bat", "bat"),
    ("cmd", "bat"),
    ("tla", "tlaplus"),
    ("tlaplus", "tlaplus"),
    ("sol", "solidity"),
    ("odin", "odin"),
    ("star", "starlark"),
    ("bzl", "starlark"),
    ("sv", "verilog"),
    ("svh", "verilog"),
    ("vh", "verilog"),
    ("vim", "vim"),
    ("ipynb", "ipynb"),
    ("hbs", "handlebars"),
    ("hsb", "handlebars"),
    ("handlebars", "handlebars"),
    ("diff", "diff"),
    ("patch", "diff"),
    ("makefile", "make"),
    ("mk", "make"),
    ("mak", "make"),
    ("cmake", "cmake"),
    ("justfile", "just"),
    ("txt", "text"),
    ("text", "text"),
    ("log", "log"),
    ("csv", "csv"),
    ("tsv", "tsv"),
    ("cfg", "conf"),
    ("conf", "conf"),
    ("config", "conf"),
    ("properties", "conf"),
    ("env", "env"),
    ("gitignore", "conf"),
    ("gitattributes", "conf"),
    ("gitmodules", "conf"),
    ("editorconfig", "conf"),
    ("npmrc", "conf"),
    ("prettierrc", "conf"),
    ("eslintrc", "conf"),
    ("prettierignore", "conf"),
    ("eslintignore", "conf"),
];

/// Fence info-string synonyms not covered by [`EXT_LANG`].
static FENCE_EXTRA: &[(&str, &str)] = &[
    ("rust", "rust"),
    ("python", "python"),
    ("javascript", "javascript"),
    ("typescript", "typescript"),
    ("java", "java"),
    ("kotlin", "kotlin"),
    ("ruby", "ruby"),
    ("elixir", "elixir"),
    ("haskell", "haskell"),
    ("ocaml", "ocaml"),
    ("clojure", "clojure"),
    ("protobuf", "protobuf"),
    ("hcl", "hcl"),
    ("terraform", "hcl"),
    ("erlang", "erlang"),
    ("shell", "bash"),
    ("shellscript", "bash"),
    ("obj-c", "objc"),
    ("objective-c", "objc"),
    ("python3", "python"),
    ("batch", "bat"),
    ("golang", "go"),
    ("c++", "cpp"),
    ("docker", "dockerfile"),
    ("plaintext", "text"),
    ("plain", "text"),
    ("txt", "text"),
    ("output", "text"),
    ("result", "text"),
    ("console", "text"),
    ("none", "text"),
    ("terminal", "text"),
];

/// Highlight language for a markdown fence info string ("```rust").
#[must_use]
pub fn lang_from_fence(info: &str) -> Option<&'static str> {
    let tok = info.split_whitespace().next()?.to_ascii_lowercase();
    fence_lookup(&tok)
}

fn fence_lookup(tok: &str) -> Option<&'static str> {
    FENCE_EXTRA
        .iter()
        .chain(EXT_LANG.iter())
        .find(|(k, _)| *k == tok)
        .map(|(_, id)| *id)
}

fn ext_lookup(key: &str) -> Option<&'static str> {
    EXT_LANG
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, id)| *id)
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Highlight language for a file path (`getLanguageFromPath`).
#[must_use]
pub fn lang_from_path(path: &str) -> Option<&'static str> {
    let base = file_name(path).to_ascii_lowercase();
    if base.starts_with(".env.") {
        return Some("env");
    }
    if base == "dockerfile" || base.starts_with("dockerfile.") || base == "containerfile" {
        return Some("dockerfile");
    }
    if base == ".emacs" {
        return Some("emacs-lisp");
    }
    if base == "justfile" {
        return Some("just");
    }
    if base == "cmakelists.txt" {
        return Some("cmake");
    }
    let key = match base.rfind('.') {
        Some(i) => &base[i + 1..],
        None => base.as_str(),
    };
    ext_lookup(key)
}

/// Whether the path renders as markdown (`isMarkdownPath`).
#[must_use]
pub fn is_markdown_path(path: &str) -> bool {
    lang_from_path(path) == Some("markdown")
}

/// LSP language identifier for a path, falling back to `plaintext`
/// (`detectLanguageId`).
#[must_use]
pub fn detect_language_id(path: &str) -> &'static str {
    let base = file_name(path).to_ascii_lowercase();
    if base == "dockerfile" || base.starts_with("dockerfile.") || base == "containerfile" {
        return "dockerfile";
    }
    if base == ".emacs" {
        return "emacs-lisp";
    }
    if base == "makefile" || base == "gnumakefile" {
        return "makefile";
    }
    if base == "justfile" {
        return "just";
    }
    let ext = match base.rfind('.') {
        Some(i) => &base[i + 1..],
        None => "",
    };
    if base == "cmakelists.txt" || ext == "cmake" {
        return "cmake";
    }
    match lang_from_path(path) {
        Some(id) => id,
        None => "plaintext",
    }
}

/// Resolve a highlight id to a parseable grammar, refusing languages whose
/// WebAssembly grammar is not installed offline.
fn resolve_lang(id: &str) -> Option<SupportLang> {
    let lang = SupportLang::from_alias(id)?;
    if let Some(wasm) = lang.wasm_grammar() {
        if !wasm.is_installed() {
            return None;
        }
    }
    Some(lang)
}

// ── Tree walking and classification ─────────────────────────────────────

/// Word-like anonymous tokens treated as keywords (union over the supported
/// grammars; unmatched literals fall back to punctuation).
const KEYWORDS: &[&str] = &[
    "abstract",
    "and",
    "as",
    "async",
    "await",
    "base",
    "bool",
    "break",
    "byte",
    "case",
    "catch",
    "chan",
    "char",
    "class",
    "const",
    "continue",
    "crate",
    "def",
    "default",
    "del",
    "defer",
    "dict",
    "do",
    "double",
    "elif",
    "else",
    "end",
    "enum",
    "except",
    "extends",
    "extern",
    "False",
    "false",
    "final",
    "finally",
    "float",
    "fn",
    "for",
    "foreach",
    "from",
    "fun",
    "function",
    "global",
    "go",
    "if",
    "impl",
    "implements",
    "import",
    "in",
    "include",
    "instanceof",
    "int",
    "interface",
    "is",
    "lambda",
    "let",
    "loop",
    "match",
    "mod",
    "module",
    "move",
    "mut",
    "namespace",
    "new",
    "nil",
    "None",
    "none",
    "not",
    "null",
    "nullptr",
    "operator",
    "or",
    "package",
    "pass",
    "private",
    "protected",
    "pub",
    "public",
    "raise",
    "ref",
    "require",
    "return",
    "Self",
    "self",
    "static",
    "str",
    "struct",
    "super",
    "switch",
    "template",
    "then",
    "this",
    "throw",
    "throws",
    "trait",
    "True",
    "true",
    "try",
    "type",
    "typedef",
    "typeof",
    "union",
    "unsafe",
    "use",
    "using",
    "var",
    "virtual",
    "void",
    "when",
    "where",
    "while",
    "with",
    "yield",
];

const OP_CHARS: &[char] = &[
    '+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|', '^', '~', '?', ':',
];

/// Parse `code` with the grammar for `lang`; `None` → caller renders plain.
fn parse_tree(code: &str, lang: Option<&str>) -> Option<tree_sitter::Tree> {
    let sl = resolve_lang(lang?)?;
    let ts = sl.grammar().load().ok()?;
    grammar::parse(&ts, code).ok()?
}

/// Pre-order walk producing `(start, end, category)` spans over `code`.
fn collect_spans(tree: &tree_sitter::Tree, code: &str) -> Vec<(usize, usize, Cat)> {
    let mut out = Vec::new();
    let mut cursor = tree.walk();
    'walk: loop {
        let node = cursor.node();
        let cat = classify(&node, code);
        match cat {
            Some(c) => out.push((node.start_byte(), node.end_byte(), c)),
            None if node.child_count() == 0 => {
                out.push((node.start_byte(), node.end_byte(), Cat::Plain));
            }
            None => {
                if cursor.goto_first_child() {
                    continue 'walk;
                }
            }
        }
        loop {
            if cursor.goto_next_sibling() {
                continue 'walk;
            }
            if !cursor.goto_parent() {
                break 'walk;
            }
        }
    }
    out
}

/// Classify a node into a category, or `None` to descend into children.
fn classify(node: &tree_sitter::Node<'_>, code: &str) -> Option<Cat> {
    let kind = node.kind();
    if !node.is_named() {
        // Anonymous node kinds are the literal source text.
        let mut symbolic = true;
        let mut operator = false;
        for c in kind.chars() {
            if c.is_alphanumeric() || c == '_' {
                symbolic = false;
            } else if OP_CHARS.contains(&c) {
                operator = true;
            }
        }
        if symbolic {
            return Some(if operator { Cat::Operator } else { Cat::Punct });
        }
        return Some(if KEYWORDS.contains(&kind) {
            Cat::Keyword
        } else {
            Cat::Punct
        });
    }
    // Named nodes.
    if kind == "info_string" || kind == "insertion" || kind == "deletion" {
        return Some(match kind {
            "insertion" => Cat::Inserted,
            "deletion" => Cat::Deleted,
            _ => Cat::Punct,
        });
    }
    if kind.contains("comment") {
        return Some(Cat::Comment);
    }
    if kind.contains("string") || kind == "character" || kind.contains("escape") {
        return Some(Cat::Str);
    }
    if kind.contains("number") || kind == "integer" || kind == "float" || kind == "decimal" {
        return Some(Cat::Number);
    }
    if matches!(
        kind,
        "true" | "false" | "null" | "none" | "boolean" | "null_literal" | "boolean_literal"
    ) {
        return Some(Cat::Keyword);
    }
    if kind.ends_with("identifier") || kind == "constant" || kind == "type" {
        return Some(ident_cat(node, code));
    }
    None
}

/// Category for an identifier-like node, using its parent context.
fn ident_cat(node: &tree_sitter::Node<'_>, code: &str) -> Cat {
    let kind = node.kind();
    if kind == "type_identifier" || kind == "type" {
        return Cat::Type;
    }
    let text = &code[node.start_byte()..node.end_byte()];
    let parent = node.parent();
    let pk = parent.as_ref().map_or("", tree_sitter::Node::kind);
    // Callee position: named identifier children of a call node that are not
    // wrapped in an arguments node are the callee in every grammar here.
    if pk.contains("call") {
        return Cat::Function;
    }
    if let Some(p) = &parent {
        if p.child_by_field_name("name")
            .is_some_and(|n| n.id() == node.id())
        {
            let declaring = pk.contains("function")
                || pk.contains("method")
                || pk.contains("def")
                || pk.contains("proc");
            return if declaring {
                Cat::Function
            } else {
                Cat::Variable
            };
        }
    }
    let shouty = text.len() > 1
        && text.chars().any(|c: char| c.is_ascii_uppercase())
        && text
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit());
    if shouty {
        Cat::Type
    } else {
        Cat::Variable
    }
}

// ── Rendering spans into ratatui lines ──────────────────────────────────

fn to_text(code: &str, spans: &[(usize, usize, Cat)], styles: Styles) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut si = 0usize;
    let mut line_start = 0usize;
    for src in code.split('\n') {
        let line_end = line_start + src.len();
        let mut row: Vec<Span<'static>> = Vec::new();
        let mut p = line_start;
        while si < spans.len() && spans[si].0 < line_end {
            let (s, e, cat) = spans[si];
            if s >= e {
                si += 1;
                continue;
            }
            let start = s.max(p);
            let end = e.min(line_end);
            if start > p {
                row.push(plain_span(&code[p..start]));
            }
            if end > start {
                row.push(Span::styled(code[start..end].to_owned(), styles.style(cat)));
            }
            if e <= line_end {
                si += 1;
                p = e;
            } else {
                // The span continues on the next line: finish this row
                // without consuming it (looping on it would never advance).
                p = line_end;
                break;
            }
        }
        if p < line_end {
            row.push(plain_span(&code[p..line_end]));
        }
        lines.push(Line::from(row));
        line_start = line_end + 1;
    }
    // A trailing newline must not produce a phantom final empty line.
    if code.ends_with('\n') && lines.last().is_some_and(|l| l.spans.is_empty()) {
        lines.pop();
    }
    Text::from(lines)
}

fn plain_span(text: &str) -> Span<'static> {
    Span::raw(text.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::ThemeValue;
    use ratatui::style::Color;

    fn theme() -> Theme {
        let mut t = Theme::default();
        t.set_token("syntaxComment", ThemeValue::Color(Color::Green));
        t.set_token("syntaxKeyword", ThemeValue::Color(Color::Blue));
        t.set_token("syntaxString", ThemeValue::Color(Color::Yellow));
        t
    }

    #[test]
    fn fence_and_path_detection() {
        assert_eq!(lang_from_fence("rust"), Some("rust"));
        assert_eq!(lang_from_fence("py"), Some("python"));
        assert_eq!(lang_from_fence("shell"), Some("bash"));
        assert_eq!(lang_from_fence("  TS  extra"), Some("typescript"));
        assert_eq!(lang_from_fence(""), None);
        assert_eq!(lang_from_path("src/foo.rs"), Some("rust"));
        assert_eq!(lang_from_path("Dockerfile"), Some("dockerfile"));
        assert_eq!(lang_from_path("conf/Dockerfile.prod"), Some("dockerfile"));
        assert_eq!(lang_from_path(".env.local"), Some("env"));
        assert_eq!(lang_from_path("notes.MD"), Some("markdown"));
        assert!(is_markdown_path("README.md"));
        assert!(!is_markdown_path("README.rs"));
        assert_eq!(detect_language_id("Makefile"), "makefile");
        assert_eq!(detect_language_id("x.unknown"), "plaintext");
    }

    #[test]
    fn highlight_categories() {
        let t = theme();
        let text = highlight("let x = 1; // hi\n", Some("rust"), &t);
        let comment = text
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.trim().starts_with("//"))
            .expect("comment span");
        assert_eq!(comment.style.fg, Some(Color::Green));
        let kw = text
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content == "let")
            .expect("keyword span");
        assert_eq!(kw.style.fg, Some(Color::Blue));
    }

    #[test]
    fn stream_matches_batch() {
        let t = theme();
        let code = "fn main() {\n    let s = \"a b\";\n    println!(\"{s}\");\n}\n";
        let batch = highlight(code, Some("rust"), &t);
        let mut stream = HighlightStream::new(Some("rust"), &t);
        let mut acc: Vec<Line<'static>> = Vec::new();
        for line in code.split_inclusive('\n') {
            if let Some(text) = stream.push(line) {
                acc.extend(text.lines);
            }
        }
        acc.extend(stream.finish().lines);
        let streamed = Text::from(acc);
        assert_eq!(streamed.lines.len(), batch.lines.len());
        for (s, b) in streamed.iter().zip(batch.iter()) {
            let flat: String = s.spans.iter().map(|sp| sp.content.as_ref()).collect();
            let flatb: String = b.spans.iter().map(|sp| sp.content.as_ref()).collect();
            assert_eq!(flat, flatb);
        }
    }

    #[test]
    fn unknown_language_is_plain() {
        let t = theme();
        let text = highlight("just text\n", Some("klingon"), &t);
        assert_eq!(text.lines.len(), 1);
        assert!(text
            .iter()
            .all(|l| l.spans.iter().all(|s| s.style == Style::default())));
    }

    #[test]
    fn multiline_span_does_not_loop() {
        // An unparsable multi-line region comes back as one ERROR node.
        let text = highlight("$$\n$$", Some("py"), &theme());
        assert_eq!(text.lines.len(), 2);
    }
}
