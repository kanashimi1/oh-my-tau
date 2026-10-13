//! LaTeX-to-unicode inline math — the port of `pi-tui/src/latex-to-unicode.ts`.
//!
//! [`to_unicode`] converts a single-line LaTeX math fragment into plain
//! unicode text: Greek letters, arrows, relations, big operators, `\sqrt`,
//! simple `\frac` (unicode vulgar fraction when both arguments are single
//! "simple" cells, `(a)/(b)` otherwise), single-char super/subscripts and
//! `\mathbb` double-struck sets. Anything unknown is kept legible: font and
//! sizing commands are stripped, `\operatorname{sin}`-style commands degrade
//! to their name, unmatched braces are removed.

use std::collections::BTreeMap;

/// Convert a single-line LaTeX math source to unicode text.
#[must_use]
pub fn to_unicode(src: &str) -> String {
    let syms = symbols();
    let mut out = String::with_capacity(src.len());
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            // Command or escaped char.
            let mut j = i + 1;
            let cmd: String = if j < chars.len() && chars[j].is_ascii_alphabetic() {
                while j < chars.len() && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                chars[i + 1..j].iter().collect()
            } else if j < chars.len() {
                j += 1;
                chars[i + 1..j].iter().collect()
            } else {
                String::new()
            };
            i = j;
            match cmd.as_str() {
                "frac" | "dfrac" | "tfrac" => {
                    let (num, ni) = read_group(&chars, i);
                    let (den, nj) = read_group(&chars, ni);
                    out.push_str(&frac(&num, &den));
                    i = nj;
                }
                "sqrt" => {
                    let (arg, ni) = read_group(&chars, i);
                    // \sqrt[n]{x} → ⁿ√(x)
                    out.push('√');
                    out.push('(');
                    out.push_str(&to_unicode(&arg));
                    out.push(')');
                    i = ni;
                }
                "mathbb" | "mathcal" | "mathfrak" | "mathscr" | "mathrm" | "mathbf" | "mathit"
                | "mathsf" | "operatorname" | "text" | "mathrmbf" | "boldsymbol" => {
                    let (arg, ni) = read_group(&chars, i);
                    out.push_str(&font_scope(&cmd, &arg));
                    i = ni;
                }
                "left" | "right" | "big" | "Big" | "bigl" | "bigr" | "Bigl" | "Bigr" | "biggl"
                | "biggr" | "displaystyle" | "limits" | "quad" | "qquad" => {
                    // \quad → space; the rest vanish.
                    if cmd == "quad" || cmd == "qquad" {
                        out.push(' ');
                    }
                }
                _ => {
                    if let Some(repl) = syms.get(cmd.as_str()) {
                        out.push_str(repl);
                    } else if cmd.len() == 1
                        && !cmd.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                    {
                        out.push_str(&cmd); // escaped char like \{ \} \%
                    } else if !cmd.is_empty() {
                        // Unknown operator (\sin, \log, ...): keep the name.
                        out.push_str(&cmd);
                    }
                }
            }
            continue;
        }
        if c == '^' || c == '_' {
            let (arg, ni) = read_group(&chars, i + 1);
            let table = if c == '^' { SUPERSCRIPTS } else { SUBSCRIPTS };
            let mut applied = String::new();
            let mut ok = !arg.is_empty();
            for ac in arg.chars() {
                if let Some((_, v)) = table.iter().find(|(k, _)| *k == ac) {
                    applied.push_str(v);
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push_str(&applied);
            } else {
                out.push(c);
                out.push_str(&arg);
            }
            i = ni;
            continue;
        }
        if c == '{' || c == '}' {
            i += 1;
            continue;
        }
        if c == '~' {
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out.trim().to_owned()
}

/// Read a `{...}` group (or the next single token) starting at `i`.
/// Returns the raw inner source and the index after the group.
fn read_group(chars: &[char], mut i: usize) -> (String, usize) {
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= chars.len() {
        return (String::new(), i);
    }
    if chars[i] == '{' {
        let mut depth = 1;
        let start = i + 1;
        i += 1;
        while i < chars.len() && depth > 0 {
            if chars[i] == '{' {
                depth += 1;
            } else if chars[i] == '}' {
                depth -= 1;
            }
            i += 1;
        }
        let end = if depth == 0 { i - 1 } else { i };
        (chars[start..end].iter().collect(), i)
    } else if chars[i] == '\\' && i + 1 < chars.len() {
        // Single command token: \alpha
        let mut j = i + 2;
        while j < chars.len() && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        (chars[i..j].iter().collect(), j)
    } else {
        (chars[i].to_string(), i + 1)
    }
}

/// `\frac{a}{b}`: unicode vulgar fraction for simple single cells, else (a)/(b).
fn frac(num: &str, den: &str) -> String {
    const VULGAR: &[((char, char), char)] = &[
        (('1', '2'), '½'),
        (('1', '3'), '⅓'),
        (('2', '3'), '⅔'),
        (('1', '4'), '¼'),
        (('3', '4'), '¾'),
        (('1', '5'), '⅕'),
        (('2', '5'), '⅖'),
        (('3', '5'), '⅗'),
        (('4', '5'), '⅘'),
        (('1', '6'), '⅙'),
        (('5', '6'), '⅚'),
        (('1', '8'), '⅛'),
        (('3', '8'), '⅜'),
        (('5', '8'), '⅝'),
        (('7', '8'), '⅞'),
    ];
    let mut nc = num.chars();
    let mut dc = den.chars();
    if let (Some(n), Some(d)) = (nc.next(), dc.next()) {
        if nc.next().is_none() && dc.next().is_none() {
            if let Some((_, v)) = VULGAR.iter().find(|((vn, vd), _)| *vn == n && *vd == d) {
                return v.to_string();
            }
        }
    }
    format!("({})/({})", to_unicode(num), to_unicode(den))
}

/// Font-command scopes: `\mathbb{R}` → ℝ; other fonts strip the command.
fn font_scope(cmd: &str, arg: &str) -> String {
    if cmd == "mathbb" {
        // Block-aligned letters use U+1D538+; the hole letters ℂℍℕℙℚℝℤ have
        // their own codepoints and must be matched on the original char.
        const HOLE: &[(char, char)] = &[
            ('C', 'ℂ'),
            ('H', 'ℍ'),
            ('N', 'ℕ'),
            ('P', 'ℙ'),
            ('Q', 'ℚ'),
            ('R', 'ℝ'),
            ('Z', 'ℤ'),
        ];
        return arg
            .chars()
            .map(|c| {
                if let Some((_, v)) = HOLE.iter().find(|(k, _)| *k == c) {
                    return *v;
                }
                match c {
                    'A'..='Z' => char::from_u32(0x1D538 + (c as u32 - 'A' as u32)).unwrap_or(c),
                    '1' => '𝟙',
                    _ => c,
                }
            })
            .collect();
    }
    if cmd == "mathcal" || cmd == "mathscr" {
        // The 0x1D49C block has unassigned holes; map common letters explicitly.
        const SCRIPT: &[(char, char)] = &[
            ('A', '𝒜'),
            ('B', 'ℬ'),
            ('C', '𝒞'),
            ('D', '𝒟'),
            ('E', 'ℰ'),
            ('F', 'ℱ'),
            ('G', '𝒢'),
            ('H', 'ℋ'),
            ('I', 'ℐ'),
            ('J', '𝒥'),
            ('K', '𝒦'),
            ('L', 'ℒ'),
            ('M', 'ℳ'),
            ('N', '𝒩'),
            ('O', '𝒪'),
            ('P', '𝒫'),
            ('Q', '𝒬'),
            ('R', 'ℛ'),
            ('S', '𝒮'),
            ('T', '𝒯'),
            ('U', '𝒰'),
            ('V', '𝒱'),
            ('W', '𝒲'),
            ('X', '𝒳'),
            ('Y', '𝒴'),
            ('Z', '𝒵'),
            ('e', 'ℯ'),
            ('g', 'ℊ'),
            ('o', 'ℴ'),
        ];
        return arg
            .chars()
            .map(|c| SCRIPT.iter().find(|(k, _)| *k == c).map_or(c, |(_, v)| *v))
            .collect();
    }
    to_unicode(arg)
}

const SUPERSCRIPTS: &[(char, &str)] = &[
    ('0', "⁰"),
    ('1', "¹"),
    ('2', "²"),
    ('3', "³"),
    ('4', "⁴"),
    ('5', "⁵"),
    ('6', "⁶"),
    ('7', "⁷"),
    ('8', "⁸"),
    ('9', "⁹"),
    ('+', "⁺"),
    ('-', "⁻"),
    ('=', "⁼"),
    ('(', "⁽"),
    (')', "⁾"),
    ('n', "ⁿ"),
    ('i', "ⁱ"),
    ('a', "ᵃ"),
    ('b', "ᵇ"),
    ('c', "ᶜ"),
    ('o', "ᵒ"),
    ('x', "ˣ"),
];

const SUBSCRIPTS: &[(char, &str)] = &[
    ('0', "₀"),
    ('1', "₁"),
    ('2', "₂"),
    ('3', "₃"),
    ('4', "₄"),
    ('5', "₅"),
    ('6', "₆"),
    ('7', "₇"),
    ('8', "₈"),
    ('9', "₉"),
    ('+', "₊"),
    ('-', "₋"),
    ('=', "₌"),
    ('(', "₍"),
    (')', "₎"),
    ('a', "ₐ"),
    ('e', "ₑ"),
    ('o', "ₒ"),
    ('x', "ₓ"),
    ('n', "ₙ"),
    ('i', "ᵢ"),
];

/// The symbol lookup table (TS `latex-to-unicode.ts` SYMB table subset).
#[allow(clippy::too_many_lines)] // one entry per symbol
fn symbols() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        // Greek lowercase
        ("alpha", "α"),
        ("beta", "β"),
        ("gamma", "γ"),
        ("delta", "δ"),
        ("epsilon", "ε"),
        ("varepsilon", "ε"),
        ("zeta", "ζ"),
        ("eta", "η"),
        ("theta", "θ"),
        ("vartheta", "ϑ"),
        ("iota", "ι"),
        ("kappa", "κ"),
        ("lambda", "λ"),
        ("mu", "μ"),
        ("nu", "ν"),
        ("xi", "ξ"),
        ("pi", "π"),
        ("varpi", "ϖ"),
        ("rho", "ρ"),
        ("varrho", "ϱ"),
        ("sigma", "σ"),
        ("varsigma", "ς"),
        ("tau", "τ"),
        ("upsilon", "υ"),
        ("phi", "φ"),
        ("varphi", "ϕ"),
        ("chi", "χ"),
        ("psi", "ψ"),
        ("omega", "ω"),
        // Greek uppercase
        ("Gamma", "Γ"),
        ("Delta", "Δ"),
        ("Theta", "Θ"),
        ("Lambda", "Λ"),
        ("Xi", "Ξ"),
        ("Pi", "Π"),
        ("Sigma", "Σ"),
        ("Upsilon", "Υ"),
        ("Phi", "Φ"),
        ("Psi", "Ψ"),
        ("Omega", "Ω"),
        // Big operators
        ("sum", "Σ"),
        ("prod", "Π"),
        ("coprod", "∐"),
        ("int", "∫"),
        ("iint", "∫∫"),
        ("iiint", "∫∫∫"),
        ("oint", "∮"),
        ("bigcup", "⋃"),
        ("bigcap", "⋂"),
        ("bigvee", "⋁"),
        ("bigwedge", "⋀"),
        ("bigoplus", "⊕"),
        ("bigotimes", "⊗"),
        // Relations
        ("leq", "≤"),
        ("le", "≤"),
        ("geq", "≥"),
        ("ge", "≥"),
        ("neq", "≠"),
        ("ne", "≠"),
        ("equiv", "≡"),
        ("approx", "≈"),
        ("cong", "≅"),
        ("simeq", "≃"),
        ("sim", "∼"),
        ("propto", "∝"),
        ("prec", "≺"),
        ("succ", "≻"),
        ("preceq", "⪯"),
        ("succeq", "⪰"),
        ("ll", "≪"),
        ("gg", "≫"),
        ("subset", "⊂"),
        ("supset", "⊃"),
        ("subseteq", "⊆"),
        ("supseteq", "⊇"),
        ("nsubseteq", "⊈"),
        ("in", "∈"),
        ("notin", "∉"),
        ("ni", "∋"),
        ("vdash", "⊢"),
        ("dashv", "⊣"),
        ("models", "⊨"),
        ("perp", "⊥"),
        ("parallel", "∥"),
        ("mid", "∣"),
        ("nmid", "∤"),
        ("asymp", "≍"),
        ("doteq", "≐"),
        ("triangleq", "≜"),
        // Arrows
        ("to", "→"),
        ("rightarrow", "→"),
        ("leftarrow", "←"),
        ("gets", "←"),
        ("Rightarrow", "⇒"),
        ("Leftarrow", "⇐"),
        ("Leftrightarrow", "⇔"),
        ("iff", "⇔"),
        ("implies", "⇒"),
        ("mapsto", "↦"),
        ("hookrightarrow", "↪"),
        ("twoheadrightarrow", "↠"),
        ("uparrow", "↑"),
        ("downarrow", "↓"),
        ("updownarrow", "↕"),
        ("Uparrow", "⇑"),
        ("Downarrow", "⇓"),
        ("nearrow", "↗"),
        ("searrow", "↘"),
        ("swarrow", "↙"),
        ("nwarrow", "↖"),
        ("longrightarrow", "⟶"),
        ("longleftarrow", "⟵"),
        ("longleftrightarrow", "⟷"),
        ("Longrightarrow", "⟹"),
        ("longLeftarrow", "⟸"),
        // Binary operators
        ("times", "×"),
        ("cdot", "·"),
        ("pm", "±"),
        ("mp", "∓"),
        ("div", "÷"),
        ("ast", "∗"),
        ("star", "⋆"),
        ("circ", "∘"),
        ("bullet", "∙"),
        ("oplus", "⊕"),
        ("ominus", "⊖"),
        ("otimes", "⊗"),
        ("oslash", "⊘"),
        ("odot", "⊙"),
        ("dagger", "†"),
        ("ddagger", "‡"),
        ("wr", "≀"),
        ("diamond", "⋄"),
        ("triangle", "△"),
        ("setminus", "∖"),
        ("cup", "∪"),
        ("cap", "∩"),
        ("vee", "∨"),
        ("wedge", "∧"),
        ("land", "∧"),
        ("lor", "∨"),
        // Delimiters / punctuation
        ("langle", "⟨"),
        ("rangle", "⟩"),
        ("lceil", "⌈"),
        ("rceil", "⌉"),
        ("lfloor", "⌊"),
        ("rfloor", "⌋"),
        ("lbrace", "{"),
        ("rbrace", "}"),
        ("lvert", "|"),
        ("rvert", "|"),
        ("Vert", "‖"),
        ("vert", "|"),
        ("backslash", "\\"),
        ("prime", "′"),
        ("angle", "∠"),
        ("measuredangle", "∡"),
        // Logic / sets / misc
        ("forall", "∀"),
        ("exists", "∃"),
        ("nexists", "∄"),
        ("neg", "¬"),
        ("lnot", "¬"),
        ("emptyset", "∅"),
        ("varnothing", "∅"),
        ("complement", "∁"),
        ("aleph", "ℵ"),
        ("beth", "ℶ"),
        ("ell", "ℓ"),
        ("Re", "ℜ"),
        ("Im", "ℑ"),
        ("wp", "℘"),
        ("hbar", "ℏ"),
        ("infty", "∞"),
        ("partial", "∂"),
        ("nabla", "∇"),
        ("top", "⊤"),
        ("bot", "⊥"),
        ("therefore", "∴"),
        ("because", "∵"),
        ("QED", "∎"),
        ("checkmark", "✓"),
        ("dag", "†"),
        ("S", "§"),
        ("P", "¶"),
        ("copyright", "©"),
        ("degree", "°"),
        // Dots
        ("ldots", "…"),
        ("dots", "…"),
        ("cdots", "⋯"),
        ("vdots", "⋮"),
        ("ddots", "⋱"),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greek_and_operators() {
        assert_eq!(to_unicode("\\alpha \\times \\beta"), "α × β");
        assert_eq!(to_unicode("\\sum_{i=1}^n i"), "Σᵢ₌₁ⁿ i");
    }

    #[test]
    fn frac_simple_and_compound() {
        assert_eq!(to_unicode("\\frac{1}{2}"), "½");
        assert_eq!(to_unicode("\\frac{a+b}{c}"), "(a+b)/(c)");
    }

    #[test]
    fn sqrt_and_mathbb() {
        assert_eq!(to_unicode("\\sqrt{x+1}"), "√(x+1)");
        assert_eq!(to_unicode("\\mathbb{R} \\to \\mathbb{C}"), "ℝ → ℂ");
    }

    #[test]
    fn unknown_operator_keeps_name() {
        assert_eq!(to_unicode("\\sin x + \\log y"), "sin x + log y");
    }

    #[test]
    fn arrows_and_relations() {
        assert_eq!(to_unicode("A \\Rightarrow B \\iff C"), "A ⇒ B ⇔ C");
        assert_eq!(to_unicode("x \\in S, y \\notin S"), "x ∈ S, y ∉ S");
    }

    #[test]
    fn nested_groups() {
        assert_eq!(to_unicode("\\frac{\\sqrt{2}}{2}"), "(√(2))/(2)");
    }
}
