//! Theme JSON schema — port of `theme/schema.ts` + `schema-validation.ts`.
//!
//! The required-token catalog mirrors the TS `themeColorsSchema`; `thinkingMax`
//! is optional. Values are `#rgb`/`#rrggbb` hex strings, 256-color indices, or
//! references into the theme's `vars` map.

/// Foreground tokens required by every theme (TS `themeColorsSchema`, minus
/// the optional `thinkingMax`).
pub const REQUIRED_COLORS: &[&str] = &[
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    "selectedBg",
    "userMessageBg",
    "userMessageText",
    "customMessageBg",
    "customMessageText",
    "customMessageLabel",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "toolTitle",
    "toolOutput",
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffContext",
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "bashMode",
    "pythonMode",
    "statusLineBg",
    "statusLineSep",
    "statusLineModel",
    "statusLinePath",
    "statusLineGitClean",
    "statusLineGitDirty",
    "statusLineContext",
    "statusLineSpend",
    "statusLineStaged",
    "statusLineDirty",
    "statusLineUntracked",
    "statusLineOutput",
    "statusLineCost",
    "statusLineSubagents",
];

/// Optional foreground token (valid but may be absent).
pub const OPTIONAL_COLORS: &[&str] = &["thinkingMax"];

/// Background tokens resolved into `Theme` bg-ish entries (kept as plain
/// tokens in the Rust port; components address them by name).
#[cfg(test)]
pub const BG_COLORS: &[&str] = &[
    "selectedBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "statusLineBg",
];

/// Valid symbol presets (TS `SymbolPreset`).
pub const SYMBOL_PRESETS: &[&str] = &["unicode", "nerd", "ascii"];

/// `true` when `preset` names a known symbol preset.
#[must_use]
pub fn is_valid_symbol_preset(preset: &str) -> bool {
    SYMBOL_PRESETS.contains(&preset)
}

/// camelCase → snake_case alias for every catalog token (static so the
/// loader never leaks keys).
pub static ALIASES: &[(&str, &str)] = &[
    ("borderAccent", "border_accent"),
    ("borderMuted", "border_muted"),
    ("thinkingText", "thinking_text"),
    ("selectedBg", "selected_bg"),
    ("userMessageBg", "user_message_bg"),
    ("userMessageText", "user_message_text"),
    ("customMessageBg", "custom_message_bg"),
    ("customMessageText", "custom_message_text"),
    ("customMessageLabel", "custom_message_label"),
    ("toolPendingBg", "tool_pending_bg"),
    ("toolSuccessBg", "tool_success_bg"),
    ("toolErrorBg", "tool_error_bg"),
    ("toolTitle", "tool_title"),
    ("toolOutput", "tool_output"),
    ("mdHeading", "md_heading"),
    ("mdLink", "md_link"),
    ("mdLinkUrl", "md_link_url"),
    ("mdCode", "md_code"),
    ("mdCodeBlock", "md_code_block"),
    ("mdCodeBlockBorder", "md_code_block_border"),
    ("mdQuote", "md_quote"),
    ("mdQuoteBorder", "md_quote_border"),
    ("mdHr", "md_hr"),
    ("mdListBullet", "md_list_bullet"),
    ("toolDiffAdded", "tool_diff_added"),
    ("toolDiffRemoved", "tool_diff_removed"),
    ("toolDiffContext", "tool_diff_context"),
    ("syntaxComment", "syntax_comment"),
    ("syntaxKeyword", "syntax_keyword"),
    ("syntaxFunction", "syntax_function"),
    ("syntaxVariable", "syntax_variable"),
    ("syntaxString", "syntax_string"),
    ("syntaxNumber", "syntax_number"),
    ("syntaxType", "syntax_type"),
    ("syntaxOperator", "syntax_operator"),
    ("syntaxPunctuation", "syntax_punctuation"),
    ("thinkingOff", "thinking_off"),
    ("thinkingMinimal", "thinking_minimal"),
    ("thinkingLow", "thinking_low"),
    ("thinkingMedium", "thinking_medium"),
    ("thinkingHigh", "thinking_high"),
    ("thinkingXhigh", "thinking_xhigh"),
    ("bashMode", "bash_mode"),
    ("pythonMode", "python_mode"),
    ("statusLineBg", "status_line_bg"),
    ("statusLineSep", "status_line_sep"),
    ("statusLineModel", "status_line_model"),
    ("statusLinePath", "status_line_path"),
    ("statusLineGitClean", "status_line_git_clean"),
    ("statusLineGitDirty", "status_line_git_dirty"),
    ("statusLineContext", "status_line_context"),
    ("statusLineSpend", "status_line_spend"),
    ("statusLineStaged", "status_line_staged"),
    ("statusLineDirty", "status_line_dirty"),
    ("statusLineUntracked", "status_line_untracked"),
    ("statusLineOutput", "status_line_output"),
    ("statusLineCost", "status_line_cost"),
    ("statusLineSubagents", "status_line_subagents"),
    ("thinkingMax", "thinking_max"),
];

/// Snake_case alias for a catalog token, or `None` when identical/unknown.
#[must_use]
pub fn snake_alias(token: &str) -> Option<&'static str> {
    ALIASES.iter().find(|(c, _)| *c == token).map(|(_, s)| *s)
}

/// Convert a TS camelCase token name to this crate's snake_case convention
/// (`statusLineSep` → `status_line_sep`), so components using either style
/// resolve.
#[must_use]
pub fn camel_to_snake(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_case_aliases() {
        assert_eq!(camel_to_snake("statusLineSep"), "status_line_sep");
        assert_eq!(camel_to_snake("accent"), "accent");
        assert_eq!(camel_to_snake("mdHeading"), "md_heading");
        assert_eq!(camel_to_snake("thinkingXhigh"), "thinking_xhigh");
    }

    #[test]
    fn preset_validation() {
        assert!(is_valid_symbol_preset("unicode"));
        assert!(!is_valid_symbol_preset("emoji"));
    }

    #[test]
    fn required_colors_unique() {
        let mut all = REQUIRED_COLORS.to_vec();
        all.sort_unstable();
        let n = all.len();
        all.dedup();
        assert_eq!(all.len(), n, "duplicate token in catalog");
        assert!(BG_COLORS.iter().all(|bg| REQUIRED_COLORS.contains(bg)));
    }
}
