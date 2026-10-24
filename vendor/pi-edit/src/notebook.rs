//! Jupyter notebook (`.ipynb`) codec: projects cells onto an editable text
//! form (`# %% [type] cell:N` markers) and serializes edits back to JSON
//! with `JSON.stringify(nb, null, 1)`-identical bytes.
//!
//! Port of `packages/coding-agent/src/edit/notebook.ts`.

use std::{collections::HashSet, fmt::Write as _, path::Path, sync::LazyLock};

use regex::Regex;
use serde_json::{Map, Value};

/// Notebook decode/encode failure. `Display` is the exact model-facing text.
#[derive(Debug, thiserror::Error)]
pub enum NotebookError {
    #[error("Invalid JSON in notebook: {display}")]
    InvalidJson { display: String },
    #[error("Invalid notebook structure (expected object): {display}")]
    NotObject { display: String },
    #[error("Invalid notebook structure (missing cells array): {display}")]
    MissingCells { display: String },
    #[error("Invalid notebook cell {index} in {display}")]
    InvalidCell { index: usize, display: String },
    #[error(
        "Invalid notebook editable representation for {display}: expected first line to be \"# %% \
		 [code] cell:0\", \"# %% [markdown] cell:0\", or \"# %% [raw] cell:0\"."
    )]
    InvalidEditableText { display: String },
}

/// A cell type supported by the editable notebook projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotebookCellType {
    /// An executable code cell.
    Code,
    /// A Markdown cell.
    Markdown,
    /// A raw-text cell.
    Raw,
}

impl NotebookCellType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "code" => Some(Self::Code),
            "markdown" => Some(Self::Markdown),
            "raw" => Some(Self::Raw),
            _ => None,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Markdown => "markdown",
            Self::Raw => "raw",
        }
    }
}

/// One cell parsed from the editable notebook projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedVirtualCell {
    /// Parsed cell kind.
    pub cell_type: NotebookCellType,
    /// Original cell index, when the marker named one.
    pub cell_index: Option<usize>,
    /// Unescaped cell source.
    pub source: String,
}

static CELL_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^# %% \[(code|markdown|raw)\](?: cell:(\d+))?$")
        .expect("valid notebook cell marker regex")
});
static ESCAPABLE_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^# %%+ \[(?:code|markdown|raw)\](?: cell:\d+)?$")
        .expect("valid notebook escape regex")
});
static ESCAPED_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^# %%%+ \[(?:code|markdown|raw)\](?: cell:\d+)?$")
        .expect("valid notebook unescape regex")
});

fn escape_marker_like_source_lines(source: &str) -> String {
    if !source.contains("# %%") {
        return source.to_owned();
    }
    source
        .split('\n')
        .map(|line| {
            if ESCAPABLE_MARKER_RE.is_match(line) {
                line.replacen("# %", "# %%", 1)
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unescape_marker_like_line(line: &str) -> String {
    if ESCAPED_MARKER_RE.is_match(line) {
        line.replacen("# %%", "# %", 1)
    } else {
        line.to_owned()
    }
}

/// True when the extension is `.ipynb` (case-insensitive).
pub fn is_notebook_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("ipynb"))
}

/// Split notebook source text into nbformat `source` lines (each keeps its
/// trailing `\n`; a final unterminated line is kept as-is).
pub fn split_notebook_source(content: &str) -> Vec<String> {
    if content.is_empty() {
        Vec::new()
    } else {
        content.split_inclusive('\n').map(str::to_owned).collect()
    }
}

fn source_to_text(source: Option<&Value>) -> String {
    match source {
        None => String::new(),
        Some(Value::String(source)) => source.clone(),
        Some(Value::Array(lines)) => lines.iter().filter_map(Value::as_str).collect(),
        Some(_) => String::new(),
    }
}

fn validate_notebook<'a>(
    value: &'a Value,
    display: &str,
) -> Result<&'a Map<String, Value>, NotebookError> {
    let object = value.as_object().ok_or_else(|| NotebookError::NotObject {
        display: display.to_owned(),
    })?;
    let cells = object
        .get("cells")
        .and_then(Value::as_array)
        .ok_or_else(|| NotebookError::MissingCells {
            display: display.to_owned(),
        })?;
    for (index, cell) in cells.iter().enumerate() {
        let valid = cell
            .as_object()
            .and_then(|cell| cell.get("cell_type"))
            .and_then(Value::as_str)
            .and_then(NotebookCellType::parse)
            .is_some();
        if !valid {
            return Err(NotebookError::InvalidCell {
                index,
                display: display.to_owned(),
            });
        }
    }
    Ok(object)
}

/// Decode notebook JSON into the editable text form.
pub fn notebook_to_editable_text(json: &str, display: &str) -> Result<String, NotebookError> {
    let json = json.strip_prefix('\u{feff}').unwrap_or(json);
    let notebook: Value = serde_json::from_str(json).map_err(|_| NotebookError::InvalidJson {
        display: display.to_owned(),
    })?;
    let object = validate_notebook(&notebook, display)?;
    let cells = object
        .get("cells")
        .and_then(Value::as_array)
        .expect("validated cells");
    Ok(cells
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            let cell = cell.as_object().expect("validated cell");
            let cell_type = cell
                .get("cell_type")
                .and_then(Value::as_str)
                .expect("validated cell type");
            let source = escape_marker_like_source_lines(&source_to_text(cell.get("source")));
            if source.is_empty() {
                format!("# %% [{cell_type}] cell:{index}")
            } else {
                format!("# %% [{cell_type}] cell:{index}\n{source}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Parse an editable notebook projection into virtual cells.
pub fn parse_notebook_editable_text(
    text: &str,
    display: &str,
) -> Result<Vec<ParsedVirtualCell>, NotebookError> {
    let mut cells = Vec::new();
    let mut current: Option<(NotebookCellType, Option<usize>, Vec<String>)> = None;
    let lines: Vec<&str> = if text.is_empty() {
        Vec::new()
    } else {
        text.split('\n').collect()
    };

    for line in lines {
        if let Some(captures) = CELL_MARKER_RE.captures(line) {
            if let Some((cell_type, cell_index, source_lines)) = current.take() {
                cells.push(ParsedVirtualCell {
                    cell_type,
                    cell_index,
                    source: source_lines.join("\n"),
                });
            }
            let cell_type =
                NotebookCellType::parse(captures.get(1).expect("cell type capture").as_str())
                    .expect("validated marker type");
            let cell_index = captures
                .get(2)
                .and_then(|value| value.as_str().parse().ok());
            current = Some((cell_type, cell_index, Vec::new()));
        } else if let Some((_, _, source_lines)) = current.as_mut() {
            source_lines.push(unescape_marker_like_line(line));
        } else {
            return Err(NotebookError::InvalidEditableText {
                display: display.to_owned(),
            });
        }
    }
    if let Some((cell_type, cell_index, source_lines)) = current {
        cells.push(ParsedVirtualCell {
            cell_type,
            cell_index,
            source: source_lines.join("\n"),
        });
    }
    Ok(cells)
}

/// Create a new nbformat cell with default metadata and code-cell state.
pub fn create_notebook_cell(cell_type: NotebookCellType, source: &str) -> Value {
    let mut cell = Map::new();
    cell.insert("cell_type".into(), Value::String(cell_type.as_str().into()));
    cell.insert("metadata".into(), Value::Object(Map::new()));
    cell.insert(
        "source".into(),
        Value::Array(
            split_notebook_source(source)
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    if cell_type == NotebookCellType::Code {
        cell.insert("execution_count".into(), Value::Null);
        cell.insert("outputs".into(), Value::Array(Vec::new()));
    }
    Value::Object(cell)
}

/// Create an empty nbformat 4.5 notebook document.
pub fn create_empty_notebook() -> Value {
    let mut notebook = Map::new();
    notebook.insert("cells".into(), Value::Array(Vec::new()));
    notebook.insert("metadata".into(), Value::Object(Map::new()));
    notebook.insert("nbformat".into(), Value::from(4));
    notebook.insert("nbformat_minor".into(), Value::from(5));
    Value::Object(notebook)
}

/// Apply editable text to a validated notebook while preserving document and
/// reused-cell fields and their insertion order.
pub fn apply_notebook_editable_text(
    notebook: &Value,
    text: &str,
    display: &str,
) -> Result<Value, NotebookError> {
    let object = validate_notebook(notebook, display)?;
    let original_cells = object
        .get("cells")
        .and_then(Value::as_array)
        .expect("validated cells");
    let parsed_cells = parse_notebook_editable_text(text, display)?;
    let mut used_original_cells = HashSet::new();
    let mut next_cells = Vec::with_capacity(parsed_cells.len());

    for parsed in parsed_cells {
        let original = parsed
            .cell_index
            .filter(|index| *index < original_cells.len() && used_original_cells.insert(*index))
            .map(|index| &original_cells[index]);
        if let Some(original) = original {
            let mut cell = original.as_object().expect("validated cell").clone();
            cell.insert(
                "cell_type".into(),
                Value::String(parsed.cell_type.as_str().into()),
            );
            cell.insert(
                "source".into(),
                Value::Array(
                    split_notebook_source(&parsed.source)
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                ),
            );
            if parsed.cell_type == NotebookCellType::Code {
                if cell.get("execution_count").is_none_or(Value::is_null) {
                    cell.insert("execution_count".into(), Value::Null);
                }
                if cell.get("outputs").is_none_or(Value::is_null) {
                    cell.insert("outputs".into(), Value::Array(Vec::new()));
                }
            } else {
                cell.remove("execution_count");
                cell.remove("outputs");
            }
            next_cells.push(Value::Object(cell));
        } else {
            next_cells.push(create_notebook_cell(parsed.cell_type, &parsed.source));
        }
    }

    let mut next_notebook = object.clone();
    next_notebook.insert("cells".into(), Value::Array(next_cells));
    Ok(Value::Object(next_notebook))
}

/// Apply edited text back onto the original notebook JSON (`None` = the
/// notebook did not exist; an empty nbformat 4.5 document is used) and
/// serialize with 1-space indentation.
pub fn serialize_edited_notebook_text(
    original_json: Option<&str>,
    text: &str,
    display: &str,
) -> Result<String, NotebookError> {
    let notebook = match original_json {
        Some(json) => {
            let json = json.strip_prefix('\u{feff}').unwrap_or(json);
            serde_json::from_str(json).map_err(|_| NotebookError::InvalidJson {
                display: display.to_owned(),
            })?
        }
        None => create_empty_notebook(),
    };
    let next_notebook = apply_notebook_editable_text(&notebook, text, display)?;
    Ok(stringify_indent1(&next_notebook))
}

fn js_number_to_string(number: &serde_json::Number) -> String {
    if let Some(value) = number.as_i64() {
        return value.to_string();
    }
    if let Some(value) = number.as_u64() {
        return value.to_string();
    }
    let value = number.as_f64().expect("JSON numbers are finite");
    if value == 0.0 {
        return "0".into();
    }
    let rendered = value.abs().to_string();
    let (mantissa, explicit_exponent) =
        rendered
            .split_once(['e', 'E'])
            .map_or((rendered.as_str(), 0), |(mantissa, exponent)| {
                (
                    mantissa,
                    exponent.parse::<i32>().expect("valid float exponent"),
                )
            });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len());
    let mut digits = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect::<String>();
    let leading_zeros = digits.len() - digits.trim_start_matches('0').len();
    let mut decimal_position = i32::try_from(decimal).expect("short float mantissa")
        - i32::try_from(leading_zeros).expect("short float mantissa")
        + explicit_exponent;
    digits.drain(..leading_zeros);
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        digits.push('0');
        decimal_position = 1;
    }
    let scientific_exponent = decimal_position - 1;
    let mut output = String::new();
    if value.is_sign_negative() {
        output.push('-');
    }
    if decimal_position > 0 && decimal_position <= 21 {
        let position = usize::try_from(decimal_position).expect("positive decimal position");
        if position >= digits.len() {
            output.push_str(&digits);
            output.extend(std::iter::repeat_n('0', position - digits.len()));
        } else {
            output.push_str(&digits[..position]);
            output.push('.');
            output.push_str(&digits[position..]);
        }
    } else if decimal_position <= 0 && decimal_position > -6 {
        output.push_str("0.");
        output.extend(std::iter::repeat_n(
            '0',
            usize::try_from(-decimal_position).expect("nonnegative zero count"),
        ));
        output.push_str(&digits);
    } else {
        output.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            output.push('.');
            output.push_str(&digits[1..]);
        }
        output.push('e');
        if scientific_exponent >= 0 {
            output.push('+');
        }
        write!(output, "{scientific_exponent}").expect("writing to String cannot fail");
    }
    output
}

fn write_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0000}'..='\u{001f}' => write!(output, "\\u{:04x}", u32::from(character))
                .expect("writing to String cannot fail"),
            _ => output.push(character),
        }
    }
    output.push('"');
}

fn write_indent(output: &mut String, depth: usize) {
    output.extend(std::iter::repeat_n(' ', depth));
}

fn write_json_value(output: &mut String, value: &Value, depth: usize) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&js_number_to_string(value)),
        Value::String(value) => write_json_string(output, value),
        Value::Array(values) => {
            if values.is_empty() {
                output.push_str("[]");
                return;
            }
            output.push_str("[\n");
            for (index, value) in values.iter().enumerate() {
                write_indent(output, depth + 1);
                write_json_value(output, value, depth + 1);
                if index + 1 != values.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            write_indent(output, depth);
            output.push(']');
        }
        Value::Object(values) => {
            if values.is_empty() {
                output.push_str("{}");
                return;
            }
            output.push_str("{\n");
            for (index, (key, value)) in values.iter().enumerate() {
                write_indent(output, depth + 1);
                write_json_string(output, key);
                output.push_str(": ");
                write_json_value(output, value, depth + 1);
                if index + 1 != values.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            write_indent(output, depth);
            output.push('}');
        }
    }
}

/// `JSON.stringify(value, null, 1)`-identical serialization.
pub fn stringify_indent1(value: &Value) -> String {
    let mut output = String::new();
    write_json_value(&mut output, value, 0);
    output
}
