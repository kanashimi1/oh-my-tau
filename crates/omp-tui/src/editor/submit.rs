//! Submit semantics — the editor side of the submit router
//! (`input-controller.ts:942-1290`, `interactive-mode.ts` `startPendingSubmission`).
//!
//! The editor decides *what kind* of submit a keypress means and packages
//! the expanded text; the app owns the actual routing (steer vs follow-up
//! vs queue vs slash dispatch) because it knows whether the agent is
//! streaming.

/// Which submit key was pressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubmitKind {
    /// Plain Enter: normal submit (steer if streaming — the app decides).
    #[default]
    Normal,
    /// `ctrl+enter` / `ctrl+q` (`app.message.followUp`): queue for delivery
    /// after the current turn; never steers.
    FollowUp,
}

/// What the editor hands the app when a submit key fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmitRequest {
    /// Expanded text: paste markers and attachment markers replaced with
    /// their payloads (`getExpandedText`).
    pub text: String,
    /// Which submit key produced this request.
    pub kind: SubmitKind,
    /// Whether the *visible* buffer was empty before expansion (empty
    /// submit while streaming means "interrupt", `:input-controller` §3).
    pub was_empty: bool,
    /// Whether the trimmed text is `.` or `c` — the synthetic continue
    /// prompt (`:4`).
    pub is_continue: bool,
    /// A `->`/`=>` queue shorthand with the prefix stripped
    /// (`parseQueueShorthand`, `queue-input.ts:17`).
    pub queue_body: Option<String>,
    /// A leading slash command split into `(name, args)` for the app's
    /// builtin dispatch (`:8`).
    pub slash: Option<(String, String)>,
}

impl SubmitRequest {
    /// Classify a raw visible buffer into a request.
    #[must_use]
    pub fn classify(visible: &str, expanded: String, kind: SubmitKind) -> Self {
        let trimmed = visible.trim();
        let was_empty = visible.is_empty();
        let is_continue = trimmed == "." || trimmed == "c";
        let queue_body = parse_queue_shorthand(trimmed).map(str::to_owned);
        let slash = parse_leading_slash(trimmed);
        Self {
            text: expanded,
            kind,
            was_empty,
            is_continue,
            queue_body,
            slash,
        }
    }
}

/// Parse a `->` / `=>` queue shorthand (`queue-input.ts:17`). The marker
/// must be the very first non-space token followed by at least one space.
#[must_use]
pub fn parse_queue_shorthand(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("->")
        .or_else(|| trimmed.strip_prefix("=>"))?;
    let rest = rest.strip_prefix(' ')?;
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

/// Split a leading slash command into `(name, args)`; `None` when the text
/// does not start with `/` or is just `/`.
#[must_use]
pub fn parse_leading_slash(trimmed: &str) -> Option<(String, String)> {
    let rest = trimmed.strip_prefix('/')?;
    let mut parts = rest.splitn(2, ' ');
    let name = parts.next()?.to_owned();
    if name.is_empty() {
        return None;
    }
    let args = parts.next().unwrap_or("").to_owned();
    Some((name, args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_plain() {
        let r = SubmitRequest::classify("hello", "hello".into(), SubmitKind::Normal);
        assert_eq!(r.text, "hello");
        assert!(!r.was_empty);
        assert!(!r.is_continue);
        assert!(r.queue_body.is_none());
        assert!(r.slash.is_none());
    }

    #[test]
    fn classify_empty() {
        let r = SubmitRequest::classify("", String::new(), SubmitKind::Normal);
        assert!(r.was_empty);
    }

    #[test]
    fn classify_continue_prompt() {
        assert!(SubmitRequest::classify(".", ".".into(), SubmitKind::Normal).is_continue);
        assert!(SubmitRequest::classify("c", "c".into(), SubmitKind::Normal).is_continue);
        assert!(!SubmitRequest::classify("cat", "cat".into(), SubmitKind::Normal).is_continue);
    }

    #[test]
    fn classify_queue_shorthand() {
        let r = SubmitRequest::classify(
            "-> do the thing",
            "-> do the thing".into(),
            SubmitKind::Normal,
        );
        assert_eq!(r.queue_body.as_deref(), Some("do the thing"));
        assert!(
            SubmitRequest::classify("->", "->".into(), SubmitKind::Normal)
                .queue_body
                .is_none()
        );
    }

    #[test]
    fn classify_slash_command() {
        let r = SubmitRequest::classify("/model gpt", "/model gpt".into(), SubmitKind::Normal);
        assert_eq!(r.slash, Some(("model".into(), "gpt".into())));
        assert!(SubmitRequest::classify("/", "/".into(), SubmitKind::Normal)
            .slash
            .is_none());
    }

    #[test]
    fn follow_up_kind_preserved() {
        let r = SubmitRequest::classify("hi", "hi".into(), SubmitKind::FollowUp);
        assert_eq!(r.kind, SubmitKind::FollowUp);
    }
}
