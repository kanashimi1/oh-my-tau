//! Port of `test/user-metrics.test.ts` (plus the JS-semantics unit tests).

use super::*;

fn m(text: &str) -> UserMessageMetrics {
    compute_user_message_metrics(text)
}

#[test]
fn returns_zeros_for_empty_or_whitespace_only_text() {
    assert_eq!(m(""), EMPTY_USER_METRICS);
    assert_eq!(m("   \n\t "), EMPTY_USER_METRICS);
}

#[test]
fn counts_multi_word_caps_as_yelling() {
    assert_eq!(m("STOP DOING THAT NOW").yelling, 1);
}

#[test]
fn mostly_lowercase_sentence_is_not_yelling() {
    assert_eq!(
        m("Hi there, please STOP doing THAT immediately, it is really annoying.").yelling,
        0
    );
}

#[test]
fn ignores_short_uppercase_fragments_below_the_letter_floor() {
    assert_eq!(m("OK").yelling, 0);
    assert_eq!(m("WIP.").yelling, 0);
}

#[test]
fn counts_multiple_yelling_sentences_split_by_terminators() {
    assert_eq!(m("WHY IS THIS BROKEN? FIX IT NOW!! please.").yelling, 2);
}

#[test]
fn does_not_flag_camel_case_or_acronyms_in_lowercase_prose() {
    assert_eq!(m("call getHTMLParser then exit").yelling, 0);
}

#[test]
fn requires_multi_word_caps_or_an_elongated_shout() {
    assert_eq!(m("Follow AGENTS.md and SYSTEM.md carefully.").yelling, 0);
    assert_eq!(m("HELLO").yelling, 0);
    assert_eq!(m("WHAT THE HELL").yelling, 1);
    assert_eq!(m("CMOOON").yelling, 1);
}

#[test]
fn profanity_is_case_insensitive_and_word_bounded() {
    // Regression: a non-raw template literal once compiled `\b` as backspace,
    // so the pattern matched nothing in real prose.
    assert_eq!(m("oh FUCK this is bullshit, damn it").profanity, 3);
    assert_eq!(m("import classes from module").profanity, 0);
}

#[test]
fn technical_and_opinion_vocabulary_is_not_profanity() {
    assert_eq!(m("this is garbage, useless and horrible work").profanity, 0);
    assert_eq!(m("dummy data, blast radius, and config knob").profanity, 0);
}

#[test]
fn version_control_prose_is_not_profanity() {
    // `git` was dropped from the list (the slang plural `gits` stays).
    assert_eq!(m("git status shows the rebase failed").profanity, 0);
    assert_eq!(m("run git commit then git push").profanity, 0);
}

#[test]
fn drama_runs_and_elongated_interjections_count_as_anguish_but_dot_trails_do_not() {
    assert_eq!(m("why!!! seriously??? omg!?!?!?").anguish, 3);
    assert_eq!(m("nooo whyyy").anguish, 2);
    assert_eq!(m("ok!! sure??").anguish, 0);
    assert_eq!(m("waiting.. still loading...").anguish, 0);
}

#[test]
fn plain_interjections_are_anguish_not_profanity() {
    for text in ["ugh", "argh", "grr"] {
        let got = m(text);
        assert_eq!(got.anguish, 1, "{text}");
        assert_eq!(got.profanity, 0, "{text}");
    }
    let elongated = m("ughh");
    assert_eq!(elongated.anguish, 1);
    assert_eq!(elongated.profanity, 0);
}

#[test]
fn absorbs_shift_key_mishits_into_one_drama_burst() {
    assert!(m("what!!!111").anguish >= 1);
    assert!(m("are you serious!?!?!??111").anguish >= 1);
    assert_eq!(m("port 8111 please").anguish, 0);
}

#[test]
fn sad_emoticons_count_without_matching_code_punctuation() {
    assert!(m("nope still same :(").anguish >= 1);
    assert_eq!(m(":((").anguish, 1);
    assert_eq!(m("match foo:(bar) in code").anguish, 0);
}

#[test]
fn negation_fires_on_line_leading_correction_openers() {
    assert_eq!(m("no this is the renderer").negation, 1);
    assert_eq!(m("nope, still wrong").negation, 1);
    assert_eq!(m("nah look at this").negation, 1);
    assert_eq!(m("wrong file").negation, 1);
    assert_eq!(m("nvm got it").negation, 1);
}

#[test]
fn bare_leading_no_is_gated_to_interjection_uses() {
    assert_eq!(m("no extensions to the landing page").negation, 0);
    assert_eq!(m("no-op change for the flag").negation, 0);
    assert_eq!(m("no - not that one").negation, 1);
    assert_eq!(m("no\u{2014}wait").negation, 1);
    assert_eq!(m("no i meant the other one").negation, 1);
    assert_eq!(m("no, wait").negation, 1);
}

#[test]
fn negation_ignores_words_sharing_a_prefix() {
    assert_eq!(m("now everything works").negation, 0);
    assert_eq!(m("nobody knows why").negation, 0);
    assert_eq!(m("normal operation resumed").negation, 0);
}

#[test]
fn negation_anchors_only_at_the_start_of_the_message() {
    // Pasted error text / bullet lists tripped the old `^...$/m` anchor.
    assert_eq!(
        m("i instantly get Finalizing ->\nNo speech detected").negation,
        0
    );
    assert_eq!(
        m("Authentication failed\n\nWrong user name or password").negation,
        0
    );
}

#[test]
fn image_markers_are_stripped_so_leading_negation_still_fires() {
    assert_eq!(m("[Image #1] nope still broken").negation, 1);
}

#[test]
fn negation_fires_on_rejection_and_nonsense_phrases() {
    assert_eq!(m("thats not what i wanted").negation, 1);
    assert_eq!(m("that's not right").negation, 1);
    assert_eq!(m("this is not what i meant at all").negation, 1);
    assert_eq!(m("ok but this makes no sense").negation, 1);
    assert_eq!(m("ok but this makes zero sense").negation, 1);
    assert_eq!(m("the docs explain it well and it makes sense").negation, 0);
}

#[test]
fn repetition_counts_explicit_recall_verbs() {
    assert_eq!(m("i meant the other file").repetition, 1);
    assert_eq!(m("i told you to skip it").repetition, 1);
    assert_eq!(m("i asked you for json not yaml").repetition, 1);
    assert_eq!(m("like i said earlier").repetition, 1);
}

#[test]
fn i_asked_requires_you_to_suppress_third_party_usage() {
    assert_eq!(m("i asked the committee to review").repetition, 0);
    assert_eq!(m("so i asked a bunch of experts").repetition, 0);
    assert_eq!(
        m("you're not doing AST rewriting like i asked").repetition,
        1
    );
}

#[test]
fn still_counts_only_with_a_negative_or_sameness_marker() {
    assert_eq!(m("the agent still works fine").repetition, 0);
    assert_eq!(m("it still doesnt work").repetition, 1);
    assert_eq!(m("still the same issue").repetition, 1);
    assert_eq!(m("still failing on darwin").repetition, 1);
}

#[test]
fn blame_fires_on_accusatory_second_person_verbs() {
    assert_eq!(m("you broke the layout").blame, 1);
    assert_eq!(m("you didnt update AGENTS").blame, 1);
    assert_eq!(m("you missed a callsite").blame, 1);
    assert_eq!(m("you forgot to commit").blame, 1);
    assert_eq!(m("you keep doing that").blame, 1);
}

#[test]
fn blame_ignores_bare_you() {
    assert_eq!(m("can you fix the bug?").blame, 0);
    assert_eq!(m("could you also add a test").blame, 0);
}

#[test]
fn blame_fires_on_blamey_why_questions_only() {
    assert_eq!(m("why did you list the request as is?").blame, 1);
    assert_eq!(m("why would u delete the config?").blame, 1);
    assert_eq!(m("why is this slow").blame, 0);
    assert_eq!(m("why does it use webpack").blame, 0);
}

#[test]
fn stop_xing_counts_only_at_sentence_start() {
    assert_eq!(m("stop touching git").blame, 1);
    assert_eq!(m("please stop making yolo changes").blame, 0);
    assert_eq!(m("ok. stop reverting things").blame, 1);
    assert_eq!(m("the loop keeps stopping").blame, 0);
}

#[test]
fn behavior_signals_zero_out_on_long_structured_prompts() {
    let long = [
        "no this is wrong, you broke it, i meant the other one.",
        "please undo and try again.",
        "acceptance: green tests.",
        "thanks!",
    ]
    .join("\n");
    let got = m(&long);
    assert_eq!(got.negation, 0);
    assert_eq!(got.repetition, 0);
    assert_eq!(got.blame, 0);
    assert_eq!(got.anguish, 0);
    assert_eq!(got.profanity, 0);
    assert_eq!(got.yelling, 0);
    assert!(got.chars > 0);
    assert!(got.words > 0);
}

#[test]
fn captures_multiple_signals_on_one_short_message() {
    let got = m("no, you broke it AGAIN. i told you it still doesnt work");
    assert_eq!(got.negation, 1);
    assert_eq!(got.blame, 1);
    assert!(got.repetition >= 2); // `i told you` + `still doesnt`
}

#[test]
fn chars_counts_utf16_code_units_like_js() {
    assert_eq!(m("\u{1f600}").chars, 2); // JS "\u{1f600}".length === 2
    assert_eq!(m("hi there").chars, 8);
    assert_eq!(m("hi there").words, 2);
}

#[test]
fn profanity_table_entry_count_matches_the_ts_source() {
    assert_eq!(tables::profanity().len(), tables::PROFANITY_COUNT);
    assert_eq!(tables::ANGUISH_PATTERNS.len(), 13);
}

#[test]
fn judge_prose_strips_structured_content_and_caps_length() {
    assert_eq!(judge_prose("  hi  "), "hi");
    assert_eq!(judge_prose("see `rm -rf /` now"), "see   now");
    assert_eq!(judge_prose("```\ncode\n```\n\nreal prose"), "real prose");
    assert_eq!(judge_prose("<tool><x/></tool>hello"), "hello");
    assert_eq!(
        judge_prose("[Image #1]\n> quoted\nsee @file.rs and https://x.dev/y"),
        "see   and"
    );

    let long = "a".repeat(PROSE_MAX_CHARS + 500);
    let capped = judge_prose(&long);
    assert_eq!(capped.encode_utf16().count(), PROSE_MAX_CHARS);
}

#[test]
fn markup_only_messages_yield_empty_prose() {
    assert_eq!(judge_prose("```\nall code\n```"), "");
}
