//! Word tables ported from `stats/src/user-metrics.ts`.
//!
//! The profanity list is kept as whitespace-separated text (order preserved,
//! because the generated alternation is leftmost-first exactly like the JS one)
//! so it stays diffable against the TS source. {@link PROFANITY_COUNT} locks the
//! entry count in a test, so a bad transcription fails loudly instead of
//! silently changing scores.

/// Profanity / aggression vocabulary, word-boundary + case-insensitive.
///
/// Broad English coverage: f-/s-word families and their censored variants,
/// mild swears, intelligence-based insults, body-part insults, British/
/// Australian/Irish slang, religious exclamations and chat acronyms. Curated
/// to exclude racial/homophobic slurs and words whose dominant use in a
/// coding corpus is technical (`dummy` data, `blast` radius, config `knob`,
/// `trash` bin, CRUD, `garbage` files) or plain opinion (`useless`, `awful`,
/// `meh`, `hate`). Pure frustration interjections (`ugh`, `argh`, `grr`) score
/// as anguish instead.
const PROFANITY_WORDS: &str = "\
    fuck fucks fucked fucking fuckin fucker fuckers fuckup fuckups fuckhead fuckheads fuckface \
    fuckwit fuckwits fucktard fuckery fuckoff motherfucker motherfuckers motherfucking \
    clusterfuck ratfuck unfuck \
    fk fks fking fkin fker fck fcks fcking fckin fcker fuk fuking fukin eff effs effed effing \
    frick fricks fricked fricking frickin freaking freakin freaked \
    shit shits shat shitty shittier shittiest shite shites shited shitting shitter shitters \
    shithead shitheads shitshow shitstorm shitstain shitfaced shitload shitbag shitcan \
    shitcanned shitpost shitposting bullshit bullshits bullshitting bullshitter horseshit \
    batshit dogshit dipshit jackshit dumbshit holyshit \
    damn damns damned damning dammit goddamn goddamned goddamnit goddammit darn darns darned \
    darnit dang danged dangit hell hells heck hecks heckin gosh bloody bollocks bollox \
    crap craps crappy crappier crappiest crapped crapping crapload crapola \
    piss pisses pissed pissing pisser pisspoor pisstake pisshead \
    ass asses asshole assholes asshat asshats asswipe asswipes assclown assbag asskisser \
    dumbass dumbasses jackass jackasses smartass smartasses badass badasses lazyass fatass \
    hardass halfass halfassed arse arsed arsehole arseholes arsewipe \
    bitch bitches bitched bitching bitchy bitchier bitchiest sonofabitch biatch biotch \
    cunt cunts cunty cuntish twat twats twatty bastard bastards \
    dick dicks dickhead dickheads dickish dickwad dickwads dickface dickbag prick pricks \
    prickish cock cocks cocky cockier cockiest cockhead cockblock cocksucker cocksuckers \
    knobhead knobheads knobend wanker wankers wankery tosser tossers jerkoff jerkoffs \
    douche douches douchebag douchebags douchey scumbag scumbags scum sleazebag sleazeball \
    slimeball lowlife lowlifes deadbeat \
    idiot idiots idiotic idiocy stupid stupider stupidest stupidity moron morons moronic \
    imbecile imbeciles retard retards retarded dumb dumber dumbest dumbo fool fools foolish \
    foolery clown clowns clownish buffoon buffoons simpleton halfwit halfwits nitwit nitwits \
    dimwit dimwits dolt dolts doltish knucklehead knuckleheads blockhead blockheads \
    lamebrain airhead airheads scatterbrain numbnuts numbskull numpty numpties muppet muppets \
    pillock pillocks plonker plonkers prat prats berk berks ninny ninnies dingbat dingbats \
    putz putzes schmuck schmucks jerk jerks jerkface gits sod sodding bugger buggered \
    suck sucks sucked sucking sucky suckage trashy \
    jesus christ jeez jeezus sheesh godsake \
    wtf wth wtaf stfu gtfo omfg omg ffs jfc kys fml smh smdh smfh idgaf idfc lmfao fubar snafu";

/// Entry count of [`PROFANITY_WORDS`] as authored in the TS list.
#[cfg(test)]
pub(super) const PROFANITY_COUNT: usize = 327;

/// Words considered profane/aggressive, in TS order.
pub(super) fn profanity() -> Vec<&'static str> {
    PROFANITY_WORDS.split_whitespace().collect()
}

/// `\b(?:fuck|fucks|...)\b` - the word-boundary, case-insensitive profanity match.
pub(super) fn profanity_pattern() -> String {
    format!(r"\b(?:{})\b", profanity().join("|"))
}

/// Anguish/exasperation interjections. Each alternative is a case-insensitive
/// word-bounded pattern. Interjections whose short form collides with normal
/// prose (`no`, `ahh`, `why`, `yes`, `god`) require *real* elongation;
/// unambiguous ones (`ugh`, `argh`, `grr`) match their plain form too. Picked
/// to avoid hex / base64 contamination via the surrounding `\b` plus
/// letter-only alternatives.
pub(super) const ANGUISH_PATTERNS: &[&str] = &[
    "no{3,}",      // nooo, noooooo
    "a+h{2,}",     // ahh, aaaahhh
    "u+r?g+h+",    // ugh, ughh, urgh, uuugh
    "a+r+g+h+",    // argh, aaargh, arrgghhh
    "g+r{2,}",     // grr, grrrr
    "st+o{3,}p+",  // stooop, sttooopp
    "w+h+y{3,}",   // whyyy, whyyyyy
    "f+u{3,}c*k*", // fuuu, fuuuck
    "wtf{3,}",     // wtfff
    "o+m+g{2,}",   // omgg, omggg
    "ye+s{3,}",    // yesss, yeessss
    "g+o+d{3,}",   // goddd, goddddd
    "br+u+h{2,}",  // bruhh, bruuuhh
];

/// `\b(?:no{3,}|a+h{2,}|...)\b` - the elongated-interjection match.
pub(super) fn anguish_pattern() -> String {
    format!(r"\b(?:{})\b", ANGUISH_PATTERNS.join("|"))
}
