//! Small JS-semantics helpers shared by the parser: `Date.parse` for session
//! ISO timestamps and `String.prototype.length` (UTF-16 code units).

/// JS `Date.parse` for the ISO-8601 shapes session JSONL uses
/// (`2026-10-10T06:09:53.689Z`, offsets, or date-only). Returns `None` where
/// JS yields NaN.
#[must_use]
pub fn date_parse(value: &str) -> Option<i64> {
    let s = value.trim();
    let bytes = s.as_bytes();
    if bytes.len() < 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: i64 = s.get(5..7)?.parse().ok()?;
    let day: i64 = s.get(8..10)?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let mut hour = 0i64;
    let mut minute = 0i64;
    let mut second = 0i64;
    let mut milli = 0i64;
    let mut offset_min = 0i64;
    // A date-time with no designator is local time (JS); date-only is UTC.
    let mut local_time = false;

    let rest = s.get(10..).unwrap_or("");
    let mut chars = rest.char_indices().peekable();
    if let Some((_, c)) = chars.peek() {
        if *c == 'T' || *c == 't' || *c == ' ' {
            let rest = &rest[1..];
            let b = rest.as_bytes();
            if b.len() < 5 || b[2] != b':' {
                return None;
            }
            hour = rest.get(0..2)?.parse().ok()?;
            minute = rest.get(3..5)?.parse().ok()?;
            let mut idx = 5;
            if b.len() >= 8 && b[5] == b':' {
                second = rest.get(6..8)?.parse().ok()?;
                idx = 8;
            }
            // Fractional seconds.
            if b.len() > idx && b[idx] == b'.' {
                let mut j = idx + 1;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                let frac = rest.get(idx + 1..j)?;
                let mut digits = frac.bytes().map(|d| (d - b'0') as i64).collect::<Vec<_>>();
                digits.resize(3, 0);
                milli = digits[0] * 100 + digits[1] * 10 + digits[2];
                idx = j;
            }
            // Timezone designator.
            let tz = &b[idx..];
            match tz.first() {
                Some(b'Z') | Some(b'z') => {}
                Some(b'+') | Some(b'-') => {
                    let sign = if tz[0] == b'+' { 1 } else { -1 };
                    let tz = std::str::from_utf8(tz).ok()?;
                    let th: i64 = tz.get(1..3)?.parse().ok()?;
                    let tm: i64 = if tz.len() >= 6 && tz.as_bytes()[3] == b':' {
                        tz.get(4..6)?.parse().ok()?
                    } else if tz.len() >= 5 {
                        tz.get(3..5)?.parse().ok()?
                    } else {
                        0
                    };
                    offset_min = sign * (th * 60 + tm);
                }
                None => local_time = true,
                _ => return None,
            }
        }
    }

    let days = days_from_civil(year, month, day);
    let naive = ((days * 24 + hour) * 60 + minute) * 60_000 + second * 1000 + milli;
    if local_time {
        return local_to_utc_ms(naive);
    }
    Some(naive - offset_min * 60_000)
}

/// Local wall-clock time (`naive`, the local fields as if they were UTC) to a
/// UTC instant, with JS `Date` rules: an ambiguous time takes the earlier
/// instant, and a time in a DST gap uses the offset in force before the gap.
/// Zone rules come from SQLite's `localtime` modifier, the same C-library
/// source the TS process uses; `None` if SQLite cannot answer.
fn local_to_utc_ms(naive: i64) -> Option<i64> {
    const DAY_MS: i64 = 86_400_000;
    let conn = rusqlite::Connection::open_in_memory().ok()?;
    // Local minus UTC, in ms, in force at the UTC instant `utc_ms`.
    let offset_at = |utc_ms: i64| -> Option<i64> {
        let secs = utc_ms.div_euclid(1000);
        let local_secs: i64 = conn
            .query_row(
                "SELECT CAST(strftime('%s', ?1, 'unixepoch', 'localtime') AS INTEGER)",
                [secs],
                |row| row.get(0),
            )
            .ok()?;
        Some((local_secs - secs) * 1000)
    };
    // Offsets a day either side: they differ only if a transition is near `naive`.
    let before = offset_at(naive - DAY_MS)?;
    let after = offset_at(naive + DAY_MS)?;
    let t_before = naive - before;
    let t_after = naive - after;
    let valid_before = offset_at(t_before)? == before;
    let valid_after = offset_at(t_after)? == after;
    Some(match (valid_before, valid_after) {
        (true, true) => t_before.min(t_after),
        (true, false) => t_before,
        (false, true) => t_after,
        (false, false) => t_before,
    })
}

/// Howard Hinnant's `days_from_civil` (proleptic Gregorian).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// JS `String.prototype.length`: UTF-16 code units, not chars or bytes.
#[must_use]
pub fn js_len(text: &str) -> i64 {
    text.encode_utf16().count() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_iso_z() {
        assert_eq!(
            date_parse("2026-10-10T06:09:53.689Z"),
            Some(1_791_612_593_689)
        );
        assert_eq!(date_parse("2026-10-10T06:09:53Z"), Some(1_791_612_593_000));
    }

    #[test]
    fn parses_offsets_and_invalid() {
        // +02:00 shifts earlier.
        assert_eq!(
            date_parse("2026-10-10T08:09:53+02:00"),
            Some(1_791_612_593_000)
        );
        // Date-only is UTC midnight in JS (unlike date-time without offset).
        assert_eq!(date_parse("2026-10-10"), Some(1_791_590_400_000));
        assert_eq!(date_parse("garbage"), None);
        assert_eq!(date_parse("2026-13-40T00:00:00Z"), None);
    }

    /// `(TZ, [(input, Date.parse(input))])`; oracle: `bun -e` under the same TZ.
    /// POSIX `EST5EDT,M3.2.0,M11.1.0` is America/New_York for these dates (no zoneinfo needed).
    const LOCAL_CASES: &[(&str, &[(&str, i64)])] = &[
        (
            "UTC",
            &[
                ("2026-10-10T06:09:53", 1_791_612_593_000),
                ("2026-10-10 06:09", 1_791_612_540_000),
                ("2026-07-01T12:00:00.250", 1_782_907_200_250),
            ],
        ),
        (
            "MSK-3",
            &[
                ("2026-10-10T06:09:53", 1_791_601_793_000),
                ("2026-10-10 06:09", 1_791_601_740_000),
                ("2026-07-01T12:00:00.250", 1_782_896_400_250),
            ],
        ),
        (
            "EST5EDT,M3.2.0,M11.1.0",
            &[
                ("2026-10-10T06:09:53", 1_791_626_993_000),
                // DST gap (02:30 does not exist): offset before the gap, EST.
                ("2026-03-08T02:30:00", 1_772_955_000_000),
                // Ambiguous (01:30 happens twice): the earlier instant, EDT.
                ("2026-11-01T01:30:00", 1_793_511_000_000),
                ("2026-11-01T02:30:00", 1_793_518_200_000),
            ],
        ),
    ];

    /// Date-time without offset is local time (JS), date-only is UTC. The zone
    /// is process-global, so each zone runs in a child test process with `TZ` set.
    #[test]
    fn date_time_without_offset_is_local_time_like_js() {
        assert_eq!(date_parse("2026-10-10"), Some(1_791_590_400_000));
        let exe = std::env::current_exe().expect("test binary");
        for (tz, _) in LOCAL_CASES {
            let out = std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "parser::util::tests::local_time_child",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("TZ", tz)
                .env("OMP_STATS_LOCAL_TIME_CHILD", "1")
                .output()
                .unwrap_or_else(|e| panic!("spawn child {exe:?}: {e}"));
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(out.status.success(), "TZ={tz}: {stdout}");
            assert!(
                stdout.contains("1 passed"),
                "TZ={tz} child did not run: {stdout}"
            );
        }
    }

    #[test]
    fn local_time_child() {
        if std::env::var_os("OMP_STATS_LOCAL_TIME_CHILD").is_none() {
            return;
        }
        let tz = std::env::var("TZ").expect("TZ set by parent");
        let (_, cases) = LOCAL_CASES
            .iter()
            .find(|(name, _)| *name == tz)
            .expect("known TZ");
        for (input, expected) in cases.iter() {
            assert_eq!(date_parse(input), Some(*expected), "TZ={tz} {input}");
        }
    }
}
