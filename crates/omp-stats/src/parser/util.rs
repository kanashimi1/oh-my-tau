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
                None => {}
                _ => return None,
            }
        }
    }

    let days = days_from_civil(year, month, day);
    let ms = ((days * 24 + hour) * 60 + minute - offset_min) * 60_000 + second * 1000 + milli;
    Some(ms)
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
        assert_eq!(date_parse("2026-10-10T06:09:53.689Z"), Some(1_791_612_593_689));
        assert_eq!(date_parse("2026-10-10T06:09:53Z"), Some(1_791_612_593_000));
    }

    #[test]
    fn parses_offsets_and_invalid() {
        // +02:00 shifts earlier.
        assert_eq!(date_parse("2026-10-10T08:09:53+02:00"), Some(1_791_612_593_000));
        // UTC midnight (JS parses date-only as *local* midnight; the port
        // pins UTC — localhost dashboards assume the producer host).
        assert_eq!(date_parse("2026-10-10"), Some(1_791_590_400_000));
        assert_eq!(date_parse("garbage"), None);
        assert_eq!(date_parse("2026-13-40T00:00:00Z"), None);
    }
}
