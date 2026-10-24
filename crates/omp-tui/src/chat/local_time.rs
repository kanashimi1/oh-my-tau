//! Local wall-clock formatting without `unsafe` libc calls: the UTC offset
//! comes from the system tz database (`$TZ`, else `/etc/localtime`, TZif
//! format), the way JS `Date#getHours()` resolves it. Used by the usage row's
//! `YYYY-MM-DD HH:MM:SS` stamp (TS `formatUsageTimestamp`).

use std::path::PathBuf;
use std::sync::LazyLock;

/// `YYYY-MM-DD HH:MM:SS` in the local time zone for an epoch-ms instant.
#[must_use]
pub fn format_local_timestamp(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let local = secs + ZONE.offset_at(secs);
    let days = local.div_euclid(86_400);
    let rem = local.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Proleptic Gregorian date of a day count since 1970-01-01 (H. Hinnant).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// UTC offsets of the local zone: explicit transitions plus the fallback
/// offset after the last one.
#[derive(Debug, Default)]
struct Zone {
    /// `(transition epoch secs, offset secs)`, ascending.
    transitions: Vec<(i64, i64)>,
    /// Offset before the first transition / when no data.
    initial: i64,
    /// Offset after the last transition (TZif footer std offset when the
    /// footer has no DST rule, else the last transition's offset).
    after_last: Option<i64>,
}

impl Zone {
    fn offset_at(&self, secs: i64) -> i64 {
        match self.transitions.iter().rposition(|(at, _)| *at <= secs) {
            None => self.initial,
            Some(i) if i + 1 == self.transitions.len() => {
                self.after_last.unwrap_or(self.transitions[i].1)
            }
            Some(i) => self.transitions[i].1,
        }
    }
}

/// The process's local zone, loaded once.
static ZONE: LazyLock<Zone> = LazyLock::new(|| load_zone().unwrap_or_default());

fn load_zone() -> Option<Zone> {
    let path = match std::env::var("TZ") {
        Ok(tz) if tz.is_empty() || tz == "UTC" || tz == "UTC0" || tz == ":UTC" => {
            return Some(Zone::default());
        }
        Ok(tz) => {
            let name = tz.strip_prefix(':').unwrap_or(&tz);
            if name.starts_with('/') {
                PathBuf::from(name)
            } else {
                PathBuf::from("/usr/share/zoneinfo").join(name)
            }
        }
        Err(_) => PathBuf::from("/etc/localtime"),
    };
    parse_tzif(&std::fs::read(path).ok()?)
}

/// Parse a TZif file (RFC 8536): the 64-bit block for v2+, else v1.
fn parse_tzif(data: &[u8]) -> Option<Zone> {
    if data.get(..4)? != b"TZif" {
        return None;
    }
    let version = *data.get(4)?;
    let (header, v1_len) = header_at(data, 0, 4)?;
    if version >= b'2' {
        let start = 44 + v1_len;
        let (header64, len64) = header_at(data, start, 8)?;
        let mut zone = block(data, start + 44, &header64, 8)?;
        let footer = data.get(start + 44 + len64..)?;
        zone.after_last = footer_std_offset(footer);
        return Some(zone);
    }
    block(data, 44, &header, 4)
}

/// TZif header counts `(isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt)`.
type Counts = [usize; 6];

fn header_at(data: &[u8], at: usize, time_size: usize) -> Option<(Counts, usize)> {
    let mut counts = [0usize; 6];
    for (i, count) in counts.iter_mut().enumerate() {
        let off = at + 20 + i * 4;
        let raw = u32::from_be_bytes(data.get(off..off + 4)?.try_into().ok()?);
        *count = usize::try_from(raw).ok()?;
    }
    let [isut, isstd, leap, time, typ, chars] = counts;
    let len = time * time_size + time + typ * 6 + chars + leap * (time_size + 4) + isstd + isut;
    Some((counts, len))
}

fn block(data: &[u8], at: usize, counts: &Counts, time_size: usize) -> Option<Zone> {
    let [_, _, _, timecnt, typecnt, _] = *counts;
    let times_at = at;
    let idx_at = times_at + timecnt * time_size;
    let types_at = idx_at + timecnt;
    let utoff = |ty: usize| -> Option<i64> {
        let off = types_at + ty * 6;
        Some(i64::from(i32::from_be_bytes(
            data.get(off..off + 4)?.try_into().ok()?,
        )))
    };
    let mut transitions = Vec::with_capacity(timecnt);
    for i in 0..timecnt {
        let off = times_at + i * time_size;
        let at_secs = if time_size == 8 {
            i64::from_be_bytes(data.get(off..off + 8)?.try_into().ok()?)
        } else {
            i64::from(i32::from_be_bytes(data.get(off..off + 4)?.try_into().ok()?))
        };
        let ty = usize::from(*data.get(idx_at + i)?);
        transitions.push((at_secs, utoff(ty)?));
    }
    let initial = if typecnt > 0 { utoff(0)? } else { 0 };
    Some(Zone {
        transitions,
        initial,
        after_last: None,
    })
}

/// The footer's POSIX TZ string (`\nMSK-3\n`): its std offset when it
/// carries no DST rule (a DST rule needs the full POSIX evaluator; the last
/// explicit transition is used then).
fn footer_std_offset(footer: &[u8]) -> Option<i64> {
    let text = std::str::from_utf8(footer).ok()?.trim_matches('\n');
    let rest = if let Some(quoted) = text.strip_prefix('<') {
        &quoted[quoted.find('>')? + 1..]
    } else {
        text.trim_start_matches(|c: char| c.is_ascii_alphabetic())
    };
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '+' | '-' | ':')))
        .unwrap_or(rest.len());
    if end != rest.len() || rest.is_empty() {
        return None; // DST name/rule follows
    }
    let (sign, digits) = match rest.as_bytes()[0] {
        b'-' => (1, &rest[1..]),
        b'+' => (-1, &rest[1..]),
        _ => (-1, rest),
    };
    let mut parts = digits.split(':').map(|p| p.parse::<i64>().unwrap_or(0));
    let h = parts.next().unwrap_or(0);
    let m = parts.next().unwrap_or(0);
    let s = parts.next().unwrap_or(0);
    // POSIX offsets are west-positive: `MSK-3` is UTC+3.
    Some(sign * (h * 3600 + m * 60 + s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_742), (2026, 10, 16));
    }

    #[test]
    fn footer_offsets() {
        assert_eq!(footer_std_offset(b"\nMSK-3\n"), Some(3 * 3600));
        assert_eq!(
            footer_std_offset(b"\n<+0530>-5:30\n"),
            Some(5 * 3600 + 1800)
        );
        assert_eq!(footer_std_offset(b"\nUTC0\n"), Some(0));
        assert_eq!(footer_std_offset(b"\nEST5EDT,M3.2.0,M11.1.0\n"), None);
    }

    #[test]
    fn zone_lookup_uses_transitions_then_footer() {
        let zone = Zone {
            transitions: vec![(100, 3600), (200, 7200)],
            initial: 0,
            after_last: Some(10_800),
        };
        assert_eq!(zone.offset_at(50), 0);
        assert_eq!(zone.offset_at(150), 3600);
        assert_eq!(zone.offset_at(250), 10_800);
    }

    #[test]
    fn timestamp_shape() {
        let s = format_local_timestamp(1_792_111_302_143);
        assert_eq!(s.len(), 19, "{s}");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], " ");
    }
}
