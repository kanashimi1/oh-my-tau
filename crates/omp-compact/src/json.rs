//! Small helpers over `serde_json::Value` entries and messages (omp session JSON is handled as
//! values here: it carries roles and kinds `pa_types` does not model, such as `developer`).

use serde_json::{Map, Value};

#[must_use]
pub fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// Entry `type` (`""` when absent).
#[must_use]
pub fn entry_type(entry: &Value) -> &str {
    str_field(entry, "type").unwrap_or("")
}

#[must_use]
pub fn entry_id(entry: &Value) -> &str {
    str_field(entry, "id").unwrap_or("")
}

/// Role of a `message` entry's payload (`""` when absent).
#[must_use]
pub fn message_role(message: &Value) -> &str {
    str_field(message, "role").unwrap_or("")
}

/// JS truthiness of an optional JSON value.
#[must_use]
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Some(Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

/// JS `new Date(iso).getTime()` for the ISO-8601 forms omp writes (`YYYY-MM-DDTHH:MM:SS[.fff]Z`,
/// optionally with a numeric offset). `None` is JS `NaN`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn parse_iso_ms(iso: &str) -> Option<i64> {
    let (date, time) = iso.split_once(['T', ' '])?;
    let mut d = date.splitn(3, '-');
    let (y, m, day) = (
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
    );
    let (clock, offset_min) = if let Some(c) = time.strip_suffix('Z') {
        (c, 0)
    } else if let Some(pos) = time.rfind(['+', '-']).filter(|p| *p > 0) {
        let (c, off) = time.split_at(pos);
        let sign = if off.starts_with('-') { -1 } else { 1 };
        let (oh, om) = off[1..].split_once(':').unwrap_or((&off[1..], "0"));
        (
            c,
            sign * (oh.parse::<i64>().ok()? * 60 + om.parse::<i64>().ok()?),
        )
    } else {
        (time, 0)
    };
    let mut t = clock.splitn(3, ':');
    let h = t.next()?.parse::<i64>().ok()?;
    let mi = t.next()?.parse::<i64>().ok()?;
    let sec = t.next().unwrap_or("0");
    let (secs, frac) = sec.split_once('.').unwrap_or((sec, ""));
    let secs = secs.parse::<i64>().ok()?;
    let millis = if frac.is_empty() {
        0
    } else {
        format!("{:0<3}", &frac[..frac.len().min(3)])
            .parse::<i64>()
            .ok()?
    };
    // Howard Hinnant's days-from-civil.
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 24 + h) * 60 + mi - offset_min) * 60_000 + secs * 1000 + millis)
}

/// Epoch milliseconds as a JSON value (`null` for `NaN`, as `JSON.stringify` writes it).
#[must_use]
pub fn ms_value(ms: Option<i64>) -> Value {
    ms.map_or(Value::Null, Value::from)
}

/// Insert `value` unless it is `None` (JS `undefined` properties vanish on serialization).
pub fn put_opt(map: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(v) = value {
        map.insert(key.to_owned(), v);
    }
}
