//! Session accent derivation — port of `pi-tui/src/theme/session-color.ts`.
//!
//! A stable per-session accent color: the session name hash picks only the
//! hue, from a dark/light-specific admissible arc; lightness and chroma are
//! carried over from the theme's own accent, re-normalized to the target
//! hue's sRGB gamut cusp.

use std::sync::LazyLock;

use super::color::{hex_to_oklch, oklch_cusp, oklch_to_hex, relative_luminance, Oklch};

/// Fallback cusp-chroma fraction when the theme accent is unparseable or
/// near-gray (TS `FALLBACK_CUSP_CHROMA_FRACTION`).
const FALLBACK_CUSP_CHROMA_FRACTION: f64 = 0.7;
/// Chroma bounds for the carried theme-accent chroma.
const MIN_CHROMA: f64 = 0.05;
const MAX_CHROMA: f64 = 0.21;
/// Lightness bounds on dark themes so a carried accent stays visible.
const DARK_MIN_LIGHTNESS: f64 = 0.65;
const DARK_MAX_LIGHTNESS: f64 = 0.88;
/// Minimum WCAG contrast ratio between a light-theme accent and its surface.
const ACCENT_MIN_CONTRAST: f64 = 3.0;
/// Minimum angular hue distance from any theme color (collision avoidance).
const MIN_HUE_DISTANCE: f64 = 10.0;
/// OKLCH chroma below which hue is meaningless (near-gray).
const MIN_CHROMA_FOR_HUE: f64 = 0.03;

/// djb2 hash, 32-bit (TS `nameToHash`).
#[must_use]
pub fn name_to_hash(name: &str) -> u32 {
    let mut hash: u32 = 5381;
    for ch in name.chars() {
        let mut h = i64::from(hash);
        h = ((h << 5) + h) ^ i64::from(ch as u32);
        hash = (h & 0xFFFF_FFFF) as u32;
    }
    hash
}

/// Inclusive hue interval `[start, end]` in degrees.
type HueInterval = (u32, u32);

/// Dark themes: full wheel minus hues whose vividness peak exceeds the dark
/// lightness cap (excludes yellow/chartreuse ≈94–138° and over-light cyan
/// ≈158–200°; TS `DARK_HUE_INTERVALS`).
fn dark_hue_intervals() -> Vec<HueInterval> {
    let mut intervals = Vec::new();
    let mut start: Option<u32> = None;
    for h in 0..=360u32 {
        let admissible = h < 360 && oklch_cusp(f64::from(h)).0 <= DARK_MAX_LIGHTNESS;
        if admissible {
            start.get_or_insert(h);
        } else if let Some(s) = start.take() {
            intervals.push((s, h - 1));
        }
    }
    intervals
}

/// Light themes: declared cool band cyan → blue → purple.
const LIGHT_HUE_INTERVALS: [HueInterval; 1] = [(195, 330)];

fn arc_length(intervals: &[HueInterval]) -> u32 {
    intervals.iter().map(|&(a, b)| b - a + 1).sum()
}

/// Map an arc position (`0..len-1`) to its hue degree.
fn arc_to_hue(intervals: &[HueInterval], pos: u32) -> f64 {
    let mut p = pos;
    for &(a, b) in intervals {
        let n = b - a + 1;
        if p < n {
            return f64::from(a + p);
        }
        p -= n;
    }
    intervals
        .last()
        .map(|&(_, b)| f64::from(b))
        .unwrap_or_default()
}

fn hue_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(360.0 - d)
}

/// OKLCH hue of a hex color; `None` for near-gray colors.
fn hex_to_hue(hex: &str) -> Option<f64> {
    let c = hex_to_oklch(hex);
    if !c.h.is_finite() || c.c < MIN_CHROMA_FOR_HUE {
        return None;
    }
    Some(c.h)
}

/// Find a hue at least `MIN_HUE_DISTANCE` from all occupied hues, walked
/// outward in arc positions (TS `findSafeHue`).
fn find_safe_hue(pos: u32, occupied: &[f64], intervals: &[HueInterval]) -> f64 {
    let total = arc_length(intervals);
    let hue_at = |p: u32| arc_to_hue(intervals, p.min(total.saturating_sub(1)));
    let target = hue_at(pos);
    if occupied.is_empty()
        || occupied
            .iter()
            .all(|&h| hue_distance(target, h) >= MIN_HUE_DISTANCE)
    {
        return target;
    }
    for d in 1..total {
        for dir in [1i64, -1] {
            let cand_i = (i64::from(pos) + i64::from(d) * dir).rem_euclid(i64::from(total));
            let candidate = arc_to_hue(intervals, cand_i as u32);
            if occupied
                .iter()
                .all(|&h| hue_distance(candidate, h) >= MIN_HUE_DISTANCE)
            {
                return candidate;
            }
        }
    }
    target
}

/// Largest relative luminance an accent may have while still meeting
/// `ACCENT_MIN_CONTRAST` against a surface.
fn accent_luminance_cap(surface_luminance: f64) -> f64 {
    ((surface_luminance + 0.05) / ACCENT_MIN_CONTRAST - 0.05).max(0.0)
}

/// Theme-derived inputs (TS `SessionAccentTheme`).
#[derive(Debug, Clone, Default)]
pub struct SessionAccentTheme {
    /// Theme accent hex; the session accent adopts its lightness and chroma.
    pub accent_hex: String,
    /// Major theme color hexes checked for hue collision.
    pub color_hexes: Vec<String>,
    /// WCAG luminance of the status-line surface on light themes; `None` on dark.
    pub surface_luminance: Option<f64>,
}

/// Derive a stable hex accent color from a session name and the active theme
/// (TS `getSessionAccentHex`/`computeSessionAccentHex`).
#[must_use]
pub fn session_accent_hex(name: &str, theme: &SessionAccentTheme) -> String {
    let is_dark = theme.surface_luminance.is_none();
    let light: [HueInterval; 1] = LIGHT_HUE_INTERVALS;
    let intervals: &[HueInterval] = if is_dark { &DARK_HUE } else { &light };

    // Session name picks a position along the admissible arc.
    let pos = name_to_hash(name) % arc_length(intervals);

    // Shift away from theme-color collisions, staying within the arc.
    let theme_hues: Vec<f64> = theme
        .color_hexes
        .iter()
        .filter_map(|h| hex_to_hue(h))
        .collect();
    let target_hue = find_safe_hue(pos, &theme_hues, intervals);

    // Carry the theme accent's perceived weight, re-normalized to the target
    // hue's gamut cusp.
    let accent = hex_to_oklch(&theme.accent_hex);
    let usable = accent.l.is_finite()
        && accent.c.is_finite()
        && accent.h.is_finite()
        && accent.c >= MIN_CHROMA_FOR_HUE;
    let (target_l, target_c) = oklch_cusp(target_hue);
    let (lightness, chroma_fraction) = if usable {
        let (source_l, source_c) = oklch_cusp(accent.h);
        let l = if accent.l <= source_l {
            (accent.l / source_l) * target_l
        } else {
            target_l + ((accent.l - source_l) / (1.0 - source_l)) * (1.0 - target_l)
        };
        (l, accent.c / source_c)
    } else {
        (target_l, FALLBACK_CUSP_CHROMA_FRACTION)
    };
    let chroma = (chroma_fraction * target_c).clamp(MIN_CHROMA, MAX_CHROMA);

    // Lightness/contrast: clamped visible on dark, bisected for AA on light.
    match theme.surface_luminance {
        None => {
            let l = lightness.clamp(DARK_MIN_LIGHTNESS, DARK_MAX_LIGHTNESS);
            oklch_to_hex(Oklch {
                l,
                c: chroma,
                h: target_hue,
            })
        }
        Some(surface) => {
            let cap = accent_luminance_cap(surface);
            let top = oklch_to_hex(Oklch {
                l: lightness,
                c: chroma,
                h: target_hue,
            });
            if relative_luminance(&top).unwrap_or(0.0) <= cap {
                return top;
            }
            let (mut lo, mut hi) = (0.0_f64, lightness);
            for _ in 0..20 {
                let mid = (lo + hi) / 2.0;
                let lum = relative_luminance(&oklch_to_hex(Oklch {
                    l: mid,
                    c: chroma,
                    h: target_hue,
                }))
                .unwrap_or(0.0);
                if lum > cap {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            oklch_to_hex(Oklch {
                l: lo,
                c: chroma,
                h: target_hue,
            })
        }
    }
}

/// Lazily-computed dark hue arc (computed once per process).
static DARK_HUE: LazyLock<Vec<HueInterval>> = LazyLock::new(dark_hue_intervals);

#[cfg(test)]
mod tests {
    use super::*;

    fn theme(accent: &str, surface: Option<f64>) -> SessionAccentTheme {
        SessionAccentTheme {
            accent_hex: accent.into(),
            color_hexes: vec![accent.into()],
            surface_luminance: surface,
        }
    }

    #[test]
    fn djb2_matches_ts() {
        // "abc" → 193409669 per the TS algorithm (djb2, 32-bit wrap each step).
        assert_eq!(name_to_hash("abc"), 193_409_669);
    }

    #[test]
    fn stable_and_distinct_across_names() {
        let t = theme("#febc38", None);
        let a = session_accent_hex("alpha", &t);
        assert_eq!(a, session_accent_hex("alpha", &t));
        let hues: Vec<String> = ["s1", "s2", "s3", "s4", "s5", "s6"]
            .iter()
            .map(|n| session_accent_hex(n, &t))
            .collect();
        assert!(hues.iter().any(|h| *h != a), "hash spreads hues: {hues:?}");
    }

    #[test]
    fn dark_accent_is_visible() {
        let t = theme("#777d88", None);
        for name in ["one", "two", "three", "four"] {
            let hex = session_accent_hex(name, &t);
            let l = hex_to_oklch(&hex).l;
            assert!(l >= DARK_MIN_LIGHTNESS - 0.01, "{name} -> {hex} L={l}");
        }
    }

    #[test]
    fn light_accent_meets_contrast() {
        let surface = relative_luminance("#e5e5e7").unwrap();
        let t = theme("#178fb9", Some(surface));
        for name in ["one", "two", "three"] {
            let hex = session_accent_hex(name, &t);
            let cap = accent_luminance_cap(surface);
            assert!(
                relative_luminance(&hex).unwrap() <= cap + 1e-6,
                "{name} -> {hex}"
            );
        }
    }

    #[test]
    fn hue_avoids_theme_accent() {
        let t = theme("#febc38", None);
        let accent_hue = hex_to_hue("#febc38").unwrap();
        for name in ["a", "b", "c", "d", "e", "f", "g", "h"] {
            let hex = session_accent_hex(name, &t);
            let h = hex_to_hue(&hex);
            if let Some(h) = h {
                assert!(
                    hue_distance(h, accent_hue) >= MIN_HUE_DISTANCE - 0.5,
                    "{name} {hex}"
                );
            }
        }
    }
}
