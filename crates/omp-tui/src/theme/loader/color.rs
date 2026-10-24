//! Color parsing and OKLCH math — port of `pi-utils/src/color.ts` (subset)
//! and `theme/color.ts`. Used by the theme loader and session-accent
//! derivation.

/// An sRGB color with 0..255 channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// OKLCH color (perceptual lightness 0..1, chroma, hue in degrees 0..360).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Oklch {
    pub l: f64,
    pub c: f64,
    pub h: f64,
}

/// Parse `#rgb` / `#rrggbb` into RGB; `None` when malformed.
#[must_use]
pub fn parse_hex(hex: &str) -> Option<Rgb> {
    let h = hex.strip_prefix('#')?;
    let (r, g, b) = match h.len() {
        3 => (h.get(0..1)?, h.get(1..2)?, h.get(2..3)?),
        // `#rrggbbaa` also appears in bundled themes; alpha is dropped.
        6 | 8 => (h.get(0..2)?, h.get(2..4)?, h.get(4..6)?),
        _ => return None,
    };
    let ch = |s: &str| -> Option<u8> {
        if s.len() == 1 {
            let v = u8::from_str_radix(s, 16).ok()?;
            Some(v * 17)
        } else {
            u8::from_str_radix(s, 16).ok()
        }
    };
    Some(Rgb {
        r: ch(r)?,
        g: ch(g)?,
        b: ch(b)?,
    })
}

/// Format RGB as `#rrggbb`.
#[must_use]
pub fn to_hex(c: Rgb) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)
}

/// xterm-256 palette entry 0..255 as RGB (TS `paletteToRgb`).
#[cfg(test)]
#[must_use]
pub fn palette_to_rgb(index: u8) -> Option<Rgb> {
    const ANSI16: [[u8; 3]; 16] = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [192, 192, 192],
        [128, 128, 128],
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ];
    const CUBE: [u8; 6] = [0, 95, 135, 175, 215, 255];
    match index {
        0..=15 => ANSI16.get(usize::from(index)).map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        }),
        16..=231 => {
            let n = u32::from(index) - 16;
            Some(Rgb {
                r: CUBE[(n / 36) as usize % 6],
                g: CUBE[(n / 6) as usize % 6],
                b: CUBE[(n % 6) as usize],
            })
        }
        _ => {
            let gray = 8 + (u16::from(index) - 232) * 10;
            Some(Rgb {
                r: gray as u8,
                g: gray as u8,
                b: gray as u8,
            })
        }
    }
}

fn linearize(channel: u8) -> f64 {
    let c = f64::from(channel) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn delinearize(linear: f64) -> u8 {
    let c = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Convert a hex color to OKLCH (TS `hexToOklch`). Unparseable input yields
/// L=0, C=0 (near-gray; hue callers must check chroma).
#[must_use]
pub fn hex_to_oklch(hex: &str) -> Oklch {
    let Some(rgb) = parse_hex(hex) else {
        return Oklch {
            l: 0.0,
            c: 0.0,
            h: 0.0,
        };
    };
    let (r, g, b) = (linearize(rgb.r), linearize(rgb.g), linearize(rgb.b));
    let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
    let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
    let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();
    let lab_l = 0.210_454_255_3 * l + 0.793_617_785 * m - 0.004_072_046_8 * s;
    let lab_a = 1.977_998_495_1 * l - 2.428_592_205 * m + 0.450_593_709_9 * s;
    let lab_b = 0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766 * s;
    let c = lab_a.hypot(lab_b);
    let mut h = lab_b.atan2(lab_a).to_degrees();
    if h < 0.0 {
        h += 360.0;
    }
    Oklch { l: lab_l, c, h }
}

/// OKLab back to linear sRGB; channels may fall outside 0..1 when out of gamut.
fn oklab_to_linear_rgb(l: f64, a: f64, b: f64) -> [f64; 3] {
    let l_ = (l + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m_ = (l - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s_ = (l - 0.089_484_177_5 * a - 1.291_485_548 * b).powi(3);
    [
        4.076_741_662_1 * l_ - 3.307_711_591_3 * m_ + 0.230_969_929_2 * s_,
        -1.268_438_004_6 * l_ + 2.609_757_401_1 * m_ - 0.341_319_396_5 * s_,
        -0.004_196_086_3 * l_ - 0.703_418_614_7 * m_ + 1.707_614_701 * s_,
    ]
}

/// Max OKLab saturation (C/L) inside sRGB for a hue direction, via Björn
/// Ottosson's polynomial fit plus one Halley refinement (TS
/// `computeMaxSaturation`).
fn compute_max_saturation(a: f64, b: f64) -> f64 {
    let (k0, k1, k2, k3, k4, wl, wm, ws);
    if -1.881_703_28 * a - 0.809_364_93 * b > 1.0 {
        k0 = 1.190_862_77;
        k1 = 1.765_767_28;
        k2 = 0.596_626_41;
        k3 = 0.755_151_97;
        k4 = 0.567_712_45;
        wl = 4.076_741_662_1;
        wm = -3.307_711_591_3;
        ws = 0.230_969_929_2;
    } else if 1.814_441_04 * a - 1.194_452_76 * b > 1.0 {
        k0 = 0.739_565_15;
        k1 = -0.459_544_04;
        k2 = 0.082_854_27;
        k3 = 0.125_410_7;
        k4 = 0.145_032_04;
        wl = -1.268_438_004_6;
        wm = 2.609_757_401_1;
        ws = -0.341_319_396_5;
    } else {
        k0 = 1.357_336_52;
        k1 = -0.009_157_99;
        k2 = -1.151_302_1;
        k3 = -0.505_596_06;
        k4 = 0.006_921_67;
        wl = -0.004_196_086_3;
        wm = -0.703_418_614_7;
        ws = 1.707_614_701;
    }
    let s = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    let kl = 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let km = -0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let ks = -0.089_484_177_5 * a - 1.291_485_548 * b;
    let l_ = 1.0 + s * kl;
    let m_ = 1.0 + s * km;
    let s_ = 1.0 + s * ks;
    let f = wl * l_.powi(3) + wm * m_.powi(3) + ws * s_.powi(3);
    let f1 = wl * 3.0 * kl * l_ * l_ + wm * 3.0 * km * m_ * m_ + ws * 3.0 * ks * s_ * s_;
    let f2 = wl * 6.0 * kl * kl * l_ + wm * 6.0 * km * km * m_ + ws * 6.0 * ks * ks * s_;
    s - (f * f1) / (f1 * f1 - 0.5 * f * f2)
}

/// The sRGB gamut cusp for an OKLCH hue (TS `oklchCusp`): lightness/chroma
/// where the hue reaches maximum chroma inside sRGB.
#[must_use]
pub fn oklch_cusp(h: f64) -> (f64, f64) {
    let rad = h.to_radians();
    let (a, b) = (rad.cos(), rad.sin());
    let s_cusp = compute_max_saturation(a, b);
    let rgb = oklab_to_linear_rgb(1.0, s_cusp * a, s_cusp * b);
    let l_cusp = (1.0 / rgb[0].max(rgb[1]).max(rgb[2])).cbrt();
    (l_cusp, l_cusp * s_cusp)
}

const GAMUT_EPSILON: f64 = 1e-4;

fn in_srgb_gamut(rgb: [f64; 3]) -> bool {
    rgb.iter()
        .all(|&ch| ch >= -GAMUT_EPSILON && ch <= 1.0 + GAMUT_EPSILON)
}

/// Convert OKLCH to a `#rrggbb` hex string, gamut-mapping by chroma
/// bisection (TS `oklchToHex`).
#[must_use]
pub fn oklch_to_hex(color: Oklch) -> String {
    let l = color.l.clamp(0.0, 1.0);
    let rad = color.h.to_radians();
    let (cos, sin) = (rad.cos(), rad.sin());
    let at = |c: f64| oklab_to_linear_rgb(l, c * cos, c * sin);

    let mut rgb = at(color.c);
    if !in_srgb_gamut(rgb) {
        let (mut lo, mut hi) = (0.0_f64, color.c);
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            if in_srgb_gamut(at(mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        rgb = at(lo);
    }
    to_hex(Rgb {
        r: delinearize(rgb[0]),
        g: delinearize(rgb[1]),
        b: delinearize(rgb[2]),
    })
}

/// WCAG 2.x relative luminance of a hex color, normalized 0..1 (TS
/// `relativeLuminance`). `None` for unparseable input.
#[must_use]
pub fn relative_luminance(hex: &str) -> Option<f64> {
    let rgb = parse_hex(hex)?;
    let lin = [rgb.r, rgb.g, rgb.b].map(linearize);
    Some(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let c = parse_hex("#0088fa").unwrap();
        assert_eq!((c.r, c.g, c.b), (0, 0x88, 0xfa));
        assert_eq!(to_hex(c), "#0088fa");
        assert_eq!(parse_hex("#08f").map(to_hex), Some("#0088ff".into()));
        assert!(parse_hex("0088fa").is_none());
        assert!(parse_hex("#0088f").is_none());
    }

    #[test]
    fn oklch_roundtrip_white() {
        let w = hex_to_oklch("#ffffff");
        assert!((w.l - 1.0).abs() < 0.01);
        assert!(w.c < 0.01);
        let back = oklch_to_hex(w);
        let again = hex_to_oklch(&back);
        assert!((w.l - again.l).abs() < 0.01 && (w.c - again.c).abs() < 0.01);
    }

    #[test]
    fn cusp_yellow_is_bright_blue_is_dim() {
        let (ly, _) = oklch_cusp(100.0); // yellow
        let (lb, _) = oklch_cusp(264.0); // blue
        assert!(ly > 0.9, "yellow cusp L {ly}");
        assert!(lb < 0.55, "blue cusp L {lb}");
    }

    #[test]
    fn luminance_scales() {
        let dark = relative_luminance("#000000").unwrap();
        let light = relative_luminance("#ffffff").unwrap();
        assert!(dark < 0.01 && light > 0.99);
    }

    #[test]
    fn palette_entries() {
        assert_eq!(palette_to_rgb(0).map(|c| c.r), Some(0));
        let cube = palette_to_rgb(16).unwrap();
        assert_eq!((cube.r, cube.g, cube.b), (0, 0, 0));
        let cube2 = palette_to_rgb(196).unwrap();
        assert_eq!((cube2.r, cube2.g, cube2.b), (255, 0, 0));
    }
}
