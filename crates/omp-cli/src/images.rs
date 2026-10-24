//! Image loading for the `read` tool and `@file` prompt attachments, ported
//! from omp's `utils/image-resize.ts`. Images are decoded with the `image`
//! crate, downsized to vision-model caps and re-encoded to the smallest of
//! PNG/JPEG via a quality + scale ladder (500 KiB budget, 1568 px cap,
//! 200 px minimum edge).

use std::path::Path;

use anyhow::{bail, Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};
use image::imageops::FilterType;
use image::{ExtendedColorType, ImageReader};
use pa_agent::types::ImageContent;

/// Hard cap on image file size (20 MB), matching omp `read.ts` `MAX_IMAGE_SIZE`.
pub const MAX_IMAGE_SIZE: u64 = 20 * 1024 * 1024;

const DEFAULT_MAX_BYTES: usize = 500 * 1024;
const DEFAULT_MIN_DIMENSION: u32 = 200;
const DEFAULT_MAX_DIMENSION: u32 = 1568;
const DEFAULT_JPEG_QUALITY: u8 = 80;

/// Resize/re-encode parameters; defaults mirror omp `DEFAULT_OPTIONS`.
#[derive(Debug, Clone)]
pub struct ImageResizeOptions {
    pub max_width: u32,
    pub max_height: u32,
    pub max_bytes: usize,
    pub jpeg_quality: u8,
    pub min_dimension: u32,
}

impl Default for ImageResizeOptions {
    fn default() -> Self {
        Self {
            max_width: DEFAULT_MAX_DIMENSION,
            max_height: DEFAULT_MAX_DIMENSION,
            max_bytes: DEFAULT_MAX_BYTES,
            jpeg_quality: DEFAULT_JPEG_QUALITY,
            min_dimension: DEFAULT_MIN_DIMENSION,
        }
    }
}

/// A resized image: base64 payload plus before/after geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResizedImage {
    pub data: String,
    pub mime_type: String,
    pub original_width: u32,
    pub original_height: u32,
    pub width: u32,
    pub height: u32,
    pub was_resized: bool,
    pub decode_failed: bool,
}

impl ResizedImage {
    fn original(data: String, mime_type: &str, width: u32, height: u32) -> Self {
        Self {
            data,
            mime_type: mime_type.to_string(),
            original_width: width,
            original_height: height,
            width,
            height,
            was_resized: false,
            decode_failed: false,
        }
    }

    /// omp `formatDimensionNote`: tells the model when pixel coordinates shifted.
    pub fn dimension_note(&self) -> Option<String> {
        if !self.was_resized
            || self.original_width == 0
            || self.original_height == 0
            || (self.width == self.original_width && self.height == self.original_height)
        {
            return None;
        }
        Some(format!(
            "Note: the image was resized from {}x{} to {}x{}.",
            self.original_width, self.original_height, self.width, self.height
        ))
    }
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(c.get(1).copied().unwrap_or(0)) << 8)
            | u32::from(c.get(2).copied().unwrap_or(0));
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// sniff the format from the bytes; `image`'s guesser covers PNG/JPEG/GIF/WebP.
fn detect_format(bytes: &[u8]) -> Option<image::ImageFormat> {
    ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .format()
}

fn encode_png(rgb: &image::DynamicImage) -> Vec<u8> {
    let mut out = Vec::new();
    let encoder =
        PngEncoder::new_with_quality(&mut out, CompressionType::Fast, PngFilter::Adaptive);
    let _ = rgb.write_with_encoder(encoder);
    out
}

fn encode_jpeg(rgb: &image::DynamicImage, quality: u8) -> Vec<u8> {
    let mut out = Vec::new();
    let rgb8 = rgb.to_rgb8();
    let _ = JpegEncoder::new_with_quality(&mut out, quality).encode(
        rgb8.as_raw(),
        rgb8.width(),
        rgb8.height(),
        ExtendedColorType::Rgb8,
    );
    out
}

/// Pick the smallest of the encoded candidates (omp `pickSmallest`).
fn smallest(candidates: Vec<(Vec<u8>, &'static str)>) -> (Vec<u8>, String) {
    candidates
        .into_iter()
        .min_by_key(|(buf, _)| buf.len())
        .map(|(buf, mime)| (buf, mime.to_string()))
        .expect("at least one candidate")
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // result is min-clamped against the u32 caps by every caller
fn scaled(width: u32, height: u32, scale: f64) -> (u32, u32) {
    (
        (f64::from(width) * scale).round() as u32,
        (f64::from(height) * scale).round() as u32,
    )
}

/// Resize and recompress an image to fit the caps, following omp's strategy:
/// fast path when already within all limits and under 1/4 of the byte budget,
/// then PNG/JPEG encode + pick-smallest, then a quality ladder, then a scale
/// ladder. On decode failure the original bytes are passed through.
#[allow(clippy::cast_possible_truncation)] // u64 intermediates are bounded by the u32 caps (max_width/max_height)
pub fn resize_image(bytes: &[u8], fallback_mime: &str, opts: &ImageResizeOptions) -> ResizedImage {
    let fmt = detect_format(bytes);
    let source_mime = fmt.map_or_else(
        || fallback_mime.to_string(),
        |f| f.to_mime_type().to_string(),
    );
    let (original_width, original_height) = match fmt {
        Some(f) => match ImageReader::with_format(std::io::Cursor::new(bytes), f).into_dimensions()
        {
            Ok(d) => d,
            Err(_) => return ResizedImage::original(base64(bytes), fallback_mime, 0, 0),
        },
        None => return ResizedImage::original(base64(bytes), fallback_mime, 0, 0),
    };

    // Fast path: within dimensions and comfortably under budget.
    let min_dimension = opts.min_dimension.min(opts.max_width).min(opts.max_height);
    if original_width >= min_dimension
        && original_height >= min_dimension
        && original_width <= opts.max_width
        && original_height <= opts.max_height
        && bytes.len() <= opts.max_bytes / 4
    {
        return ResizedImage::original(
            base64(bytes),
            &source_mime,
            original_width,
            original_height,
        );
    }

    let Ok(decoded) = image::load_from_memory(bytes) else {
        return ResizedImage::original(
            base64(bytes),
            fallback_mime,
            original_width,
            original_height,
        );
    };

    // Target dimensions: clamp to caps, then lift undersized inputs uniformly.
    let mut target_w = original_width;
    let mut target_h = original_height;
    if target_w > opts.max_width {
        target_h = (u64::from(target_h) * u64::from(opts.max_width) / u64::from(target_w)) as u32;
        target_w = opts.max_width;
    }
    if target_h > opts.max_height {
        target_w = (u64::from(target_w) * u64::from(opts.max_height) / u64::from(target_h)) as u32;
        target_h = opts.max_height;
    }
    if target_w < min_dimension || target_h < min_dimension {
        let short_edge = target_w.min(target_h).max(1);
        let upscale = (f64::from(min_dimension) / f64::from(short_edge))
            .min(f64::from(opts.max_width) / f64::from(target_w))
            .min(f64::from(opts.max_height) / f64::from(target_h));
        if upscale > 1.0 {
            let (w, h) = scaled(target_w, target_h, upscale);
            target_w = w.min(opts.max_width);
            target_h = h.min(opts.max_height);
        }
    }

    let resized_of =
        |w: u32, h: u32| decoded.resize_exact(w.max(1), h.max(1), FilterType::Triangle);
    let encode_smallest = |img: &image::DynamicImage| {
        smallest(vec![
            (encode_png(img), "image/png"),
            (encode_jpeg(img, opts.jpeg_quality), "image/jpeg"),
        ])
    };
    let encode_lossy = |img: &image::DynamicImage, quality: u8| {
        smallest(vec![(encode_jpeg(img, quality), "image/jpeg")])
    };

    let mut best = encode_smallest(&resized_of(target_w, target_h));
    let mut final_w = target_w;
    let mut final_h = target_h;
    if best.0.len() <= opts.max_bytes {
        return encoded(best, original_width, original_height, final_w, final_h);
    }

    // Lossy quality ladder.
    for quality in [70u8, 60, 50, 40] {
        best = encode_lossy(&resized_of(target_w, target_h), quality);
        if best.0.len() <= opts.max_bytes {
            return encoded(best, original_width, original_height, final_w, final_h);
        }
    }

    // Scale ladder with decreasing quality; give up below a 100 px edge.
    for scale in [1.0f64, 0.75, 0.5, 0.35, 0.25] {
        let (w, h) = scaled(target_w, target_h, scale);
        if w < 100 || h < 100 {
            break;
        }
        final_w = w;
        final_h = h;
        for quality in [70u8, 60, 50, 40] {
            best = encode_lossy(&resized_of(w, h), quality);
            if best.0.len() <= opts.max_bytes {
                return encoded(best, original_width, original_height, final_w, final_h);
            }
        }
    }
    encoded(best, original_width, original_height, final_w, final_h)
}

/// Extension-based check: does this path look like a raster image the read
/// tool should return as an image content block?
pub fn looks_like_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp")
    )
}

fn mime_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    }
}

/// Load an image file as an `ImageContent` resized per omp defaults.
/// Rejects oversized files (20 MB) the same way omp's read tool does.
pub fn read_image(path: &Path) -> Result<ImageContent> {
    let (content, _) = read_image_detailed(path, &ImageResizeOptions::default())?;
    Ok(content)
}

/// Like [`read_image`] but also returns the dimension note for resized images.
pub fn read_image_detailed(
    path: &Path,
    opts: &ImageResizeOptions,
) -> Result<(ImageContent, Option<String>)> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    if bytes.len() as u64 > MAX_IMAGE_SIZE {
        bail!(
            "Image file too large: {} bytes exceeds {} limit.",
            bytes.len(),
            MAX_IMAGE_SIZE
        );
    }
    let mime = mime_for_path(path);
    let resized = resize_image(&bytes, mime, opts);
    let note = resized.dimension_note();
    Ok((
        ImageContent {
            data: resized.data,
            mime_type: resized.mime_type,
        },
        note,
    ))
}

fn encoded(
    best: (Vec<u8>, String),
    original_width: u32,
    original_height: u32,
    width: u32,
    height: u32,
) -> ResizedImage {
    ResizedImage {
        data: base64(&best.0),
        mime_type: best.1,
        original_width,
        original_height,
        width,
        height,
        was_resized: true,
        decode_failed: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn png_bytes(w: u32, h: u32, color: [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |_, _| image::Rgb(color));
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode png");
        out
    }

    #[test]
    fn small_image_passes_through() {
        let png = png_bytes(300, 300, [255, 0, 0]);
        let r = resize_image(&png, "image/png", &ImageResizeOptions::default());
        assert!(!r.was_resized);
        assert_eq!(r.mime_type, "image/png");
        assert_eq!(r.width, 300);
        assert_eq!(r.dimension_note(), None);
    }

    #[test]
    fn oversized_image_is_downscaled_and_encoded() {
        // 3000x2000 gradient PNG — large enough that re-encode is mandatory.
        let img = image::RgbImage::from_fn(3000, 2000, |x, y| {
            image::Rgb([
                u8::try_from(x % 256).unwrap(),
                u8::try_from(y % 256).unwrap(),
                128,
            ])
        });
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let r = resize_image(&png, "image/png", &ImageResizeOptions::default());
        assert!(r.was_resized);
        assert!(r.width <= 1568 && r.height <= 1568);
        assert!(r.original_width == 3000 && r.original_height == 2000);
        let note = r.dimension_note().expect("note present");
        assert!(note.contains("3000x2000"));
    }

    #[test]
    fn tiny_image_is_upscaled_to_min_dimension() {
        let png = png_bytes(4, 4, [0, 0, 255]);
        let r = resize_image(&png, "image/png", &ImageResizeOptions::default());
        assert!(r.was_resized);
        assert!(r.width >= 200 && r.height >= 200);
        assert!(r.width <= 1568 && r.height <= 1568);
    }

    #[test]
    fn read_image_returns_image_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.png");
        std::fs::write(&path, png_bytes(300, 300, [0, 200, 0])).unwrap();
        let content = read_image(&path).unwrap();
        assert_eq!(content.mime_type, "image/png");
        assert!(!content.data.is_empty());
        // base64 decodes back to a valid PNG.
        let mut raw = Vec::new();
        base64_decode(&content.data).read_to_end(&mut raw).unwrap();
        assert_eq!(detect_format(&raw), Some(image::ImageFormat::Png));
    }

    #[test]
    fn read_image_rejects_oversized_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.png");
        std::fs::write(
            &path,
            vec![0u8; usize::try_from(MAX_IMAGE_SIZE + 1).unwrap()],
        )
        .unwrap();
        assert!(read_image(&path).is_err());
    }

    #[test]
    fn looks_like_image_paths() {
        assert!(looks_like_image(Path::new("a/b.PNG")));
        assert!(looks_like_image(Path::new("x.jpeg")));
        assert!(!looks_like_image(Path::new("x.txt")));
        assert!(!looks_like_image(Path::new("x")));
    }

    fn base64_decode(s: &str) -> impl Read {
        struct Dec(std::str::Chars<'static>);
        impl Read for Dec {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                const T: &[u8; 64] =
                    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                if out.is_empty() {
                    return Ok(0);
                }
                let vals = |c: u8| {
                    T.iter()
                        .position(|&t| t == c)
                        .map_or(0, |p| u32::try_from(p).expect("alphabet index"))
                };
                let mut n = 0usize;
                while n + 3 <= out.len() {
                    let mut quad = [0u8; 4];
                    let mut got = 0usize;
                    while got < 4 {
                        match self.0.next() {
                            Some(c) if c.is_ascii_whitespace() || c == '=' => {}
                            Some(c) => {
                                quad[got] = c as u8;
                                got += 1;
                            }
                            None => break,
                        }
                    }
                    if got < 2 {
                        break;
                    }
                    let v = vals(quad[0]) << 18
                        | vals(quad[1]) << 12
                        | vals(quad[2]) << 6
                        | vals(quad[3]);
                    for i in 0..(got - 1).min(3) {
                        out[n] = ((v >> (16 - 8 * i)) & 0xff) as u8;
                        n += 1;
                    }
                }
                Ok(n)
            }
        }
        let owned: &'static str = Box::leak(s.to_string().into_boxed_str());
        Dec(owned.chars())
    }
}
