//! `omp-rs judge-sessions <dir|file> [--out FILE]` — score historical sessions with the Laya
//! turn judge (same view builder and ONNX scorer as the `OMP_LAYA_JUDGE=1` shadow hook).
//! One JSON row per episode (one real user message and the steps until the next one).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use omp_judge::{offline, scorer::ModelPaths};

const USAGE: &str = "usage: omt judge-sessions <dir|file> [--out FILE] [--limit N] [--ckpt DIR] [--onnx FILE] [--threads N]";

/// Run the subcommand; returns the process exit code.
///
/// # Errors
/// Bad arguments, unreadable input, model load or output failure.
pub fn run(args: &[String]) -> Result<i32> {
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut paths = ModelPaths::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .cloned()
                .with_context(|| format!("{name} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--out" => out = Some(val("--out")?.into()),
            "--limit" => limit = Some(val("--limit")?.parse().context("--limit")?),
            "--ckpt" => paths.ckpt = val("--ckpt")?.into(),
            "--onnx" => paths.onnx = val("--onnx")?.into(),
            "--threads" => paths.threads = val("--threads")?.parse().context("--threads")?,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(0);
            }
            flag if flag.starts_with('-') => bail!("unknown flag {flag}\n{USAGE}"),
            _ if input.is_none() => input = Some(a.into()),
            _ => bail!("unexpected argument {a}\n{USAGE}"),
        }
    }
    let Some(input) = input else { bail!("{USAGE}") };
    let mut files = offline::collect_files(&input)?;
    if let Some(n) = limit {
        files.truncate(n);
    }
    let mut sink: Box<dyn Write> = match &out {
        Some(p) => Box::new(BufWriter::new(
            File::create(p).with_context(|| format!("create {}", p.display()))?,
        )),
        None => Box::new(BufWriter::new(std::io::stdout().lock())),
    };
    let sum = offline::judge_files(&files, &mut sink, &paths, &mut |f, n| {
        eprintln!("{}: {n} episode(s)", f.display())
    })?;
    sink.flush()?;
    eprintln!(
        "judged {} episode(s) from {} file(s), {} failed",
        sum.episodes, sum.files, sum.failed
    );
    Ok(i32::from(sum.failed > 0 && sum.episodes == 0))
}
