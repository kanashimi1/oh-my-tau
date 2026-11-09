//! `omt laya-serve --ckpt DIR --onnx FILE --threads N --sock PATH` — the shared laya judge
//! daemon (see `omp_judge::shared`). Started on demand by the first `omt` that needs the
//! judge; every other `omt` on the machine uses it over the socket instead of loading its
//! own ~1.7 GB copy of the model. Exits after 30 min without requests.

use anyhow::{bail, Context as _, Result};
use omp_judge::scorer::ModelPaths;

const USAGE: &str = "usage: omt laya-serve --ckpt DIR --onnx FILE --threads N --sock PATH";

/// Run the subcommand; returns the process exit code.
///
/// # Errors
/// Bad arguments, model load or socket failure.
pub fn run(args: &[String]) -> Result<i32> {
    let mut paths = ModelPaths::default();
    let mut sock = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .cloned()
                .with_context(|| format!("{name} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--ckpt" => paths.ckpt = val("--ckpt")?.into(),
            "--onnx" => paths.onnx = val("--onnx")?.into(),
            "--threads" => paths.threads = val("--threads")?.parse().context("--threads")?,
            "--sock" => sock = Some(std::path::PathBuf::from(val("--sock")?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(0);
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    let sock = sock.unwrap_or_else(|| omp_judge::shared::socket_path(&paths));
    omp_judge::shared::serve(&paths, &sock)?;
    Ok(0)
}
