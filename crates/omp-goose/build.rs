//! Compiles the committed goose-generated C kernel (no goose toolchain needed).

fn main() {
    let src = "generated/omt_dedup.c";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-changed=goose/omt_dedup.goose");
    // Generated code: its unused-variable warnings are not actionable here.
    cc::Build::new()
        .file(src)
        .warnings(false)
        .compile("omt_goose");
}
