#!/bin/sh
# Build a fully static `omt` for Linux musl with zig as the C toolchain.
#   scripts/build-musl.sh [x86_64|aarch64]   (default: host arch)
# Needs: zig >= 0.13 on PATH, `rustup target add <arch>-unknown-linux-musl`.
# The musl build drops the `onnx` feature (no prebuilt ONNX Runtime for musl);
# everything else is identical. Output: target/<arch>-unknown-linux-musl/release/omt
set -eu
arch="${1:-$(uname -m)}"
case "$arch" in
  x86_64|amd64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) echo "unsupported arch: $arch" >&2; exit 2 ;;
esac
target="${arch}-unknown-linux-musl"
zt="${arch}-linux-musl"
wrap="$(mktemp -d)"
trap 'rm -rf "$wrap"' EXIT
for tool in cc c++; do
  # cc-rs passes --target=<rust triple>, which zig does not understand: drop it.
  cat > "$wrap/zig-$tool" <<SH
#!/bin/sh
for a in "\$@"; do case "\$a" in --target=*) ;; *) set -- "\$@" "\$a" ;; esac; shift; done
exec zig $tool -target $zt "\$@"
SH
  chmod +x "$wrap/zig-$tool"
done
printf '#!/bin/sh\nexec zig ar "$@"\n' > "$wrap/zig-ar"
chmod +x "$wrap/zig-ar"
up=$(echo "$target" | tr 'a-z-' 'A-Z_')
low=$(echo "$target" | tr '-' '_')
export "CC_${low}=$wrap/zig-cc" "CXX_${low}=$wrap/zig-c++" "AR_${low}=$wrap/zig-ar"
export "CARGO_TARGET_${up}_LINKER=$wrap/zig-cc"
# zig supplies musl's crt objects; Rust's self-contained copies would duplicate _start.
export "CARGO_TARGET_${up}_RUSTFLAGS=-C link-self-contained=no ${OMT_EXTRA_RUSTFLAGS:-}"
cargo build --release --target "$target" -p omp-cli --no-default-features
