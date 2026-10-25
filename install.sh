#!/bin/sh
# Install omt (oh my tau).
#   curl -fsSL https://raw.githubusercontent.com/kanashimi1/oh-my-tau/main/install.sh | sh
# Picks the release asset for Linux x86_64/aarch64 and macOS x86_64/arm64,
# verifies the sha256 against SHA256SUMS.txt, then installs to
# ${OMT_INSTALL_DIR:-$HOME/.local/bin}/omt.
set -eu

REPO="kanashimi1/oh-my-tau"
TAG="${OMT_VERSION:-v0.12.0}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
DEST_DIR="${OMT_INSTALL_DIR:-$HOME/.local/bin}"

os=$(uname -s)
arch=$(uname -m)

case "$os" in
    Linux)  platform="linux" ;;
    Darwin) platform="macos" ;;
    *) platform="" ;;
esac
case "$arch" in
    x86_64 | amd64)   machine="x64" ;;
    aarch64 | arm64)  machine="arm64" ;;
    *) machine="" ;;
esac

if [ -z "$platform" ] || [ -z "$machine" ]; then
    echo "omt: no prebuilt binary for ${os} ${arch} (supported: linux/macos x86_64 + aarch64/arm64)." >&2
    echo "Build from source: git clone https://github.com/${REPO} && cd oh-my-tau && cargo build --release -p omp-cli" >&2
    echo "The binary is target/release/omt." >&2
    exit 1
fi

# Linux: glibc build by default; the fully static musl build where glibc is
# absent (Alpine, distroless) or when OMT_LIBC=musl. The musl build lacks only
# the optional ONNX-based Laya judge.
libc="${OMT_LIBC:-}"
if [ "$platform" = "linux" ] && [ -z "$libc" ]; then
    if ldd --version 2>&1 | grep -qi musl || ! ldd --version >/dev/null 2>&1; then
        libc="musl"
    fi
fi
ASSET="omt-${platform}-${machine}"
if [ "$platform" = "linux" ] && [ "$libc" = "musl" ]; then
    ASSET="${ASSET}-musl"
fi

command -v curl >/dev/null 2>&1 || { echo "omt: curl is required" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$@"; }
else
    echo "omt: sha256sum or shasum is required" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

curl -fsSL -o "$tmp/SHA256SUMS.txt" "${BASE}/SHA256SUMS.txt"

# Download one asset into $tmp and verify it against SHA256SUMS.txt.
fetch() {
    echo "Downloading $1 ${TAG} ..."
    curl -fL --progress-bar -o "$tmp/$1" "${BASE}/$1"
    expected=$(awk -v f="$1" '$2 == f || $2 == "*" f { print $1; exit }' "$tmp/SHA256SUMS.txt")
    if [ -z "$expected" ]; then
        echo "omt: $1 not listed in SHA256SUMS.txt" >&2
        exit 1
    fi
    actual=$(sha256 "$tmp/$1" | awk '{ print $1 }')
    if [ "$expected" != "$actual" ]; then
        echo "omt: sha256 mismatch (expected ${expected}, got ${actual}); nothing installed" >&2
        exit 1
    fi
    echo "sha256 ok: ${actual}"
    chmod +x "$tmp/$1"
}

fetch "$ASSET"
# A glibc build that cannot run here (glibc too old): use the static musl build.
if [ "$platform" = "linux" ] && [ "$libc" != "musl" ] && ! "$tmp/$ASSET" --version >/dev/null 2>&1; then
    echo "omt: ${ASSET} does not run on this system (glibc too old?); using the static musl build"
    ASSET="${ASSET}-musl"
    fetch "$ASSET"
fi

mkdir -p "$DEST_DIR"
mv -f "$tmp/$ASSET" "$DEST_DIR/omt"
echo "Installed ${DEST_DIR}/omt"

case ":${PATH}:" in
    *":${DEST_DIR}:"*) ;;
    *) echo "Add it to PATH:  export PATH=\"${DEST_DIR}:\$PATH\"" ;;
esac
