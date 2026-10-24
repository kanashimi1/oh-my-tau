#!/bin/sh
# Install omt (oh my tau) v0.10.0 for linux x86_64.
#   curl -fsSL https://raw.githubusercontent.com/kanashimi1/oh-my-tau/main/install.sh | sh
# Installs to ${OMT_INSTALL_DIR:-$HOME/.local/bin}/omt after verifying the sha256.
set -eu

TAG="v0.10.0"
ASSET="omt-linux-x64"
BASE="https://github.com/kanashimi1/oh-my-tau/releases/download/${TAG}"
DEST_DIR="${OMT_INSTALL_DIR:-$HOME/.local/bin}"

os=$(uname -s)
arch=$(uname -m)
if [ "$os" != "Linux" ] || { [ "$arch" != "x86_64" ] && [ "$arch" != "amd64" ]; }; then
    echo "omt: no prebuilt binary for ${os} ${arch} (only linux x86_64)." >&2
    echo "Build from source: git clone https://github.com/kanashimi1/oh-my-tau && cd oh-my-tau && cargo build --release -p omp-cli" >&2
    echo "The binary is target/release/omt." >&2
    exit 1
fi

command -v curl >/dev/null 2>&1 || { echo "omt: curl is required" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "omt: sha256sum is required" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

echo "Downloading ${ASSET} ${TAG} ..."
curl -fL --progress-bar -o "$tmp/$ASSET" "${BASE}/${ASSET}"
curl -fsSL -o "$tmp/SHA256SUMS.txt" "${BASE}/SHA256SUMS.txt"

expected=$(awk -v f="$ASSET" '$2 == f || $2 == "*" f { print $1; exit }' "$tmp/SHA256SUMS.txt")
if [ -z "$expected" ]; then
    echo "omt: ${ASSET} not listed in SHA256SUMS.txt" >&2
    exit 1
fi
actual=$(sha256sum "$tmp/$ASSET" | awk '{ print $1 }')
if [ "$expected" != "$actual" ]; then
    echo "omt: sha256 mismatch (expected ${expected}, got ${actual}); nothing installed" >&2
    exit 1
fi
echo "sha256 ok: ${actual}"

mkdir -p "$DEST_DIR"
chmod +x "$tmp/$ASSET"
mv -f "$tmp/$ASSET" "$DEST_DIR/omt"
echo "Installed ${DEST_DIR}/omt"

case ":${PATH}:" in
    *":${DEST_DIR}:"*) ;;
    *) echo "Add it to PATH:  export PATH=\"${DEST_DIR}:\$PATH\"" ;;
esac
