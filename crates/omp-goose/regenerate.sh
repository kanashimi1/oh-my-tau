#!/usr/bin/env bash
# Regenerates generated/omt_dedup.{c,h} from goose/omt_dedup.goose.
#
# Requires the goose compiler: either `goose` on PATH, or set $GOOSE to the
# goose binary. The exact command used to produce the committed generated files
# (goose build tree, -O2) was:
#
#   goose -o generated/omt_dedup.c --header generated/omt_dedup.h \
#       --standalone goose/omt_dedup.goose -O2
#
# The "generated" C is self-contained (the whole goose runtime, --standalone)
# and is what build.rs compiles; the source here is only needed to regenerate.
set -euo pipefail
goose="${GOOSE:-goose}"
cd "$(dirname "$0")"
"$goose" -o generated/omt_dedup.c --header generated/omt_dedup.h \
    --standalone goose/omt_dedup.goose -O2
echo "wrote generated/omt_dedup.c and generated/omt_dedup.h"
