#!/usr/bin/env bash
# TurnPeak repro: resume the 39.5 MB session, one mock turn, fine-grained RSS sampling.
set -u
ROOT=/home/kaneki/fleet-split/omp-rs
BIN=$ROOT/target/shared/release/omp-rs
PORT=48771
SRC=/home/kaneki/.omp/agent/sessions/-fleet-split/2026-09-11T00-31-09-193Z_01a08de0-1989-7728-9b22-24e4b11f5f0d.jsonl
WORK=$ROOT/bench/tmp/turnpeak
rm -rf "$WORK"; mkdir -p "$WORK/home/.omp/agent/sessions/-bench"
cp "$SRC" "$WORK/home/.omp/agent/sessions/-bench/"
cat > "$WORK/home/.omp/agent/models.yml" <<EOF
providers:
  mock:
    baseUrl: http://127.0.0.1:$PORT/v1
    apiKey: mock-key-not-a-secret
    api: openai-completions
    models:
      - id: text
        name: mock text
        contextWindow: 128000
        maxTokens: 4096
EOF

MOCK_LOG=$WORK/mock.log PORT=$PORT bun $ROOT/bench/mock-server.ts &
MOCK_PID=$!
sleep 0.5

OUT=$WORK/out.log
: > "$OUT"
( cd "$WORK/home" && HOME="$WORK/home" OMP_RS_RESUME_TRACE=1 OMP_RS_TURN_TRACE=1 \
    "$BIN" "run the benchmark" -p --model mock/text --max-time 120 \
    --session-dir "$WORK/home/.omp/agent/sessions/-bench" \
    --resume "$WORK/home/.omp/agent/sessions/-bench/$(basename "$SRC")" \
    > "$OUT" 2> "$WORK/err.log" ) &
RUN_PID=$!

# sampler: 10ms cadence, self + children RSS
SAMPLES=$WORK/rss.tsv
: > "$SAMPLES"
peak=0
while kill -0 $RUN_PID 2>/dev/null; do
  total=0
  for p in $(pgrep -P $RUN_PID) $RUN_PID; do
    [ -r /proc/$p/status ] || continue
    kb=$(awk '/^VmRSS:/{print $2}' /proc/$p/status)
    total=$((total + ${kb:-0}))
  done
  ts=$(date +%s%N)
  echo -e "$ts\t$total" >> "$SAMPLES"
  [ "$total" -gt "$peak" ] && peak=$total
  sleep 0.01
done
wait $RUN_PID; CODE=$?
kill $MOCK_PID 2>/dev/null

# annotate stdout frame lines with timestamps: rewrite OUT with time prefix per line
echo "exit=$CODE peak_kb=$peak samples=$(wc -l < $SAMPLES)"
awk -F'\t' 'BEGIN{prev=0} {if($2>max){max=$2; mt=$1}} END{printf "sampled_peak_kb=%d at t=%d ms into run\n", max, (mt-systime_init)/1}' "$SAMPLES" 2>/dev/null || true
sort -t$'\t' -k2 -n "$SAMPLES" | tail -3
