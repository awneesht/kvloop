#!/usr/bin/env bash
# Closed-loop run of one arm:
#   vLLM (arm flags) on :8100  <-  logging proxy on :8000  <-  harness
#
# Usage:  HARNESS_CMD='swegemma eval ... --out "$OUT/results"' bash scripts/run_arm.sh ARM RUN_ID
#
# HARNESS_CMD is evaluated after OUT is set, so it can write into "$OUT/results".
# Point the harness at http://127.0.0.1:8000/v1 (the proxy), exactly as it would
# point at vLLM. Set INJECT_TOKEN_IDS=0 if the harness rejects extra response fields.
set -euo pipefail

ARM="${1:?arm name}"
RUN_ID="${2:-r1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/scorer_flags.sh"

OUT="${RUNS_DIR:-runs}/$ARM/$RUN_ID"
mkdir -p "$OUT/results"
CONFIG_JSON="${CONFIG_JSON:-$MODEL/config.json}"
read -r -a ARM_FLAGS <<< "$(kvloop --config-json "$CONFIG_JSON" serve-args "$ARM")"
UP_PORT="${UP_PORT:-8100}"
PX_PORT="${PX_PORT:-8000}"

echo "[kvloop] arm=$ARM run=$RUN_ID flags: ${ARM_FLAGS[*]:-(none)}"
vllm serve "$MODEL" --port "$UP_PORT" "${SCORER_FLAGS[@]}" "${ARM_FLAGS[@]}" > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill "${PX_PID:-}" "$VLLM_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$UP_PORT/health" > /dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died; tail of log:"; tail -40 "$OUT/vllm.log"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$UP_PORT/health" > /dev/null || { echo "vLLM not healthy after 30 min"; exit 1; }

INJECT=()
[[ "${INJECT_TOKEN_IDS:-1}" == "1" ]] && INJECT=(--inject-token-ids)
python -m kvloop.proxy --upstream "http://127.0.0.1:$UP_PORT" --port "$PX_PORT" --log "$OUT/requests.jsonl" "${INJECT[@]}" &
PX_PID=$!
sleep 3

# sample KV usage / queue depth / prefix-cache counters every 30 s (Prometheus text)
( while kill -0 "$VLLM_PID" 2>/dev/null; do
    { echo "# t=$(date +%s)"
      curl -s "http://127.0.0.1:$UP_PORT/metrics" \
        | grep -E '^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_running|num_requests_waiting|prefix_cache_(hits|queries))'
    } >> "$OUT/metrics.prom"
    sleep 30
  done ) &

T0=$(date +%s)
set +e
eval "${HARNESS_CMD:?set HARNESS_CMD}" 2>&1 | tee "$OUT/harness.log"
RC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)

python - "$OUT" "$ARM" "$RUN_ID" "$T0" "$T1" "$RC" "${ARM_FLAGS[*]:-}" <<'EOF'
import json, sys
out, arm, run, t0, t1, rc, flags = sys.argv[1:8]
json.dump({"arm": arm, "run": run, "t_start": int(t0), "t_end": int(t1), "wall_s": int(t1) - int(t0),
           "harness_rc": int(rc), "vllm_flags": flags.split()}, open(f"{out}/meta.json", "w"), indent=2)
EOF
echo "[kvloop] done: $OUT (harness rc=$RC, $((T1 - T0)) s)"
