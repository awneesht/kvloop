#!/usr/bin/env bash
# Feasibility probe: does each closed-loop arm start on this hardware with
# Gemma 4 and the scorer's vLLM, and how big is the KV pool it gets?
# Records the KV-cache token count, max concurrency, attention backend, and
# whether `return_token_ids` works. Run this FIRST; it retires the main risks.
#
# Usage: bash scripts/probe_arms.sh [arm ...]     (default: all vLLM-capable arms)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/scorer_flags.sh"
CONFIG_JSON="${CONFIG_JSON:-$MODEL/config.json}"
OUT="${RUNS_DIR:-runs}/probe"
mkdir -p "$OUT"
ARMS=("$@")
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=(bf16 fp8_all fp8_global fp8_sliding)
PORT="${UP_PORT:-8100}"

for ARM in "${ARMS[@]}"; do
  read -r -a FLAGS <<< "$(kvloop --config-json "$CONFIG_JSON" serve-args "$ARM")"
  LOG="$OUT/$ARM.vllm.log"
  echo "=== $ARM: ${FLAGS[*]:-(none)}"
  vllm serve "$MODEL" --port "$PORT" "${SCORER_FLAGS[@]}" "${FLAGS[@]}" > "$LOG" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { OK=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 10
  done
  RESP=""
  if [[ $OK == 1 ]]; then
    RESP=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$SERVED_NAME\", \"max_tokens\": 64, \"temperature\": 0, \"return_token_ids\": true,
      \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python function that reverses a string.\"}]}")
  fi
  printf '%s' "$RESP" > "$OUT/$ARM.resp.json"
  python - "$ARM" "$OK" "$LOG" "$OUT/$ARM.json" "$OUT/$ARM.resp.json" "${FLAGS[*]:-}" <<'EOF'
import json, re, sys
arm, ok, log, out, resp_path, flags = sys.argv[1:7]
txt = open(log, errors="replace").read()
resp_raw = open(resp_path, errors="replace").read()
try:
    resp = json.loads(resp_raw) if resp_raw.strip() else {}
except Exception:
    resp = {"raw": resp_raw[:500]}
kv = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", txt)
backend = re.findall(r"[Uu]sing ([A-Za-z0-9_]+) (?:attention )?backend", txt)
ch = (resp.get("choices") or [{}])[0]
rec = {"arm": arm, "started": ok == "1", "vllm_flags": flags.split(),
       "kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None,
       "max_concurrency": float(conc[-1][1]) if conc else None,
       "attention_backends": sorted(set(backend)),
       "return_token_ids_ok": bool(resp.get("prompt_token_ids") or ch.get("token_ids")),
       "sample": ((ch.get("message") or {}).get("content") or "")[:200],
       "errors": [l for l in txt.splitlines() if "Error" in l or "error:" in l][-5:]}
json.dump(rec, open(out, "w"), indent=2)
print(json.dumps({k: rec[k] for k in ("arm", "started", "kv_cache_tokens", "max_concurrency", "return_token_ids_ok")}))
EOF
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 5
done
