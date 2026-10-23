#!/usr/bin/env bash
# Feasibility probe: does each closed-loop arm start on this hardware with Gemma 4 and the
# scorer's vLLM, how big a KV pool does it get, how fast does it decode, and can it serve a
# near-32K prompt? Run this FIRST; it retires the main risks.
#
# Usage:
#   bash scripts/probe_arms.sh [arm ...]                                     # 4x L4, scorer setting
#   TP=1 GPU_UTIL=0.92 CUDA_VISIBLE_DEVICES=0 PROBE_TAG=tp1 bash scripts/probe_arms.sh   # one 24 GB GPU
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/scorer_flags.sh"
CONFIG_JSON="${CONFIG_JSON:-$MODEL/config.json}"
TAG="${PROBE_TAG:-tp$TP}"
OUT="${RUNS_DIR:-runs}/probe/$TAG"
mkdir -p "$OUT"
ARMS=("$@")
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=(bf16 fp8_all fp8_global fp8_sliding)
PORT="${UP_PORT:-8100}"
LONG_TOKENS="${LONG_TOKENS:-28000}"

for ARM in "${ARMS[@]}"; do
  read -r -a FLAGS <<< "$(kvloop --config-json "$CONFIG_JSON" serve-args "$ARM")"
  LOG="$OUT/$ARM.vllm.log"
  echo "=== [$TAG] $ARM: ${FLAGS[*]:-(none)}"
  vllm serve "$MODEL" --port "$PORT" "${SCORER_FLAGS[@]}" "${FLAGS[@]}" > "$LOG" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { OK=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 10
  done
  python - "$ARM" "$OK" "$LOG" "$OUT/$ARM.json" "$PORT" "$SERVED_NAME" "$LONG_TOKENS" "${FLAGS[*]:-}" <<'EOF'
import json, re, sys, time, urllib.request

arm, ok, log, out, port, served, long_tokens, flags = sys.argv[1:9]
url = f"http://127.0.0.1:{port}/v1/chat/completions"


def call(body, timeout=900):
    t0 = time.time()
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.time() - t0, None
    except Exception as e:  # HTTP 400 for over-long prompts lands here
        detail = getattr(e, "read", lambda: b"")().decode(errors="replace")[:300]
        return {}, time.time() - t0, f"{e} {detail}"


rec = {"arm": arm, "started": ok == "1", "vllm_flags": flags.split()}
if ok == "1":
    msg = [{"role": "user", "content": "Write a Python function that reverses a string."}]
    r, _, err = call({"model": served, "messages": msg, "max_tokens": 64, "temperature": 0, "return_token_ids": True})
    ch = (r.get("choices") or [{}])[0]
    rec["return_token_ids_ok"] = bool(r.get("prompt_token_ids") or ch.get("token_ids"))
    rec["sample"] = ((ch.get("message") or {}).get("content") or "")[:200]
    # decode speed: 512 forced tokens, warm
    body = {"model": served, "messages": msg, "max_tokens": 512, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    call(body)
    r, dt, err = call(body)
    n = (r.get("usage") or {}).get("completion_tokens") or 0
    rec["decode_tok_s_short_ctx"] = round(n / dt, 2) if dt and n else None
    # near-32K prompt: does it fit, and how fast is prefill + decode there?
    filler = "The quick brown fox jumps over the lazy dog. " * (int(long_tokens) // 10)
    body = {"model": served, "max_tokens": 128, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": filler + "\nSummarise the text above in one word."}]}
    r, dt, err = call(body)
    u = r.get("usage") or {}
    rec["long_prompt_tokens"] = u.get("prompt_tokens")
    rec["long_prompt_ok"] = err is None and bool(u)
    rec["long_prompt_s"] = round(dt, 1)
    rec["long_prompt_error"] = err
txt = open(log, errors="replace").read()
kv = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", txt)
rec.update({
    "kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None,
    "max_concurrency": float(conc[-1][1]) if conc else None,
    "attention_backends": sorted(set(re.findall(r"[Uu]sing ([A-Za-z0-9_]+) (?:attention )?backend", txt))),
    "errors": [l for l in txt.splitlines() if "Error" in l or "error:" in l][-5:],
})
json.dump(rec, open(out, "w"), indent=2)
keys = ("arm", "started", "kv_cache_tokens", "max_concurrency", "decode_tok_s_short_ctx", "long_prompt_ok",
        "return_token_ids_ok")
print(json.dumps({k: rec.get(k) for k in keys}))
EOF
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 5
done
