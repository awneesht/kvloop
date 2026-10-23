# vLLM flags that mirror the competition scorer (vLLM 0.19.1, 4x L4).
# Source: public summaries of the harness HARNESS_README (TP 4, 0.80 utilisation,
# 32K context, gemma4 tool-call and reasoning parsers, thinking on).
# ACTION: diff this against the VllmServer section of HARNESS_README.md in the
# competition data and copy any flag that differs. Every arm adds its own
# --kv-cache-dtype flags on top of these.

MODEL="${MODEL:-/kaggle/input/gemma-4-31b-it-qat-w4a16-ct}"
SERVED_NAME="${SERVED_NAME:-gemma-4-31b-it-qat-w4a16-ct}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_LEN="${MAX_LEN:-32768}"

SCORER_FLAGS=(
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN"
  --served-model-name "$SERVED_NAME"
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
)
