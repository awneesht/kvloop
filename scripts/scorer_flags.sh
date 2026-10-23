# vLLM flags that mirror the competition scorer (vLLM 0.19.1, 4x L4).
# Confirmed against HARNESS_README.md on Kaggle 2026-10-23: TP 4, gpu_memory_utilization 0.90,
# max_model_len 32768, gemma4 tool-call and reasoning parsers. Reasoning-config flags follow
# public entrant scripts. Every arm adds its own --kv-cache-dtype flags on top.
#
# The harness hard-codes api_base 127.0.0.1:8000, so run_arm.sh puts the kvloop
# proxy on :8000 and vLLM on :8100.
#
# Single-GPU deployment (consumer setting): TP=1 GPU_UTIL=0.92 CUDA_VISIBLE_DEVICES=0

MODEL="${MODEL:-/kaggle/input/gemma-4-31b-it-qat-w4a16-ct}"
SERVED_NAME="${SERVED_NAME:-gemma-4-31b-it-qat-w4a16-ct}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.90}"   # HARNESS_README: gpu_memory_utilization = 0.90
MAX_LEN="${MAX_LEN:-32768}"
REASONING_START="${REASONING_START:-<|channel>}"

SCORER_FLAGS=(
  --served-model-name "$SERVED_NAME" gemma4
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN"
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
  --reasoning-config "{\"reasoning_start_str\": \"$REASONING_START\", \"reasoning_end_str\": \"<channel|>\"}"
  --default-chat-template-kwargs '{"enable_thinking": true}'
)
if [[ "$SERVED_NAME" == "gemma-4-12b-it-qat-w4a16-ct" ]]; then
  SCORER_FLAGS+=(--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')
fi
