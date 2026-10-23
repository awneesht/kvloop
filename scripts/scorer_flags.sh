# vLLM flags that mirror the competition scorer (vLLM 0.19.1, 4x L4).
# Reconstructed from HARNESS_README §3.1/§3.3/§3.4 as quoted in public entrant repos
# (e.g. github.com/fatu/gemma4-dev-agent scripts/serve_vllm.sh). Diff against your
# copy of HARNESS_README.md. Every arm adds its own --kv-cache-dtype flags on top.
#
# The harness hard-codes api_base 127.0.0.1:8000, so run_arm.sh puts the kvloop
# proxy on :8000 and vLLM on :8100.
#
# Single-GPU deployment (consumer setting): TP=1 GPU_UTIL=0.92 CUDA_VISIBLE_DEVICES=0

MODEL="${MODEL:-/kaggle/input/gemma-4-31b-it-qat-w4a16-ct}"
SERVED_NAME="${SERVED_NAME:-gemma-4-31b-it-qat-w4a16-ct}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.80}"
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
