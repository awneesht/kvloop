# adapters/

LoRA adapters go here as `adapters/<name>/{adapter_config.json, adapter_model.safetensors}` and are
referenced with `adapter: <name>` on an LlmAgent. v1 ships **no** adapter (plan §3.4 Q4: adapters may be
silently zeroed by the patched vLLM; gate G-LoRA on Oct 25). `*.safetensors` is gitignored. This README is
not reachable from `agent.yaml`, so the packer does not ship it.
