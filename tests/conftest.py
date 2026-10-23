import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def tiny_gemma4():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Gemma4TextConfig"):
        pytest.skip("transformers without Gemma 4")
    cfg = tr.Gemma4TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=64,
        layer_types=["sliding_attention"] * 5 + ["full_attention"], max_position_embeddings=1024,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=97,
    )
    torch.manual_seed(0)
    model = tr.AutoModelForCausalLM.from_config(cfg).eval()
    # scale K/V projections up so FP8 with scale=1 sees realistic magnitudes
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(("k_proj.weight", "v_proj.weight")):
                p.mul_(8.0)
    return model


class CharTokenizer:
    """Maps id i -> chr(ord('a') + i % 26); enough for segment labelling tests."""

    def decode(self, ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)


@pytest.fixture
def char_tok():
    return CharTokenizer()


def make_record(messages, content=None, tool_calls=None, prompt_ids=None, completion_ids=None,
                t=0.0, status=200, usage=None, finish="stop", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning
    choice = {"index": 0, "message": msg, "finish_reason": finish}
    if completion_ids is not None:
        choice["token_ids"] = completion_ids
    resp = {"choices": [choice], "usage": usage or {"prompt_tokens": len(prompt_ids or []),
                                                    "completion_tokens": len(completion_ids or [])}}
    if prompt_ids is not None:
        resp["prompt_token_ids"] = prompt_ids
    return {"id": f"r{t}", "t_start": t, "t_end": t + 1.0, "status": status, "stream": False,
            "path": "/v1/chat/completions", "request": {"messages": messages}, "response": resp}


def tc(name, args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
