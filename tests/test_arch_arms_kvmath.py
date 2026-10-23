import pytest

from kvloop.arch import AttentionLayout
from kvloop.arms import DEFAULT_ARMS, Arm, get_arm
from kvloop.kvmath import concurrency, kv_bytes


def test_gemma4_31b_layout():
    lay = AttentionLayout.gemma4_31b()
    assert lay.num_layers == 60
    assert len(lay.indices("sliding")) == 50
    assert lay.indices("global") == list(range(5, 60, 6))
    assert lay.layer_types[-1] == "full_attention"


def test_layout_from_config_dict_nested():
    cfg = {"text_config": {"num_hidden_layers": 6, "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
                           "num_key_value_heads": 16, "head_dim": 256, "num_global_key_value_heads": 4,
                           "global_head_dim": 512, "sliding_window": 1024, "attention_k_eq_v": True}}
    lay = AttentionLayout.from_config_dict(cfg)
    assert lay.global_kv_heads == 4 and lay.global_head_dim == 512 and lay.global_k_eq_v
    assert lay.indices("global") == [5]


def test_vllm_flags():
    lay = AttentionLayout.gemma4_31b()
    assert get_arm("bf16").vllm_args(lay) == []
    assert get_arm("fp8_all").vllm_args(lay) == ["--kv-cache-dtype", "fp8"]
    assert get_arm("fp8_global").vllm_args(lay) == ["--kv-cache-dtype", "fp8", "--kv-cache-dtype-skip-layers",
                                                    "sliding_window"]
    sl = get_arm("fp8_sliding").vllm_args(lay)
    assert sl[:3] == ["--kv-cache-dtype", "fp8", "--kv-cache-dtype-skip-layers"]
    assert sl[3:] == [str(i) for i in range(5, 60, 6)]
    with pytest.raises(ValueError):
        get_arm("int4_all").vllm_args(lay)


def test_bad_layers_rejected():
    with pytest.raises(ValueError):
        Arm("x", "fp8_e4m3", "middle")


def test_kv_bytes_gemma4_31b():
    lay = AttentionLayout.gemma4_31b()
    bf16 = kv_bytes(lay, get_arm("bf16"), 32768)
    # sliding: 50 layers * 1024 tok * (2*16*256) * 2 B ; global: 10 * 32768 * (2*4*512) * 2 B
    assert bf16.sliding_bytes == 50 * 1024 * 8192 * 2
    assert bf16.global_bytes == 10 * 32768 * 4096 * 2
    assert 0.75 < bf16.global_share < 0.77  # globals dominate at 32K
    fp8 = kv_bytes(lay, get_arm("fp8_all"), 32768)
    assert fp8.total_bytes == pytest.approx(bf16.total_bytes / 2)
    g = kv_bytes(lay, get_arm("fp8_global"), 32768)
    s = kv_bytes(lay, get_arm("fp8_sliding"), 32768)
    assert g.total_bytes < s.total_bytes  # compressing globals saves more at long context
    short = kv_bytes(lay, get_arm("bf16"), 1024)
    assert short.global_share < 0.2  # ...but sliding layers dominate short contexts


def test_concurrency_scales_with_compression():
    lay = AttentionLayout.gemma4_31b()
    pool = 40 * 2**30
    c16 = concurrency(lay, get_arm("bf16"), 32768, pool)
    c8 = concurrency(lay, get_arm("fp8_all"), 32768, pool)
    assert c8 == pytest.approx(2 * c16)
    assert all(a.name for a in DEFAULT_ARMS)
