import pytest

torch = pytest.importorskip("torch")

from kvloop.fakequant_cache import make_fakequant_cache  # noqa: E402
from kvloop.quant import SCHEMES, fp8, get_scheme, int_asym, int_sym  # noqa: E402


def test_quantizers_shape_dtype_and_error_order():
    torch.manual_seed(0)
    x = (torch.randn(1, 2, 64, 32) * 3).to(torch.bfloat16)
    errs = {}
    for name in ("fp8_e4m3", "int8_tok", "int4_tok", "kivi2"):
        s = get_scheme(name)
        y = s.key(x)
        assert y.shape == x.shape and y.dtype == x.dtype
        errs[name] = (y.float() - x.float()).abs().mean().item()
    assert errs["int8_tok"] < errs["int4_tok"] < errs["kivi2"]
    assert errs["fp8_e4m3"] < errs["int4_tok"]


def test_int_levels():
    x = torch.linspace(-1, 1, 64).reshape(1, 1, 1, 64)
    assert torch.unique(int_asym(x, 2)).numel() <= 4
    assert torch.unique(int_sym(x, 4)).numel() <= 15


def test_fp8_scale_one_saturates():
    x = torch.tensor([1000.0, -1000.0, 1.0]).reshape(1, 1, 1, 3)
    y = fp8(x, "e4m3", 1.0)
    assert y[0, 0, 0, 0] == 448 and y[0, 0, 0, 1] == -448


def test_all_schemes_registered():
    assert {"bf16", "fp8_e4m3", "int8_tok", "int4_tok", "kivi2"} <= set(SCHEMES)


def test_bf16_cache_is_exact(tiny_gemma4):
    m = tiny_gemma4
    x = torch.randint(0, 97, (1, 80))
    with torch.no_grad():
        ref = m(x).logits
        out = m(x, past_key_values=make_fakequant_cache(m.config, "bf16"), use_cache=True).logits
    assert torch.equal(ref, out)


@pytest.mark.parametrize("layers,expected", [("all", 6), ("sliding", 5), ("global", 1)])
def test_layer_selection(tiny_gemma4, layers, expected):
    c = make_fakequant_cache(tiny_gemma4.config, "int4_tok", layers)
    assert c.kvloop_info["wrapped"] == expected


def test_compression_changes_logits_monotonically(tiny_gemma4):
    m = tiny_gemma4
    torch.manual_seed(1)
    x = torch.randint(0, 97, (1, 96))
    with torch.no_grad():
        ref = m(x).logits.float()
        d = {}
        for s in ("int8_tok", "int4_tok", "kivi2"):
            o = m(x, past_key_values=make_fakequant_cache(m.config, s, "all"), use_cache=True).logits.float()
            d[s] = (o - ref).abs().mean().item()
    assert 0 < d["int8_tok"] < d["int4_tok"] < d["kivi2"]


def test_kivi_residual_semantics(tiny_gemma4):
    """Layer 0 sees identical inputs in both runs, so its raw K is the BF16 K.
    After N tokens fed in uneven chunks, complete 32-token groups must be quantised
    and the trailing N % 32 tokens must be untouched."""
    m = tiny_gemma4
    torch.manual_seed(2)
    N = 50
    x = torch.randint(0, 97, (1, N))
    base = make_fakequant_cache(m.config, "bf16")
    kv = make_fakequant_cache(m.config, "kivi2", "all")
    with torch.no_grad():
        m(x, past_key_values=base, use_cache=True)
        for s, e in ((0, 7), (7, 20), (20, 33), (33, 50)):
            m(x[:, s:e], past_key_values=kv, use_cache=True)
    raw = base.layers[0].keys
    got = kv.layers[0].keys
    assert got.shape == raw.shape
    sch = get_scheme("kivi2")
    torch.testing.assert_close(got[..., :32, :], sch.key(raw[..., :32, :]))
    torch.testing.assert_close(got[..., 32:, :], raw[..., 32:, :])
    torch.testing.assert_close(kv.layers[0].values[..., :32, :], sch.value(base.layers[0].values[..., :32, :]))


def test_kivi_decode_fills_group(tiny_gemma4):
    """Token-by-token decode: the 32nd token triggers quantisation of the whole group."""
    m = tiny_gemma4
    x = torch.randint(0, 97, (1, 32))
    base = make_fakequant_cache(m.config, "bf16")
    kv = make_fakequant_cache(m.config, "kivi2", "all")
    with torch.no_grad():
        m(x, past_key_values=base, use_cache=True)
        for i in range(32):
            m(x[:, i:i + 1], past_key_values=kv, use_cache=True)
    torch.testing.assert_close(kv.layers[0].keys, get_scheme("kivi2").key(base.layers[0].keys))
