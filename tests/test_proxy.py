import json
import socket
import threading
import time

import pytest

from kvloop.proxy import StreamAggregator

httpx = pytest.importorskip("httpx")
pytest.importorskip("starlette")
uvicorn = pytest.importorskip("uvicorn")


def test_stream_aggregator():
    agg = StreamAggregator()
    chunks = [
        {"model": "g", "prompt_token_ids": [1, 2], "choices": [{"delta": {"reasoning_content": "think "}, "token_ids": [5]}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_", "arguments": "{\"pa"}}]},
                      "token_ids": [6, 7]}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": "th\": 1}"}}]},
                      "finish_reason": "tool_calls", "token_ids": [8]}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 4}},
    ]
    for c in chunks:
        agg.feed_line("data: " + json.dumps(c))
    agg.feed_line("data: [DONE]")
    r = agg.result()
    msg = r["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": 1}'}
    assert msg["reasoning_content"] == "think "
    assert r["choices"][0]["token_ids"] == [5, 6, 7, 8]
    assert r["prompt_token_ids"] == [1, 2] and r["usage"]["completion_tokens"] == 4


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _fake_upstream(seen):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    async def chat(request: Request):
        body = await request.json()
        seen.append(body)
        if body.get("stream"):
            async def gen():
                for d in ({"content": "hel"}, {"content": "lo"}):
                    yield "data: " + json.dumps({"choices": [{"delta": d, "token_ids": [9]}]}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop",
                                          "token_ids": [3]}], "prompt_token_ids": [1, 2],
                             "usage": {"prompt_tokens": 2, "completion_tokens": 1}})

    async def models(request):
        return JSONResponse({"data": [{"id": "m"}]})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/models", models)])


def test_proxy_roundtrip(tmp_path):
    from kvloop.proxy import build_app

    seen = []
    up_port, px_port = _free_port(), _free_port()
    log = tmp_path / "req.jsonl"
    servers = [uvicorn.Server(uvicorn.Config(_fake_upstream(seen), port=up_port, log_level="error")),
               uvicorn.Server(uvicorn.Config(build_app(f"http://127.0.0.1:{up_port}", log, inject_token_ids=True),
                                             port=px_port, log_level="error"))]
    threads = [threading.Thread(target=s.run, daemon=True) for s in servers]
    for t in threads:
        t.start()
    base = f"http://127.0.0.1:{px_port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/v1/models").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    msgs = [{"role": "user", "content": "hi"}]
    r = httpx.post(base + "/v1/chat/completions", json={"model": "m", "messages": msgs})
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    with httpx.stream("POST", base + "/v1/chat/completions", json={"model": "m", "messages": msgs, "stream": True}) as s:
        text = "".join(s.iter_text())
    assert "hel" in text and "[DONE]" in text
    for _ in range(50):
        if log.exists() and len(log.read_text().splitlines()) >= 2:
            break
        time.sleep(0.05)
    for s in servers:
        s.should_exit = True
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(recs) == 2
    assert all(b.get("return_token_ids") is True for b in seen)  # injected upstream
    assert recs[0]["response"]["prompt_token_ids"] == [1, 2]
    assert recs[1]["stream"] and recs[1]["response"]["choices"][0]["message"]["content"] == "hello"
    assert recs[1]["response"]["choices"][0]["token_ids"] == [9, 9]
