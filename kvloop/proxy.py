"""Logging reverse proxy for an OpenAI-compatible server (vLLM).

The harness talks to the proxy exactly as it would talk to vLLM; the proxy
forwards every request unchanged (apart from optional `return_token_ids`) and
appends one JSON line per chat/completions call:

    {"id", "t_start", "t_end", "latency_s", "ttft_s", "status", "stream",
     "request": {...}, "response": {...}}

With `inject_token_ids=True` it asks vLLM to return the exact prompt and
completion token ids, which makes teacher-forced replay exact (no chat-template
reconstruction). Turn it off if the harness rejects the extra response fields.

Run:  python -m kvloop.proxy --upstream http://127.0.0.1:8100 --port 8000 --log run/requests.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

LOGGED_PATHS = ("/v1/chat/completions", "/v1/completions")


class StreamAggregator:
    """Rebuilds a chat completion response from SSE chunks."""

    def __init__(self):
        self.content = []
        self.reasoning = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason = None
        self.usage = None
        self.token_ids: list[int] = []
        self.prompt_token_ids = None
        self.model = None
        self.n_chunks = 0

    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if payload == "[DONE]" or not payload:
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        self.n_chunks += 1
        self.model = obj.get("model", self.model)
        if obj.get("usage"):
            self.usage = obj["usage"]
        if obj.get("prompt_token_ids") is not None and self.prompt_token_ids is None:
            self.prompt_token_ids = obj["prompt_token_ids"]
        for ch in obj.get("choices") or []:
            if ch.get("token_ids"):
                self.token_ids.extend(ch["token_ids"])
            if ch.get("prompt_token_ids") is not None and self.prompt_token_ids is None:
                self.prompt_token_ids = ch["prompt_token_ids"]
            if ch.get("finish_reason"):
                self.finish_reason = ch["finish_reason"]
            d = ch.get("delta") or {}
            if d.get("content"):
                self.content.append(d["content"])
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                self.reasoning.append(r)
            for tc in d.get("tool_calls") or []:
                slot = self.tool_calls.setdefault(tc.get("index", 0), {"id": None, "type": "function",
                                                                       "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

    def result(self) -> dict:
        msg = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        choice = {"index": 0, "message": msg, "finish_reason": self.finish_reason}
        if self.token_ids:
            choice["token_ids"] = self.token_ids
        out = {"model": self.model, "choices": [choice], "usage": self.usage, "stream_chunks": self.n_chunks}
        if self.prompt_token_ids is not None:
            out["prompt_token_ids"] = self.prompt_token_ids
        return out


def build_app(upstream: str, log_path: str | Path, inject_token_ids: bool = False, timeout: float = 3600.0):
    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import Route

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(timeout, connect=30.0))
    hop = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}

    async def write(rec: dict) -> None:
        async with lock:
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    async def handler(request: Request):
        path = request.url.path
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in hop}
        logged = request.method == "POST" and path in LOGGED_PATHS
        req_json = None
        if logged:
            try:
                req_json = json.loads(body)
                if inject_token_ids:
                    req_json["return_token_ids"] = True
                    body = json.dumps(req_json).encode()
            except json.JSONDecodeError:
                logged = False
        rid, t0 = uuid.uuid4().hex, time.time()
        stream = bool(req_json and req_json.get("stream"))
        up_req = client.build_request(request.method, path, params=request.query_params,
                                      headers=headers, content=body)
        up = await client.send(up_req, stream=True)
        resp_headers = {k: v for k, v in up.headers.items() if k.lower() not in hop}

        if not logged:
            data = await up.aread()
            await up.aclose()
            return Response(data, status_code=up.status_code, headers=resp_headers)

        if stream:
            agg = StreamAggregator()
            first = {"t": None}

            async def gen():
                buf = ""
                try:
                    async for chunk in up.aiter_bytes():
                        if first["t"] is None:
                            first["t"] = time.time()
                        text = chunk.decode("utf-8", errors="replace")
                        buf += text
                        *lines, buf = buf.split("\n")
                        for ln in lines:
                            agg.feed_line(ln)
                        yield chunk
                    if buf:
                        agg.feed_line(buf)
                finally:
                    await up.aclose()
                    t1 = time.time()
                    await write({"id": rid, "t_start": t0, "t_end": t1, "latency_s": t1 - t0,
                                 "ttft_s": (first["t"] - t0) if first["t"] else None,
                                 "status": up.status_code, "stream": True, "path": path,
                                 "request": req_json, "response": agg.result()})

            return StreamingResponse(gen(), status_code=up.status_code, headers=resp_headers)

        data = await up.aread()
        await up.aclose()
        t1 = time.time()
        try:
            resp_json = json.loads(data)
        except json.JSONDecodeError:
            resp_json = {"raw": data.decode("utf-8", errors="replace")[:20000]}
        await write({"id": rid, "t_start": t0, "t_end": t1, "latency_s": t1 - t0, "ttft_s": None,
                     "status": up.status_code, "stream": False, "path": path,
                     "request": req_json, "response": resp_json})
        return Response(data, status_code=up.status_code, headers=resp_headers)

    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
    return Starlette(routes=[Route("/{path:path}", handler, methods=methods)])


def main(argv=None):
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", default="http://127.0.0.1:8100")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log", required=True)
    p.add_argument("--inject-token-ids", action="store_true")
    a = p.parse_args(argv)
    uvicorn.run(build_app(a.upstream, a.log, a.inject_token_ids), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
