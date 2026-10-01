"""Tiny OpenAI-compatible streaming mock for validating the harness off-GPU.

Models one "replica": a KV-capacity semaphore (MAX_SEQS), prefill time
proportional to prompt length, and a decode step time that grows with the
number of active sequences.

    python tests/mock_server.py --port 8000 --max-seqs 8
"""

import argparse
import asyncio
import json

from aiohttp import web

state = {"active": 0}


async def completions(request):
    body = await request.json()
    prompt = body.get("prompt") or body["messages"][-1]["content"]
    n_in, n_out = len(prompt.split()), body.get("max_tokens", 16)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    async with request.app["kv"]:
        async with request.app["prefill_lock"]:  # one prefill at a time
            await asyncio.sleep(n_in * request.app["prefill_us"] / 1e6)
        state["active"] += 1
        try:
            for i in range(n_out):
                if i:
                    await asyncio.sleep((4 + 0.15 * state["active"]) / 1e3)
                chunk = {"choices": [{"index": 0, "text": " tok"}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        finally:
            state["active"] -= 1
    usage = {"prompt_tokens": n_in, "completion_tokens": n_out}
    await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--prefill-us", type=float, default=20.0, help="per prompt token")
    a = ap.parse_args()
    app = web.Application()
    app["kv"] = asyncio.Semaphore(a.max_seqs)
    app["prefill_lock"] = asyncio.Lock()
    app["prefill_us"] = a.prefill_us
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", completions)
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
