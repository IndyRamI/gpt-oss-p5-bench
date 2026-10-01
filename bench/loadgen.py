"""Phase-isolated load generator for an OpenAI-compatible endpoint.

prefill mode  - max_tokens=1, closed-loop at fixed concurrency (or open-loop
                Poisson with --rates). Latency ~= TTFT ~= queue + prefill.
                Headline: TTFT percentiles and input tokens/s.
decode mode   - synchronized bursts of C requests with long forced outputs
                (ignore_eos). After the last request in the burst gets its
                first token, no more prefills are pending, so the window
                [last first-token, first completion] is pure decode at batch C.
                Headline: decode tokens/s and TPOT/ITL at batch C.

Writes one JSON per sweep level into --out.
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time

import aiohttp

from dataset import build_prompts, load_tokenizer


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs, scale=1.0):
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return {}
    return {"mean": statistics.fmean(xs), "p50": pct(xs, 50), "p90": pct(xs, 90),
            "p99": pct(xs, 99), "max": max(xs), "n": len(xs)}


async def one_request(session, args, prompt, max_tokens, keep_chunks):
    if args.api == "chat":
        url = f"{args.base_url}/v1/chat/completions"
        payload = {"messages": [{"role": "user", "content": prompt}]}
    else:
        url = f"{args.base_url}/v1/completions"
        payload = {"prompt": prompt}
    payload.update({
        "model": args.model, "max_tokens": max_tokens, "min_tokens": max_tokens,
        "ignore_eos": True, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    })

    rec = {"t_send": time.perf_counter(), "ttft": None, "t_done": None,
           "chunks": 0, "prompt_tokens": None, "completion_tokens": None, "error": None}
    chunk_times = []
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return rec
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if msg.get("usage"):
                    rec["prompt_tokens"] = msg["usage"].get("prompt_tokens")
                    rec["completion_tokens"] = msg["usage"].get("completion_tokens")
                for ch in msg.get("choices") or []:
                    d = ch.get("delta") or {}
                    text = ch.get("text") or d.get("content") or d.get(
                        "reasoning_content") or d.get("reasoning")
                    if text:
                        now = time.perf_counter()
                        if rec["ttft"] is None:
                            rec["ttft"] = now - rec["t_send"]
                        rec["chunks"] += 1
                        chunk_times.append(now)
        rec["t_done"] = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:200]
    if keep_chunks:
        rec["chunk_times"] = chunk_times
    return rec


# ---------------------------------------------------------------- prefill

async def run_prefill_level(session, args, prompts, concurrency=None, rate=None):
    n = max(16, args.requests_mult * (concurrency or 4))
    if rate:
        n = max(n, int(rate * args.rate_duration))
    work = [prompts[i % len(prompts)] for i in range(n)]

    # warmup (CUDA graphs, allocator) - not recorded
    await asyncio.gather(*[one_request(session, args, p, 1, False)
                           for p in prompts[: min(4, concurrency or 4)]])

    t0 = time.perf_counter()
    if rate:  # open loop, Poisson arrivals
        tasks = []
        for p in work:
            tasks.append(asyncio.create_task(one_request(session, args, p, 1, False)))
            await asyncio.sleep(random.expovariate(rate))
        recs = await asyncio.gather(*tasks)
    else:     # closed loop
        sem = asyncio.Semaphore(concurrency)

        async def bounded(p):
            async with sem:
                return await one_request(session, args, p, 1, False)

        recs = await asyncio.gather(*[bounded(p) for p in work])
    wall = time.perf_counter() - t0

    ok = [r for r in recs if not r["error"]]
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    summary = {
        "ok": len(ok), "errors": len(recs) - len(ok), "wall_s": wall,
        "ttft_ms": dist([r["ttft"] for r in ok], 1e3),
        "e2e_ms": dist([r["t_done"] - r["t_send"] for r in ok], 1e3),
        "prompt_tokens": dist([r["prompt_tokens"] for r in ok]),
        "input_tok_s": in_tok / wall if wall else None,
        "req_s": len(ok) / wall if wall else None,
    }
    return summary, recs


# ---------------------------------------------------------------- decode

def decode_window_metrics(recs, output_len):
    ok = [r for r in recs if not r["error"] and r["chunk_times"]]
    if not ok:
        return {"ok": 0, "errors": len(recs)}
    first = [r["chunk_times"][0] for r in ok]
    last = [r["chunk_times"][-1] for r in ok]
    w0, w1 = max(first), min(last)

    # tokens per streamed chunk (1.0 unless the server batches stream output)
    tpc = statistics.fmean(
        (r["completion_tokens"] or r["chunks"]) / r["chunks"] for r in ok)

    m = {"ok": len(ok), "errors": len(recs) - len(ok), "tokens_per_chunk": tpc}
    m["ttft_ms"] = dist([r["ttft"] for r in ok], 1e3)
    m["tpot_ms"] = dist([(r["chunk_times"][-1] - r["chunk_times"][0]) /
                         max(1, (r["completion_tokens"] or output_len) - 1)
                         for r in ok], 1e3)
    gaps = [b - a for r in ok for a, b in zip(r["chunk_times"], r["chunk_times"][1:])]
    m["itl_ms"] = dist([g / tpc for g in gaps], 1e3)
    # how many requests were decoding simultaneously before the first one finished
    m["effective_batch"] = sum(1 for f in first if f < w1)

    if w1 - w0 > 0.5:  # clean pure-decode window exists
        n = sum(sum(1 for t in r["chunk_times"] if w0 <= t <= w1) for r in ok)
        m.update(window_clean=True, window_s=w1 - w0,
                 decode_tok_s=n * tpc / (w1 - w0))
    else:
        # Server could not hold all C sequences at once (KV capacity /
        # max_num_seqs) - prefills and decodes overlapped. Report the blended
        # rate and flag it; this is itself a key finding for TP1 at 15K ctx.
        total = sum((r["completion_tokens"] or len(r["chunk_times"])) - 1 for r in ok)
        span = max(last) - min(first)
        m.update(window_clean=False, window_s=max(0.0, w1 - w0),
                 decode_tok_s=total / span if span else None,
                 # not_admitted: some sequences only started after others had
                 #   finished (KV capacity, max_num_seqs, or prefill backlog).
                 # window_too_short: all admitted, raise --output-len.
                 unclean_reason=("window_too_short" if m["effective_batch"] == len(ok)
                                 else "not_admitted"))
    m["per_user_tok_s"] = 1e3 / m["tpot_ms"]["p50"] if m["tpot_ms"] else None
    return m


async def run_decode_level(session, args, prompts, concurrency):
    # warmup at this batch size, short outputs
    await asyncio.gather(*[one_request(session, args, prompts[i % len(prompts)], 16, False)
                           for i in range(min(concurrency, 8))])
    bursts, all_recs = [], []
    for b in range(args.bursts):
        batch = [prompts[(b * concurrency + i) % len(prompts)] for i in range(concurrency)]
        recs = await asyncio.gather(*[
            one_request(session, args, p, args.output_len, True) for p in batch])
        bursts.append(decode_window_metrics(recs, args.output_len))
        all_recs.extend(recs)
    # Pick the median burst by decode_tok_s as the representative summary
    good = sorted((b for b in bursts if b.get("decode_tok_s")),
                  key=lambda b: b["decode_tok_s"])
    summary = dict(good[len(good) // 2]) if good else {"ok": 0}
    summary["bursts"] = bursts
    for r in all_recs:  # keep files small: store relative offsets, ms resolution
        ct = r.pop("chunk_times", [])
        r["chunk_offsets_ms"] = [round((t - r["t_send"]) * 1e3, 2) for t in ct]
    return summary, all_recs


# ---------------------------------------------------------------- main

async def main_async(args):
    levels = args.rates if args.rates else args.concurrency
    need = max(levels) * (args.bursts if args.mode == "decode" else args.requests_mult)
    tok = load_tokenizer(args.tokenizer)
    print(f"[loadgen] building {min(need, args.max_prompts)} prompts of {args.input_len} tokens")
    prompts = build_prompts(min(need, args.max_prompts), args.input_len, tok,
                            args.dataset, args.prompt_field, natural=args.natural_len)
    os.makedirs(args.out, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.request_timeout)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        probe = await one_request(session, args, "Hello", 2, False)
        if probe["error"]:
            raise SystemExit(f"[loadgen] endpoint check failed: {probe['error']}")
        for lvl in levels:
            t = time.time()
            if args.mode == "prefill":
                kw = {"rate": lvl} if args.rates else {"concurrency": lvl}
                summary, recs = await run_prefill_level(session, args, prompts, **kw)
            else:
                summary, recs = await run_decode_level(session, args, prompts, lvl)
            key = f"rate{lvl}" if args.rates else f"c{lvl}"
            meta = {"mode": args.mode, "level": lvl,
                    "level_kind": "rate" if args.rates else "concurrency",
                    "input_len": args.input_len,
                    "output_len": 1 if args.mode == "prefill" else args.output_len,
                    "api": args.api, "dataset": args.dataset, "started": t,
                    "tag": args.tag}
            with open(os.path.join(args.out, f"{args.mode}_{key}.json"), "w") as f:
                json.dump({"meta": meta, "summary": summary, "requests": recs}, f)
            brief = {k: summary.get(k) for k in
                     ("ok", "errors", "input_tok_s", "decode_tok_s", "window_clean",
                      "effective_batch", "per_user_tok_s")}
            ttft = (summary.get("ttft_ms") or {}).get("p50")
            print(f"[loadgen] {args.mode} {key}: ttft_p50={ttft and round(ttft)}ms "
                  f"{json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in brief.items() if v is not None})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--api", choices=["completions", "chat"], default="completions")
    ap.add_argument("--dataset", help="JSONL test-bed file; omit for synthetic prompts")
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--natural-len", action="store_true",
                    help="use dataset prompts at their own length (truncate only)")
    ap.add_argument("--tokenizer", default="openai/gpt-oss-120b",
                    help="HF tokenizer id/path, or 'none' for word-approximation")
    ap.add_argument("--input-len", type=int, default=15000)
    ap.add_argument("--output-len", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--rates", type=float, nargs="+",
                    help="prefill only: open-loop request rates (req/s) instead of concurrency")
    ap.add_argument("--rate-duration", type=float, default=60)
    ap.add_argument("--requests-mult", type=int, default=4)
    ap.add_argument("--bursts", type=int, default=2)
    ap.add_argument("--max-prompts", type=int, default=512)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
