"""Deploy gpt-oss-120b on Ray Serve LLM with a given topology + vLLM profile.

Usage (on the P5, after `ray start --head`):
    python serve/launch.py --topology tp2x2 --profile decode --out results/<run>/

The Serve app outlives this script (Serve is detached); tear it down with
`serve shutdown -y`. On success, writes <out>/server.json describing exactly
what was deployed, including vLLM's reported KV-cache capacity.
"""

import argparse
import glob
import json
import os
import re
import time
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KV_PATTERNS = [
    re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
]


def build_llm_config(cfg, topology, profile, engine_metrics):
    from ray.serve.llm import LLMConfig

    topo = cfg["topologies"][topology]
    model = cfg["model"]
    source = model["model_source"]
    if not os.path.exists(source):
        print(f"[launch] {source} not found, using {model['hf_id']} from HF hub")
        source = model["hf_id"]

    engine_kwargs = dict(cfg["profiles"][profile])
    engine_kwargs["tensor_parallel_size"] = topo["tp"]

    kwargs = dict(
        model_loading_config=dict(model_id=model["model_id"], model_source=source),
        accelerator_type=model["accelerator_type"],
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=topo["replicas"],
                max_replicas=topo["replicas"],
                initial_replicas=topo["replicas"],
            ),
            # Let vLLM's scheduler do the queueing, not the Serve router.
            max_ongoing_requests=512,
        ),
        engine_kwargs=engine_kwargs,
        runtime_env=dict(env_vars=cfg.get("env_vars") or {}),
    )
    if engine_metrics:
        kwargs["log_engine_metrics"] = True
    return LLMConfig(**kwargs), engine_kwargs


def wait_ready(base_url, model_id, timeout_s):
    """serve.run returns once replicas are RUNNING; still do one real request."""
    body = json.dumps(
        {"model": model_id, "prompt": "Hello", "max_tokens": 4, "temperature": 0}
    ).encode()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{base_url}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - keep polling on any failure
            print(f"[launch] not ready yet: {e}")
        time.sleep(10)
    raise TimeoutError(f"server not ready after {timeout_s}s")


def scrape_kv_capacity(since_ts):
    """Best effort: pull vLLM's KV-cache startup lines out of the Ray logs."""
    found = {"kv_cache_tokens": [], "max_concurrency_lines": []}
    for path in glob.glob("/tmp/ray/session_latest/logs/**/*", recursive=True):
        if not os.path.isfile(path) or os.path.getmtime(path) < since_ts:
            continue
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    if m := KV_PATTERNS[0].search(line):
                        found["kv_cache_tokens"].append(int(m.group(1).replace(",", "")))
                    elif KV_PATTERNS[1].search(line):
                        found["max_concurrency_lines"].append(line.strip()[-160:])
        except OSError:
            continue
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/experiment.yaml"))
    ap.add_argument("--topology", required=True)
    ap.add_argument("--profile", required=True, choices=["prefill", "decode"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--ready-timeout", type=int, default=1800)
    ap.add_argument("--engine-metrics", action="store_true",
                    help="set LLMConfig.log_engine_metrics (Prometheus vLLM metrics)")
    args = ap.parse_args()

    import ray
    from ray import serve
    from ray.serve.llm import build_openai_app

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)

    llm_config, engine_kwargs = build_llm_config(
        cfg, args.topology, args.profile, args.engine_metrics
    )
    ray.init(address="auto")
    t0 = time.time()
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app, name="gptoss", route_prefix="/", blocking=False)
    wait_ready(args.base_url, cfg["model"]["model_id"], args.ready_timeout)
    startup_s = time.time() - t0
    print(f"[launch] ready in {startup_s:.0f}s")

    info = {
        "topology": args.topology,
        **cfg["topologies"][args.topology],
        "profile": args.profile,
        "engine_kwargs": engine_kwargs,
        "env_vars": cfg.get("env_vars") or {},
        "startup_s": round(startup_s, 1),
        "ray_version": ray.__version__,
        **scrape_kv_capacity(t0),
    }
    try:
        import vllm

        info["vllm_version"] = vllm.__version__
    except ImportError:
        pass
    with open(os.path.join(args.out, "server.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
