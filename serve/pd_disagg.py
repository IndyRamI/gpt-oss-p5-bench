"""Phase 4 (optional, experimental): true prefill/decode disaggregation.

Prefill replicas use the `prefill` profile, decode replicas the `decode`
profile, and KV caches move between them over NIXL. Default split mirrors the
iso-4 budget: 1 prefill replica at TP2 + 1 decode replica at TP2.

    python serve/pd_disagg.py --prefill tp2x1 --decode tp2x1 --out results/<run>/pd_2p2d/decode
    python bench/loadgen.py --mode decode ... --out results/<run>/pd_2p2d/decode

Requires NIXL (`uv pip install nixl`, preinstalled in rayproject/ray-llm images).
The PD API in Ray Serve LLM is still moving between releases - check
https://docs.ray.io/en/latest/serve/llm/user-guides/prefill-decode.html against
your installed Ray version if this fails to build.
"""

import argparse
import json
import os
import re

import yaml

from launch import ROOT, scrape_kv_capacity, wait_ready


def parse(spec):
    m = re.fullmatch(r"tp(\d+)x(\d+)", spec)
    if not m:
        raise SystemExit(f"bad spec {spec!r}, want e.g. tp2x1")
    return int(m.group(1)), int(m.group(2))


def make(cfg, profile, tp, replicas, role):
    from ray.serve.llm import LLMConfig

    model = cfg["model"]
    src = model["model_source"] if os.path.exists(model["model_source"]) else model["hf_id"]
    ek = dict(cfg["profiles"][profile], tensor_parallel_size=tp,
              kv_transfer_config={"kv_connector": "NixlConnector", "kv_role": role})
    return LLMConfig(
        model_loading_config=dict(model_id=model["model_id"], model_source=src),
        accelerator_type=model["accelerator_type"],
        deployment_config=dict(
            autoscaling_config=dict(min_replicas=replicas, max_replicas=replicas,
                                    initial_replicas=replicas),
            max_ongoing_requests=512),
        engine_kwargs=ek,
        runtime_env=dict(env_vars=cfg.get("env_vars") or {}),
    ), ek


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/experiment.yaml"))
    ap.add_argument("--prefill", default="tp2x1")
    ap.add_argument("--decode", default="tp2x1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    import time

    import ray
    from ray import serve
    from ray.serve.llm import build_pd_openai_app

    cfg = yaml.safe_load(open(args.config))
    (ptp, pr), (dtp, dr) = parse(args.prefill), parse(args.decode)
    pcfg, pek = make(cfg, "prefill", ptp, pr, "kv_producer")
    dcfg, dek = make(cfg, "decode", dtp, dr, "kv_consumer")

    ray.init(address="auto")
    t0 = time.time()
    app = build_pd_openai_app(dict(prefill_config=pcfg, decode_config=dcfg))
    serve.run(app, name="gptoss", route_prefix="/", blocking=False)
    wait_ready(args.base_url, cfg["model"]["model_id"], 1800)

    os.makedirs(args.out, exist_ok=True)
    info = {"topology": f"pd_{args.prefill}_{args.decode}", "tp": None,
            "replicas": None, "budget": ptp * pr + dtp * dr,
            "prefill": {"tp": ptp, "replicas": pr, "engine_kwargs": pek},
            "decode": {"tp": dtp, "replicas": dr, "engine_kwargs": dek},
            "startup_s": round(time.time() - t0, 1), "ray_version": ray.__version__,
            **scrape_kv_capacity(t0)}
    json.dump(info, open(os.path.join(args.out, "server.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
