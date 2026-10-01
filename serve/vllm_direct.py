"""Baseline: same engine profile served by plain `vllm serve` (no Ray Serve).

Run one TP group directly to measure Ray Serve's routing/proxy overhead:
    python serve/vllm_direct.py --tp 1 --profile prefill            # port 8001
    python bench/loadgen.py --mode prefill --base-url http://127.0.0.1:8001 ...
Compare against the same TP with replicas=1 under Ray Serve (e.g. tp4x1 vs --tp 4).
"""

import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_flags(engine_kwargs):
    flags = []
    for k, v in engine_kwargs.items():
        name = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            flags.append(name if v else "--no-" + k.replace("_", "-"))
        elif isinstance(v, dict):
            flags += [name, json.dumps(v)]
        else:
            flags += [name, str(v)]
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/experiment.yaml"))
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--profile", choices=["prefill", "decode"], required=True)
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model = cfg["model"]
    src = model["model_source"] if os.path.exists(model["model_source"]) else model["hf_id"]
    cmd = ["vllm", "serve", src, "--served-model-name", model["model_id"],
           "--tensor-parallel-size", str(args.tp), "--port", str(args.port),
           *to_flags(cfg["profiles"][args.profile])]
    os.environ.update({k: str(v) for k, v in (cfg.get("env_vars") or {}).items()})
    print(" ".join(cmd), file=sys.stderr)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
