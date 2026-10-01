# Experiment plan: gpt-oss-120b on one p5.48xlarge, ~15K-token prompts

## 1. Question

At a fixed GPU budget, how should gpt-oss-120b be split between tensor
parallelism and data-parallel replicas (Ray Serve) for 15K-token prompts?
Prefill and decode are measured **separately**, each with its own vLLM
config, because they bottleneck on different things:

| Phase | Bound by | What we want | vLLM profile |
|---|---|---|---|
| Prefill (15K in → 1 out) | Tensor-core FLOPs | low TTFT, high input tok/s/GPU | `prefill`: 16K-token chunks, ≤64 seqs |
| Decode (15K ctx → 2048 out) | HBM bandwidth (weights + KV) and KV capacity | high output tok/s/GPU at acceptable per-user tok/s | `decode`: 8K chunks, 256 seqs, big CUDA-graph range |

## 2. Hardware and model facts that drive the design

**p5.48xlarge**: 8× H100 SXM 80 GB (3.35 TB/s HBM3, ~989 dense BF16 TFLOPS each),
NVLink/NVSwitch at 900 GB/s per GPU, 8×3.84 TB local NVMe.

**gpt-oss-120b**: 117B total / 5.1B active parameters, MoE with 128 experts and top-4
routing, 36 layers, 64 query heads, 8 KV heads, head_dim 64. Layers alternate between
128-token sliding-window attention and full attention. Expert weights are
MXFP4 and everything else is BF16. The checkpoint is about 65 GB.

### About "MXFP4_MXFP8" on P5

The MXFP4-weight × MXFP8-activation MoE kernel
(`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`) is **Blackwell-only (SM100)**. On H100,
vLLM runs the MXFP4 experts on the **Marlin** kernel: 4-bit weights are
dequantized and the matmuls run with BF16 activations (W4A16). So on P5 you are
benchmarking MXFP4 weights with BF16 activations. If you need true MXFP4×MXFP8
numbers, rerun this repo unchanged on a p6-b200 and set that env var in
`configs/experiment.yaml → env_vars`. Each run records `env.txt` and
`server.json`, so the two runs can be told apart.

## 3. Back-of-envelope predictions (to check against measurements)

These are estimates. The value of the experiment comes from where the measurements disagree with them.

**KV cache per token.** Only the 18 full-attention layers grow with context:
`18 layers × 2 (K,V) × 8 heads × 64 × 2 B = 36 KiB/token`. That makes one 15K
sequence about **0.55 GB** of KV. Sliding-window layers add only about 5 MB per sequence.

**KV capacity per replica** (gpu_memory_utilization 0.92 ≈ 73 GB per GPU):

| TP | Weights/GPU | Left for KV/GPU (approx) | KV per token per GPU | ≈ concurrent 15K seqs / replica |
|---|---|---|---|---|
| 1 | ~65 GB | ~5–8 GB | 36 KiB | **~10–15** |
| 2 | ~33 GB | ~36 GB | 18 KiB | ~130 |
| 4 | ~17 GB | ~52 GB | 9 KiB | ~370 (capped by max_num_seqs=256) |

**Hypothesis H1:** at 15K context, TP1 is *KV-capacity-bound*. tp1x4 can hold only
about 40–60 sequences across 4 GPUs, while tp2x2 holds about 260. The decode curves for TP1
should flatten early, and the harness should flag `not_admitted` at c≥64. Confirm it
with the "Deployed capacity" table in the report, which comes from vLLM's
`GPU KV cache size` log line.

**Prefill compute per prompt.** Linear layers take `2 × 5.1B × 15K ≈ 153 TFLOP`.
Full-attention layers take `18 × 2 × 15K² × 4096 ≈ 33 TFLOP` (causal). That totals about
**186 TFLOP per prompt**. At 35–50% MFU on H100, TP1 TTFT should be about
**0.4–0.55 s**, which is roughly 25–35K input tok/s/GPU at saturation.

**Hypothesis H2:** TP2 should cut single-request TTFT by about 1.8×. The 72
all-reduces of about 86 MB each cost around 20 ms over NVLink. TP4 should cut it by about 3×. Prefill
**tok/s per GPU** should be best at TP1 and degrade slowly as TP grows. For
throughput-oriented prefill, tp1x4 should win as long as the 15K prompts fit,
and they do.

**Decode bytes per step.** Every step reads the active weights. At batch 1 that is about 2 GB of
BF16 attention, about 1.9 GB of MXFP4 experts and about 1.2 GB for the unembedding, so
roughly 5 GB. That puts the ideal at about 1.5 ms per token, and roughly 250–350 tok/s per user is realistic.
As the batch grows, more distinct experts get touched. The chance that a given expert is untouched is
`(1−4/128)^B`: about 60% at B=16 and about 13% at B=64. By B≈64 a step reads close to
the full ~60 GB of experts. Each step also reads `B × 0.55 GB` of KV.

**Hypothesis H3:** decode is dominated by expert-weight streaming plus KV reads.
TP divides both across GPUs, so TPOT should drop almost linearly with TP. TP1
also cannot batch far enough to amortize the expert sweep (see H1). The expectation
is that **tp2x2 beats tp1x4 on decode tok/s/GPU at 15K context**, which is the
opposite of the short-context intuition. tp4x1 should give the best per-user tok/s.

## 4. Methodology

### Prefill measurement (`bench/loadgen.py --mode prefill`)
- `max_tokens=1`, so end-to-end time ≈ TTFT = queueing + prefill.
- Closed loop at total concurrency {1, 2, 4, 8, 16, 32}, with 4×C requests (at least 16) after
  warmup.
- Headline metrics: TTFT p50/p90/p99 and **input tok/s/GPU**. The knee where TTFT
  starts growing linearly with C is that topology's prefill saturation point.
- Optional open-loop mode (`--rates 0.5 1 2 4`) gives TTFT under Poisson arrivals,
  which is closer to production.

### Decode measurement (`bench/loadgen.py --mode decode`): the "decode window"
- Send a synchronized burst of C requests with 2048 forced output tokens
  (`ignore_eos`, `min_tokens`).
- Record a timestamp for every streamed token.
- Once the **last** request in the burst has its first token, no prefills are
  pending. From then until the **first** request finishes, the server runs pure
  decode at batch C. Throughput and ITL are measured inside that window.
- If the window never opens, the run is flagged:
  - `not_admitted`: some sequences only started after others finished. The usual causes are
    KV capacity or max_num_seqs, but a long prefill backlog can also do it. This is H1 showing up.
  - `window_too_short`: every sequence was admitted, but the output was too short to measure. Raise
    `output_len`.
- Headline metrics: **output tok/s/GPU** against **per-user tok/s (1/TPOT)**, shown as a Pareto
  plot. Up and to the right is better. This is the standard way to compare
  serving configurations.

### Controls
- Prefix caching is off in both profiles, and every prompt gets a unique leading
  tag. Every request pays the full 15K prefill.
- Prompts are built to exactly 15,000 tokens with the gpt-oss tokenizer, so all
  topologies see identical work. `--natural-len` uses your test-bed lengths
  as-is instead.
- `stream_interval` stays at 1. The vLLM Hopper recipe uses 20, which would
  make client-side ITL meaningless. The harness also corrects for this through
  `tokens_per_chunk`.
- GPU utilization, power and SM clock are sampled every 1 s into `gpu.csv` for each phase.
- Every run saves `experiment.yaml`, `env.txt` (driver, topology, package
  versions) and `server.json` (exact engine kwargs, KV capacity, startup time).
- Ray Serve overhead: `serve/vllm_direct.py` serves the same profile with plain
  `vllm serve`. Compare tp4x1 under Serve against `--tp 4` direct. A gap of more than about 3% means
  the router or proxy is a confound.

## 5. Matrix and time budget

| Stage | Configs | Est. wall time |
|---|---|---|
| A. Bring-up and smoke | tp1x4/prefill, c=1 | 30 min |
| B. **iso-4 (your ask)** | tp1x4, tp2x2, tp4x1 × {prefill, decode} | ~2.5 h |
| C. iso-8 (whole node) | tp1x8, tp2x4, tp4x2, tp8x1 × both | ~3.5 h |
| D. Serve overhead | vllm_direct --tp 4 vs tp4x1 | 30 min |
| E. Optional P/D disaggregation | 1×TP2 prefill + 1×TP2 decode over NIXL vs tp2x2 | 1 h |
| F. Optional tuning | best topology: sweep max_num_batched_tokens {4K, 8K, 16K}, max_num_seqs | 1–2 h |

Each deployment takes about 3–6 min to start (weights load from NVMe and CUDA graphs get captured).
The torch.compile cache in `~/.cache/vllm` makes restarts faster.

## 6. What the final analysis should answer

1. For **TTFT SLOs** (for example p90 < 1 s at a given arrival rate), which topology
   serves the most req/s per GPU?
2. For **decode**, which topology sits on the Pareto frontier at your target per-user
   speed (for example ≥ 50 tok/s/user)?
3. Were H1–H3 right? Did TP1's KV capacity limit decode? Where are the
   prefill knees?
4. Is the best prefill topology different from the best decode topology? If it is,
   that is the quantitative case for **P/D disaggregation** (stage E). Prefill
   would run on the compute-efficient TP and decode on the bandwidth- and capacity-efficient TP.
5. How large is Ray Serve's overhead compared with plain vLLM?

Write these answers into the "Analysis notes" section of each run's
`REPORT.md`, and add a cross-run summary to `results/README.md`.
