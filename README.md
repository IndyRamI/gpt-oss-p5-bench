# gpt-oss-p5-bench

This repo benchmarks **gpt-oss-120b (MXFP4)** on one **AWS p5.48xlarge** (8× H100)
served with **Ray Serve LLM + vLLM**. It compares TP/replica splits such as
TP1×4 replicas, TP2×2 and TP4×1 on ~15K-token prompts. Prefill and decode are
measured separately, each with its own vLLM config.

- **Experiment design, hypotheses and back-of-envelope math:** [docs/PLAN.md](docs/PLAN.md). Read this first.
- On H100 the MoE runs MXFP4 weights with BF16 activations. The MXFP4×MXFP8 kernel is
  Blackwell-only. See [PLAN §2](docs/PLAN.md#about-mxfp4_mxfp8-on-p5).

```
configs/experiment.yaml   topologies, vLLM prefill/decode profiles, sweeps  <- edit this
serve/launch.py           deploy one topology+profile on Ray Serve LLM
serve/vllm_direct.py      same profile on plain `vllm serve` (overhead baseline)
serve/pd_disagg.py        optional: prefill/decode disaggregation over NIXL
bench/loadgen.py          phase-isolated load generator (prefill | decode)
bench/dataset.py          builds exact-length prompts from your JSONL or synthetic
scripts/run_matrix.sh     full matrix: deploy -> sweep -> telemetry -> teardown
analysis/analyze.py       summary.csv + charts + REPORT.md per run
tests/smoke.sh            off-GPU check of loadgen + analysis against a mock server
```

## Runbook

### 0. AWS prerequisites (one-time)
- **Quota:** p5.48xlarge has 192 vCPUs. Under Service Quotas → EC2, raise *Running On-Demand P
  instances* to at least 192 in your region. On-demand P5 capacity is often scarce. If
  `run-instances` returns `InsufficientInstanceCapacity`, buy an
  **EC2 Capacity Block for ML** (EC2 console → Capacity Reservations → Capacity
  Blocks) for the hours you need. The full matrix needs about 8 h. See PLAN §5.
- Create a key pair and a security group that allows SSH only from your IP. Don't
  open 8000 or 8265; use SSH tunnels.

### 1. Launch the instance
```bash
AMI=$(aws ssm get-parameter --region us-east-1 \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
aws ec2 run-instances --region us-east-1 --instance-type p5.48xlarge --image-id "$AMI" \
  --key-name <your-key> --security-group-ids <sg-id> --subnet-id <subnet-in-AZ-with-capacity> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gpt-oss-bench}]'
# For a Capacity Block add:
#   --instance-market-options MarketType=capacity-block \
#   --capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=<cr-id>}
```
Connect with tunnels for the Ray dashboard (8265) and the endpoint (8000):
```bash
ssh -i <key>.pem -L 8265:localhost:8265 -L 8000:localhost:8000 ubuntu@<public-ip>
```

### 2. Set up the instance (~15 min, most of it the 65 GB download)
```bash
git clone https://github.com/IndyRamI/gpt-oss-p5-bench.git && cd gpt-oss-p5-bench
bash setup/install.sh
source .venv/bin/activate
ray start --head --dashboard-host 0.0.0.0
```
The model goes to `/opt/dlami/nvme/models/gpt-oss-120b`, the DLAMI's instance-store
NVMe. It loads much faster from there than from EBS. Instance store is **wiped on stop**,
so `install.sh` downloads the model again after a restart.

### 3. Bring-up check (one config, ~10 min)
```bash
python serve/launch.py --topology tp1x4 --profile prefill --out results/bringup
cat results/bringup/server.json    # check kv_cache_tokens, vllm_version
python bench/loadgen.py --mode prefill --concurrency 1 4 --out results/bringup
serve shutdown -y
```
Check that TTFT at c=1 is roughly in the 0.4–1 s range (PLAN §3) and that `errors` is 0.

### 4. Bring your test-bed data
Copy your prompts to the instance as JSONL, one object per line, with the text under `prompt`.
A chat `messages` list in that field also works. Prompts are packed or truncated to exactly
15,000 tokens. Pass `--natural-len` to `loadgen.py` to keep their real lengths instead.
```bash
scp -i <key>.pem testbed.jsonl ubuntu@<ip>:~/gpt-oss-p5-bench/data/
```
Without `DATASET`, the harness falls back to synthetic prompts. Prefer real data:
MoE expert routing, and therefore decode bandwidth, depends a little on content.

### 5. Run the matrix
Run it inside `tmux` so it survives SSH disconnects:
```bash
tmux new -s bench
source .venv/bin/activate
DATASET=data/testbed.jsonl GPU_BUDGET=4 ./scripts/run_matrix.sh     # tp1x4, tp2x2, tp4x1  (~2.5 h)
DATASET=data/testbed.jsonl GPU_BUDGET=8 ./scripts/run_matrix.sh     # optional whole-node set
```
Useful variants:
```bash
TOPOLOGIES="tp2x2" PHASES="decode" ./scripts/run_matrix.sh          # rerun one cell
API=chat ./scripts/run_matrix.sh                                    # chat endpoint (harmony template)
```
Tune either phase's vLLM config in `configs/experiment.yaml → profiles`. Any vLLM
engine arg can go there. The config is copied into each run directory.

### 6. Optional stages (PLAN §5 D–F)
```bash
# Ray Serve overhead baseline: plain vLLM at TP4 on port 8001
python serve/vllm_direct.py --tp 4 --profile decode &
python bench/loadgen.py --mode decode --base-url http://127.0.0.1:8001 \
  --concurrency 1 8 32 --out results/<run>/direct_tp4/decode
# P/D disaggregation (experimental; needs NIXL)
python serve/pd_disagg.py --prefill tp2x1 --decode tp2x1 --out results/<run>/pd_tp2p_tp2d/decode
python bench/loadgen.py --mode decode --concurrency 1 4 8 16 32 64 --out results/<run>/pd_tp2p_tp2d/decode
python analysis/analyze.py results/<run>
```

### 7. Bring results home and commit
Keep GitHub credentials off the instance. Pull the results back to your laptop:
```bash
rsync -avz -e "ssh -i <key>.pem" ubuntu@<ip>:~/gpt-oss-p5-bench/results/ results/
git add results/<run-id> && git commit -m "Results: <run-id> iso-4 matrix" && git push
```
Each run directory contains `REPORT.md` (findings, capacity table, charts and tables),
`summary.csv`, and the raw per-request JSON. Write your interpretation in the
report's **Analysis notes** section, guided by the questions in PLAN §6.

### 8. Stop the instance
```bash
aws ec2 terminate-instances --instance-ids <id>     # or stop; NVMe model copy is lost either way
```

## Local development (no GPU)
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash tests/smoke.sh        # mock server -> loadgen -> analysis; writes results/smoke/
```

## Reading the output
- **prefill.png:** TTFT p50/p99 and input tok/s/GPU against concurrency. Look for the knee.
- **decode.png:** tok/s/GPU and TPOT against batch. A hollow marker means no clean decode
  window. Check `unclean_reason` in `summary.csv`.
- **decode_pareto.png:** efficiency (tok/s/GPU) against interactivity (tok/s/user).
  The frontier is the answer to "which topology for decode".
