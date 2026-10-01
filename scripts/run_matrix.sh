#!/usr/bin/env bash
# Run the full topology x phase matrix on one P5 node.
#
#   GPU_BUDGET=4 ./scripts/run_matrix.sh                 # tp1x4, tp2x2, tp4x1
#   GPU_BUDGET=8 ./scripts/run_matrix.sh                 # tp1x8 ... tp8x1
#   TOPOLOGIES="tp2x2" PHASES="decode" ./scripts/run_matrix.sh
#   DATASET=/data/testbed.jsonl ./scripts/run_matrix.sh  # use your 15K prompts
#
# For each (topology, phase): deploy with that phase's vLLM profile, run the
# phase's sweep, record GPU telemetry, tear down. Results land in
# results/<RUN_ID>/<topology>/<phase>/.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/experiment.yaml}
GPU_BUDGET=${GPU_BUDGET:-4}
PHASES=${PHASES:-"prefill decode"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)-$(hostname -s)}
DATASET=${DATASET:-}
PROMPT_FIELD=${PROMPT_FIELD:-prompt}
API=${API:-completions}

cfg() { python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(eval(sys.argv[1]))" "$1"; }

TOPOLOGIES=${TOPOLOGIES:-$(cfg "' '.join(k for k,v in c['topologies'].items() if v['budget']==$GPU_BUDGET)")}
ROOT_OUT=results/$RUN_ID
mkdir -p "$ROOT_OUT"
cp "$CONFIG" "$ROOT_OUT/experiment.yaml"
{ nvidia-smi -q | head -30; nvidia-smi topo -m; pip freeze | grep -Ei '^(ray|vllm|torch|transformers|flashinfer|triton)'; } \
  > "$ROOT_OUT/env.txt" 2>&1 || true

echo "run=$RUN_ID topologies=[$TOPOLOGIES] phases=[$PHASES]"

for topo in $TOPOLOGIES; do
  for phase in $PHASES; do
    out=$ROOT_OUT/$topo/$phase
    mkdir -p "$out"
    echo "=== $topo / $phase ==="
    serve shutdown -y >/dev/null 2>&1 || true
    sleep 5

    if ! python serve/launch.py --config "$CONFIG" --topology "$topo" \
          --profile "$phase" --out "$out" 2>&1 | tee "$out/launch.log"; then
      echo "launch failed for $topo/$phase, skipping (see $out/launch.log)"
      continue
    fi

    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm \
      --format=csv -lms 1000 > "$out/gpu.csv" &
    smi_pid=$!

    common=(--mode "$phase" --api "$API" --input-len "$(cfg "c['sweeps']['$phase']['input_len']")"
            --out "$out" --tag "$topo")
    [[ -n "$DATASET" ]] && common+=(--dataset "$DATASET" --prompt-field "$PROMPT_FIELD")

    if [[ $phase == prefill ]]; then
      python bench/loadgen.py "${common[@]}" \
        --concurrency $(cfg "' '.join(map(str, c['sweeps']['prefill']['concurrency']))") \
        2>&1 | tee "$out/loadgen.log" || true
    else
      python bench/loadgen.py "${common[@]}" \
        --output-len "$(cfg "c['sweeps']['decode']['output_len']")" \
        --bursts "$(cfg "c['sweeps']['decode']['bursts_per_level']")" \
        --concurrency $(cfg "' '.join(map(str, c['sweeps']['decode']['concurrency']))") \
        2>&1 | tee "$out/loadgen.log" || true
    fi

    kill $smi_pid 2>/dev/null || true
    serve shutdown -y >/dev/null 2>&1 || true
  done
done

python analysis/analyze.py "$ROOT_OUT"
echo "done: $ROOT_OUT/REPORT.md"
