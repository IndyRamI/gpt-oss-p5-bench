#!/usr/bin/env bash
# Off-GPU end-to-end check of loadgen + analysis against mock servers.
# Two fake "topologies" differ only in KV capacity, which is enough to
# exercise the clean / not-clean decode-window logic.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/smoke
rm -rf "$OUT"
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

run() {  # topo port max_seqs tp replicas
  local topo=$1 port=$2
  python tests/mock_server.py --port "$port" --max-seqs "$3" &
  for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$port" && break; sleep 0.2; done
  for phase in prefill decode; do
    mkdir -p "$OUT/$topo/$phase"
    echo "{\"topology\":\"$topo\",\"tp\":$4,\"replicas\":$5,\"budget\":4,\"profile\":\"$phase\",\"kv_cache_tokens\":[$(( $3 * 15500 ))]}" \
      > "$OUT/$topo/$phase/server.json"
  done
  python bench/loadgen.py --mode prefill --base-url "http://127.0.0.1:$port" --tokenizer none \
    --input-len 2000 --concurrency 1 2 4 --out "$OUT/$topo/prefill" --tag "$topo"
  python bench/loadgen.py --mode decode --base-url "http://127.0.0.1:$port" --tokenizer none \
    --input-len 2000 --output-len 128 --bursts 1 --concurrency 1 4 16 --out "$OUT/$topo/decode" --tag "$topo"
}

run tp1x4 18001 4 1 4
run tp2x2 18002 16 2 2
python analysis/analyze.py "$OUT"
