#!/usr/bin/env bash
# One-time setup on a fresh p5.48xlarge running the AWS Deep Learning AMI
# (Ubuntu 22.04/24.04, NVIDIA driver preinstalled). Idempotent.
#
#   bash setup/install.sh
#
# Afterwards: source .venv/bin/activate && ray start --head
set -euo pipefail
cd "$(dirname "$0")/.."

NVME=${NVME:-/opt/dlami/nvme}         # DLAMI mounts the 8x3.84TB instance store here
MODEL_DIR=${MODEL_DIR:-$NVME/models/gpt-oss-120b}
RAY_VERSION=${RAY_VERSION:-}          # e.g. "==2.50.0" to pin; empty = latest

echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m | head -10

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== Python env (.venv)"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
# ray[llm] pins a compatible vLLM; let it choose rather than mixing versions.
uv pip install "ray[serve,llm]${RAY_VERSION}" transformers aiohttp pyyaml matplotlib \
  "huggingface_hub[hf_transfer]"
python -c "import ray, vllm, torch; print('ray', ray.__version__, '| vllm', vllm.__version__, '| torch', torch.__version__, '| cuda ok:', torch.cuda.is_available())"

echo "== Model -> $MODEL_DIR (~65 GB, skips original/ and metal/ duplicates)"
sudo mkdir -p "$MODEL_DIR" && sudo chown -R "$USER" "$NVME/models"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download openai/gpt-oss-120b --local-dir "$MODEL_DIR" --exclude "original/*" "metal/*"
du -sh "$MODEL_DIR"

# Tokenizer cache for the load generator (small)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_DIR')"

cat <<EOF

Setup complete. Next:
  source .venv/bin/activate
  ray start --head --dashboard-host 0.0.0.0
  python serve/launch.py --topology tp1x4 --profile prefill --out results/try    # smoke on real GPUs
  serve shutdown -y
EOF
