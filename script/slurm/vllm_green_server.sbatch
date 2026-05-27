#!/bin/bash
#SBATCH --job-name=u2t-vllm-green
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/vllm-green-%j.out
#SBATCH --error=logs/vllm-green-%j.err

# Hosts a local GREEN scorer (radllama2-7b) via vLLM's OpenAI-compatible API.
# Update config/project.json's openai_server.base_url to point at this node:port
# (the URL is printed to the log) when running GREEN evaluation.
#
# Stop with `scancel <jobid>`.

set -euo pipefail
cd /lus/lfs1aip2/projects/u6hx/huanan/u2Tokenizer
source script/slurm/_env.sh

# Reuse the existing vllm conda env if present.
if conda env list | awk '{print $1}' | grep -qx vllm; then
    conda activate vllm
fi

MODEL_ID=${GREEN_MODEL:-StanfordAIMI/GREEN-radllama2-7b}
PORT=${PORT:-8000}
HOST_IP=$(hostname -I | awk '{print $1}')

echo "[vllm-green] serving $MODEL_ID at http://$HOST_IP:$PORT/v1"
echo "[vllm-green] set config/project.json.openai_server.base_url to that URL"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.85
