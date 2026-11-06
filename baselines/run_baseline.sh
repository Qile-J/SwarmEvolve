#!/bin/bash
# One baseline run on one GPU, with a fixed wall-clock budget.
#
# Usage:
#   baselines/run_baseline.sh <openevolve|shinkaevolve|codeevolve> <problem> <model> [budget minutes, 60] [seed, 42]
#
# What it does:
#   1. starts vLLM servers on this machine: the Qwen3 model (and Qwen3-Embedding-0.6B for ShinkaEvolve),
#   2. runs the framework until `timeout` stops it after the budget,
#   3. scores every program the framework writes with the shared evaluation (benchmarks/evaluation.py), which logs
#      each evaluation to <run>/eval_log.jsonl.
# Needs: `vllm` and the main Python environment on PATH, and the environments of baselines/setup_baselines.sh.

set -u
cd "$(dirname "$0")/.."

framework=$1 problem=$2 model=$3 budget=${4:-60} seed=${5:-42}
out=results/$problem/${framework}_$(basename $model)_thinking_seed${seed}_$(date +%Y%m%dT%H%M%S)   # the run directory
mkdir -p $out
echo "$out"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OPENAI_API_KEY=EMPTY
export EVAL_PYTHON=${EVAL_PYTHON:-$(command -v python)}   # the main environment runs every evaluated program

# The framework's own environment: OpenEvolve and ShinkaEvolve share one, CodeEvolve needs Python 3.13
python_of() { [ $1 = codeevolve ] && echo baselines/venv_codeevolve/bin/python || echo baselines/venv/bin/python; }

# A free TCP port, chosen by the OS
free_port() { python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])"; }

# serve <port> <log file> <vllm args...>: start a vLLM server and wait until it answers (up to 15 minutes)
serve() {
  local port=$1 log=$2; shift 2
  vllm serve "$@" --port $port >> $log 2>&1 &
  local server=$!
  for i in $(seq 180); do curl -s localhost:$port/v1/models > /dev/null && return 0; kill -0 $server 2>/dev/null || break; sleep 5; done
  echo "vLLM did not start (see $log)"; exit 1
}
trap 'pkill -P $$' EXIT   # the servers end with this script

# 1. The vLLM servers
port=$(free_port)
if [ $framework = shinkaevolve ]; then
  # ShinkaEvolve's embedding model first (two servers profiling GPU memory at the same time fail);
  # the main server then takes 0.78 of the GPU memory instead of 0.88
  embed_port=$(free_port)
  serve $embed_port $out/vllm_embedding.log Qwen/Qwen3-Embedding-0.6B --runner pooling --max-model-len 8192 --gpu-memory-utilization 0.1
fi
# The same settings for every framework: 32k context, prefix caching, Qwen3's thinking mode (reasoning parsed apart)
serve $port $out/vllm.log $model --max-model-len 32768 --gpu-memory-utilization $([ $framework = shinkaevolve ] && echo 0.78 || echo 0.88) \
    --enable-prefix-caching --reasoning-parser qwen3

# 2. The run: run.json records its start (scores are counted from here), then the framework runs for `budget` minutes.
#    At the end, `timeout` sends SIGTERM to the framework's whole process group.
export BENCHMARK_PROBLEM=$problem EVAL_LOG=$PWD/$out/eval_log.jsonl
echo "{\"framework\": \"$framework\", \"problem\": \"$problem\", \"model\": \"$model\", \"mode\": \"thinking\", \"budget_minutes\": $budget, \"seed\": $seed, \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\", \"start\": $(date +%s.%N)}" > $out/run.json
case $framework in
  openevolve)   timeout $((budget * 60))s $(python_of $framework) baselines/run_openevolve.py --problem $problem --model $model --port $port --output $out --seed $seed ;;
  shinkaevolve) timeout $((budget * 60))s $(python_of $framework) baselines/run_shinkaevolve.py --problem $problem --model $model --port $port --embed-port $embed_port --output $out --seed $seed ;;
  codeevolve)   timeout $((budget * 60))s $(python_of $framework) baselines/run_codeevolve.py --problem $problem --model $model --port $port --output $out --seed $seed 2>> $out/codeevolve_stderr.log ;;
  *) echo "unknown framework $framework"; exit 1 ;;
esac
status=$?   # 124: the budget ran out (the normal end)
echo "{\"end\": $(date +%s.%N), \"exit\": $status}" > $out/end.json
