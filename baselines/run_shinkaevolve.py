"""Run ShinkaEvolve on one problem against local vLLM servers, until it is stopped (baselines/run_baseline.sh).

    python baselines/run_shinkaevolve.py --problem circle_packing_26 --model Qwen/Qwen3-8B --port 8000 --embed-port 8001 --output DIR
"""
# The configuration is ShinkaEvolve's examples/circle_packing/shinka_small.yaml: async runner, 2 islands, weighted
# parent selection, diff / full / crossover patches, embedding novelty filter, LLM novelty judge. Ours: every LLM role
# on the one local Qwen3 model, Qwen3-Embedding-0.6B for embeddings, the meta scratchpad every 10 generations,
# 8 proposal and 8 evaluation jobs.

import argparse
import os
from pathlib import Path

import yaml
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

ROOT = Path(__file__).resolve().parents[1]   # the repository root

parser = argparse.ArgumentParser()
parser.add_argument('--problem', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--port', type=int, required=True)         # the vLLM server of the model
parser.add_argument('--embed-port', type=int, required=True)   # the vLLM server of the embedding model
parser.add_argument('--output', type=Path, required=True)      # the run directory
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--workers', type=int, default=8)          # parallel proposals and evaluations
args = parser.parse_args()

problem = ROOT / 'benchmarks' / 'problems' / args.problem
config = yaml.safe_load((ROOT / 'baselines' / 'ShinkaEvolve' / 'examples' / 'circle_packing' / 'shinka_small.yaml').read_text())
llm = f'local/{args.model}@http://localhost:{args.port}/v1'   # ShinkaEvolve's notation for a local OpenAI-style server
config['evo_config'].update({
    'task_sys_msg': (problem / 'task.md').read_text(),
    'num_generations': 100000,   # no generation limit: the wall-clock budget ends the run
    'max_api_costs': 1e9,
    'llm_models': [llm],
    'llm_kwargs': {'temperatures': [0.6, 0.8, 1.0], 'max_tokens': 12000},
    'meta_rec_interval': 10,
    'meta_llm_models': [llm],
    'meta_llm_kwargs': {'temperatures': [0.6], 'max_tokens': 8192},
    'novelty_llm_models': [llm],
    'novelty_llm_kwargs': {'temperatures': [0.6], 'max_tokens': 4096},
    'embedding_model': f'local/Qwen/Qwen3-Embedding-0.6B@http://localhost:{args.embed_port}/v1',
    'llm_dynamic_selection': None,
    'init_program_path': str(problem / 'initial_program.py'),
    'results_dir': str(args.output / 'shinka_output'),
})
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'shinka_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')

runner = ShinkaEvolveRunner(
    evo_config=EvolutionConfig(**config['evo_config']),
    job_config=LocalJobConfig(eval_program_path=str(ROOT / 'baselines' / 'shinka_evaluate.py'), time='00:05:00'),
    db_config=DatabaseConfig(**{k: v for k, v in config['db_config'].items() if k in DatabaseConfig.__dataclass_fields__}),
    max_evaluation_jobs=args.workers, max_proposal_jobs=args.workers, max_db_workers=4, debug=False, verbose=True)
runner.run()
