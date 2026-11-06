"""Run OpenEvolve on one problem against a local vLLM server, until it is stopped (baselines/run_baseline.sh).

    python baselines/run_openevolve.py --problem circle_packing_26 --model Qwen/Qwen3-8B --port 8000 --output DIR
"""
# The configuration is OpenEvolve's own for the AlphaEvolve problems (examples/alphaevolve_math_problems/*/config.yaml):
# population 40, 5 islands, migration every 40 iterations, MAP-Elites on score and complexity, 3 top + 2 diverse
# programs in the prompt, diff-based edits. Ours: the problem statement as the system message, 8 parallel evaluations.

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]   # the repository root

parser = argparse.ArgumentParser()
parser.add_argument('--problem', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--port', type=int, required=True)        # the vLLM server's port
parser.add_argument('--output', type=Path, required=True)     # the run directory
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--workers', type=int, default=8)          # parallel evaluations
parser.add_argument('--full-rewrite', action='store_true', help='full rewrites instead of diffs')
args = parser.parse_args()

problem = ROOT / 'benchmarks' / 'problems' / args.problem
config = {
    'max_iterations': 100000,   # no iteration limit: the wall-clock budget ends the run
    'checkpoint_interval': 10,
    'random_seed': None,   # with a seed, OpenEvolve sends the same seed with every request: parallel workers would sample
                           # identical programs
    'log_level': 'INFO',
    'diff_based_evolution': not args.full_rewrite,
    'max_code_length': 20000,
    'llm': {'api_base': f'http://localhost:{args.port}/v1', 'api_key': 'EMPTY',
            'models': [{'name': args.model, 'weight': 1.0}],
            'temperature': 0.7, 'top_p': 0.95, 'max_tokens': 12000, 'timeout': 1200, 'retries': 2},
    'prompt': {'system_message': (problem / 'task.md').read_text(), 'num_top_programs': 3, 'num_diverse_programs': 2},
    'database': {'population_size': 40, 'num_islands': 5, 'migration_interval': 40,
                 'feature_dimensions': ['score', 'complexity']},
    'evaluator': {'timeout': 150, 'max_retries': 1, 'cascade_evaluation': False, 'parallel_evaluations': args.workers},
}
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'openevolve_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

# OpenEvolve's own entry point (baselines/openevolve_entry.py adds a shutdown fix), with our evaluator
subprocess.run([sys.executable, str(ROOT / 'baselines' / 'openevolve_entry.py'), str(problem / 'initial_program.py'),
                str(ROOT / 'baselines' / 'openevolve_evaluator.py'),
                '--config', str(args.output / 'openevolve_config.yaml'),
                '--output', str(args.output / 'openevolve_output')], check=True)
