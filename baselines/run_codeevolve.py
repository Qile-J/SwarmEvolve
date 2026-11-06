"""Run CodeEvolve on one problem against a local vLLM server, until it is stopped (baselines/run_baseline.sh).

    python baselines/run_codeevolve.py --problem circle_packing_26 --model Qwen/Qwen3-8B --port 8000 --output DIR
"""
# The configuration is CodeEvolve's own for open-weight Qwen models (CodeEvolve/configs/templates/config_qwen.yaml):
# initial population 10, exploration rate 0.2 with its plateau scheduler, rank-roulette selection, ring migration every
# 25 epochs, meta-prompting, MAP-Elites on evaluation time, 2 inspirations, chat depth 3, temperatures 0.9 (exploration)
# and 0.7 (exploitation).
# Ours: every LLM role on the one local Qwen3 model (up to 12000 tokens, as for OpenEvolve and ShinkaEvolve), the
# problem statement as the evolvable system prompt, and the shared evaluation (baselines/codeevolve_evaluate.py).
# Changed for this setup:
#   - 8 islands, i.e. 8 parallel evaluations as for the other frameworks (the template has 10)
#   - no memory limit (the template has 1 GB; the other frameworks have none)
#   - no epoch limit or early stopping: the wall-clock budget ends the run
#   - a checkpoint after every epoch (the template: every 10), as in the paper's runs; checkpoints do not change the search

import argparse
import os
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]   # the repository root

parser = argparse.ArgumentParser()
parser.add_argument('--problem', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--port', type=int, required=True)        # the vLLM server's port
parser.add_argument('--output', type=Path, required=True)     # the run directory
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--workers', type=int, default=8)          # islands (parallel evaluations)
args = parser.parse_args()

problem = ROOT / 'benchmarks' / 'problems' / args.problem
config = yaml.safe_load((ROOT / 'baselines' / 'CodeEvolve' / 'configs' / 'templates' / 'config_qwen.yaml').read_text())
llm = {'model_name': args.model, 'top_p': 0.95, 'retries': 3, 'weight': 1, 'max_tok': 12000}
config.update({
    'SEED': args.seed,
    # the system prompt CodeEvolve evolves (meta-prompting changes the part between the markers)
    'SYS_MSG': '# PROMPT-BLOCK-START\n' + (problem / 'task.md').read_text().strip() + '\n# PROMPT-BLOCK-END\n',
    # the shared evaluation stops programs at 60 s; CodeEvolve's own limit is a little longer
    'BUDGET_CONFIG': {'eval_timeout': 65, 'max_mem_bytes': None, 'resource_check_interval_s': 0.1},
    'EXPLORATION_ENSEMBLE': [{**llm, 'temp': 0.9}],
    'EXPLOITATION_ENSEMBLE': [{**llm, 'temp': 0.7}],
    'SAMPLER_AUX_LM': {**llm, 'temp': 0.7},
    'MAP_ELITES': {'elite_map_type': 'grid',
                   'features': [{'name': 'eval_seconds', 'min_val': 0, 'max_val': 60, 'num_bins': 20}]},
})
config['EVOLVE_CONFIG'].update({'fitness_key': 'combined_score', 'num_epochs': 100000, 'early_stopping_rounds': 100000,
                                'num_islands': args.workers, 'ckpt': 1})

# CodeEvolve copies its input folder for every evaluation: only the initial program and the evaluate script go there
inputs = args.output / 'codeevolve_input'
(inputs / config['CODEBASE_PATH']).mkdir(parents=True, exist_ok=True)
shutil.copy(problem / 'initial_program.py', inputs / config['CODEBASE_PATH'] / config['INIT_FILE_DATA']['filename'])
shutil.copy(ROOT / 'baselines' / 'codeevolve_evaluate.py', inputs / config['EVAL_FILE_NAME'])
(args.output / 'codeevolve_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
os.environ.update(API_BASE=f'http://localhost:{args.port}/v1', API_KEY='EMPTY', BENCHMARKS_DIR=str(ROOT / 'benchmarks'))

# Replace this process with CodeEvolve's entry point (the budget's timeout then signals CodeEvolve directly).
# Its terminal dashboard redraws the whole screen twice a second, so stdout goes to /dev/null; every island keeps its
# own log under codeevolve_output/island_*/logs.
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
os.execv(sys.executable, [sys.executable, '-m', 'codeevolve.cli', '--inpt_dir', str(inputs),
                          '--cfg_path', str(args.output / 'codeevolve_config.yaml'),
                          '--out_dir', str(args.output / 'codeevolve_output'), '--load_ckpt', '0', '--y'])
