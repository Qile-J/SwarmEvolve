"""ShinkaEvolve's evaluate script: python shinka_evaluate.py --program_path P --results_dir D (its interface)."""
# Runs the shared evaluation and writes ShinkaEvolve's metrics.json and correct.json; the error and the program's
# output go into its text feedback.
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from evaluation import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--program_path', required=True)
parser.add_argument('--results_dir', required=True)
args = parser.parse_args()
os.makedirs(args.results_dir, exist_ok=True)

metrics = evaluate(args.program_path)
error = metrics.get('error') or None
numbers = {k: v for k, v in metrics.items()
           if isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'combined_score'}
feedback = '\n'.join(part for part in (error and f'Error: {error}',
                                       metrics.get('output_tail') and f"Program output (tail):\n{metrics['output_tail']}")
                     if part)

result = {'combined_score': float(metrics['combined_score']), 'public': numbers, 'private': {}, 'text_feedback': feedback}
correct = error is None and metrics['combined_score'] > 0
Path(args.results_dir, 'metrics.json').write_text(json.dumps(result, indent=1))
Path(args.results_dir, 'correct.json').write_text(json.dumps({'correct': correct, 'error': error}, indent=1))
print(json.dumps({'combined_score': result['combined_score'], 'correct': correct}))
