"""CodeEvolve's evaluate script: python evaluate.py <program> <results file> (its interface)."""
# Runs the shared evaluation and writes the numeric metrics as CodeEvolve's results JSON.
#   - A failed program (an error and score 0) exits with code 1; the error and the program's output go to stderr,
#     which CodeEvolve shows to its LLM.
#   - An error next to a positive score also goes to stderr (CodeEvolve shows it as a warning) and the metrics count.
import json
import os
import sys

sys.path.insert(0, os.environ['BENCHMARKS_DIR'])
from evaluation import evaluate

metrics = evaluate(sys.argv[1])
numbers = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
with open(sys.argv[2], 'w') as f:
    json.dump(numbers, f, indent=1)

error = metrics.get('error')
feedback = '\n'.join(part for part in (error and f'Error: {error}',
                                       metrics.get('output_tail') and f"Program output (tail):\n{metrics['output_tail']}")
                     if part)
if feedback:
    print(feedback, file=sys.stderr)
sys.exit(1 if error and not numbers.get('combined_score') else 0)
