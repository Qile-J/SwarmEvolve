"""OpenEvolve's evaluator: the shared evaluation, with the error and program output passed on as artifacts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from evaluation import evaluate as shared_evaluate
from openevolve.evaluation_result import EvaluationResult


def evaluate(program_path):
    metrics = shared_evaluate(program_path)
    # Artifacts appear in OpenEvolve's next prompts; metrics must be numbers
    artifacts = {key: metrics[key] for key in ('error', 'output_tail') if metrics.get(key)}
    numbers = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return EvaluationResult(metrics=numbers, artifacts=artifacts)
