"""The evaluation every framework uses (SwarmEvolve, OpenEvolve, ShinkaEvolve, CodeEvolve)."""
# evaluate(program_path) runs the problem's evaluator on a program and returns its metrics; combined_score is the score
# (0 for an invalid program). The evaluator runs in a fresh process with single-threaded math libraries and a 60 s limit.
#
# With $EVAL_LOG set, every evaluation appends one JSON line (time, score, error) to that file and saves the program
# under <log folder>/programs/<sha>.py. All frameworks' scores are read from these logs.
# The problem is $BENCHMARK_PROBLEM (a folder of benchmarks/problems) unless it is passed explicitly.

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent

# One interpreter (with numpy, scipy, jax, ...) runs every framework's programs
PYTHON = Path(os.environ.get('EVAL_PYTHON', sys.executable))

TIME_LIMIT = 60             # seconds per evaluation
MARKER = '@@METRICS@@'      # run_evaluator.py prints the metrics after this marker

# Every evaluation gets one CPU core: no multithreaded BLAS / OpenMP / XLA, no GPU
SINGLE_THREAD = {**{name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                 'JAX_PLATFORMS': 'cpu', 'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',
                 'PYTHONNOUSERSITE': '1', 'CUDA_VISIBLE_DEVICES': ''}


def evaluate(program_path, problem=None, log=None):
    problem = problem or os.environ['BENCHMARK_PROBLEM']
    log = log or os.environ.get('EVAL_LOG')
    program_path = Path(program_path).resolve()
    code = program_path.read_text()
    begin = time.time()
    output = ''

    # Run the evaluator in a child process (its own process group, so a timeout can kill everything it started)
    with tempfile.TemporaryDirectory() as work:
        process = subprocess.Popen(
            [str(PYTHON), str(HERE / 'run_evaluator.py'), str(HERE / 'problems' / problem), str(program_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=work,
            env={**os.environ, **SINGLE_THREAD}, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=TIME_LIMIT)
            lines = output.splitlines()
            found = [line for line in lines if line.startswith(MARKER)]
            # No metrics line: the evaluator crashed; its output tail becomes the error
            metrics = json.loads(found[-1][len(MARKER):]) if found else \
                {'combined_score': 0.0, 'error': 'evaluator crashed:\n' + output[-1500:]}
            output = '\n'.join(line for line in lines if not line.startswith(MARKER))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            metrics = {'combined_score': 0.0, 'error': f'time limit of {TIME_LIMIT} s exceeded'}

    # A missing or NaN score counts as 0
    score = metrics.get('combined_score', 0.0)
    if not isinstance(score, (int, float)) or score != score:
        metrics['combined_score'] = score = 0.0
    metrics['eval_seconds'] = round(time.time() - begin, 2)

    # The shared log: one line per evaluation, and the program's code by its hash
    if log:
        sha = hashlib.sha1(code.encode()).hexdigest()[:16]
        folder = Path(log).parent / 'programs'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{sha}.py').write_text(code)
        record = {'time': time.time(), 'score': score, 'sha': sha, 'error': str(metrics.get('error', ''))[:300],
                  'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        with open(log, 'a') as f:
            f.write(json.dumps(record) + '\n')

    metrics['output_tail'] = output[-1500:]   # the last lines the program printed (feedback for the frameworks)
    return metrics


if __name__ == '__main__':   # python benchmarks/evaluation.py <problem> <program>
    print(json.dumps(evaluate(sys.argv[2], sys.argv[1]), indent=1))
