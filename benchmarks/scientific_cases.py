"""Shared evaluation of the scientific-computing problems: one child process per test case, scored by digits."""
# The program's entry function runs once per test case, in a fresh child process that receives only the case's inputs
# (never the reference answer) and has its own time limit, so a slow or crashing case costs only that case.
# Each case scores its digits of accuracy, min(cap, -log10 relative error) / cap; combined_score is the mean over cases.
# The per-case report (digits, error, seconds, or why it failed) is printed for the frameworks; reference values never are.

import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

# The child process: load the program, call its entry function with the case's inputs, save the output array
CHILD = '''import importlib.util, pickle, sys
import numpy as np
spec = importlib.util.spec_from_file_location('program', sys.argv[1])
program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)
with open(sys.argv[3], 'rb') as f:
    inputs = pickle.load(f)
out = np.asarray(getattr(program, sys.argv[2])(**inputs))
if out.dtype == object:
    raise TypeError(f'{sys.argv[2]} must return a numeric array, got an object array of shape {out.shape}')
np.save(sys.argv[4], out, allow_pickle=False)
'''


def run_case(program_path, entry, inputs, seconds):
    # Run entry(**inputs) of the program in a child process with a time limit.
    # Returns (output array or None, error message, seconds).
    with tempfile.TemporaryDirectory() as work:
        Path(work, 'child.py').write_text(CHILD)
        with open(Path(work, 'inputs.pkl'), 'wb') as f:
            pickle.dump(inputs, f)

        begin = time.time()
        process = subprocess.Popen(
            [sys.executable, 'child.py', str(Path(program_path).resolve()), entry, 'inputs.pkl', 'out.npy'],
            cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return None, f'time limit of {seconds:g} s exceeded', time.time() - begin
        elapsed = time.time() - begin

        # A crash: report the last line the child printed
        if process.returncode != 0 or not Path(work, 'out.npy').exists():
            lines = [line for line in output.strip().splitlines() if line.strip()]
            return None, (lines[-1] if lines else f'exit code {process.returncode}')[:300], elapsed
        return np.load(Path(work, 'out.npy'), allow_pickle=False), '', elapsed


def digits(error, cap):
    # Correct digits of a relative error, between 0 and cap
    return float(min(cap, max(0.0, -np.log10(max(error, 1e-300)))))


def evaluate_cases(program_path, entry, cases, check, cap, seconds):
    # cases: [(name, label, inputs, reference)]; check(output, reference, inputs) returns the relative error, or raises
    # ValueError with a message for a malformed output. Prints the per-case report; returns the metrics.
    metrics, scores, failures, slowest = {}, [], [], 0.0
    for name, label, inputs, reference in cases:
        out, error, elapsed = run_case(program_path, entry, inputs, seconds)
        slowest = max(slowest, elapsed)

        if out is not None:
            try:
                relative = check(out, reference, inputs)
                if not np.isfinite(relative):
                    raise ValueError('the output contains NaN or infinite values')
            except (ValueError, TypeError) as e:
                out, error = None, str(e)[:300]

        if out is None:
            scores.append(0.0)
            failures.append(f'{name}: {error}')
            print(f'case {name} ({label}): 0 of {cap} digits: {error} ({elapsed:.1f} s)')
        else:
            d = digits(relative, cap)
            scores.append(d / cap)
            print(f'case {name} ({label}): {d:.2f} of {cap} digits (relative error {relative:.1e}) in {elapsed:.1f} s')
        metrics[f'digits_{name}'] = round(scores[-1] * cap, 3)

    metrics['combined_score'] = float(np.mean(scores))
    # The slowest case against the limit per case (frameworks may use these to judge a program's speed)
    metrics['run_seconds'], metrics['time_limit'] = round(slowest, 2), seconds

    # Score 0: an error message, with identical errors of several cases reported once
    if metrics['combined_score'] == 0:
        grouped = {}
        for failure in failures:
            name, message = failure.split(': ', 1)
            grouped.setdefault(message, []).append(name)
        metrics['error'] = '; '.join(
            (f'all {len(cases)} cases' if len(names) == len(cases) else ', '.join(names)) + f': {message}'
            for message, names in grouped.items())[:600] or 'no case reached any digit of accuracy'
    return metrics
