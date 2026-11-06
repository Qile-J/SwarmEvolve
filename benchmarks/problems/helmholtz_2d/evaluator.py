"""2D Helmholtz -Laplacian u - k^2 u = f on the unit square with the impedance condition du/dn - i k u = 0 on the whole
boundary, f two Gaussian sources of width ~1/k, for several k. References: Chebyshev collocation (boundary values
eliminated per direction) solved as a Sylvester equation, converged to 1e-8 (benchmarks/calibration/
make_pde_references.py), in reference.npz. Score per case: digits of the relative 2-norm error over 2000 random points,
capped at CAP; combined_score is the mean."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scientific_cases import evaluate_cases

CAP, SECONDS = 6, 9
REFERENCE_CASES = [('k_10', {'k': 10}, (60, 90)), ('k_20', {'k': 20}, (90, 130)), ('k_40', {'k': 40}, (140, 200)),
                   ('k_60', {'k': 60}, (180, 260)), ('k_80', {'k': 80}, (220, 320))]


def points():
    return np.random.default_rng(20261020).uniform(0, 1, (2000, 2))


def cases():
    reference, p = np.load(Path(__file__).with_name('reference.npz')), points()
    return [(name, f"k = {params['k']:g}", {'k': float(params['k']), 'x': p[:, 0], 'y': p[:, 1]}, reference[name])
            for name, params, _ in REFERENCE_CASES]


def check(u, reference, inputs):
    u = np.asarray(u).astype(complex).ravel()
    if u.shape != reference.shape:
        raise ValueError(f'solve returned shape {u.shape}, expected {reference.shape}')
    return np.linalg.norm(u - reference) / np.linalg.norm(reference)


def evaluate(program_path):
    return evaluate_cases(program_path, 'solve', cases(), check, CAP, SECONDS)
