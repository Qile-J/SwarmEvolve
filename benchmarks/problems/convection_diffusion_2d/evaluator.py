"""2D convection-diffusion -eps Laplacian u + b . grad u = 1 on the unit square, u = 0 on the boundary, b = (1, 0.5),
for several eps (boundary layers of width ~eps at the outflow sides x = 1 and y = 1). References: Chebyshev collocation
solved as a Sylvester equation, converged to 1e-8 (benchmarks/calibration/make_pde_references.py), in reference.npz.
Score per case: digits of the relative 2-norm error over 2000 random points, capped at CAP; combined_score is the mean."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scientific_cases import evaluate_cases

CAP, SECONDS, FLOW = 6, 9, (1.0, 0.5)
REFERENCE_CASES = [('eps_0.1', {'eps': 0.1}, (60, 90)), ('eps_0.01', {'eps': 0.01}, (160, 220)),
                   ('eps_0.003', {'eps': 0.003}, (300, 400)), ('eps_0.001', {'eps': 0.001}, (500, 700)),
                   ('eps_0.0003', {'eps': 0.0003}, (900, 1200))]


def points():
    return np.random.default_rng(20261020).uniform(0, 1, (2000, 2))


def cases():
    reference, p = np.load(Path(__file__).with_name('reference.npz')), points()
    return [(name, f"eps = {params['eps']:g}", {'eps': params['eps'], 'b': FLOW, 'x': p[:, 0], 'y': p[:, 1]}, reference[name])
            for name, params, _ in REFERENCE_CASES]


def check(u, reference, inputs):
    u = np.asarray(u, dtype=float).ravel()
    if u.shape != reference.shape:
        raise ValueError(f'solve returned shape {u.shape}, expected {reference.shape}')
    return np.linalg.norm(u - reference) / np.linalg.norm(reference)


def evaluate(program_path):
    return evaluate_cases(program_path, 'solve', cases(), check, CAP, SECONDS)
