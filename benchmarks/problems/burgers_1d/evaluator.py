"""1D viscous Burgers u_t + u u_x = nu u_xx on [-1, 1], periodic, u(x, 0) = -sin(pi x), at time T = 1, for several nu.
The reference is the exact Cole-Hopf solution, u = -int sin(pi(x-y)) F(x-y) G(y) dy / int F(x-y) G(y) dy with
F(y) = exp(-cos(pi y) / (2 pi nu)) and G(y) = exp(-y^2 / (4 nu T)), integrated with the trapezoidal rule in log scale
(checked against 4x the nodes and against mpmath). Score per case: digits of the relative 2-norm error over 1000 random
points, capped at CAP; combined_score is the mean."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scientific_cases import evaluate_cases

CAP, SECONDS, T = 6, 9, 1.0
VISCOSITIES = (0.1, 0.03, 0.01, 0.003, 0.001)


def cole_hopf(x, nu, t, nodes=8001):
    z = np.linspace(-14, 14, nodes)   # y = sqrt(4 nu t) z, so G = exp(-z^2)
    y = x[:, None] - np.sqrt(4 * nu * t) * z[None, :]
    exponent = -np.cos(np.pi * y) / (2 * np.pi * nu) - z[None, :] ** 2
    weight = np.exp(exponent - exponent.max(axis=1, keepdims=True))
    return -(np.sin(np.pi * y) * weight).sum(axis=1) / weight.sum(axis=1)


def cases():
    x = np.sort(np.random.default_rng(20261020).uniform(-1, 1, 1000))
    return [(f'nu_{nu:g}', f'nu = {nu:g}', {'nu': nu, 'T': T, 'x': x}, cole_hopf(x, nu, T)) for nu in VISCOSITIES]


def check(u, reference, inputs):
    u = np.asarray(u, dtype=float).ravel()
    if u.shape != reference.shape:
        raise ValueError(f'solve returned shape {u.shape}, expected {reference.shape}')
    return np.linalg.norm(u - reference) / np.linalg.norm(reference)


def evaluate(program_path):
    return evaluate_cases(program_path, 'solve', cases(), check, CAP, SECONDS)
