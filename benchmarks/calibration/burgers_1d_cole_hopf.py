"""Expert check for burgers_1d: the exact Cole-Hopf solution by trapezoidal quadrature in log scale."""
import numpy as np


def solve(nu, T, x):
    z = np.linspace(-14, 14, 8001)
    y = np.asarray(x)[:, None] - np.sqrt(4 * nu * T) * z[None, :]
    e = -np.cos(np.pi * y) / (2 * np.pi * nu) - z[None, :] ** 2
    w = np.exp(e - e.max(axis=1, keepdims=True))
    return -(np.sin(np.pi * y) * w).sum(axis=1) / w.sum(axis=1)
