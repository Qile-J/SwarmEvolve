# EVOLVE-BLOCK-START
"""2D convection-diffusion with central finite differences on a uniform grid and a sparse direct solve."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


def solve(eps, b, x, y):
    """-eps (u_xx + u_yy) + b[0] u_x + b[1] u_y = 1 on (0,1)^2, u = 0 on the boundary.
    Returns u at the points (x[i], y[i]) (arrays of the same shape)."""
    N = 64                                   # grid intervals per direction
    h = 1.0 / N
    n = N - 1                                # interior points per direction
    I = sp.identity(n)
    D2 = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]) / h**2
    D1 = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [-1, 1]) / (2 * h)
    A = -eps * (sp.kron(D2, I) + sp.kron(I, D2)) + b[0] * sp.kron(D1, I) + b[1] * sp.kron(I, D1)
    u = spsolve(A.tocsc(), np.ones(n * n)).reshape(n, n)
    grid = np.linspace(0, 1, N + 1)
    full = np.zeros((N + 1, N + 1))
    full[1:-1, 1:-1] = u
    return RegularGridInterpolator((grid, grid), full)(np.column_stack([x, y]))
# EVOLVE-BLOCK-END
