# EVOLVE-BLOCK-START
"""2D Helmholtz equation with second-order finite differences (ghost points for the impedance condition)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


def source(k, X, Y):
    a = k**2 / 2
    return np.exp(-a * ((X - 0.3)**2 + (Y - 0.6)**2)) - 0.6 * np.exp(-a * ((X - 0.7)**2 + (Y - 0.25)**2))


def solve(k, x, y):
    """-(u_xx + u_yy) - k^2 u = f on (0,1)^2 with du/dn - i k u = 0 on the boundary.
    Returns the complex solution u at the points (x[i], y[i])."""
    N = 100                                  # grid intervals per direction, nodes include the boundary
    h = 1.0 / N
    m = N + 1
    # 1D second difference with the impedance condition: the ghost value u_{-1} = u_1 + 2 h i k u_0
    T = sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1], format='lil').astype(complex)
    T[0, 1], T[-1, -2] = 2, 2
    T[0, 0] += 2j * k * h
    T[-1, -1] += 2j * k * h
    T = T.tocsr() / h**2
    I = sp.identity(m)
    A = -(sp.kron(T, I) + sp.kron(I, T)) - k**2 * sp.identity(m * m)
    grid = np.linspace(0, 1, m)
    X, Y = np.meshgrid(grid, grid, indexing='ij')
    u = spsolve(A.tocsc(), source(k, X, Y).ravel().astype(complex)).reshape(m, m)
    points = np.column_stack([x, y])
    return (RegularGridInterpolator((grid, grid), u.real)(points)
            + 1j * RegularGridInterpolator((grid, grid), u.imag)(points))
# EVOLVE-BLOCK-END
