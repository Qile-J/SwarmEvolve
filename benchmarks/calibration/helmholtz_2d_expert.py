"""Expert check (Chebyshev collocation + Sylvester), never shown to agents."""
import numpy as np
from scipy.linalg import solve_sylvester


def chebyshev(N):
    """Points x_j = (1 - cos(pi j / N)) / 2 on [0, 1] (ascending) and the first-derivative matrix."""
    t = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    D = np.outer(c, 1 / c) / (t[:, None] - t[None, :] + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return (1 - t) / 2, -2 * D


def interpolation_matrix(nodes, points):
    """Barycentric Lagrange interpolation matrix for Chebyshev points (weights (-1)^j, halved at the ends)."""
    w = (-1.0) ** np.arange(len(nodes))
    w[0] /= 2
    w[-1] /= 2
    diff = points[:, None] - nodes[None, :]
    exact = np.isclose(diff, 0, atol=1e-15)
    diff[exact] = 1
    M = w / diff
    M /= M.sum(axis=1, keepdims=True)
    rows = exact.any(axis=1)
    M[rows] = exact[rows]
    return M


def convection_diffusion(eps, bx, by, N, points):
    """-eps Laplacian u + (bx, by) . grad u = 1 on (0,1)^2, u = 0 on the boundary."""
    x, D = chebyshev(N)
    D2 = D @ D
    I = slice(1, N)
    Ax = (-eps * D2 + bx * D)[I, I]
    Ay = (-eps * D2 + by * D)[I, I]
    U = solve_sylvester(Ax, Ay.T, np.ones((N - 1, N - 1)))
    full = np.zeros((N + 1, N + 1))
    full[I, I] = U
    Lx, Ly = interpolation_matrix(x, points[:, 0]), interpolation_matrix(x, points[:, 1])
    return np.einsum('pi,ij,pj->p', Lx, full, Ly)


def helmholtz_source(k, X, Y):
    a = k**2 / 2
    return np.exp(-a * ((X - 0.3) ** 2 + (Y - 0.6) ** 2)) - 0.6 * np.exp(-a * ((X - 0.7) ** 2 + (Y - 0.25) ** 2))


def helmholtz(k, N, points):
    """-Laplacian u - k^2 u = f on (0,1)^2, impedance du/dn - i k u = 0 on the whole boundary."""
    x, D = chebyshev(N)
    D2 = D @ D
    I, B = np.arange(1, N), np.array([0, N])
    # boundary values of a line from its interior values: u_x + i k u = 0 at x = 0, u_x - i k u = 0 at x = 1
    M = D[np.ix_(B, B)] + np.diag([1j * k, -1j * k])
    E = -np.linalg.solve(M, D[np.ix_(B, I)])          # (2, N-1): [u_0, u_N] = E u_interior
    A = D2[np.ix_(I, I)] + D2[np.ix_(I, B)] @ E       # reduced second derivative on interior points
    X, Y = np.meshgrid(x[I], x[I], indexing='ij')
    shift = -A - k**2 / 2 * np.eye(N - 1)
    U = solve_sylvester(shift, shift.T, helmholtz_source(k, X, Y))
    full = np.zeros((N + 1, N + 1), dtype=complex)
    full[np.ix_(I, I)] = U
    full[np.ix_(B, I)] = E @ U                         # x-boundaries (rows 0, N) at interior y
    full[np.ix_(I, B)] = (E @ U.T).T                   # y-boundaries at interior x
    full[np.ix_(B, B)] = E @ full[np.ix_(I, B)]        # corners from the x-condition on the y-boundary rows
    Lx, Ly = interpolation_matrix(x, points[:, 0]), interpolation_matrix(x, points[:, 1])
    return np.einsum('pi,ij,pj->p', Lx, full, Ly)


def solve(k, x, y):
    return helmholtz(k, int(40 + 2.2 * k), np.column_stack([x, y]))
