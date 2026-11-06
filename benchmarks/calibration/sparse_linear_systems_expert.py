"""Expert check for sparse_linear_systems: smoothed-aggregation AMG as a preconditioner (CG for symmetric matrices with
a positive diagonal, GMRES for nonsymmetric 3D ones), checked by the residual; sparse LU for the rest; one refinement step."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def solve(A, b):
    import pyamg
    A = A.tocsr()
    n = A.shape[0]
    symmetric = abs(A - A.T).max() <= 1e-12 * abs(A).max()
    if n > 100000 or (symmetric and A.diagonal().min() > 0 and n > 70000):
        if symmetric:
            ml = pyamg.smoothed_aggregation_solver(A, max_coarse=500)
            x = ml.solve(b, tol=1e-13, accel='cg', maxiter=300)
        else:
            ml = pyamg.smoothed_aggregation_solver(A, symmetry='nonsymmetric', max_coarse=500)
            x = ml.solve(b, tol=1e-13, accel='gmres', maxiter=200)
        if np.linalg.norm(b - A @ x) <= 1e-10 * np.linalg.norm(b):
            return x
    lu = splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A' if symmetric else 'COLAMD')
    x = lu.solve(b)
    return x + lu.solve(b - A @ x)
