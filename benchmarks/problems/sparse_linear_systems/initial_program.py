# EVOLVE-BLOCK-START
"""Sparse linear solver: Jacobi-preconditioned conjugate gradients."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg


def solve(A, b):
    """Solve A x = b. A: scipy.sparse CSR matrix (n x n), b: array (n,). Returns x, an array of shape (n,)."""
    M = sp.diags(1.0 / A.diagonal())
    x, info = cg(A, b, M=M, rtol=1e-10, maxiter=2000)
    return x
# EVOLVE-BLOCK-END
