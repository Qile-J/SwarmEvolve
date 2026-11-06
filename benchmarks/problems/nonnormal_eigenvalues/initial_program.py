# EVOLVE-BLOCK-START
"""Eigenvalues of a square integer matrix with LAPACK (numpy.linalg.eigvals)."""
import numpy as np


def eigenvalues(A):
    """A: integer numpy array (n x n). Returns all n eigenvalues (complex array of shape (n,)), with multiplicity, any order."""
    return np.linalg.eigvals(A.astype(float))
# EVOLVE-BLOCK-END
