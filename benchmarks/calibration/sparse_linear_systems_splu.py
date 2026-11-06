"""Calibration: sparse LU (scipy spsolve) for every system."""
from scipy.sparse.linalg import spsolve


def solve(A, b):
    return spsolve(A.tocsc(), b)
