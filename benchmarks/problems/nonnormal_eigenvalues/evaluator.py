"""All eigenvalues of integer matrices, several of them strongly non-normal, against exactly known eigenvalues.
Score per matrix: digits of the largest relative error after pairing returned and true eigenvalues one-to-one (minimum
total relative error), capped at CAP; combined_score is the mean."""
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scientific_cases import evaluate_cases

CAP, SECONDS = 10, 8


def toeplitz_eigenvalues(m, a, sub, sup):
    return a + 2 * np.sqrt(sub * sup + 0j) * np.cos(np.arange(1, m + 1) * np.pi / (m + 1))


def tridiagonal(m, a, sub, sup):
    return a * np.eye(m, dtype=np.int64) + np.diag(np.full(m - 1, sub), -1) + np.diag(np.full(m - 1, sup), 1)


def two_by_two(rng):
    # integer block whose eigenvalues are irrational (real pair or complex pair)
    while True:
        p, q, r, s = (int(v) for v in rng.integers(-30, 31, 4))
        disc = (p - s) ** 2 + 4 * q * r
        if q * r != 0 and disc != 0 and int(round(abs(disc) ** 0.5)) ** 2 != abs(disc):
            root = np.sqrt(complex(disc))
            return np.array([[p, q], [r, s]]), np.array([(p + s + root) / 2, (p + s - root) / 2])


def cases():
    rng = np.random.default_rng(20261020)
    out = []
    # 1. dense circulant (normal): eigenvalues are the DFT of its first row
    c = rng.integers(-9, 10, 200)
    c[0] = np.abs(c[1:]).sum() // 4 + 50
    out.append(('circulant', 'dense circulant, n = 200', np.array([np.roll(c, j) for j in range(200)]), np.fft.fft(c)))
    # 2. tridiagonal Toeplitz with sub-diagonal 1 and super-diagonal 4
    out.append(('tridiagonal', 'tridiagonal Toeplitz, diagonal 9, sub-diagonal 1, super-diagonal 4, n = 200',
                tridiagonal(200, 9, 1, 4), toeplitz_eigenvalues(200, 9, 1, 4)))
    # 3. Kronecker sum of two nonsymmetric tridiagonal Toeplitz matrices (an upwinded 2D convection-diffusion operator)
    T1, T2 = tridiagonal(24, 9, -1, -9), tridiagonal(22, 9, -9, -1)
    A = np.kron(T1, np.eye(22, dtype=np.int64)) + np.kron(np.eye(24, dtype=np.int64), T2)
    true = (toeplitz_eigenvalues(24, 9, -1, -9)[:, None] + toeplitz_eigenvalues(22, 9, -9, -1)[None, :]).ravel()
    out.append(('kronecker', '2D convection-diffusion stencil on a 24 x 22 grid, n = 528', A, true))
    # 4. tridiagonal with random positive off-diagonals of very different sizes below and above, symmetrically permuted
    n = 200
    d, sub, sup = rng.integers(-12, 13, n), rng.integers(1, 3, n - 1), rng.integers(4, 9, n - 1)
    T = np.diag(d) + np.diag(sub, -1) + np.diag(sup, 1)
    true = eigvalsh_tridiagonal(d.astype(float), np.sqrt(sub * sup.astype(float))) + 0j
    order = rng.permutation(n)
    out.append(('permuted_tridiagonal', 'symmetrically permuted nonsymmetric tridiagonal, n = 200', T[np.ix_(order, order)], true))
    # 5. block upper-triangular with 2x2 diagonal blocks and random coupling, then a symmetric permutation
    n, T, true = 200, np.zeros((200, 200), dtype=np.int64), []
    for i in range(0, n, 2):
        B, ev = two_by_two(rng)
        T[i:i + 2, i:i + 2] = B
        true += list(ev)
    T += np.triu(rng.integers(-20, 21, (n, n)) * (rng.random((n, n)) < 0.15), 2) * (T == 0)
    order = rng.permutation(n)
    out.append(('block_triangular', 'permuted block upper-triangular, n = 200', T[np.ix_(order, order)], np.array(true)))
    # 6. dense matrix similar to a block-diagonal one through a well-conditioned unimodular integer matrix (control)
    n, B, true = 180, np.zeros((180, 180), dtype=np.int64), []
    for i in range(0, n, 2):
        block, ev = two_by_two(rng)
        B[i:i + 2, i:i + 2] = block
        true += list(ev)
    U, Uinv = np.eye(n, dtype=np.int64), np.eye(n, dtype=np.int64)
    for _ in range(3 * n):   # U <- U (I + s e_i e_j^T), s = +-1: U stays unimodular and its inverse is tracked exactly
        i, j = rng.choice(n, 2, replace=False)
        s = int(rng.choice([-1, 1]))
        U[:, j] += s * U[:, i]
        Uinv[i, :] -= s * Uinv[j, :]
    out.append(('similar_dense', 'dense, similar to a block-diagonal matrix, n = 180', U @ B @ Uinv, np.array(true)))
    return [(name, label, {'A': A.astype(np.int64)}, true) for name, label, A, true in out]


def check(values, true, inputs):
    values = np.asarray(values).astype(complex).ravel()
    if values.shape != true.shape:
        raise ValueError(f'eigenvalues returned shape {values.shape}, expected {true.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('the output contains NaN or infinite values')
    cost = np.abs(values[:, None] - true[None, :]) / np.abs(true)[None, :]
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols].max()


def evaluate(program_path):
    return evaluate_cases(program_path, 'eigenvalues', cases(), check, CAP, SECONDS)
