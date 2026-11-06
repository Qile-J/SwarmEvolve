"""Sparse linear systems A x = b from discretized PDEs and a graph, each with a planted random solution x*.
Score per system: digits of the relative error ||x - x*|| / ||x*||, capped at CAP; combined_score is the mean."""
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scientific_cases import evaluate_cases

CAP, SECONDS = 8, 9


def second_difference(n):
    h = 1 / (n + 1)
    return sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]) / h**2


def poisson(n, dim):
    T, I = second_difference(n), sp.identity(n)
    if dim == 2:
        return sp.kron(T, I) + sp.kron(I, T)
    return sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)


def jump(n, dim, contrast, blocks, rng):
    # -div(c grad u), c constant on a random blocks^dim checkerboard with values 10^U(0, log10 contrast); harmonic-mean faces
    cells = 10 ** rng.uniform(0, np.log10(contrast), (blocks,) * dim)
    index = np.arange(n) * blocks // n
    c = cells[np.ix_(*(index,) * dim)]
    h = 1 / (n + 1)
    padded = np.pad(c, 1, mode='edge')
    ids = np.arange(n**dim).reshape((n,) * dim)
    rows, cols, vals, diagonal = [], [], [], np.zeros((n,) * dim)
    for axis in range(dim):
        for step in (-1, 1):
            neighbor = np.take(padded, np.arange(n) + 1 + step, axis=axis)
            neighbor = neighbor[tuple(slice(1, -1) if a != axis else slice(None) for a in range(dim))]
            face = 2 * c * neighbor / (c + neighbor) / h**2
            diagonal += face
            valid = np.ones((n,) * dim, dtype=bool)
            edge = [slice(None)] * dim
            edge[axis] = -1 if step == 1 else 0
            valid[tuple(edge)] = False
            rows.append(ids[valid]); cols.append(np.roll(ids, -step, axis=axis)[valid]); vals.append(-face[valid])
    rows.append(ids.ravel()); cols.append(ids.ravel()); vals.append(diagonal.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n**dim,) * 2)


def convection_diffusion(n, dim, peclet):
    # -eps Laplacian + w . grad u, first-order upwind, w a rotating flow (plus a constant vertical drift in 3D): nonsymmetric
    eps, h = 1 / peclet, 1 / (n + 1)
    s = (np.arange(n) + 1) * h
    grids = np.meshgrid(*(s,) * dim, indexing='ij')
    w = [2 * grids[1] - 1, 1 - 2 * grids[0]] + ([0.5 * np.ones_like(grids[0])] if dim == 3 else [])
    I = sp.identity(n)
    forward = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1]) / h
    backward = sp.diags([np.ones(n), -np.ones(n - 1)], [0, -1]) / h

    def along(D, axis):
        factors = [I] * dim
        factors[axis] = D
        out = factors[0]
        for F in factors[1:]:
            out = sp.kron(out, F)
        return out
    A = eps * poisson(n, dim)
    for axis, wa in enumerate(w):
        wa = wa.ravel()
        A = A + sp.diags(np.maximum(wa, 0)) @ along(backward, axis) + sp.diags(np.minimum(wa, 0)) @ along(forward, axis)
    return A


def cases():
    rng = np.random.default_rng(20261020)
    systems = [
        ('poisson_2d', '2D Poisson, 256 x 256 grid', poisson(256, 2)),
        ('jump_3d', '3D diffusion, coefficient jumps up to 1e6, 64^3 grid', jump(64, 3, 1e6, 8, rng)),
        ('helmholtz_2d', '2D Helmholtz, k = 40.3, 300 x 300 grid', poisson(300, 2) - 40.3**2 * sp.identity(300**2)),
        ('convection_2d', '2D convection-diffusion, Peclet 1000, 300 x 300 grid', convection_diffusion(300, 2, 1000)),
        ('convection_3d', '3D convection-diffusion, Peclet 100, 48^3 grid', convection_diffusion(48, 3, 100)),
    ]
    out = []
    for name, label, A in systems:
        A = sp.csr_matrix(A)
        A.sort_indices()
        x = rng.standard_normal(A.shape[0])
        out.append((name, f'{label}, n = {A.shape[0]}', {'A': A, 'b': A @ x}, x))
    return out


def check(x, reference, inputs):
    x = np.asarray(x, dtype=float).ravel() if not np.iscomplexobj(x) else np.asarray(x).ravel()
    if x.shape != reference.shape:
        raise ValueError(f'solve returned shape {x.shape}, expected {reference.shape}')
    return np.linalg.norm(x - reference) / np.linalg.norm(reference)


def evaluate(program_path):
    return evaluate_cases(program_path, 'solve', cases(), check, CAP, SECONDS)
