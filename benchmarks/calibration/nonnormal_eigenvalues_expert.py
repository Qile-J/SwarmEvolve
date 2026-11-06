"""Expert check for nonnormal_eigenvalues (never shown to agents): strongly connected components split a reducible
matrix into its diagonal blocks; a block whose pattern is symmetric with A_ij A_ji > 0 and whose cycle products match
is diagonally similar to the symmetric matrix sqrt(A_ij A_ji), solved with eigvalsh; anything else uses eigvals."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components


def block_eigenvalues(B):
    n = len(B)
    if n <= 2:
        return np.linalg.eigvals(B)
    off = B - np.diag(np.diag(B))
    if np.all((off != 0) == (off.T != 0)) and np.all(off * off.T >= 0):
        # log of the diagonal similarity along a BFS tree, then check every edge for consistency
        order, parent = breadth_first_order(csr_matrix(off != 0), 0, directed=False)
        if len(order) == n:
            logd = np.zeros(n)
            for v in order[1:]:
                p = parent[v]
                logd[v] = logd[p] + 0.5 * (np.log(abs(off[p, v])) - np.log(abs(off[v, p])))
            i, j = np.nonzero(off)
            if np.allclose(logd[j] - logd[i], 0.5 * (np.log(abs(off[i, j])) - np.log(abs(off[j, i]))), atol=1e-9):
                S = np.sign(off) * np.sqrt(np.abs(off * off.T)) + np.diag(np.diag(B))
                return np.linalg.eigvalsh(S) + 0j
    return np.linalg.eigvals(B)


def eigenvalues(A):
    A = A.astype(float)
    count, labels = connected_components(csr_matrix(A != 0), directed=True, connection='strong')
    return np.concatenate([block_eigenvalues(A[np.ix_(labels == c, labels == c)]) for c in range(count)])
