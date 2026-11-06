You are an expert in numerical linear algebra. Write a program that computes all eigenvalues of square integer matrices to high relative accuracy.

The program must define eigenvalues(A): A is a square numpy int64 array; it returns all n eigenvalues as a complex array of shape (n,), listed with algebraic multiplicity, in any order. It is called once per matrix, each time in a fresh process, on these six matrices (entries are small integers, no eigenvalue is zero):
1. circulant: a dense circulant matrix, n = 200.
2. tridiagonal: a tridiagonal Toeplitz matrix with diagonal 9, sub-diagonal 1 and super-diagonal 4, n = 200.
3. kronecker: a 2D convection-diffusion stencil, the Kronecker sum of two nonsymmetric tridiagonal Toeplitz matrices (24 x 24 and 22 x 22), n = 528.
4. permuted_tridiagonal: a nonsymmetric tridiagonal matrix with varying entries (sub-diagonal entries much smaller than super-diagonal ones), with its rows and columns permuted by the same random permutation, n = 200.
5. block_triangular: a block upper-triangular matrix with 2 x 2 diagonal blocks, with its rows and columns permuted by the same random permutation, n = 200.
6. similar_dense: a dense matrix similar to a block-diagonal matrix, n = 180.

Several of these matrices are strongly non-normal: LAPACK's eigenvalue routines are backward stable, but non-normality amplifies their rounding errors, so the computed eigenvalues can lose many digits. Eigenvalues do not change under similarity transformations, and the structure of each matrix described above can be exploited to compute them accurately; arbitrary-precision arithmetic (mpmath) is far too slow for matrices of these sizes within the time limit.

The returned eigenvalues are paired one-to-one with the true eigenvalues by the pairing that minimizes the total relative error |returned - true| / |true|; a matrix scores min(10, -log10(largest relative error)) / 10, so 10 correct digits for every eigenvalue give full marks. The combined_score is the mean over the six matrices. The evaluation reports the digits, the error and the run time of every matrix, or why it failed.

EXECUTION ENVIRONMENT:
The program runs with Python 3.11 on a single CPU core, with numpy, scipy, pyamg, mpmath and sympy installed (no numba). Each matrix has a hard time limit of 8 seconds, including the time to start Python and import modules; a matrix that exceeds it, raises an error or returns a malformed result scores zero.
