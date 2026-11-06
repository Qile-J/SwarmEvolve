You are an expert in numerical linear algebra and scientific computing. Write a fast and accurate solver for sparse linear systems A x = b.

The program must define solve(A, b): A is a square scipy.sparse CSR matrix (float64) and b a float64 array; it returns x, an array of shape (n,). It is called once per system, each time in a fresh process, on these five systems from finite-difference discretizations of PDEs on the unit square or cube with Dirichlet boundary conditions:
1. poisson_2d: 2D Poisson equation, 5-point stencil on a 256 x 256 grid (n = 65536), symmetric positive definite.
2. jump_3d: 3D diffusion -div(c grad u) with a piecewise constant coefficient c that jumps by up to a factor 1e6 between blocks of an 8 x 8 x 8 checkerboard, 7-point stencil on a 64 x 64 x 64 grid (n = 262144), symmetric positive definite and ill-conditioned.
3. helmholtz_2d: 2D Helmholtz operator -Laplacian - k^2 with k = 40.3, 5-point stencil on a 300 x 300 grid (n = 90000), symmetric indefinite.
4. convection_2d: 2D convection-diffusion -eps Laplacian u + w . grad u with a rotating flow w, Peclet number 1/eps = 1000, first-order upwind differences on a 300 x 300 grid (n = 90000), nonsymmetric.
5. convection_3d: 3D convection-diffusion with a rotating flow plus a constant vertical drift, Peclet number 100, first-order upwind differences on a 48 x 48 x 48 grid (n = 110592), nonsymmetric.

Each right-hand side is b = A x* for a hidden random solution x*. A system scores min(8, -log10(||x - x*|| / ||x*||)) / 8: 8 correct digits give full marks for that system. The combined_score is the mean over the five systems. The evaluation reports the digits, the error and the run time of every system, or why it failed.

EXECUTION ENVIRONMENT:
The program runs with Python 3.11 on a single CPU core, with numpy, scipy, pyamg, mpmath and sympy installed (no numba). Each system has a hard time limit of 9 seconds, including the time to start Python and import modules; a system that exceeds it, raises an error or returns a malformed result scores zero.
