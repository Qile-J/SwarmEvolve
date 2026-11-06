You are an expert in numerical methods for partial differential equations. Write a fast and accurate solver for the 1D viscous Burgers equation

    u_t + u u_x = nu u_xx,   x in [-1, 1] with periodic boundary conditions,   u(x, 0) = -sin(pi x),

up to the final time T = 1. As nu decreases, a steep viscous shock of width proportional to nu forms at x = 0 (from t = 1/pi on).

The program must define solve(nu, T, x): nu and T are floats and x is a float array of 1000 points in [-1, 1]; it returns the solution u(x, T) at those points, an array of shape (1000,). It is called once per case, each time in a fresh process, for five cases: nu = 0.1, 0.03, 0.01, 0.003 and 0.001, each with T = 1.

The answer is compared with a high-accuracy reference solution. A case scores min(6, -log10(||u - u_ref|| / ||u_ref||)) / 6 (2-norm over the 1000 points), so 6 correct digits give full marks for that case. The combined_score is the mean over the five cases. The evaluation reports the digits, the error and the run time of every case, or why it failed.

EXECUTION ENVIRONMENT:
The program runs with Python 3.11 on a single CPU core, with numpy, scipy, pyamg, mpmath and sympy installed (no numba). Each case has a hard time limit of 9 seconds, including the time to start Python and import modules; a case that exceeds it, raises an error or returns a malformed result scores zero.
