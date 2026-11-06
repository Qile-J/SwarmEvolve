You are an expert in numerical methods for partial differential equations. Write a fast and accurate solver for the steady 2D convection-diffusion equation

    -eps (u_xx + u_yy) + b1 u_x + b2 u_y = 1   on the unit square (0, 1) x (0, 1),   u = 0 on the boundary,

with the constant flow b = (b1, b2) = (1, 0.5). For small eps the solution has boundary layers of width proportional to eps along the outflow sides x = 1 and y = 1.

The program must define solve(eps, b, x, y): eps is a float, b the tuple (1.0, 0.5), and x, y float arrays of 2000 points inside the square; it returns the solution u at the points (x[i], y[i]), an array of shape (2000,). It is called once per case, each time in a fresh process, for five cases: eps = 0.1, 0.01, 0.003, 0.001 and 0.0003.

The answer is compared with a high-accuracy reference solution. A case scores min(6, -log10(||u - u_ref|| / ||u_ref||)) / 6 (2-norm over the 2000 points), so 6 correct digits give full marks for that case. The combined_score is the mean over the five cases. The evaluation reports the digits, the error and the run time of every case, or why it failed.

EXECUTION ENVIRONMENT:
The program runs with Python 3.11 on a single CPU core, with numpy, scipy, pyamg, mpmath and sympy installed (no numba). Each case has a hard time limit of 9 seconds, including the time to start Python and import modules; a case that exceeds it, raises an error or returns a malformed result scores zero.
