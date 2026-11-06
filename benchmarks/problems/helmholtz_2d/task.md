You are an expert in numerical methods for partial differential equations and wave propagation. Write a fast and accurate solver for the 2D Helmholtz equation

    -(u_xx + u_yy) - k^2 u = f   on the unit square (0, 1) x (0, 1),   du/dn - i k u = 0 on the boundary (impedance condition, n the outward normal),

with two Gaussian sources whose width scales with the wavelength:

    f(x, y) = exp(-(k^2 / 2) ((x - 0.3)^2 + (y - 0.6)^2)) - 0.6 exp(-(k^2 / 2) ((x - 0.7)^2 + (y - 0.25)^2)).

The solution u is complex. Low-order discretizations suffer from pollution (dispersion) error that grows with k at a fixed number of points per wavelength.

The program must define solve(k, x, y): k is a float and x, y float arrays of 2000 points inside the square; it returns the complex solution u at the points (x[i], y[i]), an array of shape (2000,). It is called once per case, each time in a fresh process, for five cases: k = 10, 20, 40, 60 and 80.

The answer is compared with a high-accuracy reference solution. A case scores min(6, -log10(||u - u_ref|| / ||u_ref||)) / 6 (2-norm over the 2000 points), so 6 correct digits give full marks for that case. The combined_score is the mean over the five cases. The evaluation reports the digits, the error and the run time of every case, or why it failed.

EXECUTION ENVIRONMENT:
The program runs with Python 3.11 on a single CPU core, with numpy, scipy, pyamg, mpmath and sympy installed (no numba). Each case has a hard time limit of 9 seconds, including the time to start Python and import modules; a case that exceeds it, raises an error or returns a malformed result scores zero.
