# EVOLVE-BLOCK-START
"""1D viscous Burgers equation with explicit finite differences (forward Euler, central differences)."""
import numpy as np


def solve(nu, T, x):
    """u_t + u u_x = nu u_xx on [-1, 1], periodic, u(x, 0) = -sin(pi x). Returns u(x, T) at the points x (array)."""
    N = 256
    grid = np.linspace(-1, 1, N, endpoint=False)
    dx = 2 / N
    u = -np.sin(np.pi * grid)
    t = 0.0
    while t < T:
        dt = min(0.4 * dx / max(np.abs(u).max(), 1e-12), 0.4 * dx**2 / nu, T - t)
        flux = 0.5 * u**2
        u = u + dt * (-(np.roll(flux, -1) - np.roll(flux, 1)) / (2 * dx)
                      + nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2)
        t += dt
    return np.interp(x, grid, u, period=2)
# EVOLVE-BLOCK-END
