"""Calibration for burgers_1d: Fourier pseudo-spectral (2/3 dealiasing) with integrating-factor RK4, N by nu."""
import numpy as np


def solve(nu, T, x):
    N = int(min(16384, max(256, 2 ** np.ceil(np.log2(6 / nu)))))
    grid = np.linspace(-1, 1, N, endpoint=False)
    k = np.pi * np.fft.rfftfreq(N, d=1 / N)
    dealias = k < (2 / 3) * k.max()
    v = np.fft.rfft(-np.sin(np.pi * grid))
    def nonlinear(v):
        u = np.fft.irfft(v, N)
        return -0.5j * k * np.fft.rfft(u * u) * dealias
    steps = int(np.ceil(T / (0.5 * (2 / N) / 1.0)))
    dt = T / steps
    E, E2 = np.exp(-nu * k**2 * dt), np.exp(-nu * k**2 * dt / 2)
    for _ in range(steps):
        a = nonlinear(v)
        b = nonlinear(E2 * (v + dt / 2 * a))
        c = nonlinear(E2 * v + dt / 2 * b)
        d = nonlinear(E * v + dt * E2 * c)
        v = E * v + dt / 6 * (E * a + 2 * E2 * (b + c) + d)
    # evaluate the Fourier series at the points
    u_hat = v / N
    modes = np.arange(len(k))
    phase = np.exp(1j * np.pi * np.outer(np.asarray(x) + 1, modes))
    weights = np.full(len(k), 2.0); weights[0] = 1.0
    if N % 2 == 0: weights[-1] = 1.0
    return (phase * (u_hat * weights)).real.sum(axis=1)
