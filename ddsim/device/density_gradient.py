from __future__ import annotations
import math
from dataclasses import dataclass, replace
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from ddsim.core import constants as C
from ddsim.device.builder import Device
from ddsim.device.schrodinger_poisson import Column, QuantumState, column
from ddsim.physics.quantum import HBAR

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DensityGradient:
    gamma: float = 3.6

    mass: float = 1.0903

    gamma_ox: float = 1.0

    mass_ox: float = 0.14

    barrier: float = 3.15

    mass_tunnel: float = 0.4

    @property
    def b(self) -> float:
        return self.gamma * HBAR**2 / (6 * C.q * self.mass * C.m_0) * 1e4

    @property
    def b_ox(self) -> float:
        return self.gamma_ox * HBAR**2 / (6 * C.q * self.mass_ox * C.m_0) * 1e4

    @property
    def decay(self) -> float:
        return HBAR / math.sqrt(2 * self.mass_tunnel * C.m_0 * C.q * self.barrier) * 1e2


def residual(c: Column, x: Array, top: int, model: DensityGradient, n_i: float, V_T: float) -> tuple[Array, sp.csr_matrix]:
    N = c.y.size
    psi, L = x[:N], np.zeros(N)
    L[: top + 1] = x[N:]
    k = c.eps / c.h
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def put(i: int, j: int, v: float) -> None:
        rows.append(i)
        cols.append(j)
        vals.append(v)

    si = c.silicon > 0.0
    n = np.where(si, n_i * np.exp((psi - L) / V_T), 0.0)
    p = np.where(si, n_i * np.exp(-psi / V_T), 0.0)
    f = np.zeros(N + top + 1)
    for e in range(N - 1):
        flux = k[e] * (psi[e + 1] - psi[e])
        f[e] += flux
        f[e + 1] -= flux
        for i, j in ((e, e), (e, e + 1), (e + 1, e), (e + 1, e + 1)):
            put(i, j, k[e] * (1.0 if i != j else -1.0))
    f[:N] += C.q * c.silicon * (p - n + c.doping)
    for i in range(N):
        put(i, i, C.q * c.silicon[i] * (-p[i] - n[i]) / V_T)
        if si[i]:
            put(i, N + i, C.q * c.silicon[i] * n[i] / V_T)
    b = model.b
    for e in range(top):
        g = 1.0 / (V_T * c.h[e])
        d = ((psi[e + 1] - psi[e]) - (L[e + 1] - L[e])) * g
        wiggle = {e: (-g, g), e + 1: (g, -g)}
        for i, sign in ((e, 1.0), (e + 1, -1.0)):
            if i == 0:
                continue
            f[N + i] += sign * 0.5 * b * d + 0.25 * b * d * d * 0.5 * c.h[e]
            for j, (dpsi, dL) in wiggle.items():
                slope = sign * 0.5 * b + 0.5 * b * d * 0.5 * c.h[e]
                put(N + i, j, slope * dpsi)
                put(N + i, N + j, slope * dL)
    for i in range(1, top + 1):
        f[N + i] += L[i] * c.silicon[i]
        put(N + i, N + i, c.silicon[i])
    f[N + top] -= model.b_ox / model.decay
    f[0] = psi[0] - c.psi_classical[0]
    f[N - 1] = psi[N - 1] - c.psi_classical[N - 1]
    f[N] = L[0]
    keep = [(r, q, v) for r, q, v in zip(rows, cols, vals, strict=True) if r not in (0, N - 1, N)]
    J = sp.coo_matrix((np.array([v for _, _, v in keep] + [1.0, 1.0, 1.0]), (np.array([r for r, _, _ in keep] + [0, N - 1, N]), np.array([q for _, q, _ in keep] + [0, N - 1, N]))), shape=(N + top + 1, N + top + 1)).tocsr()
    return f, J


def solve_density_gradient(device: Device, model: DensityGradient | None = None, gate: float | None = None, max_iterations: int = 100) -> QuantumState:
    model = DensityGradient() if model is None else model
    c = column(device)
    if gate is not None:
        psi0 = c.psi_classical.copy()
        psi0[-1] = gate
        c = Column(y=c.y, h=c.h, eps=c.eps, silicon=c.silicon, doping=c.doping, interface=c.interface, psi_classical=psi0)
    T = device.material.T
    V_T = C.k_B * T / C.q
    n_i = device.material.n_i
    N = c.y.size
    top = int(np.max(np.flatnonzero(c.silicon > 0.0)))
    x = np.concatenate([c.psi_classical, np.zeros(top + 1)])
    it = 0
    share = 0.0
    move = 0.05
    while share < 1.0:
        trial = min(1.0, share + move)
        stage = replace(model, gamma=model.gamma * trial, gamma_ox=model.gamma_ox * trial)
        y, done = x.copy(), False
        for _ in range(max_iterations):
            it += 1
            f, J = residual(c, y, top, stage, n_i, V_T)
            step = np.clip(spsolve(J.tocsc(), -f), -0.1, 0.1)
            y = y + step
            if np.max(np.abs(step)) < 1e-10:
                done = True
                break
        if done:
            x, share = y, trial
            move = min(2 * move, 1.0)
        else:
            move /= 4
            if move < 1e-4:
                raise RuntimeError(f"the density gradient solve stalled at {share:.4f} of gamma {model.gamma:g}")
    psi, L = x[:N], np.zeros(N)
    L[: top + 1] = x[N:]
    si = c.silicon > 0.0
    n = np.where(si, n_i * np.exp((psi - L) / V_T), 0.0)
    p = np.where(si, n_i * np.exp(-psi / V_T), 0.0)
    return QuantumState(y=c.y, psi=psi, n=n, p=p, levels=(), interface=c.interface, sheet_charge=float(C.q * np.sum(c.silicon * (p - n + c.doping))), iterations=it)


def calibrate_gamma(devices: list[Device], target: list[float], low: float = 0.5, high: float = 40.0, rtol: float = 1e-3) -> float:
    if len(devices) != len(target):
        raise ValueError(f"one target charge per device, got {len(devices)} devices and {len(target)} targets")

    def miss(gamma: float) -> float:
        return float(np.mean([(solve_density_gradient(d, DensityGradient(gamma=gamma)).sheet_charge - t) / t for d, t in zip(devices, target, strict=True)]))

    a, b = low, high
    fa, fb = miss(a), miss(b)
    if fa * fb > 0.0:
        raise ValueError(f"no gamma between {low:g} and {high:g} matches the target charge: the mean miss is {fa:+.3e} at one end and {fb:+.3e} at the other")
    while b - a > rtol * a:
        m = math.sqrt(a * b)
        fm = miss(m)
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return math.sqrt(a * b)
