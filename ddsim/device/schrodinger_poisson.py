from __future__ import annotations
import math
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded
from ddsim.core import constants as C
from ddsim.device.builder import Device
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.physics.quantum import HBAR, bound_states

Array = npt.NDArray[np.float64]
Density = Callable[[Array], tuple[Array, Array]]
VALLEYS = ((C.M_L, C.M_T, 2), (C.M_T, math.sqrt(C.M_L * C.M_T), 4))


@dataclass(frozen=True)
class Column:
    y: Array

    h: Array

    eps: Array

    silicon: Array

    doping: Array

    interface: float

    psi_classical: Array


@dataclass(frozen=True)
class QuantumState:
    y: Array

    psi: Array

    n: Array

    p: Array

    levels: tuple[tuple[float, ...], ...]

    interface: float

    sheet_charge: float

    iterations: int

    @property
    def centroid(self) -> float:
        d = self.interface - self.y
        k = (d >= 0.0) & (d <= 0.5 * self.interface)
        return float(np.trapezoid(self.n[k] * d[k], d[k]) / np.trapezoid(self.n[k], d[k]))


def column(device: Device) -> Column:
    m = device.mesh
    if not isinstance(m, Mesh2D) or device.regions is None:
        raise ValueError("Schrodinger-Poisson takes a MOS capacitor built on a tensor mesh with silicon and oxide regions")
    s = solve_equilibrium(device)
    if s.newton is None or not s.newton.converged:
        raise RuntimeError("the classical equilibrium did not converge, so there is no starting potential")
    at = np.array([m.node_at(0, j) for j in range(m.ny)])
    y = m.y_axis.x
    interface = float(y[int(np.max(np.flatnonzero(device.regions.semiconductor_volume[at] > 0.0)))])
    below = y[1:] <= interface
    silicon = np.zeros(y.size)
    silicon[1:] += np.where(below, 0.5 * m.y_axis.h, 0.0)
    silicon[:-1] += np.where(below, 0.5 * m.y_axis.h, 0.0)
    return Column(y=y, h=m.y_axis.h, eps=np.where(below, C.eps_Si(), C.eps_ox()), silicon=silicon, doping=device.net_doping.data[at], interface=interface, psi_classical=s.psi.to_physical(device.scale).data[at])


def valley_Nc(T: float) -> float:
    return float(2 * sum(g for _, _, g in VALLEYS) * (2 * math.pi * C.m_0 * C.k_B * T / C.h**2) ** 1.5 * math.sqrt(C.M_L * C.M_T**2) * 1e-6)


def subband_density(depth: Array, Ec: Array, T: float, n_states: int, n_points: int) -> tuple[Array, tuple[tuple[float, ...], ...]]:
    V_T = C.k_B * T / C.q
    n = np.zeros(depth.size)
    found = []
    for mz, md, g in VALLEYS:
        s = bound_states(depth, Ec, mz, n_states=n_states, n_points=n_points)
        found.append(s.energies)
        dos = g * md * C.m_0 * C.k_B * T / (math.pi * HBAR**2) * 1e-4
        for e, w in zip(s.energies, s.psi, strict=True):
            n += dos * np.logaddexp(0.0, -e / V_T) * w**2
    return n, tuple(found)


def _poisson(c: Column, psi: Array, n_of: Density, n_i: float, V_T: float) -> Array:
    x = psi.copy()
    k = c.eps / c.h
    for _ in range(200):
        n, dn = n_of(x)
        p = n_i * np.exp(-x / V_T) * (c.silicon > 0.0)
        f = np.zeros_like(x)
        f[:-1] += k * (x[1:] - x[:-1])
        f[1:] -= k * (x[1:] - x[:-1])
        f += C.q * c.silicon * (p - n + c.doping)
        ab = np.zeros((3, x.size))
        ab[0, 1:] = k
        ab[2, :-1] = k
        ab[1, :-1] -= k
        ab[1, 1:] -= k
        ab[1] += C.q * c.silicon * (-p / V_T - dn)
        ab[1, 0] = ab[1, -1] = 1.0
        ab[0, 1] = ab[2, -2] = 0.0
        f[0] = f[-1] = 0.0
        step = np.clip(solve_banded((1, 1), ab, -f), -0.1, 0.1)
        x = x + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return x


def solve_schrodinger_poisson(device: Device, depth: float = 2.5e-6, n_states: int = 8, n_points: int = 3000, tol: float = 1e-7, max_iterations: int = 200, quantum: bool = True) -> QuantumState:
    c = column(device)
    T = device.material.T
    V_T = C.k_B * T / C.q
    n_i = device.material.n_i
    offset = V_T * math.log(valley_Nc(T) / n_i)
    d = c.interface - c.y
    box = np.flatnonzero((d >= 0.0) & (d <= depth))
    order = box[np.argsort(d[box])]
    inside = np.zeros(c.y.size, dtype=bool)
    inside[order] = True

    def classical(x: Array) -> tuple[Array, Array]:
        n = n_i * np.exp(x / V_T) * (c.silicon > 0.0)
        return n, n / V_T

    psi = c.psi_classical.copy()
    levels: tuple[tuple[float, ...], ...] = ()
    density: Density = classical
    it = 0
    if quantum:
        while it < max_iterations:
            it += 1
            nq = np.zeros(c.y.size)
            nq[order], levels = subband_density(d[order], -psi[order] + offset, T, n_states, n_points)
            anchor = psi.copy()

            def mixed(x: Array, nq: Array = nq, anchor: Array = anchor) -> tuple[Array, Array]:
                cl, _ = classical(x)
                n = np.where(inside, nq * np.exp((x - anchor) / V_T), cl)
                return n, n / V_T

            density = mixed
            new = _poisson(c, psi, density, n_i, V_T)
            moved = float(np.max(np.abs(new - psi)))
            psi = new
            if moved < tol:
                break
    else:
        psi = _poisson(c, psi, density, n_i, V_T)
    n, _ = density(psi)
    p = n_i * np.exp(-psi / V_T) * (c.silicon > 0.0)
    return QuantumState(y=c.y, psi=psi, n=n, p=p, levels=levels, interface=c.interface, sheet_charge=float(C.q * np.sum(c.silicon * (p - n + c.doping))), iterations=it)
