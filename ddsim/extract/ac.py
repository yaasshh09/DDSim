from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix
from ddsim.core import constants as C
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import GateContact, apply_dirichlet_nodes
from ddsim.discretize.coupled import (
    Unknown,
    assemble_coupled_terms,
    coupled_mass,
    edge_drop,
    pack,
    row_weights,
    scale_rows,
    unknown_index,
    unpack,
)
from ddsim.physics.bernoulli import B
from ddsim.physics.mobility import diffusivity_at
from ddsim.physics.recombination import NoRecombination, SRHRecombination
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.solve.linear import Ordering, SparseLU


@dataclass(frozen=True)
class Admittance:
    frequency: float

    contacts: tuple[str, ...]

    Y: npt.NDArray[np.complex128]

    def y(self, current_at: str, driven: str) -> complex:
        return complex(self.Y[self.contacts.index(current_at), self.contacts.index(driven)])

    @property
    def G(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.Y.real)

    @property
    def C(self) -> npt.NDArray[np.float64]:
        if self.frequency <= 0.0:
            raise ValueError(
                "capacitance is Im(Y) / omega, which needs a frequency above "
                f"zero, got {self.frequency:g} Hz"
            )
        return np.asarray(self.Y.imag / (2.0 * math.pi * self.frequency))

    def f_T(self, gate: str, drain: str) -> float:
        g = self.contacts.index(gate)
        return float(self.G[self.contacts.index(drain), g] / (2.0 * math.pi * self.C[g, g]))


@dataclass(frozen=True)
class Noise:
    frequency: float

    contacts: tuple[str, ...]

    S_I: npt.NDArray[np.complex128]

    Y: npt.NDArray[np.complex128]

    def open_circuit(self, reference: str) -> npt.NDArray[np.complex128]:
        if reference not in self.contacts:
            raise KeyError(f"no contact named {reference!r}, the contacts are {list(self.contacts)}")
        k = [i for i, c in enumerate(self.contacts) if c != reference]
        Z = np.linalg.inv(self.Y[np.ix_(k, k)])
        out = np.zeros_like(self.S_I)
        out[np.ix_(k, k)] = Z @ self.S_I[np.ix_(k, k)] @ Z.conj().T
        return out


def _pinned(device: Device) -> list[int]:
    k: list[int] = []
    for c in device.contacts:
        for i in c.nodes:
            k.append(unknown_index(i, Unknown.PSI))
            if not isinstance(c, GateContact):
                k += [unknown_index(i, Unknown.N), unknown_index(i, Unknown.P)]
    for i in device.carrier_free_nodes:
        k += [unknown_index(i, Unknown.N), unknown_index(i, Unknown.P)]
    return k


@dataclass(frozen=True)
class _Linear:
    x: npt.NDArray[np.float64]
    w: npt.NDArray[np.float64]
    pins: list[int]
    lu: SparseLU
    J: csr_matrix
    e: npt.NDArray[np.complex128]
    stamped: Callable[[list[float]], SparseAssembly]


def _linearized(device: Device, state: DeviceState, frequency: float, models: TransportModels, ordering: Ordering = "COLAMD") -> _Linear:
    if models.surface is not None:
        raise NotImplementedError(
            "surface mobility is held fixed at the DC state, so the Jacobian "
            "misses its dependence on psi, n and p and the admittance would "
            "be wrong. Solve without surface=True."
        )
    for run in (state.newton, state.gummel):
        if run is not None and not run.converged:
            raise ValueError(
                "the DC state did not converge, so its Jacobian linearizes "
                "around the wrong point and every admittance from it is wrong. "
                f"The solver said: {run.message or 'no message'}"
            )
    if frequency < 0.0:
        raise ValueError(f"frequency must be zero or above, got {frequency:g} Hz")
    s = device.scale
    mesh = device.scaled_mesh
    vol = device.charge_volume_scaled
    x = pack(state.psi.data, state.n.data, state.p.data)
    bare, scales = assemble_coupled_terms(mesh.h, vol, x, device.poisson_doping_scaled.data, models.Dn, models.Dp, models.recombination, mesh.geometry, device.degeneracy, device.ionization, device.net_doping_scaled.data,)
    N = x.size
    w = row_weights(scales, mesh.n_nodes)
    pins = _pinned(device)

    def stamped(targets: list[float]) -> SparseAssembly:
        a = SparseAssembly(residual=np.zeros(N), rows=bare.rows, cols=bare.cols, values=bare.values, shape=bare.shape)
        return scale_rows(apply_dirichlet_nodes(a, np.zeros(N), pins, targets), w)

    omega = 2.0 * math.pi * frequency * s.t_0
    r, c, m = coupled_mass(vol)
    A = stamped([0.0] * len(pins))
    lu = SparseLU(ordering)
    lu.factorize(np.concatenate([A.rows, r]), np.concatenate([A.cols, c]), np.concatenate([A.values.astype(np.complex128), 1j * omega * m / w[r]]), A.shape,)

    e = np.zeros((len(device.contacts), N), dtype=np.complex128)
    for j, k in enumerate(device.contacts):
        for i in k.nodes:
            e[j, unknown_index(i, Unknown.PSI)] = 1j * omega
            if not isinstance(k, GateContact):
                e[j, unknown_index(i, Unknown.N)] = -1.0
                e[j, unknown_index(i, Unknown.P)] = 1.0
    J = coo_matrix((bare.values, (bare.rows, bare.cols)), shape=bare.shape).tocsr()
    return _Linear(x=x, w=w, pins=pins, lu=lu, J=J, e=e, stamped=stamped)


def _y(device: Device, lin: _Linear) -> npt.NDArray[np.complex128]:
    Y = np.zeros((len(device.contacts), len(device.contacts)), dtype=np.complex128)
    for j, driven in enumerate(device.contacts):
        t = [0.0] * len(lin.pins)
        for i in driven.nodes:
            t[lin.pins.index(unknown_index(i, Unknown.PSI))] = 1.0 / device.scale.psi_0
        dx = lin.lu.solve(-lin.stamped(t).residual.astype(np.complex128))
        Y[:, j] = lin.e @ np.asarray(lin.J @ dx, dtype=np.complex128)
    return Y


def _unit(device: Device, width: float) -> float:
    s = device.scale
    return s.J_0 * s.x_0 ** (device.dimension - 1) / width


def admittance(device: Device, state: DeviceState, frequency: float, models: TransportModels | None = None, width: float = 1.0, ordering: Ordering = "COLAMD",) -> Admittance:
    if models is None:
        models = TransportModels.for_device(device)
    Y = _y(device, _linearized(device, state, frequency, models, ordering)) * _unit(device, width)
    return Admittance(frequency=frequency, contacts=tuple(k.name for k in device.contacts), Y=Y)


@dataclass(frozen=True)
class ACFrame:
    index: int

    voltage: float

    frequency: float

    conductance: float

    capacitance: float | None

    G_spread: float

    C_spread: float


@dataclass(frozen=True)
class ACPoint:
    voltage: float

    state: DeviceState

    admittances: tuple[Admittance, ...]

    G_spread: tuple[npt.NDArray[np.float64], ...]

    C_spread: tuple[npt.NDArray[np.float64], ...]


def _spread(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    top = np.maximum(np.abs(a), np.abs(b))
    return np.divide(np.abs(a - b), top, out=np.zeros_like(top), where=top > 0)


@dataclass(frozen=True)
class ACCurve:
    contact: str

    frequencies: tuple[float, ...]

    points: tuple[ACPoint, ...]

    complete: bool

    message: str = ""


_ORDERINGS: tuple[Ordering, Ordering] = ("COLAMD", "MMD_AT_PLUS_A")

ROUNDOFF = 1e-3


def check_frequencies(frequencies: Sequence[float]) -> None:
    if not frequencies:
        raise ValueError("an ac sweep needs at least one frequency [Hz]")
    for f in frequencies:
        if not (math.isfinite(f) and f >= 0.0):
            raise ValueError(f"every frequency must be a finite number of Hz, zero or above, got {f!r}")


def _dc(device: Device, models: TransportModels, max_iterations: int, on_frame: Callable[[object], None] | None) -> DeviceState | None:
    try:
        eq = solve_equilibrium(device, quasi_fermi=frozen_quasi_fermi(device), on_frame=on_frame)
        s = solve_bias_newton(device, models, guess=eq, max_iterations=max_iterations, on_frame=on_frame)
        if s.newton is not None and s.newton.converged:
            return s
    except RuntimeError:
        pass
    try:
        s = solve_bias_ramped(device, models=models, max_iterations=max_iterations, on_frame=on_frame)
    except RuntimeError:
        return None
    return s if s.newton is not None and s.newton.converged else None


def ac_sweep(device: Device, contact: str, voltages: list[float], frequencies: list[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None,) -> ACCurve:
    names = [k.name for k in device.contacts]
    if contact not in names:
        raise KeyError(f"no contact named {contact!r} on this device, which has {sorted(names)}")
    check_frequencies(frequencies)
    if models is None:
        models = TransportModels.for_device(device)
    width = 1.0 if isinstance(device.mesh, Mesh1D) else device.mesh.x_axis.length
    out: list[ACPoint] = []
    for v in voltages:
        e = device.with_bias(**{contact: v})
        s = _dc(e, models, max_iterations, on_frame)
        if s is None:
            return ACCurve(contact=contact, frequencies=tuple(frequencies), points=tuple(out), complete=False, message=f"the DC state at {v:+g} V did not converge, so there is nothing to linearize around")
        ys: list[Admittance] = []
        gs: list[npt.NDArray[np.float64]] = []
        cs: list[npt.NDArray[np.float64]] = []
        for f in frequencies:
            a, b = (admittance(e, s, f, models, width, o) for o in _ORDERINGS)
            ys.append(a)
            gs.append(_spread(a.Y.real, b.Y.real))
            cs.append(_spread(a.Y.imag, b.Y.imag))
            if on_frame is not None:
                g = a.contacts.index(contact)
                on_frame(ACFrame(index=len(out), voltage=v, frequency=f, conductance=float(a.G[g, g]), capacitance=float(a.C[g, g]) if f > 0.0 else None, G_spread=float(gs[-1][g, g]), C_spread=float(cs[-1][g, g])))
        out.append(ACPoint(voltage=v, state=s, admittances=tuple(ys), G_spread=tuple(gs), C_spread=tuple(cs)))
    return ACCurve(contact=contact, frequencies=tuple(frequencies), points=tuple(out), complete=True)


def noise(device: Device, state: DeviceState, frequency: float, models: TransportModels | None = None, width: float = 1.0,) -> Noise:
    if models is None:
        models = TransportModels.for_device(device)
    if device.degeneracy is not None:
        raise NotImplementedError(
            "diffusion noise here is 4 q^2 D n, the Boltzmann form. A "
            "degenerate gas changes the Einstein relation and so the source. "
            "Build the device with degenerate=False."
        )
    rec = models.recombination
    if not isinstance(rec, (SRHRecombination, NoRecombination)):
        raise NotImplementedError(
            f"only SRH generation-recombination noise is modeled, and this "
            f"device recombines through {type(rec).__name__}"
        )
    lin = _linearized(device, state, frequency, models)
    psi, n, p = unpack(lin.x)
    mesh = device.scaled_mesh
    h = mesh.h
    a, b = mesh.geometry.ends(h.size)
    d = edge_drop(psi, mesh.geometry)
    face = mesh.geometry.carrier_face / h
    gn = diffusivity_at(models.Dn, d, h) * face * (B(d) * n[b] + B(-d) * n[a]) / 2.0
    gp = diffusivity_at(models.Dp, d, h) * face * (B(d) * p[a] + B(-d) * p[b]) / 2.0

    free = np.ones(lin.x.size)
    free[lin.pins] = 0.0
    y = np.column_stack([lin.lu.solve(np.asarray(lin.J.T @ k, dtype=np.complex128), trans="T") for k in lin.e])
    tmp = lin.e.copy()
    tmp[:, Unknown.PSI::3] = 0.0
    t = tmp - (free / lin.w)[None, :] * y.T
    kn = np.arange(mesh.n_nodes) * 3 + Unknown.N
    kp = np.arange(mesh.n_nodes) * 3 + Unknown.P
    S = np.zeros((len(device.contacts), len(device.contacts)), dtype=np.complex128)
    for T, kappa in (
        (t[:, kn[b]] - t[:, kn[a]], 4.0 * gn),
        (t[:, kp[a]] - t[:, kp[b]], 4.0 * gp),
        (t[:, kn] + t[:, kp], rec.rate_noise(n, p) * device.charge_volume_scaled),
    ):
        S += (T * np.broadcast_to(kappa, T.shape[1:])) @ T.conj().T
    unit = _unit(device, width)
    return Noise(frequency=frequency, contacts=tuple(k.name for k in device.contacts), S_I=C.q * unit * S, Y=_y(device, lin) * unit)
