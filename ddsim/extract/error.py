from __future__ import annotations
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import numpy as np
import numpy.typing as npt
from ddsim.core.config import CONFIG
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.device.regions import region_map
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import Contact, OhmicContact, apply_contacts, apply_dirichlet_nodes
from ddsim.discretize.poisson import assemble_poisson
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.extract.cv import _charge_unit, _contact_nodes, small_signal_capacitance
from ddsim.discretize.coupled import Unknown, apply_contacts_coupled, assemble_coupled_terms, effective_potentials, pack, unknown_index, unpack
from ddsim.mesh.mesh1d import Mesh1D, _assemble
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.solve.linear import SparseLU

Models = Callable[[Device], TransportModels]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ErrorEstimate:
    value: float

    estimate: float

    indicator: Array

    refinement: int

    @property
    def relative(self) -> float:
        if self.value == 0.0:
            return math.inf
        return abs(self.estimate) / abs(self.value)


@dataclass(frozen=True)
class Adapted:
    device: Device

    state: DeviceState

    estimate: ErrorEstimate

    converged: bool

    message: str

    history: tuple[tuple[int, float, float], ...]


def _axes(device: Device) -> tuple[Array, Array | None]:
    m = device.mesh
    if isinstance(m, Mesh1D):
        return m.x, None
    return m.x_axis.x, m.y_axis.x


def _subdivided(x: Array, k: int) -> Array:
    t = np.arange(1, k) / k
    return np.sort(np.concatenate([x, (x[:-1, None] + np.diff(x)[:, None] * t[None, :]).ravel()]))


def _halved(x: Array, cells: Sequence[int]) -> Array:
    c = np.unique(np.asarray(cells, dtype=np.int64))
    return np.sort(np.concatenate([x, 0.5 * (x[c] + x[c + 1])]))


def _on(old: Array, new: Array) -> npt.NDArray[np.int64]:
    i = np.searchsorted(new, old)
    if not np.array_equal(new[np.minimum(i, new.size - 1)], old):
        raise ValueError("the new mesh does not keep every old node, so nothing on the old mesh carries over to it")
    return i.astype(np.int64)


def _moved(c: Contact, nx: int, ix: npt.NDArray[np.int64], iy: npt.NDArray[np.int64], nxf: int) -> Contact:
    s = set(c.nodes)
    out: set[int] = set()
    for v in c.nodes:
        i, j = v % nx, v // nx
        out.add(int(iy[j] * nxf + ix[i]))
        if i + 1 < nx and v + 1 in s:
            out.update(int(iy[j] * nxf + t) for t in range(ix[i], ix[i + 1] + 1))
        if v + nx in s:
            out.update(int(t * nxf + ix[i]) for t in range(iy[j], iy[j + 1] + 1))
    if isinstance(c, OhmicContact):
        return replace(c, node=out.pop())
    return replace(c, nodes=tuple(sorted(out)))


def _rebuilt(device: Device, x: Array, y: Array | None) -> Device:
    xo, yo = _axes(device)
    ix = _on(xo, x)
    if y is None or yo is None:
        cs = tuple(_moved(c, xo.size, ix, np.zeros(1, dtype=np.int64), x.size) for c in device.contacts)
        return replace(device, mesh=_assemble(x), contacts=cs)
    iy = _on(yo, y)
    m = tensor_mesh_2d(_assemble(x), _assemble(y))
    cs = tuple(_moved(c, xo.size, ix, iy, x.size) for c in device.contacts)
    r = device.regions
    if r is not None:
        a = np.searchsorted(xo, 0.5 * (x[:-1] + x[1:])) - 1
        b = np.searchsorted(yo, 0.5 * (y[:-1] + y[1:])) - 1
        r = region_map(m, r.cell_material[np.ix_(b, a)])
    return replace(device, mesh=m, contacts=cs, regions=r)


def refined(device: Device, k: int = CONFIG.errors.refinement) -> Device:
    if k < 2:
        raise ValueError(f"refining means at least 2 cells for every old one, got {k}")
    x, y = _axes(device)
    return _rebuilt(device, _subdivided(x, k), None if y is None else _subdivided(y, k))


def split(device: Device, columns: Sequence[int] = (), rows: Sequence[int] = ()) -> Device:
    x, y = _axes(device)
    if y is None:
        return _rebuilt(device, _halved(x, columns), None)
    return _rebuilt(device, _halved(x, columns), _halved(y, rows))


def split_cells(device: Device, cells: Sequence[int]) -> Device:
    x, y = _axes(device)
    if y is None:
        return split(device, cells)
    w = x.size - 1
    return split(device, [c % w for c in cells], [c // w for c in cells])


def _weights(old: Array, new: Array) -> tuple[npt.NDArray[np.int64], Array]:
    i = np.clip(np.searchsorted(old, new, side="right") - 1, 0, old.size - 2)
    return i.astype(np.int64), (new - old[i]) / (old[i + 1] - old[i])


def _fraction(t: Array, d: Array) -> Array:
    with np.errstate(all="ignore"):
        up = (np.exp(d * (t - 1.0)) - np.exp(-d)) / -np.expm1(-d)
        down = np.expm1(d * t) / np.expm1(d)
    return np.asarray(np.where(np.abs(d) < 1e-10, t, np.where(d > 0.0, up, down)))


def _along(v: Array, guide: Array | None, i: npt.NDArray[np.int64], t: Array, sign: float) -> Array:
    a, b = v[:, i], v[:, i + 1]
    if guide is None:
        return a + (b - a) * t[None, :]
    return a + (b - a) * _fraction(t[None, :], sign * (guide[:, i + 1] - guide[:, i]))


def _prolong(coarse: Device, fine: Device, v: Array, guide: Array | None = None, sign: float = 0.0) -> Array:
    xo, yo = _axes(coarse)
    xn, yn = _axes(fine)
    g = v.reshape(-1, xo.size)
    h = None if guide is None else guide.reshape(-1, xo.size)
    i, t = _weights(xo, xn)
    u = _along(g, h, i, t, sign)
    if yo is None or yn is None:
        return u.ravel()
    q = None if h is None else _along(h, None, i, t, 0.0)
    j, s = _weights(yo, yn)
    return _along(u.T, None if q is None else q.T, j, s, sign).T.ravel()


def _pins(device: Device, x: Array) -> tuple[npt.NDArray[np.int64], Array]:
    N = x.size
    e = SparseAssembly(residual=np.zeros(N), rows=np.zeros(0, dtype=np.int64), cols=np.zeros(0, dtype=np.int64), values=np.zeros(0), shape=(N, N))
    a = apply_contacts_coupled(e, x, device.contact_doping_scaled.data, device.contacts, device.scale, device.carrier_free_nodes, device.material.T, device.degeneracy)
    return a.rows, x[a.rows] - a.residual[a.rows]


def prolonged_state(coarse: Device, fine: Device, state: DeviceState) -> DeviceState:
    psi, n, p = state.psi.data, state.n.data, state.p.data
    b, c = effective_potentials(psi, n, p, coarse.degeneracy)
    u = _prolong(coarse, fine, psi)
    v = _prolong(coarse, fine, n, b, 1.0)
    w = _prolong(coarse, fine, p, c, -1.0)
    k = list(fine.carrier_free_nodes)
    v[k] = 0.0
    w[k] = 0.0
    x = pack(u, v, w)
    i, t = _pins(fine, x)
    x[i] = t
    u, v, w = unpack(x)
    return DeviceState(psi=Field(u.copy(), "V", ScalingState.SCALED, Location.NODE, name="psi"), n=Field(v.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="n"), p=Field(w.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name="p"), degeneracy=fine.degeneracy)


def _raw(device: Device, models: TransportModels, x: Array) -> SparseAssembly:
    sm = device.scaled_mesh
    psi, n, p = unpack(x)
    m = models.at_state(device, psi, n, p)
    a, _ = assemble_coupled_terms(sm.h, device.charge_volume_scaled, x, device.poisson_doping_scaled.data, m.Dn, m.Dp, m.recombination, sm.geometry, device.degeneracy, device.ionization, device.net_doping_scaled.data)
    return a


def _current_weights(device: Device, contact: str) -> Array:
    c = np.zeros(3 * device.mesh.n_nodes)
    f = device.scale.J_0 * device.scale.x_0 ** (device.dimension - 1)
    k = [z for z in device.semiconductor_contacts if z.name == contact]
    if not k:
        raise KeyError(f"no contact named {contact!r} that carries current, the device has {sorted(z.name for z in device.semiconductor_contacts)}")
    for i in k[0].nodes:
        c[unknown_index(i, Unknown.N)] = -f
        c[unknown_index(i, Unknown.P)] = f
    return c


def _to_cells(old: Array, new: Array) -> Array:
    i, t = _weights(old, new)
    n = old.size - 1
    M = np.zeros((n, new.size))
    j = np.arange(new.size)
    inside = (t > 0.0) & (t < 1.0)
    M[i[inside], j[inside]] = 1.0
    k = np.where(t >= 1.0, i + 1, i)
    for a in np.flatnonzero(~inside):
        left, right = k[a] - 1, k[a]
        if left < 0:
            M[right, a] = 1.0
        elif right >= n:
            M[left, a] = 1.0
        else:
            M[left, a] = M[right, a] = 0.5
    return M


def _cells(coarse: Device, at: Device, contribution: Array, per_node: int) -> Array:
    xo, yo = _axes(coarse)
    xn, yn = _axes(at)
    g = contribution.reshape(-1, per_node).sum(axis=1).reshape(-1, xn.size)
    out = g @ _to_cells(xo, xn).T
    if yo is not None and yn is not None:
        out = _to_cells(yo, yn) @ out
    return np.asarray(out.ravel())


Raw = Callable[[Device, Array], SparseAssembly]

Pins = Callable[[Device, Array], tuple[npt.NDArray[np.int64], Array]]


def _estimate(coarse: Device, fine: Device, x: Array, y: Array, raw: Raw, pins: Pins, weights: Callable[[Device], Array], per_node: int, k: int) -> ErrorEstimate:
    A = raw(coarse, x)
    c = weights(coarse)
    p, t = pins(coarse, x)
    value = float(c @ A.residual)
    g = np.zeros(x.size)
    np.add.at(g, A.cols, A.values * c[A.rows])
    g[p] = 0.0
    S = apply_dirichlet_nodes(A, x, p.tolist(), t.tolist())
    lu = SparseLU()
    lu.factorize(S.rows, S.cols, S.values, S.shape)
    w = lu.solve(-g, trans="T")
    w[p] = c[p]
    F = raw(fine, y).residual
    z = np.column_stack([_prolong(coarse, fine, w[j::per_node]) for j in range(per_node)]).ravel()
    i, _ = pins(fine, y)
    z[i] = weights(fine)[i]
    there = z * F
    here = c * A.residual
    return ErrorEstimate(value=value, estimate=float(np.sum(there)) - value, indicator=_cells(coarse, fine, there, per_node) - _cells(coarse, coarse, here, per_node), refinement=k)


def current_error(device: Device, state: DeviceState, contact: str, models_for: Models = TransportModels.for_device, k: int = CONFIG.errors.refinement) -> ErrorEstimate:
    fine = refined(device, k)
    s = prolonged_state(device, fine, state)

    def raw(d: Device, v: Array) -> SparseAssembly:
        return _raw(d, models_for(d), v)

    return _estimate(device, fine, pack(state.psi.data, state.n.data, state.p.data), pack(s.psi.data, s.n.data, s.p.data), raw, _pins, lambda d: _current_weights(d, contact), 3, k)


def _poisson_raw(device: Device, v: Array) -> SparseAssembly:
    phi_n, phi_p = frozen_quasi_fermi(device)
    return assemble_poisson(device.scaled_mesh, Field(v, "V", ScalingState.SCALED, Location.NODE, name="psi"), device.poisson_doping_scaled, phi_n, phi_p, charge_volume=device.charge_volume_scaled, degeneracy=device.degeneracy, ionization=device.ionization)


def _poisson_pins(device: Device, v: Array) -> tuple[npt.NDArray[np.int64], Array]:
    N = v.size
    e = SparseAssembly(residual=np.zeros(N), rows=np.zeros(0, dtype=np.int64), cols=np.zeros(0, dtype=np.int64), values=np.zeros(0), shape=(N, N))
    a = apply_contacts(e, v, device.contact_doping_scaled.data, device.contacts, device.scale, device.material.T, device.degeneracy)
    return a.rows, v[a.rows] - a.residual[a.rows]


def _charge_weights(device: Device, contact: str) -> Array:
    c = np.zeros(device.mesh.n_nodes)
    c[list(_contact_nodes(device, contact))] = _charge_unit(device, None)
    return c


def charge_error(device: Device, state: DeviceState, contact: str, k: int = CONFIG.errors.refinement) -> ErrorEstimate:
    fine = refined(device, k)
    y = _prolong(device, fine, state.psi.data)
    i, t = _poisson_pins(fine, y)
    y[i] = t
    return _estimate(device, fine, state.psi.data, y, _poisson_raw, _poisson_pins, lambda d: _charge_weights(d, contact), 1, k)


def capacitance_error(device: Device, contact: str, delta: float = CONFIG.errors.voltage_step, k: int = CONFIG.errors.refinement) -> ErrorEstimate:
    v = next(c.voltage for c in device.contacts if c.name == _contact_name(device, contact))

    def at(u: float) -> tuple[Device, DeviceState]:
        d = device.with_bias(**{contact: u})
        return d, solve_equilibrium(d, quasi_fermi=frozen_quasi_fermi(d))

    d, s = at(v)
    value = small_signal_capacitance(d, s, contact, quasi_fermi=frozen_quasi_fermi(d))
    a = charge_error(*at(v - delta), contact, k)
    b = charge_error(*at(v + delta), contact, k)
    return ErrorEstimate(value=value, estimate=(b.estimate - a.estimate) / (2.0 * delta), indicator=(b.indicator - a.indicator) / (2.0 * delta), refinement=k)


def _contact_name(device: Device, contact: str) -> str:
    _contact_nodes(device, contact)
    return contact


def _marked(score: Array, fraction: float) -> list[int]:
    o = np.argsort(score)[::-1]
    c = np.cumsum(score[o])
    if c[-1] <= 0.0:
        return []
    return [int(v) for v in o[: int(np.searchsorted(c, fraction * c[-1])) + 1]]


def _proposal(device: Device, indicator: Array, fraction: float) -> Device:
    x, y = _axes(device)
    a = np.abs(indicator)
    if y is None:
        return split(device, _marked(a, fraction))
    g = a.reshape(y.size - 1, x.size - 1)
    return split(device, _marked(g.sum(axis=0), fraction), _marked(g.sum(axis=1), fraction))


def adapt(device: Device, contact: str, rtol: float, budget: int = CONFIG.mesh.node_budget, models_for: Models = TransportModels.for_device, max_rounds: int = CONFIG.errors.max_rounds, fraction: float = CONFIG.errors.mark_fraction) -> Adapted:
    if not rtol > 0.0:
        raise ValueError(f"the tolerance has to be above zero, got {rtol:g}")
    if device.mesh.n_nodes > budget:
        raise ValueError(f"the starting mesh has {device.mesh.n_nodes} nodes, already over the budget of {budget}")
    d = device
    s = solve_bias_ramped(d, models_for(d))
    if s.newton is None or not s.newton.converged:
        raise RuntimeError(f"the starting mesh did not converge: {s.newton.message if s.newton else 'no newton run'}")
    log: list[tuple[int, float, float]] = []
    for _ in range(max_rounds):
        e = current_error(d, s, contact, models_for)
        log.append((d.mesh.n_nodes, e.value, e.estimate))
        if abs(e.estimate) <= rtol * abs(e.value):
            return Adapted(d, s, e, True, f"reached {e.relative:.2e} on {d.mesh.n_nodes} nodes", tuple(log))
        f = _proposal(d, e.indicator, fraction)
        if f.mesh.n_nodes == d.mesh.n_nodes:
            return Adapted(d, s, e, False, f"the indicator flags nothing to refine, the estimate is still {e.relative:.2e} against {rtol:.2e}", tuple(log))
        if f.mesh.n_nodes > budget:
            return Adapted(d, s, e, False, f"stopped at the node budget of {budget}: the next mesh needs {f.mesh.n_nodes} nodes, and the estimate is still {e.relative:.2e} against {rtol:.2e}", tuple(log))
        m = models_for(f)
        t = solve_bias_newton(f, m, guess=prolonged_state(d, f, s))
        if t.newton is None or not t.newton.converged:
            t = solve_bias_ramped(f, m)
        if t.newton is None or not t.newton.converged:
            return Adapted(d, s, e, False, f"the refined mesh with {f.mesh.n_nodes} nodes did not converge, so I kept the last one, at {e.relative:.2e}", tuple(log))
        d, s = f, t
    e = current_error(d, s, contact, models_for)
    return Adapted(d, s, e, abs(e.estimate) <= rtol * abs(e.value), f"stopped after {max_rounds} rounds at {e.relative:.2e}", tuple(log))
