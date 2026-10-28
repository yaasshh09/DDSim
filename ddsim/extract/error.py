from __future__ import annotations
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import numpy as np
import numpy.typing as npt
from ddsim.core.config import CONFIG
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
from ddsim.device.regions import region_map
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import Carrier, Contact, OhmicContact, SchottkyContact, apply_contacts, apply_dirichlet_nodes, pins_carrier
from ddsim.discretize.poisson import assemble_poisson
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.extract.cv import CVCurve, Response, _charge_unit, _contact_nodes, small_signal_capacitance
from ddsim.extract.iv import IVCurve
from ddsim.discretize.coupled import Unknown, apply_contacts_coupled, assemble_coupled_terms, effective_potentials, pack, unknown_index, unpack
from ddsim.mesh.mesh1d import Mesh1D, _assemble
from ddsim.mesh.mesh2d import Mesh2D, tensor_mesh_2d
from ddsim.solve.linear import SparseLU

Models = Callable[[Device], TransportModels]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ErrorEstimate:
    value: float

    estimate: float

    indicator: Array

    refinement: int

    step: float = 0.0

    step_cells: Array = field(default_factory=lambda: np.zeros(0))

    @property
    def linear(self) -> bool:
        return self.step <= CONFIG.errors.max_step

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
    if not isinstance(m, Mesh2D):
        raise ValueError("the error estimate refines the mesh line by line, which needs a tensor mesh. A triangle mesh has no lines to subdivide yet.")
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
    if isinstance(c, OhmicContact | SchottkyContact):
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


def refined(device: Device, k: int = CONFIG.errors.refinement, ky: int | None = None) -> Device:
    x, y = _axes(device)
    j = k if ky is None or y is None else ky
    if min(k, j) < 1 or max(k, j) < 2:
        raise ValueError(f"refining means at least 2 cells for every old one along some direction, got {k} along x and {j} along y")
    return _rebuilt(device, _subdivided(x, k), None if y is None else _subdivided(y, j))


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
    a = apply_contacts_coupled(e, x, device.contact_doping_scaled.data, device.contacts, device.scale, device.carrier_free_nodes, device.material.T, device.degeneracy, thermionic=False)
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
        z = float(device.contact_doping_scaled.data[i])
        if pins_carrier(k[0], z, Carrier.ELECTRON):
            c[unknown_index(i, Unknown.N)] = -f
        if pins_carrier(k[0], z, Carrier.HOLE):
            c[unknown_index(i, Unknown.P)] = f
    return c


def _attributed(old: Array, new: Array, G: Array) -> Array:
    i, t = _weights(old, new)
    n = old.size - 1
    inside = (t > 0.0) & (t < 1.0)
    out = np.zeros((G.shape[0], n))
    size = np.zeros((G.shape[0], n))
    for a in np.flatnonzero(inside):
        out[:, i[a]] += G[:, a]
        size[:, i[a]] += np.abs(G[:, a])
    k = np.where(t >= 1.0, i + 1, i)
    for a in np.flatnonzero(~inside):
        left, right = k[a] - 1, k[a]
        if left < 0:
            out[:, right] += G[:, a]
        elif right >= n:
            out[:, left] += G[:, a]
        else:
            tot = size[:, left] + size[:, right]
            w = np.where(tot > 0.0, size[:, left] / np.where(tot > 0.0, tot, 1.0), 0.5)
            out[:, left] += w * G[:, a]
            out[:, right] += (1.0 - w) * G[:, a]
    return out


def _cells(coarse: Device, fine: Device, there: Array, here: Array, per_node: int) -> Array:
    xo, yo = _axes(coarse)
    xn, yn = _axes(fine)
    g = there.reshape(-1, per_node).sum(axis=1).reshape(-1, xn.size)
    h = here.reshape(-1, per_node).sum(axis=1).reshape(-1, xo.size)
    rows = np.zeros(1, dtype=np.int64) if yo is None or yn is None else _on(yo, yn)
    g[np.ix_(rows, _on(xo, xn))] -= h
    out = _attributed(xo, xn, g)
    if yo is not None and yn is not None:
        out = _attributed(yo, yn, out.T).T
    return np.asarray(out.ravel())


Raw = Callable[[Device, Array], SparseAssembly]

Pins = Callable[[Device, Array], tuple[npt.NDArray[np.int64], Array]]


def _adjoint(device: Device, x: Array, raw: Raw, pins: Pins, c: Array, per_node: int) -> tuple[SparseAssembly, Array, Array]:
    A = raw(device, x)
    p, t = pins(device, x)
    g = np.zeros(x.size)
    np.add.at(g, A.cols, A.values * c[A.rows])
    g[p] = 0.0
    S = apply_dirichlet_nodes(A, x, p.tolist(), t.tolist())
    lu = SparseLU()
    lu.factorize(S.rows, S.cols, S.values, S.shape)
    w = lu.solve(-g, trans="T")
    w[p] = c[p]
    return A, w, np.where(device.charge_volume_scaled > 0.0, np.abs(lu.solve(-S.residual)[::per_node]), 0.0)


def _estimate(coarse: Device, fine: Device, x: Array, y: Array, raw: Raw, pins: Pins, weights: Callable[[Device], Array], per_node: int, k: int) -> ErrorEstimate:
    c = weights(coarse)
    here = c * raw(coarse, x).residual
    value = float(np.sum(here))
    F, z, d = _adjoint(fine, y, raw, pins, weights(fine), per_node)
    there = z * F.residual
    f = 1.0 / (1.0 - float(k) ** -CONFIG.errors.assumed_order)
    return ErrorEstimate(value=value, estimate=f * (float(np.sum(there)) - value), indicator=f * _cells(coarse, fine, there, here, per_node), refinement=k, step=float(np.max(d)), step_cells=_cells(coarse, fine, d, np.zeros(coarse.mesh.n_nodes), 1))


def current_error(device: Device, state: DeviceState, contact: str, models_for: Models = TransportModels.for_device, k: int = CONFIG.errors.refinement, ky: int | None = None) -> ErrorEstimate:
    fine = refined(device, k, ky)
    s = prolonged_state(device, fine, state)

    def raw(d: Device, v: Array) -> SparseAssembly:
        return _raw(d, models_for(d), v)

    return _estimate(device, fine, pack(state.psi.data, state.n.data, state.p.data), pack(s.psi.data, s.n.data, s.p.data), raw, _pins, lambda d: _current_weights(d, contact), 3, max(k, ky or k))


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


def charge_error(device: Device, state: DeviceState, contact: str, k: int = CONFIG.errors.refinement, ky: int | None = None) -> ErrorEstimate:
    fine = refined(device, k, ky)
    y = _prolong(device, fine, state.psi.data)
    i, t = _poisson_pins(fine, y)
    y[i] = t
    return _estimate(device, fine, state.psi.data, y, _poisson_raw, _poisson_pins, lambda d: _charge_weights(d, contact), 1, max(k, ky or k))


def capacitance_error(device: Device, contact: str, delta: float = CONFIG.errors.voltage_step, k: int = CONFIG.errors.refinement, ky: int | None = None) -> ErrorEstimate:
    v = next(c.voltage for c in device.contacts if c.name == _contact_name(device, contact))

    def at(u: float) -> tuple[Device, DeviceState]:
        d = device.with_bias(**{contact: u})
        return d, solve_equilibrium(d, quasi_fermi=frozen_quasi_fermi(d))

    d, s = at(v)
    value = small_signal_capacitance(d, s, contact, quasi_fermi=frozen_quasi_fermi(d))
    a = charge_error(*at(v - delta), contact, k, ky)
    b = charge_error(*at(v + delta), contact, k, ky)
    return ErrorEstimate(value=value, estimate=(b.estimate - a.estimate) / (2.0 * delta), indicator=(b.indicator - a.indicator) / (2.0 * delta), refinement=a.refinement, step=max(a.step, b.step), step_cells=a.step_cells + b.step_cells)


def _contact_name(device: Device, contact: str) -> str:
    _contact_nodes(device, contact)
    return contact


def _marked(score: Array, fraction: float) -> list[int]:
    o = np.argsort(score)[::-1]
    c = np.cumsum(score[o])
    if c[-1] <= 0.0:
        return []
    return [int(v) for v in o[: int(np.searchsorted(c, fraction * c[-1])) + 1]]


Measure = Callable[[Device, DeviceState, int, "int | None"], ErrorEstimate]


def _directional(d: Device, s: DeviceState, measure: Measure) -> tuple[ErrorEstimate, list[ErrorEstimate]]:
    if isinstance(d.mesh, Mesh1D):
        e = measure(d, s, CONFIG.errors.refinement, None)
        return e, [e]
    k = CONFIG.errors.refinement
    a, b = measure(d, s, k, 1), measure(d, s, 1, k)
    return ErrorEstimate(value=a.value, estimate=a.estimate + b.estimate, indicator=a.indicator + b.indicator, refinement=k, step=max(a.step, b.step), step_cells=a.step_cells + b.step_cells), [a, b]


def _proposal(device: Device, parts: list[ErrorEstimate], fraction: float) -> Device:
    x, y = _axes(device)
    linear = all(e.linear for e in parts)

    def score(e: ErrorEstimate) -> Array:
        return np.abs(e.indicator) if linear else e.step_cells

    if y is None:
        return split(device, _marked(score(parts[0]), fraction))
    w = x.size - 1
    a = score(parts[0]).reshape(-1, w).sum(axis=0)
    b = score(parts[1]).reshape(-1, w).sum(axis=1)
    m = _marked(np.concatenate([a, b]), fraction)
    return split(device, [v for v in m if v < w], [v - w for v in m if v >= w])


GOALS = ("current", "capacitance")


def adapt(device: Device, contact: str, rtol: float, goal: str = "current", budget: int = CONFIG.mesh.node_budget, models_for: Models = TransportModels.for_device, max_rounds: int = CONFIG.errors.max_rounds, fraction: float = CONFIG.errors.mark_fraction) -> Adapted:
    if not rtol > 0.0:
        raise ValueError(f"the tolerance has to be above zero, got {rtol:g}")
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}, use one of {GOALS}")
    if device.mesh.n_nodes > budget:
        raise ValueError(f"the starting mesh has {device.mesh.n_nodes} nodes, already over the budget of {budget}")

    def measure(d: Device, s: DeviceState, k: int, ky: int | None) -> ErrorEstimate:
        if goal == "capacitance":
            return capacitance_error(d, contact, k=k, ky=ky)
        return current_error(d, s, contact, models_for, k, ky)

    def solved(d: Device, before: tuple[Device, DeviceState] | None) -> DeviceState | None:
        if goal == "capacitance":
            return solve_equilibrium(d, quasi_fermi=frozen_quasi_fermi(d))
        m = models_for(d)
        t = None if before is None else solve_bias_newton(d, m, guess=prolonged_state(before[0], d, before[1]))
        if t is None or t.newton is None or not t.newton.converged:
            t = solve_bias_ramped(d, m)
        return t if t.newton is not None and t.newton.converged else None

    d = device
    s = solved(d, None)
    if s is None:
        raise RuntimeError("the starting mesh did not converge")
    log: list[tuple[int, float, float]] = []
    for _ in range(max_rounds):
        e, parts = _directional(d, s, measure)
        log.append((d.mesh.n_nodes, e.value, e.estimate))
        if e.linear and abs(e.estimate) <= rtol * abs(e.value):
            return Adapted(d, s, e, True, f"reached {e.relative:.2e} on {d.mesh.n_nodes} nodes", tuple(log))
        f = _proposal(d, parts, fraction)
        if f.mesh.n_nodes == d.mesh.n_nodes:
            return Adapted(d, s, e, False, f"the indicator flags nothing to refine, the estimate is still {e.relative:.2e} against {rtol:.2e}", tuple(log))
        if f.mesh.n_nodes > budget:
            return Adapted(d, s, e, False, f"stopped at the node budget of {budget}: the next mesh needs {f.mesh.n_nodes} nodes, and the estimate is still {e.relative:.2e} against {rtol:.2e}", tuple(log))
        t = solved(f, (d, s))
        if t is None:
            return Adapted(d, s, e, False, f"the refined mesh with {f.mesh.n_nodes} nodes did not converge, so I kept the last one, at {e.relative:.2e}", tuple(log))
        d, s = f, t
    e, _ = _directional(d, s, measure)
    return Adapted(d, s, e, e.linear and abs(e.estimate) <= rtol * abs(e.value), f"stopped after {max_rounds} rounds at {e.relative:.2e}", tuple(log))


@dataclass(frozen=True)
class ErrorFrame:
    index: int

    voltage: float

    value: float

    estimate: float

    linear: bool


def curve_errors(device: Device, curve: IVCurve | CVCurve, models_for: Models = TransportModels.for_device, on_frame: Callable[[object], None] | None = None) -> tuple[ErrorEstimate | None, ...]:
    out: list[ErrorEstimate | None] = []

    def keep(i: int, v: float, e: ErrorEstimate | None) -> None:
        out.append(e)
        if e is not None and on_frame is not None:
            on_frame(ErrorFrame(index=i, voltage=v, value=e.value, estimate=e.estimate, linear=e.linear))

    if isinstance(curve, CVCurve):
        for i, c in enumerate(curve.points):
            e: ErrorEstimate | None = None
            if curve.response is Response.LOW_FREQUENCY:
                e = capacitance_error(device.with_bias(**{curve.contact: c.gate_voltage}), curve.contact)
                if e.value != 0.0:
                    r = c.capacitance / e.value
                    e = replace(e, value=c.capacitance, estimate=r * e.estimate, indicator=r * e.indicator)
            keep(i, c.gate_voltage, e)
        return tuple(out)
    for i, q in enumerate(curve.points):
        keep(i, q.voltage, current_error(device.with_bias(**{curve.contact: q.voltage}), q.state, curve.measured_at or curve.contact, models_for))
    return tuple(out)
