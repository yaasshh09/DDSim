from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import GateContact, apply_dirichlet_nodes
from ddsim.discretize.coupled import (
    Unknown,
    assemble_coupled_terms,
    coupled_mass,
    pack,
    row_weights,
    scale_rows,
    unknown_index,
    unpack,
)
from ddsim.solve.linear import SparseLU


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


def admittance(device: Device, state: DeviceState, frequency: float, models: TransportModels | None = None, width: float = 1.0,) -> Admittance:
    if models is None:
        models = TransportModels.for_device(device)
    if models.surface is not None:
        raise NotImplementedError(
            "surface mobility is held fixed at the DC state, so the Jacobian "
            "misses its dependence on psi, n and p and the admittance would "
            "be quietly wrong. Solve without surface=True."
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
    lu = SparseLU()
    lu.factorize(np.concatenate([A.rows, r]), np.concatenate([A.cols, c]), np.concatenate([A.values.astype(np.complex128), 1j * omega * m / w[r]]), A.shape,)

    J = coo_matrix((bare.values, (bare.rows, bare.cols)), shape=bare.shape).tocsr()
    names = tuple(k.name for k in device.contacts)
    Y = np.zeros((len(names), len(names)), dtype=np.complex128)
    unit = s.J_0 * s.x_0 ** (device.dimension - 1) / width
    for j, driven in enumerate(device.contacts):
        t = [0.0] * len(pins)
        for i in driven.nodes:
            t[pins.index(unknown_index(i, Unknown.PSI))] = 1.0 / s.psi_0
        dx = lu.solve(-stamped(t).residual.astype(np.complex128))
        dpsi, dn, dp = unpack(np.asarray(J @ dx, dtype=np.complex128))
        for i, k in enumerate(device.contacts):
            q = complex(np.sum(dpsi[list(k.nodes)]))
            g = 0.0 if isinstance(k, GateContact) else complex(np.sum(-dn[list(k.nodes)] + dp[list(k.nodes)]))
            Y[i, j] = (g + 1j * omega * q) * unit
    return Admittance(frequency=frequency, contacts=names, Y=Y)
