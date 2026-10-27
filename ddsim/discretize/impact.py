from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.coupled import UNKNOWNS_PER_NODE, Unknown, unpack
from ddsim.discretize.geometry import EdgeGeometry
from ddsim.physics.bernoulli import B, dB_dx
from ddsim.physics.mobility import EdgeDiffusivity, diffusivity_at, diffusivity_tangent

Array = npt.NDArray[np.float64]


def _alpha(E: Array, a: Array | float, b: Array | float) -> tuple[Array, Array]:
    ok = E > 0.0
    e = np.where(ok, E, 1.0)
    v = np.where(ok, a * np.exp(-b / e), 0.0)
    return v, np.where(ok, v * b / (e * e), 0.0)


@dataclass(frozen=True)
class ImpactIonization:
    a_n: float
    b_n: float
    a_p_low: float
    b_p_low: float
    a_p_high: float
    b_p_high: float
    switch: float

    @classmethod
    def for_silicon(cls, scale: ScaleFactors) -> ImpactIonization:
        x, f = scale.x_0, scale.x_0 / scale.psi_0
        return cls(a_n=C.II_A_N * x, b_n=C.II_B_N * f, a_p_low=C.II_A_P_LOW * x, b_p_low=C.II_B_P_LOW * f, a_p_high=C.II_A_P_HIGH * x, b_p_high=C.II_B_P_HIGH * f, switch=C.II_E_P_SWITCH * f)

    def alpha_n(self, E: Array) -> tuple[Array, Array]:
        return _alpha(E, self.a_n, self.b_n)

    def alpha_p(self, E: Array) -> tuple[Array, Array]:
        high = E >= self.switch
        return _alpha(E, np.where(high, self.a_p_high, self.a_p_low), np.where(high, self.b_p_high, self.b_p_low))


def add_impact(assembly: SparseAssembly, x: Array, h: Array, geometry: EdgeGeometry, Dn: EdgeDiffusivity, Dp: EdgeDiffusivity, model: ImpactIonization, dimension: int) -> SparseAssembly:
    psi, n, p = unpack(x)
    a, b = geometry.ends(h.size)
    d = psi[b] - psi[a]
    share = h * geometry.carrier_face / (2 * dimension)

    bp, bm = np.asarray(B(d), dtype=np.float64), np.asarray(B(-d), dtype=np.float64)
    dbp, dbm = np.asarray(dB_dx(d), dtype=np.float64), np.asarray(dB_dx(-d), dtype=np.float64)

    kn = diffusivity_at(Dn, d, h) / h
    fn = bp * n[b] - bm * n[a]
    jn = kn * fn
    jn_d = kn * (dbp * n[b] + dbm * n[a])
    tn = diffusivity_tangent(Dn, d, h)
    if tn is not None:
        jn_d = jn_d + tn / h * fn
    jn_na = -kn * bm
    jn_nb = kn * bp

    kp = diffusivity_at(Dp, d, h) / h
    fp = bp * p[a] - bm * p[b]
    jp = kp * fp
    jp_d = kp * (dbp * p[a] + dbm * p[b])
    tp = diffusivity_tangent(Dp, d, h)
    if tp is not None:
        jp_d = jp_d + tp / h * fp
    jp_pa = kp * bp
    jp_pb = -kp * bm

    E = np.abs(d) / h
    E_d = np.sign(d) / h
    an, an_E = model.alpha_n(E)
    ap, ap_E = model.alpha_p(E)
    sn, sp = np.sign(jn), np.sign(jp)

    G = an * np.abs(jn) + ap * np.abs(jp)
    G_d = an_E * E_d * np.abs(jn) + an * sn * jn_d + ap_E * E_d * np.abs(jp) + ap * sp * jp_d

    res = assembly.residual.copy()
    rows: list[npt.NDArray[np.int64]] = []
    cols: list[npt.NDArray[np.int64]] = []
    vals: list[Array] = []
    k = UNKNOWNS_PER_NODE
    psi_a, psi_b = a * k + int(Unknown.PSI), b * k + int(Unknown.PSI)
    n_a, n_b = a * k + int(Unknown.N), b * k + int(Unknown.N)
    p_a, p_b = a * k + int(Unknown.P), b * k + int(Unknown.P)
    for family in (Unknown.N, Unknown.P):
        for end in (a, b):
            row = end * k + int(family)
            np.add.at(res, row, -G * share)
            for col, dG in ((psi_a, -G_d), (psi_b, G_d), (n_a, an * sn * jn_na), (n_b, an * sn * jn_nb), (p_a, ap * sp * jp_pa), (p_b, ap * sp * jp_pb)):
                rows.append(row)
                cols.append(col)
                vals.append(-dG * share)
    return SparseAssembly(residual=res, rows=np.concatenate([assembly.rows, *rows]), cols=np.concatenate([assembly.cols, *cols]), values=np.concatenate([assembly.values, *vals]), shape=assembly.shape)


def impact_residual(x: Array, h: Array, geometry: EdgeGeometry, Dn: EdgeDiffusivity, Dp: EdgeDiffusivity, model: ImpactIonization, dimension: int) -> Array:
    none = np.zeros(0, dtype=np.int64)
    zero = SparseAssembly(residual=np.zeros(x.size), rows=none, cols=none, values=np.zeros(0), shape=(x.size, x.size))
    return add_impact(zero, x, h, geometry, Dn, Dp, model, dimension).residual
