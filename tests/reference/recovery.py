from __future__ import annotations
import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from ddsim.core import constants as C


def storage_time(Na: float, Nd: float, tau: float, V0: float, VR: float, ratio: float, depletion: bool = True, nodes: int = 2000, steps: int = 5000) -> float:
    VT, ni, q = C.V_T(C.T_ROOM), C.n_i(), C.q
    Dp, Dn = C.MU_P_300 * VT, C.MU_N_300 * VT
    Lp, Ln = math.sqrt(Dp * tau), math.sqrt(Dn * tau)
    pn0, np0 = ni**2 / Nd, ni**2 / Na
    share = 1.0 + (Dn * np0 / Ln) / (Dp * pn0 / Lp)
    Vbi = VT * math.log(Na * Nd / ni**2)
    Neff = Na * Nd / (Na + Nd)

    def charge(V: float) -> float:
        return math.sqrt(2.0 * q * C.eps_Si() * Neff * max(Vbi - V, 1e-12)) if depletion else 0.0

    W = 5.0 * Lp
    x = np.linspace(0.0, W, nodes)
    h = x[1] - x[0]
    edge = pn0 * math.expm1(V0 / VT)
    p = edge * np.sinh((W - x) / Lp) / math.sinh(W / Lp)
    IF = q * share * Dp * edge / Lp / math.tanh(W / Lp)
    R = (VR + V0) / (ratio * IF)
    dt = 1.5 * tau / steps
    A = diags([np.full(nodes - 1, -Dp / h**2), np.full(nodes, 1.0 / dt + 2.0 * Dp / h**2 + 1.0 / tau), np.full(nodes - 1, -Dp / h**2)], [-1, 0, 1]).tolil()
    A[0, 1] = -2.0 * Dp / h**2
    A[nodes - 1, nodes - 2] = 0.0
    A[nodes - 1, nodes - 1] = 1.0
    lu = splu(A.tocsc())
    unit = np.zeros(nodes)
    unit[0] = 1.0
    green = lu.solve(unit)
    t, V = 0.0, V0
    while t < 1.5 * tau:
        before = V
        rest = lu.solve(np.r_[p[:-1] / dt, 0.0])
        for _ in range(60):
            into = (-VR - V) / R + (charge(V) - charge(before)) / dt
            new = rest + green * 2.0 * into / (q * share * h)
            gap = new[0] - pn0 * math.expm1(V / VT)
            slope = green[0] * 2.0 * (-1.0 / R + (charge(V + 1e-7) - charge(V)) / 1e-7 / dt) / (q * share * h) - pn0 * math.exp(V / VT) / VT
            V -= gap / slope
            if abs(gap / slope) < 1e-13:
                break
        p = new
        t += dt
        if V <= 0.0:
            return t - dt * V / (V - before)
    raise ValueError("the junction voltage never crossed zero")
