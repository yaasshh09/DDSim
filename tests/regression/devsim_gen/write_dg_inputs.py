from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ddsim.core import constants as C
from ddsim.device.mos_cap import mos_cap
from ddsim.device.schrodinger_poisson import column
from tests.regression.devsim_gen import parameters as P

PATH = os.path.join("data", "golden", f"{P.DG_NAME}_inputs.json")


def capacitor(gate: float):
    return mos_cap(gate_voltage=gate, t_ox=P.DG_T_OX, substrate_doping=P.DG_SUBSTRATE, h_min=P.DG_H_MIN, n_silicon=P.DG_N_SILICON)


def inputs() -> dict:
    columns = [column(capacitor(v)) for v in P.DG_GATES]
    c = columns[0]
    d = capacitor(P.DG_GATES[0])
    return {
        "y": c.y.tolist(),
        "interface": int(np.max(np.flatnonzero(c.silicon > 0.0))),
        "doping": c.doping.tolist(),
        "body": float(c.psi_classical[0]),
        "gates": list(P.DG_GATES),
        "gate_potentials": [float(k.psi_classical[-1]) for k in columns],
        "q": C.q,
        "k_B": C.k_B,
        "T": d.material.T,
        "n_i": d.material.n_i,
        "eps_si": C.eps_Si(),
        "eps_ox": C.eps_ox(),
        "hbar": C.h / (2 * math.pi),
        "m_0": C.m_0,
    }


def main() -> int:
    a = argparse.ArgumentParser(description="Write the mesh and boundary values benchmark 20's DEVSIM generator reads.")
    a.add_argument("--out", default=PATH)
    o = a.parse_args()
    x = inputs()
    with open(o.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(x, f, indent=1)
    print(f"wrote {o.out}: {len(x['y'])} nodes, interface at node {x['interface']}, {len(x['gates'])} gate biases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
