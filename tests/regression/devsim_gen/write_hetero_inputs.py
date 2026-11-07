from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ddsim.core import constants as C
from ddsim.device.builder import Device, build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.pn_diode import hetero_bands
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d
from ddsim.physics.bands import Semiconductor
from tests.regression.devsim_gen import parameters as P

PATH = os.path.join("data", "golden", f"{P.HETERO_NAME}_inputs.json")


def diode() -> Device:
    m = graded_mesh_1d(length=P.HETERO_LENGTH, n_nodes=P.HETERO_N_NODES, refine_at=P.HETERO_JUNCTION, h_min=P.HETERO_H_MIN)
    bands = hetero_bands(m, P.HETERO_JUNCTION, Semiconductor.sige(P.HETERO_GE), Semiconductor.silicon())
    contacts = (OhmicContact(name="anode", node=0, voltage=0.0), OhmicContact(name="cathode", node=m.n_nodes - 1, voltage=0.0))
    return build_device(mesh=m, doping=abrupt_junction(Na=P.HETERO_NA, Nd=P.HETERO_ND, position=P.HETERO_JUNCTION), contacts=contacts, bands=bands)


def inputs() -> dict:
    d = diode()
    assert d.bands is not None
    return {
        "x": d.mesh_1d.x.tolist(),
        "doping": d.net_doping.data.tolist(),
        "d_n": d.bands.d_n.tolist(),
        "d_p": d.bands.d_p.tolist(),
        "eps_r_edge": (d.bands.eps_r * C.EPS_R_SI).tolist(),
        "voltages": list(P.HETERO_VOLTAGES),
        "q": C.q,
        "k_B": C.k_B,
        "T": d.material.T,
        "n_i": d.material.n_i,
        "eps_0": C.eps_0,
        "mu_n": C.MU_N_300,
        "mu_p": C.MU_P_300,
    }


def main() -> int:
    a = argparse.ArgumentParser(description="Write the mesh, doping and band offsets benchmark 21's DEVSIM generator reads.")
    a.add_argument("--out", default=PATH)
    o = a.parse_args()
    x = inputs()
    with open(o.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(x, f, indent=1)
    print(f"wrote {o.out}: {len(x['x'])} nodes, {len(x['voltages'])} biases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
