from __future__ import annotations
from ddsim.core.config import CONFIG
import math
from typing import Any
import numpy as np
from scipy.special import erfcinv as _erfcinv
from ddsim.core import constants as C; from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Along, Erfc, Gaussian, Mirrored, Uniform
from ddsim.device.regions import stacked_regions
from ddsim.discretize.boundary import Contact, GateContact, OhmicPlate
from ddsim.mesh.mesh1d import (
    Mesh1D,
    graded_mesh_1d,
    stacked_mesh_1d,
    uniform_mesh_1d,
)
from ddsim.mesh.mesh2d import Mesh2D, tensor_mesh_2d

SOURCE = 'source'


DRAIN = "drain"
GATE = "gate"
BODY = 'body'

def _junction_mesh(
    length: float, n_nodes: int, refine_at:float, h_min: float
) -> Mesh1D:
    if h_min * (n_nodes - 1) >= length:
        return uniform_mesh_1d(length=length, n_nodes=n_nodes)
    return graded_mesh_1d(
        length=length, n_nodes=n_nodes, refine_at=refine_at, h_min=h_min
    )


def implant_lengths(x_j: float, lateral_diffusion: float, sd_peak: float, Na: float) -> tuple[float, float]:
    d = x_j / math.sqrt(2.0 * math.log(sd_peak / Na))
    c = lateral_diffusion / float(_erfcinv(2.0*Na/sd_peak))
    return d, c


def nmos(L_gate: float = CONFIG.nmos.L_gate, sd_length: float = CONFIG.nmos.sd_length, contact_length: float = CONFIG.nmos.contact_length, substrate_doping:float = CONFIG.nmos.substrate_doping, sd_peak: float = CONFIG.nmos.sd_peak, x_j: float = CONFIG.nmos.x_j, lateral_diffusion:float = CONFIG.nmos.lateral_diffusion, t_ox:float = CONFIG.nmos.t_ox, t_si: float = CONFIG.nmos.t_si, n_contact: int = CONFIG.nmos.n_contact, n_sd: int = CONFIG.nmos.n_sd, n_channel: int = CONFIG.nmos.n_channel, n_silicon: int = CONFIG.nmos.n_silicon, n_oxide: int = CONFIG.nmos.n_oxide, h_min_x: float = CONFIG.nmos.h_min_x, h_min_y:float = CONFIG.nmos.h_min_y, gate_voltage: float = 0.0, drain_voltage: float = 0.0, source_voltage: float = 0.0, body_voltage: float = 0.0, work_function: float = C.PHI_M_N_POLY, material: Material | None = None, degenerate:bool = True, Q_f: float = 0.0, T: float = C.T_ROOM, incomplete_ionization: bool = False,) -> Device:
    for k, d2 in (
        ("L_gate", L_gate),
        ("sd_length", sd_length),
        ('contact_length', contact_length),
        ('x_j', x_j),
        ('lateral_diffusion', lateral_diffusion),
        ('t_ox', t_ox),
        ("t_si", t_si),
    ):
        if d2 <= 0.0:
            raise ValueError(f"{k} must be positive, got {d2}")
    if substrate_doping >= 0.0:
        raise ValueError(
            f"nmos builds an n-channel device, so the body must be p-type and "
            f"substrate_doping negative, got {substrate_doping:g}. The sign "
            "convention is net doping, Nd - Na, everywhere in this codebase."
        )

    Na = -substrate_doping
    if sd_peak <= Na:
        raise ValueError(
            f"sd_peak must exceed the substrate doping, got sd_peak="
            f"{sd_peak:g} against {Na:g} cm^-3. A source no heavier than the "
            "body it sits in makes no junction, and there would be no depth "
            "at which to put one."
        )
    if x_j >= t_si:
        raise ValueError(
            f"x_j must be less than t_si, got x_j={x_j:g} cm in {t_si:g} cm "
            'of silicon. The source would reach the body contact.'
        )

    if 2.0 * lateral_diffusion >= L_gate:
        raise ValueError(
            f"no channel is left: the source and drain each reach "
            f"{lateral_diffusion:g} cm under a gate {L_gate:g} cm long, so "
            "the two junctions meet or cross. That geometry is a short "
            'circuit, and a solver would happily return one instead of an '
            "error. Shrink lateral_diffusion or lengthen the gate."
        )

    if contact_length >= sd_length:
        raise ValueError(
            f"contact_length must be less than sd_length, got "
            f"{contact_length:g} against {sd_length:g} cm. A contact reaching "
            'the mask edge pins the junction itself, and the built in '
            "potential stops being something the solve works out."
        )
    if n_contact < 2 or n_sd < 2 or n_channel < 2 or n_silicon < 2 or n_oxide < 2:
        raise ValueError(
            'every mesh segment needs at least 2 nodes, got n_contact='
            f"{n_contact}, n_sd={n_sd}, n_channel={n_channel}, n_silicon="
            f"{n_silicon}, n_oxide={n_oxide}. A segment with one node has no "
            'extent to carry a field across.'
        )

    w = 2.0 * sd_length+L_gate

    s = sd_length - contact_length
    ii = 0.5 * L_gate

    d = stacked_mesh_1d(uniform_mesh_1d(length=contact_length, n_nodes=n_contact), _junction_mesh(s, n_sd, refine_at=s, h_min=h_min_x), _junction_mesh(ii, n_channel, refine_at=0.0, h_min=h_min_x), _junction_mesh(ii, n_channel, refine_at=ii, h_min=h_min_x), _junction_mesh(s, n_sd, refine_at=0.0, h_min=h_min_x), uniform_mesh_1d(length=contact_length, n_nodes=n_contact),)

    z = stacked_mesh_1d(
        graded_mesh_1d(
            length=t_si, n_nodes=n_silicon, refine_at=t_si, h_min=h_min_y
        ),
        uniform_mesh_1d(length=t_ox, n_nodes=n_oxide),
    )

    tmp = tensor_mesh_2d(d, z)
    j = stacked_regions(tmp, interface_y=t_si)
    rr = n_contact - 1

    w2 = n_contact+n_sd - 2

    k2 = w2 + 2 * (n_channel - 1)
    item = k2+n_sd-1

    tt = n_silicon - 1
    r = tmp.ny - 1
    b = (OhmicPlate(name=SOURCE, nodes=tuple(tmp.node_at(arr, tt) for arr in range(rr + 1)), voltage=source_voltage,), OhmicPlate(name=DRAIN, nodes=tuple(tmp.node_at(val2, tt) for val2 in range(item, tmp.nx)), voltage=drain_voltage,), GateContact(name=GATE, nodes=tuple(tmp.node_at(i, r) for i in range(w2, k2 + 1)), voltage=gate_voltage, work_function=work_function,), OhmicPlate(name=BODY, nodes=tuple(tmp.node_at(h, 0) for h in range(tmp.nx)), voltage=body_voltage,),)

    m, xs = implant_lengths(x_j, lateral_diffusion, sd_peak, Na)
    y2 = (Along(Erfc(peak=0.5, position=sd_length, length=xs), "x") * Along(Gaussian(peak=1.0, centre=t_si, sigma=m), "y") * sd_peak)
    out2 = Uniform(substrate_doping) + y2 + Mirrored(y2, 0.5 * w)
    return build_device(mesh=tmp, doping=out2, contacts=b, material=material, regions=j, degenerate=degenerate, Q_f=Q_f, T=T, incomplete_ionization=incomplete_ionization,)


def narrow_nmos(width: float, field_width: float = 3e-5, t_field: float = 2e-5, n_width: int = 6, n_field: int = 6, n_field_oxide: int = 4, gate_voltage: float = 0.0, drain_voltage: float = 0.0, **process: Any) -> Device:
    from dataclasses import replace
    from ddsim.device.regions import OXIDE, SILICON, box_regions
    from ddsim.mesh.mesh3d import extruded
    t_ox = process.get("t_ox", CONFIG.nmos.t_ox)
    t_si = process.get("t_si", CONFIG.nmos.t_si)
    if width <= 0.0 or field_width <= 0.0:
        raise ValueError(f"width and field_width must be positive, got {width:g} and {field_width:g} cm")
    if t_field <= t_ox:
        raise ValueError(f"the field oxide has to be thicker than the gate oxide, got t_field={t_field:g} against t_ox={t_ox:g} cm")
    d = nmos(gate_voltage=gate_voltage, drain_voltage=drain_voltage, **process)
    assert isinstance(d.mesh, Mesh2D)
    y = stacked_mesh_1d(d.mesh.y_axis, uniform_mesh_1d(length=t_field - t_ox, n_nodes=n_field_oxide))
    field = uniform_mesh_1d(length=field_width, n_nodes=n_field)
    s = float(field.h[0])
    half = uniform_mesh_1d(length=0.5 * width, n_nodes=n_width) if 0.5 * width / (n_width - 1) <= 1.5 * s else graded_mesh_1d(length=0.5 * width, n_nodes=n_width, refine_at=0.5 * width, h_min=s)
    z = stacked_mesh_1d(half, field)
    base = tensor_mesh_2d(d.mesh.x_axis, y)
    m = extruded(base, z)
    top = t_si + t_field
    cells = np.full((z.n_nodes - 1, y.n_nodes - 1, base.nx - 1), SILICON, dtype=np.int64)
    cells[:, y.x[:-1] >= t_si * (1 - 1e-12), :] = OXIDE
    gate = next(c for c in d.contacts if isinstance(c, GateContact))
    lo, hi = d.mesh.node_x[list(gate.nodes)].min(), d.mesh.node_x[list(gate.nodes)].max()
    tol = 1e-9 * top
    over = (m.node_x >= lo - tol) & (m.node_x <= hi + tol)
    active = m.node_z <= 0.5 * width + tol
    metal = over & (((m.node_y >= t_si + t_ox - tol) & active) | (m.node_y >= top - tol))
    contacts: list[Contact] = [replace(gate, nodes=tuple(np.flatnonzero(metal).tolist()))]
    n = base.n_nodes
    for c in d.contacts:
        if isinstance(c, GateContact):
            continue
        assert isinstance(c, OhmicPlate)
        layers = range(z.n_nodes) if c.name == BODY else np.flatnonzero(z.x <= 0.5 * width + tol)
        flat = [int(base.node_at(int(np.argmin(np.abs(d.mesh.x_axis.x - d.mesh.node_x[v]))), int(np.argmin(np.abs(y.x - d.mesh.node_y[v]))))) for v in c.nodes]
        contacts.append(replace(c, nodes=tuple(v + int(k) * n for k in layers for v in flat)))
    return build_device(mesh=m, doping=d.doping, contacts=tuple(contacts), material=d.material, regions=box_regions(m, cells), degenerate=d.degenerate, T=d.material.T)
