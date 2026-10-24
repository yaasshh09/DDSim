from __future__ import annotations
from ddsim.core import constants as C
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device, Material, build_device
import math
import numpy as np
from ddsim.device.doping import Along, Coordinates, DopingProfile, Gaussian, Layers, Product, Sum, Uniform, Window
from ddsim.discretize.boundary import OhmicContact, OhmicPlate
from ddsim.mesh.mesh1d import graded_mesh_1d_through
from ddsim.mesh.mesh2d import tensor_mesh_2d

EMITTER = "emitter"

BASE = 'base'

COLLECTOR = "collector"


def npn_1d(emitter_doping: float = CONFIG.npn.emitter_doping, base_doping: float = CONFIG.npn.base_doping, collector_doping: float = CONFIG.npn.collector_doping, emitter_length: float = CONFIG.npn.emitter_length, base_width: float = CONFIG.npn.base_width, collector_length: float = CONFIG.npn.collector_length, n_nodes: int = CONFIG.npn.n_nodes, h_min: float = CONFIG.npn.h_min, emitter_voltage: float = 0.0, base_voltage: float = 0.0, collector_voltage: float = 0.0, material: Material | None = None, T: float = C.T_ROOM,) -> Device:
    for k, v in (("emitter_doping", emitter_doping), ("base_doping", base_doping), ('collector_doping', collector_doping), ("emitter_length", emitter_length), ('base_width', base_width), ("collector_length", collector_length)):
        if v <= 0.0:
            raise ValueError(f"{k} must be positive, got {v:g}")
    a = emitter_length
    b = a + base_width
    m = 0.5 * (a + b)
    z = graded_mesh_1d_through(length=b + collector_length, n_nodes=n_nodes, lines=(a, m, b), points=(a, b), h_min=h_min)
    i = int(abs(z.x - m).argmin())
    x = (
        OhmicContact(name=EMITTER, node=0, voltage=emitter_voltage),
        OhmicContact(name=BASE, node=i, voltage=base_voltage, majority_only=True),
        OhmicContact(name=COLLECTOR, node=z.n_nodes - 1, voltage=collector_voltage),
    )
    return build_device(mesh=z, doping=Layers(boundaries=(a, b), values=(emitter_doping, -base_doping, collector_doping)), contacts=x, material=material, T=T)


def _crossings(f: DopingProfile, at: tuple[float | None, float | None], span: float) -> tuple[float, ...]:
    t = np.linspace(0.0, span, 4001)
    x = t if at[0] is None else np.full_like(t, at[0])
    y = t if at[1] is None else np.full_like(t, at[1])
    v = f(Coordinates(x, y))
    k = np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:]))
    return tuple(float(t[i] - v[i] * (t[i + 1] - t[i]) / (v[i + 1] - v[i])) for i in k)


def npn_2d(width: float = CONFIG.npn_2d.width, depth: float = CONFIG.npn_2d.depth, emitter_window: float = CONFIG.npn_2d.emitter_window, emitter_contact: float = CONFIG.npn_2d.emitter_contact, base_contact: float = CONFIG.npn_2d.base_contact, epi_doping: float = CONFIG.npn_2d.epi_doping, buried_peak: float = CONFIG.npn_2d.buried_peak, buried_sigma: float = CONFIG.npn_2d.buried_sigma, base_peak: float = CONFIG.npn_2d.base_peak, base_sigma: float = CONFIG.npn_2d.base_sigma, emitter_peak: float = CONFIG.npn_2d.emitter_peak, emitter_sigma: float = CONFIG.npn_2d.emitter_sigma, lateral_length: float = CONFIG.npn_2d.lateral_length, nx: int = CONFIG.npn_2d.nx, ny: int = CONFIG.npn_2d.ny, h_min_x: float = CONFIG.npn_2d.h_min_x, h_min_y: float = CONFIG.npn_2d.h_min_y, emitter_voltage: float = 0.0, base_voltage: float = 0.0, collector_voltage: float = 0.0, degenerate: bool = False, material: Material | None = None, T: float = C.T_ROOM,) -> Device:
    if not 0.0 < emitter_contact <= emitter_window < base_contact < width:
        raise ValueError(
            f"the emitter contact has to sit inside the emitter window and both left of the base contact, got emitter contact to {emitter_contact:g}, "
            f"window to {emitter_window:g} and base contact from {base_contact:g} cm on a device {width:g} cm wide"
        )
    for k, v in (("depth", depth), ("epi_doping", epi_doping), ("buried_sigma", buried_sigma), ("base_peak", base_peak), ("base_sigma", base_sigma), ("emitter_peak", emitter_peak), ("emitter_sigma", emitter_sigma), ("lateral_length", lateral_length)):
        if v <= 0.0:
            raise ValueError(f"{k} must be positive, got {v:g}")
    if base_peak <= epi_doping:
        raise ValueError(f"the base peak has to beat the epi doping or there's no base, got {base_peak:g} against {epi_doping:g} cm^-3")
    f: DopingProfile = Sum((
        Uniform(epi_doping),
        Along(Gaussian(peak=buried_peak, centre=0.0, sigma=buried_sigma), "y"),
        Along(Gaussian(peak=-base_peak, centre=depth, sigma=base_sigma), "y"),
        Product((Along(Window(low=-math.inf, high=emitter_window, edge="erfc", length=lateral_length), "x"), Along(Gaussian(peak=emitter_peak, centre=depth, sigma=emitter_sigma), "y"))),
    ))
    a = _crossings(f, (0.0, None), depth)
    b = _crossings(f, (None, depth), width)
    if len(a) != 2 or len(b) != 1:
        raise ValueError(f"these profiles don't make an npn: under the emitter the doping changes sign at {a} cm and along the surface at {b} cm")
    xm = graded_mesh_1d_through(length=width, n_nodes=nx, lines=(emitter_contact, base_contact), points=b, h_min=h_min_x)
    ym = graded_mesh_1d_through(length=depth, n_nodes=ny, lines=(), points=a, h_min=h_min_y)
    m = tensor_mesh_2d(xm, ym)
    top = m.ny - 1
    x = (
        OhmicPlate(name=EMITTER, nodes=tuple(m.node_at(i, top) for i in range(m.nx) if xm.x[i] <= emitter_contact * (1 + 1e-12)), voltage=emitter_voltage),
        OhmicPlate(name=BASE, nodes=tuple(m.node_at(i, top) for i in range(m.nx) if xm.x[i] >= base_contact * (1 - 1e-12)), voltage=base_voltage),
        OhmicPlate(name=COLLECTOR, nodes=tuple(m.node_at(i, 0) for i in range(m.nx)), voltage=collector_voltage),
    )
    return build_device(mesh=m, doping=f, contacts=x, material=material, degenerate=degenerate, T=T)
