from __future__ import annotations
from ddsim.core import constants as C
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Layers
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d_through

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
