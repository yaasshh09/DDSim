from __future__ import annotations
from ddsim.core import constants as C
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Uniform
from ddsim.discretize.boundary import OhmicContact, SchottkyContact
from ddsim.mesh.mesh1d import graded_mesh_1d


def schottky_diode(Nd: float = CONFIG.schottky.Nd, length: float = CONFIG.schottky.length, barrier: float = CONFIG.schottky.barrier, n_nodes: int = CONFIG.schottky.n_nodes, h_min: float = CONFIG.schottky.h_min, anode_voltage: float = 0.0, cathode_voltage: float = 0.0, material: Material | None = None, T: float = C.T_ROOM,) -> Device:
    if Nd <= 0.0:
        raise ValueError(f"Nd must be positive, got {Nd:g}. The diode is a metal on n-type silicon.")
    z = graded_mesh_1d(length=length, n_nodes=n_nodes, refine_at=0.0, h_min=h_min)
    x = (
        SchottkyContact(name="anode", node=0, voltage=anode_voltage, barrier=barrier),
        OhmicContact(name="cathode", node=n_nodes - 1, voltage=cathode_voltage),
    )
    return build_device(mesh=z, doping=Uniform(value=Nd), contacts=x, material=material, T=T)
