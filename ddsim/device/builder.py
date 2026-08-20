"""Device composition. Geometry and doping in, a Device out.

A Device is a specification, not a solution. It knows its mesh, its material,
its contacts, and how to evaluate its doping anywhere. It holds no solver
state, which is why solving returns a DeviceState rather than mutating it.

The doping profile is kept as a callable rather than collapsed into an array,
per docs/03-architecture.md. Phase 5 refines the mesh adaptively, so the
profile has to stay re-evaluable on a mesh that does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.device.doping import DopingProfile
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import Mesh1D


@dataclass(frozen=True)
class Material:
    """Material parameters at a fixed temperature."""

    name: str
    """Material name, for reporting."""

    T: float
    """Temperature [K]."""

    eps: float
    """Permittivity [F/cm]."""

    n_i: float
    """Intrinsic carrier density [cm^-3]."""

    @classmethod
    def silicon(cls, T: float = C.T_ROOM) -> Material:
        """Silicon at temperature T [K], from docs/06-constants.md."""
        return cls(name="silicon", T=T, eps=C.eps_Si(), n_i=C.n_i(T))


@dataclass(frozen=True)
class Device:
    """A 1D device specification."""

    mesh: Mesh1D
    """The mesh, positions in cm."""

    doping: DopingProfile
    """Net doping as a callable of position [cm], returning [cm^-3]."""

    material: Material
    """Material parameters."""

    contacts: tuple[OhmicContact, ...]
    """Ohmic contacts, each pinned to a mesh node."""

    scale: ScaleFactors
    """The de Mari scale factors this device is solved in."""

    @property
    def net_doping(self) -> Field:
        """Net doping on the mesh nodes [cm^-3], physical units."""
        return Field(
            self.doping(self.mesh.x),
            "cm^-3",
            ScalingState.PHYSICAL,
            Location.NODE,
            name="net_doping",
        )

    @property
    def net_doping_scaled(self) -> Field:
        """Net doping on the mesh nodes [cm^-3], scaled by C_0."""
        return self.net_doping.to_scaled(self.scale)

    def __repr__(self) -> str:
        names = ", ".join(contact.name for contact in self.contacts)
        return (
            f"Device {self.material.name} {self.mesh.n_nodes} nodes "
            f"length={self.mesh.length:.3e} cm contacts=({names})"
        )


def build_device(
    mesh: Mesh1D,
    doping: DopingProfile,
    contacts: tuple[OhmicContact, ...],
    material: Material | None = None,
    C_0: float | None = None,
) -> Device:
    """Assemble a Device and check that it is self consistent.

    Args:
        mesh: the 1D mesh.
        doping: net doping profile, a callable of position.
        contacts: at least one ohmic contact.
        material: defaults to silicon at 300 K.
        C_0: reference concentration for scaling [cm^-3], defaults to n_i.

    C_0 is exposed here so that Phase 5 can switch to max|net doping| in one
    place if conditioning demands it, per docs/02-numerics.md.
    """
    if material is None:
        material = Material.silicon()

    if not contacts:
        raise ValueError("a device needs at least one contact")

    for contact in contacts:
        if not 0 <= contact.node < mesh.n_nodes:
            raise IndexError(
                f"contact {contact.name!r} sits on node {contact.node}, "
                f"but the mesh has {mesh.n_nodes} nodes"
            )

    names = [contact.name for contact in contacts]
    if len(set(names)) != len(names):
        raise ValueError(f"contact names must be unique, got {names}")

    scale = ScaleFactors.for_silicon(
        T=material.T,
        C_0=material.n_i if C_0 is None else C_0,
        eps=material.eps,
    )

    return Device(
        mesh=mesh,
        doping=doping,
        material=material,
        contacts=contacts,
        scale=scale,
    )
