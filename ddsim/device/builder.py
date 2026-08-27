"""Device composition. Geometry and doping in, a Device out.

A Device is a specification, not a solution. It knows its mesh, its material,
its contacts, and how to evaluate its doping anywhere. It holds no solver
state, which is why solving returns a DeviceState rather than mutating it.

The doping profile is kept as a callable rather than collapsed into an array,
per docs/03-architecture.md. Phase 5 refines the mesh adaptively, so the
profile has to stay re-evaluable on a mesh that does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.device.doping import Coordinates, DopingProfile
from ddsim.device.regions import RegionMap
from ddsim.discretize.boundary import Contact, GateContact, SemiconductorContact
from ddsim.discretize.geometry import ScaledMesh
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.mesh.mesh2d import Mesh2D

AnyMesh = Mesh1D | Mesh2D
"""A mesh of either dimension. Everything below works on both."""


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
    """A device specification, in one dimension or two."""

    mesh: AnyMesh
    """The mesh, positions in cm."""

    doping: DopingProfile
    """Net doping as a callable of position [cm], returning [cm^-3].

    Evaluated at the coordinates of every node, which on a grid is two arrays
    and on a line is one. A profile that reads x alone is handed exactly the
    array it used to be handed, so every device built before Phase 5 produces
    the same doping to the last bit. A source implant reads both, because it
    is Gaussian in depth and bounded laterally.
    """

    material: Material
    """Material parameters."""

    contacts: tuple[Contact, ...]
    """The device terminals: ohmic points, ohmic plates and gates."""

    scale: ScaleFactors
    """The de Mari scale factors this device is solved in."""

    regions: RegionMap | None = None
    """Which cell is which material, on a device made of more than one.

    None means the whole device is the one semiconductor, which is every 1D
    device built so far and every single material 2D one. A MOS stack passes
    the map, and both things that follow from it, the per edge permittivity
    and the semiconductor volume, are read from here.
    """

    @cached_property
    def net_doping(self) -> Field:
        """Net doping on the mesh nodes [cm^-3], physical units.

        Zero wherever there is no semiconductor. The zero charge volume there
        already makes that true in the equations, so this is for everything
        that reads the doping for some other reason: a plot, a lifetime, a
        contact potential. An array reading 1e16 in the middle of an insulator
        is a trap laid for all of them.

        Cached, because a Device is frozen: the mesh and the profile that this
        is evaluated from cannot change under it. with_bias returns a new
        Device, which starts with an empty cache, so a rebiased device never
        inherits a doping array from the one it was copied from.
        """
        values = self.doping(self.node_coordinates)
        if self.regions is not None:
            values = np.where(self.regions.semiconductor_volume > 0.0, values, 0.0)
        return Field(
            values,
            "cm^-3",
            ScalingState.PHYSICAL,
            Location.NODE,
            name="net_doping",
        )

    @property
    def node_coordinates(self) -> Coordinates:
        """Where every node is [cm], as the doping profile is asked for it.

        A 1D mesh is a line along x and gets no y at all rather than a column
        of zeros, so a depth dependent profile on one fails loudly instead of
        reading its peak everywhere.
        """
        depth = None if isinstance(self.mesh, Mesh1D) else self.mesh.node_y
        return Coordinates(self.node_x, depth)

    @property
    def node_x(self) -> npt.NDArray[np.float64]:
        """x position of every node [cm].

        The two meshes name it differently, `x` on a line and `node_x` on a
        grid, because on a grid it is one of two coordinates and calling it x
        alone would read as the axis. The doping profile wants one array of
        positions either way, so the difference stops here.
        """
        if isinstance(self.mesh, Mesh1D):
            return self.mesh.x
        return self.mesh.node_x

    @cached_property
    def scaled_mesh(self) -> ScaledMesh:
        """The mesh in the units the assemblies work in.

        The mesh scales itself, because the power of x_0 on the dual volume is
        the dimension and no call site should have to know which one it is in.
        See ScaledMesh in discretize/geometry.py.
        """
        if isinstance(self.mesh, Mesh1D):
            return self.mesh.scaled(self.scale)
        if self.regions is None:
            return self.mesh.scaled(self.scale)
        return self.mesh.scaled(
            self.scale,
            eps_r=self.regions.eps_r,
            semiconductor_face=self.regions.semiconductor_face,
        )

    @cached_property
    def charge_volume_scaled(self) -> npt.NDArray[np.float64]:
        """The part of each dual cell that carries charge [1], scaled.

        The whole dual cell in a single material device. In a MOS stack it is
        zero in the oxide, which turns those Poisson rows into the bare
        Laplacian an insulator wants, and half a cell at the interface, where
        half the cell is silicon and holds the inversion layer.
        """
        if self.regions is None:
            return self.scaled_mesh.volume
        power = 1 if isinstance(self.mesh, Mesh1D) else 2
        return np.asarray(self.regions.semiconductor_volume / self.scale.x_0**power)

    @cached_property
    def ohmic_contacts(self) -> tuple[SemiconductorContact, ...]:
        """The contacts, if every one of them touches semiconductor.

        A point and a plate are the same thing to the uncoupled Gummel
        blocks: each pins a density at every node the contact covers, and a
        point contact covers one node. A gate is not, and never can be. It
        sits on an insulator, so there is no doping under it to read and no
        carrier density to pin, and a block that silently left a terminal out
        would converge and mean nothing. Those blocks ask through here and get
        a refusal they can read.

        The coupled path does not come through here any more. It applies every
        contact in one pass through discretize.coupled.apply_contacts_coupled,
        which pins three unknowns at an ohmic node and one at a gate. This
        property is what is left for the parts that genuinely cannot take a
        gate, which is electron_block and hole_block in device/transport.py.
        """
        for contact in self.contacts:
            if isinstance(contact, GateContact):
                raise TypeError(
                    f"contact {contact.name!r} is a {type(contact).__name__}, "
                    "and this path handles contacts that touch semiconductor "
                    "only. The coupled transport solve pins psi, n and p at "
                    "every node of a contact, and a gate sits on an insulator "
                    "where there is no doping to read and no carrier to pin."
                )
        return tuple(self.contacts)  # type: ignore[arg-type]

    @cached_property
    def carrier_free_nodes(self) -> tuple[int, ...]:
        """Nodes holding no semiconductor, whose n and p rows have to be pinned.

        Empty on a device made of one semiconductor. On a MOS stack these are
        the nodes strictly inside the oxide: their charge volume is zero and
        their carrier face is zero, which leaves both continuity rows reading
        0 = 0. See apply_contacts_coupled.
        """
        if self.regions is None:
            return ()
        return tuple(int(node) for node in self.regions.oxide_nodes)

    @property
    def mesh_1d(self) -> Mesh1D:
        """The mesh, if it is a line.

        The transport and current extraction paths slice edges contiguously and
        assume every node has at most two neighbours, which is a 1D mesh and
        nothing else. They ask through here so that handing them a grid is a
        refusal rather than an index error somewhere deep in an assembly.
        """
        if not isinstance(self.mesh, Mesh1D):
            raise TypeError(
                "this path is 1D and the device carries a "
                f"{type(self.mesh).__name__}. The coupled Newton solve and "
                "the current extraction work in both; this is the Gummel "
                "path, which slices edges contiguously."
            )
        return self.mesh

    @cached_property
    def net_doping_scaled(self) -> Field:
        """Net doping on the mesh nodes [cm^-3], scaled by C_0."""
        return self.net_doping.to_scaled(self.scale)

    def with_bias(self, **voltages: float) -> Device:
        """A copy of this device with new contact voltages [V].

            device.with_bias(anode=0.5)

        Contacts not named keep the bias they had. A Device is frozen, so a
        bias sweep is a sequence of devices rather than one device being
        mutated, which means a converged solution can never be left attached to
        a bias it was not solved at.
        """
        known = {contact.name for contact in self.contacts}
        unknown = sorted(set(voltages) - known)
        if unknown:
            raise KeyError(
                f"no contact named {unknown} on this device, which has "
                f"{sorted(known)}"
            )

        contacts = tuple(
            replace(contact, voltage=voltages.get(contact.name, contact.voltage))
            for contact in self.contacts
        )
        return replace(self, contacts=contacts)

    def __repr__(self) -> str:
        names = ", ".join(
            f"{contact.name}={contact.voltage:g}V" for contact in self.contacts
        )
        if isinstance(self.mesh, Mesh1D):
            extent = f"length={self.mesh.length:.3e} cm"
        else:
            extent = (
                f"size={self.mesh.x_axis.length:.3e} by "
                f"{self.mesh.y_axis.length:.3e} cm"
            )
        return (
            f"Device {self.material.name} {self.mesh.n_nodes} nodes "
            f"{extent} contacts=({names})"
        )


def build_device(
    mesh: AnyMesh,
    doping: DopingProfile,
    contacts: tuple[Contact, ...],
    material: Material | None = None,
    C_0: float | None = None,
    regions: RegionMap | None = None,
) -> Device:
    """Assemble a Device and check that it is self consistent.

    Args:
        mesh: the mesh, 1D or 2D.
        doping: net doping profile, a callable of position.
        contacts: at least one contact.
        material: defaults to silicon at 300 K.
        C_0: reference concentration for scaling [cm^-3], defaults to n_i.
        regions: the material map, on a device made of more than one material.
            None means the whole mesh is the one semiconductor.

    C_0 is exposed here so that Phase 5 can switch to max|net doping| in one
    place if conditioning demands it, per docs/02-numerics.md.
    """
    if material is None:
        material = Material.silicon()

    if not contacts:
        raise ValueError("a device needs at least one contact")

    for contact in contacts:
        for node in contact.nodes:
            if not 0 <= node < mesh.n_nodes:
                raise IndexError(
                    f"contact {contact.name!r} sits on node {node}, "
                    f"but the mesh has {mesh.n_nodes} nodes"
                )

    if regions is not None:
        if regions.semiconductor_volume.size != mesh.n_nodes:
            raise ValueError(
                f"the region map covers {regions.semiconductor_volume.size} "
                f"nodes but the mesh has {mesh.n_nodes}. A region map belongs "
                "to the mesh it was built on."
            )
        if np.asarray(regions.eps_r).size != mesh.n_edges:
            raise ValueError(
                f"the region map carries {np.asarray(regions.eps_r).size} edge "
                f"permittivities but the mesh has {mesh.n_edges} edges. A "
                "region map belongs to the mesh it was built on."
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
        regions=regions,
    )
