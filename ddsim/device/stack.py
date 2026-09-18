"""A 1D stack of doped regions, the device builder of phases/PHASE-7.md Stage 4.

Regions are laid left to right, any number of them, each with a length and a
doping, with an ohmic contact at each end: pn, pin, p+n, npn and whatever a
student invents. Nothing new is solved here. It is the pn_diode recipe with
more than one junction: the same graded mesh generator, the same kind of
doping profile, the same contacts. Drawn as the Phase 2 diode, it is that
diode to the last bit.

Graded at every junction
------------------------
The mesh is graded_mesh_1d_at over the junctions: h_min at every one, one
growth rate for the whole stack, so a thin base between two junctions meets
its neighbours at the same spacing. With one junction that is graded_mesh_1d
itself, which is how the Phase 2 diode drawn as a stack stays that diode.

A boundary between two regions of the same doping is not a junction. Nothing
changes there, so there is nothing to grade towards, and the stack is the
same device as the one with those two regions drawn as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Layers
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d_at

DOPING_RANGE = (1e14, 1e19)
"""The doping a region may have [cm^-3], the range docs/01-physics.md states
for the models a 1D device is solved with."""

NODES_INSIDE = 2
"""Fewest mesh nodes a region must hold strictly inside it [1]. Two nodes
make three cells, the least that shows a slope inside the region rather
than a straight line from one boundary to the other."""


@dataclass(frozen=True)
class Region:
    """One doped slab of a stack."""

    dopant: str
    """"n" for donors, "p" for acceptors."""

    length: float
    """Thickness along the stack [cm]."""

    concentration: float
    """Dopant concentration [cm^-3], given positive."""

    @property
    def net_doping(self) -> float:
        """Nd - Na [cm^-3], the sign convention of docs/01-physics.md."""
        return self.concentration if self.dopant == "n" else -self.concentration

    def describe(self) -> str:
        """The region as a student drew it, for a refusal to name."""
        return f"{self.dopant}-type {self.concentration:g} cm^-3, {self.length:g} cm"


PHASE_2_DIODE = (Region("p", 0.5e-4, 1e16), Region("n", 0.5e-4, 1e16))
"""pn_diode's defaults drawn as a stack."""


def _check_regions(regions: tuple[Region, ...]) -> None:
    """Refuse a region that is not a slab of doped silicon the models cover."""
    if len(regions) < 2:
        raise ValueError(
            f"a stack of {len(regions)} regions has no junction: it needs at "
            "least two regions, with the doping changing between them"
        )
    low, high = DOPING_RANGE
    for number, region in enumerate(regions, start=1):
        if region.dopant not in ("n", "p"):
            raise ValueError(
                f"region {number}: the dopant is 'n' or 'p', got "
                f"{region.dopant!r}. An intrinsic layer is drawn as a lightly "
                f"doped one, {low:g} n-type say, the bottom of the range the "
                "models are used over."
            )
        if not region.length > 0.0:
            raise ValueError(
                f"region {number}: the length must be positive, got "
                f"{region.length:g} cm"
            )
        if not low <= region.concentration <= high:
            raise ValueError(
                f"region {number}: a doping of {region.concentration:g} cm^-3 "
                f"is outside {low:g} to {high:g} cm^-3, the range "
                "docs/01-physics.md states for the mobility and recombination "
                "models and the statistics this device is solved with."
            )


def _too_short(number: int, region: Region, inside: int) -> ValueError:
    return ValueError(
        f"region {number} ({region.describe()}) is shorter than the mesh can "
        f"resolve: it holds {inside} mesh nodes inside it and needs at least "
        f"{NODES_INSIDE}. Make it longer, or refine the mesh with a smaller "
        "h_min or more n_nodes."
    )


def stack(
    regions: tuple[Region, ...] = PHASE_2_DIODE,
    n_nodes: int = 201,
    h_min: float = 1e-7,
    left_voltage: float = 0.0,
    right_voltage: float = 0.0,
    material: Material | None = None,
) -> Device:
    """Doped regions laid left to right, an ohmic contact at each end.

    Args:
        regions: the regions in order from the left contact, at least two,
            with the doping changing at one boundary at least.
        n_nodes: mesh node count over the whole stack [1]. Range 51 to 1001.
            Shared between the junctions by how far each one's grading has to
            grow, so a stack of many junctions wants more.
        h_min: mesh spacing at every junction [cm]. Range 1e-8 to 1e-6, log.
            docs/02-numerics.md wants it below half the Debye length: 20 nm
            at 1e16 and 0.65 nm at 1e19, the top of the doping range, so the
            1 nm default under-resolves the heaviest regions.
        left_voltage: bias on the left contact [V]. Range -5 to 1. The same
            ends as the diode's contacts. Where a cold solve stops converging
            was measured on the Phase 2 diode, near 1.3 V forward, and not on
            other stacks.
        right_voltage: bias on the right contact [V]. Range -5 to 1. The same
            ends as the left contact.
        material: defaults to silicon at 300 K.
    """
    _check_regions(regions)
    ends = np.cumsum([region.length for region in regions])
    junctions = [
        float(end)
        for end, (before, after) in zip(
            ends[:-1], zip(regions[:-1], regions[1:], strict=True), strict=True
        )
        if before.net_doping != after.net_doping
    ]
    if not junctions:
        raise ValueError(
            "this stack has no junction: every region has the same doping, so "
            "there is nothing for the mesh to grade towards and no device to "
            "see. Change the doping of one region."
        )

    # A region can hold at most one node per h_min of its length, so this one
    # is refused before a mesh is asked for. graded_mesh_1d would refuse it
    # too, without saying which region it was.
    for number, region in enumerate(regions, start=1):
        if region.length < (NODES_INSIDE + 1) * h_min:
            raise _too_short(number, region, int(region.length / h_min))

    mesh = graded_mesh_1d_at(float(ends[-1]), n_nodes, tuple(junctions), h_min)

    starts = np.concatenate([[0.0], ends[:-1]])
    for number, (region, start, end) in enumerate(
        zip(regions, starts, ends, strict=True), start=1
    ):
        inside = int(np.count_nonzero((mesh.x > start) & (mesh.x < end)))
        if inside < NODES_INSIDE:
            raise _too_short(number, region, inside)

    contacts = (
        OhmicContact(name="left", node=0, voltage=left_voltage),
        OhmicContact(name="right", node=mesh.n_nodes - 1, voltage=right_voltage),
    )
    return build_device(
        mesh=mesh,
        doping=Layers(
            boundaries=tuple(float(end) for end in ends[:-1]),
            values=tuple(region.net_doping for region in regions),
        ),
        contacts=contacts,
        material=material,
    )
