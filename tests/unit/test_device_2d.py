"""A Device on a 2D mesh: the wiring Stage 4 of the Phase 4 plan asks for.

The physics of the MOS stack already runs, hand assembled, in
tests/analytic/test_mos_electrostatics.py. This is about the composition
around it: a Device that carries a Mesh2D and a RegionMap, hands the assembly a
mesh that has already scaled itself, and knows that only part of a dual cell
carries charge.

Three things are easy to get wrong here and all three are checked below.

The power of x_0 on the dual volume. It is x_0 in 1D and x_0 squared in 2D, and
getting it wrong makes the device the wrong size by a factor of the Debye
length while converging perfectly happily.

The charge volume against the geometric volume. The oxide has dual cells like
anywhere else, and no charge in them. If the plain volume reaches the charge
term, the oxide fills with carriers.

The doping under the oxide. Nothing there is doped, and while the zero charge
volume already makes that true numerically, a net_doping array that says 1e16
in the middle of an insulator is a trap for everything downstream that reads it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors
from ddsim.device.builder import build_device
from ddsim.device.doping import Uniform
from ddsim.device.regions import stacked_regions
from ddsim.discretize.boundary import GateContact, OhmicContact, OhmicPlate
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d, uniform_mesh_2d

NA = 1e16
"""p-type substrate [cm^-3]."""

T_SI = 1e-5
"""Silicon thickness [cm]."""

T_OX = 1e-6
"""Oxide thickness [cm]."""

WIDTH = 1e-5
"""Device width [cm]."""

DY = 5e-7
"""Uniform y spacing [cm]. Divides both thicknesses so the interface lands on
a node line."""


def stack_mesh():
    """The MOS stack mesh, three columns wide."""
    ny = int(round((T_SI + T_OX) / DY)) + 1
    return tensor_mesh_2d(
        uniform_mesh_1d(length=WIDTH, n_nodes=3),
        uniform_mesh_1d(length=T_SI + T_OX, n_nodes=ny),
    )


def mos_contacts(mesh):
    """A substrate plate along the bottom and a gate plate along the top."""
    body = OhmicPlate(
        name="body",
        nodes=tuple(mesh.node_at(i, 0) for i in range(mesh.nx)),
        voltage=0.0,
    )
    gate = GateContact(
        name="gate",
        nodes=tuple(mesh.node_at(i, mesh.ny - 1) for i in range(mesh.nx)),
        voltage=0.0,
        work_function=C.PHI_M_N_POLY,
    )
    return (body, gate)


def mos_device(mesh=None, regions=None):
    """A MOS capacitor Device, built the long way for the tests below."""
    mesh = stack_mesh() if mesh is None else mesh
    regions = stacked_regions(mesh, interface_y=T_SI) if regions is None else regions
    return build_device(
        mesh=mesh,
        doping=Uniform(-NA),
        contacts=mos_contacts(mesh),
        regions=regions,
    )


class TestScaling:
    """What the device hands the assembly, and in which units."""

    def test_a_2d_dual_volume_is_scaled_by_x_0_squared(self) -> None:
        """It is an area. Scaling it by x_0 leaves it a factor of x_0 too big."""
        mesh = uniform_mesh_2d(width=1e-4, height=1e-4, nx=5, ny=5)
        device = build_device(
            mesh=mesh,
            doping=Uniform(1e16),
            contacts=(OhmicContact(name="body", node=0, voltage=0.0),),
        )
        scale = device.scale

        np.testing.assert_allclose(
            device.scaled_mesh.volume,
            mesh.volume / scale.x_0**2,
            rtol=1e-15,
        )
        assert float(np.sum(device.scaled_mesh.volume)) == pytest.approx(
            1e-8 / scale.x_0**2, rel=1e-12
        )

    def test_a_1d_device_scales_exactly_as_it_always_did(self) -> None:
        """The whole 1D suite is pinned to these numbers, to the last bit."""
        mesh = uniform_mesh_1d(1e-4, 11)
        device = build_device(
            mesh=mesh,
            doping=Uniform(1e16),
            contacts=(OhmicContact(name="anode", node=0, voltage=0.0),),
        )

        np.testing.assert_array_equal(
            device.scaled_mesh.h, mesh.h / device.scale.x_0
        )
        np.testing.assert_array_equal(
            device.scaled_mesh.volume, mesh.volume / device.scale.x_0
        )
        np.testing.assert_array_equal(
            device.charge_volume_scaled, mesh.volume / device.scale.x_0
        )

    def test_the_oxide_edges_carry_the_oxide_permittivity(self) -> None:
        """Which is what makes normal D continuous instead of E."""
        device = mos_device()
        eps_r = np.asarray(device.scaled_mesh.geometry.eps_r)

        assert eps_r.min() == pytest.approx(C.EPS_R_OX / C.EPS_R_SI, rel=1e-12)
        assert eps_r.max() == pytest.approx(1.0, rel=1e-12)


class TestChargeVolume:
    """Only part of a dual cell carries charge once there is an insulator."""

    def test_the_charge_volume_is_zero_in_the_oxide(self) -> None:
        device = mos_device()
        oxide = device.regions.oxide_nodes

        assert oxide.size > 0, "this stack has an oxide, so it has oxide nodes"
        np.testing.assert_array_equal(device.charge_volume_scaled[oxide], 0.0)

    def test_the_charge_volume_is_not_the_geometric_volume(self) -> None:
        """If these were equal the test above would be measuring nothing."""
        device = mos_device()

        assert not np.allclose(
            device.charge_volume_scaled, device.scaled_mesh.volume
        )

    def test_the_interface_node_keeps_half_its_cell(self) -> None:
        """It has silicon under it and oxide over it, so it holds half a cell.

        The inversion layer forms here, so calling it an oxide node and zeroing
        it would delete the entire point of the device.
        """
        device = mos_device()
        mesh = device.mesh
        row = int(round(T_SI / DY))
        node = mesh.node_at(1, row)

        interior = mesh.node_at(1, row - 1)
        assert device.charge_volume_scaled[node] == pytest.approx(
            0.5 * device.charge_volume_scaled[interior], rel=1e-12
        )

    def test_the_charge_volume_sums_to_the_silicon_area(self) -> None:
        device = mos_device()
        expected = WIDTH * T_SI / device.scale.x_0**2

        assert float(np.sum(device.charge_volume_scaled)) == pytest.approx(
            expected, rel=1e-12
        )


class TestDoping:
    """A doping profile is a semiconductor property. There is none in an oxide."""

    def test_the_doping_is_zeroed_where_there_is_no_semiconductor(self) -> None:
        device = mos_device()

        np.testing.assert_array_equal(
            device.net_doping.data[device.regions.oxide_nodes], 0.0
        )

    def test_the_doping_survives_everywhere_else(self) -> None:
        device = mos_device()
        silicon = device.charge_volume_scaled > 0.0

        np.testing.assert_allclose(device.net_doping.data[silicon], -NA)

    def test_a_device_with_no_regions_keeps_every_node_doped(self) -> None:
        mesh = uniform_mesh_2d(width=1e-4, height=1e-4, nx=4, ny=4)
        device = build_device(
            mesh=mesh,
            doping=Uniform(1e16),
            contacts=(OhmicContact(name="body", node=0, voltage=0.0),),
        )

        np.testing.assert_allclose(device.net_doping.data, 1e16)


class TestValidation:
    """What build_device refuses, and why."""

    def test_a_contact_outside_the_mesh_is_refused(self) -> None:
        mesh = stack_mesh()
        with pytest.raises(IndexError, match="node"):
            build_device(
                mesh=mesh,
                doping=Uniform(-NA),
                contacts=(
                    OhmicPlate(name="body", nodes=(0, mesh.n_nodes), voltage=0.0),
                ),
            )

    def test_a_region_map_from_another_mesh_is_refused(self) -> None:
        """Its permittivities are per edge, so a mismatch is a silent disaster."""
        mesh = stack_mesh()
        other = tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH, n_nodes=4),
            uniform_mesh_1d(length=T_SI + T_OX, n_nodes=mesh.ny),
        )

        with pytest.raises(ValueError, match="region"):
            build_device(
                mesh=mesh,
                doping=Uniform(-NA),
                contacts=mos_contacts(mesh),
                regions=stacked_regions(other, interface_y=T_SI),
            )

    def test_a_region_map_with_the_wrong_edge_count_is_refused(self) -> None:
        """Same node count, a different edge count, which the check above misses.

        A 3 by 4 tensor mesh and a 2 by 6 one both carry 12 nodes, and 17
        edges against 16. Permittivity lives on edges, so a region map can
        pass the node check and still be the wrong shape for the mesh.
        """
        mesh = tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH, n_nodes=3),
            uniform_mesh_1d(length=T_SI + T_OX, n_nodes=4),
        )
        other = tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH, n_nodes=2),
            uniform_mesh_1d(length=T_SI + T_OX, n_nodes=6),
        )
        assert other.n_nodes == mesh.n_nodes, "the node check has to pass first"
        assert other.n_edges != mesh.n_edges

        with pytest.raises(ValueError, match="edge permittivities"):
            build_device(
                mesh=mesh,
                doping=Uniform(-NA),
                contacts=(OhmicPlate(name="body", nodes=(0,), voltage=0.0),),
                regions=stacked_regions(other, interface_y=T_SI + T_OX),
            )

    def test_duplicate_contact_names_are_refused(self) -> None:
        mesh = stack_mesh()
        with pytest.raises(ValueError, match="unique"):
            build_device(
                mesh=mesh,
                doping=Uniform(-NA),
                contacts=(
                    OhmicPlate(name="body", nodes=(0,), voltage=0.0),
                    OhmicPlate(name="body", nodes=(1,), voltage=0.0),
                ),
            )

    def test_the_uncoupled_blocks_refuse_a_gate(self) -> None:
        """A gate is not a contact a Gummel block can pin.

        It sits on an insulator, so there is no doping under it to read and no
        carrier density to hold at equilibrium. Better to say so than to
        assemble a system with the gate quietly left out of it, which would
        converge and mean nothing.

        Only the uncoupled blocks ask through here. The coupled path applies
        gates itself, pinning psi alone at their nodes. See
        tests/unit/test_coupled_transport.py.
        """
        device = mos_device()

        with pytest.raises(TypeError, match="touch semiconductor"):
            _ = device.ohmic_contacts

    def test_the_gummel_path_refuses_a_grid(self) -> None:
        """It slices edges contiguously, which is a line and nothing else.

        The coupled Newton solve works in either dimension now, so the
        refusal has to name which path it is rather than claiming that 2D
        transport does not exist.
        """
        device = build_device(
            mesh=uniform_mesh_2d(width=1e-4, height=1e-4, nx=3, ny=3),
            doping=Uniform(1e16),
            contacts=(OhmicPlate(name="body", nodes=(0,), voltage=0.0),),
        )

        with pytest.raises(TypeError, match="this is the Gummel"):
            _ = device.mesh_1d

    def test_the_transport_path_accepts_a_plate(self) -> None:
        """A point and a plate differ only in how many nodes they cover.

        The coupled solve pins psi, n and p at every node of a contact, and a
        point contact is the case where that is one node.
        """
        device = build_device(
            mesh=uniform_mesh_2d(width=1e-4, height=1e-4, nx=3, ny=3),
            doping=Uniform(1e16),
            contacts=(
                OhmicPlate(name="left", nodes=(0, 3, 6), voltage=0.0),
                OhmicPlate(name="right", nodes=(2, 5, 8), voltage=0.0),
            ),
        )

        assert [c.name for c in device.ohmic_contacts] == ["left", "right"]

    def test_a_single_material_device_has_no_carrier_free_nodes(self) -> None:
        """Nothing to pin where every node holds semiconductor."""
        device = build_device(
            mesh=uniform_mesh_1d(1e-4, 11),
            doping=Uniform(1e16),
            contacts=(OhmicContact(name="anode", node=0, voltage=0.0),),
        )

        assert device.carrier_free_nodes == ()

    def test_the_oxide_nodes_of_a_stack_are_the_ones_pinned(self) -> None:
        """They are exactly the nodes whose two continuity rows read 0 = 0."""
        device = mos_device()

        assert device.carrier_free_nodes == tuple(
            int(node) for node in device.regions.oxide_nodes
        )
        assert len(device.carrier_free_nodes) > 0

    def test_a_1d_device_still_reports_its_point_contacts(self) -> None:
        device = build_device(
            mesh=uniform_mesh_1d(1e-4, 11),
            doping=Uniform(1e16),
            contacts=(OhmicContact(name="anode", node=0, voltage=0.0),),
        )

        assert device.ohmic_contacts == device.contacts


class TestBias:
    """Rebiasing a 2D device with a gate."""

    def test_the_gate_bias_can_be_changed_by_name(self) -> None:
        device = mos_device()
        biased = device.with_bias(gate=1.5)

        assert biased.contacts[1].voltage == 1.5
        assert biased.contacts[1].work_function == C.PHI_M_N_POLY
        assert device.contacts[1].voltage == 0.0, "the original must not move"

    def test_rebiasing_keeps_the_regions(self) -> None:
        """A rebiased device that lost its oxide would solve as bare silicon."""
        device = mos_device().with_bias(gate=1.0)

        assert device.regions is not None
        np.testing.assert_array_equal(
            device.charge_volume_scaled[device.regions.oxide_nodes], 0.0
        )


def test_the_scaled_mesh_is_built_once() -> None:
    """It is a per node and per edge array, and every Newton step asks for it."""
    device = mos_device()

    assert device.scaled_mesh is device.scaled_mesh


def test_a_2d_device_reports_itself_sensibly() -> None:
    device = mos_device()
    text = repr(device)

    assert "silicon" in text
    assert str(device.mesh.n_nodes) in text
    assert "gate" in text


def test_scale_factors_are_shared_between_1d_and_2d() -> None:
    """Nothing about the scaling depends on dimension except the volume power."""
    device = mos_device()

    assert device.scale == ScaleFactors.for_silicon(C_0=C.n_i())
