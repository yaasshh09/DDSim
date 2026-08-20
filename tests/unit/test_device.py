"""Tests for device/builder.py, device/pn_diode.py and device/equilibrium.py.

device/ composes: geometry and doping in, a Device out that knows its mesh, its
material, its contacts and how to evaluate its doping anywhere. It holds no
solver state of its own, which is why solve_equilibrium returns a DeviceState
rather than mutating the Device.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core.field import Location, ScalingState
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Uniform
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import uniform_mesh_1d

MICRON = 1e-4
"""One micron [cm]."""


# ------------------------------------------------------------------- material


def test_silicon_material_matches_the_constants_doc() -> None:
    silicon = Material.silicon()
    assert silicon.T == 300.0  # [K]
    assert silicon.n_i == 1.0e10  # [cm^-3]
    assert silicon.eps == pytest.approx(11.7 * 8.8541878128e-14)  # [F/cm]


def test_material_at_another_temperature_moves_n_i() -> None:
    assert Material.silicon(T=400.0).n_i > Material.silicon(T=300.0).n_i


# --------------------------------------------------------------------- device


def test_build_device_evaluates_doping_on_the_mesh() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    device = build_device(
        mesh=mesh,
        doping=Uniform(1e16),
        contacts=(OhmicContact("anode", 0, 0.0), OhmicContact("cathode", 10, 0.0)),
    )
    np.testing.assert_allclose(device.net_doping.data, 1e16)  # [cm^-3]


def test_net_doping_is_a_physical_node_field() -> None:
    device = pn_diode()
    assert device.net_doping.unit == "cm^-3"
    assert device.net_doping.scaling is ScalingState.PHYSICAL
    assert device.net_doping.location is Location.NODE


def test_net_doping_scaled_divides_by_c_0() -> None:
    device = pn_diode(Na=1e16, Nd=1e16)
    scaled = device.net_doping_scaled
    assert scaled.scaling is ScalingState.SCALED
    np.testing.assert_allclose(
        scaled.data, device.net_doping.data / device.scale.C_0, rtol=1e-14
    )


def test_device_rejects_a_contact_outside_the_mesh() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    with pytest.raises(IndexError, match="node"):
        build_device(
            mesh=mesh,
            doping=Uniform(1e16),
            contacts=(OhmicContact("anode", 99, 0.0),),
        )


def test_device_requires_at_least_one_contact() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    with pytest.raises(ValueError, match="contact"):
        build_device(mesh=mesh, doping=Uniform(1e16), contacts=())


def test_device_is_immutable() -> None:
    device = pn_diode()
    with pytest.raises(AttributeError):
        device.mesh = None  # type: ignore[misc]


def test_doping_stays_re_evaluable_after_construction() -> None:
    """The profile is kept, not just its values on this mesh.

    Phase 5 refines adaptively, so the Device has to be able to produce doping
    on a mesh that did not exist when it was built.
    """
    device = pn_diode(Na=1e16, Nd=1e17, length=MICRON, junction=0.5 * MICRON)
    finer = np.linspace(0.0, MICRON, 1001)
    values = device.doping(finer)
    assert values[0] == pytest.approx(-1e16)
    assert values[-1] == pytest.approx(1e17)


# ------------------------------------------------------------------ pn diode


def test_pn_diode_is_p_type_on_the_left_and_n_type_on_the_right() -> None:
    device = pn_diode(Na=1e16, Nd=1e16)
    assert device.net_doping.data[0] < 0.0
    assert device.net_doping.data[-1] > 0.0


def test_pn_diode_has_two_named_contacts_at_the_ends() -> None:
    device = pn_diode()
    assert [c.name for c in device.contacts] == ["anode", "cathode"]
    assert device.contacts[0].node == 0
    assert device.contacts[1].node == device.mesh.n_nodes - 1


def test_pn_diode_refines_the_mesh_at_the_junction() -> None:
    device = pn_diode(junction=0.5 * MICRON, h_min=1e-7)
    finest = int(np.argmin(device.mesh.h))
    midpoint = 0.5 * (device.mesh.x[finest] + device.mesh.x[finest + 1])
    assert abs(midpoint - 0.5 * MICRON) < 2e-7  # [cm]


def test_pn_diode_contacts_default_to_zero_bias() -> None:
    device = pn_diode()
    assert all(contact.voltage == 0.0 for contact in device.contacts)


# ------------------------------------------------------------- equilibrium


def test_equilibrium_solve_converges() -> None:
    state = solve_equilibrium(pn_diode())
    assert state.newton.converged


def test_equilibrium_solve_converges_in_under_ten_iterations() -> None:
    """The acceptance criterion in phases/PHASE-1.md."""
    state = solve_equilibrium(pn_diode())
    assert state.newton.iterations < 10, state.newton.residual_history


def test_equilibrium_state_fields_are_scaled_node_fields() -> None:
    state = solve_equilibrium(pn_diode())
    for field in (state.psi, state.n, state.p):
        assert field.scaling is ScalingState.SCALED
        assert field.location is Location.NODE


def test_equilibrium_psi_can_be_converted_to_volts() -> None:
    device = pn_diode()
    state = solve_equilibrium(device)
    volts = state.psi.to_physical(device.scale)
    assert volts.scaling is ScalingState.PHYSICAL
    assert np.max(np.abs(volts.data)) < 2.0  # [V], nothing silly


def test_equilibrium_pins_psi_at_the_contacts() -> None:
    from ddsim.discretize.boundary import ohmic_psi_scaled

    device = pn_diode(Na=1e16, Nd=1e16)
    state = solve_equilibrium(device)
    doping_scaled = device.net_doping_scaled.data

    for contact in device.contacts:
        expected = ohmic_psi_scaled(float(doping_scaled[contact.node]), 0.0)
        assert state.psi.data[contact.node] == pytest.approx(expected, rel=1e-10)


def test_equilibrium_is_flat_in_uniform_material() -> None:
    """No junction means no field. The initial guess is already the answer."""
    mesh = uniform_mesh_1d(MICRON, 51)
    device = build_device(
        mesh=mesh,
        doping=Uniform(1e16),
        contacts=(OhmicContact("left", 0, 0.0), OhmicContact("right", 50, 0.0)),
    )
    state = solve_equilibrium(device)
    assert np.ptp(state.psi.data) < 1e-9


def test_equilibrium_reports_the_residual_history() -> None:
    """phases/PHASE-1.md wants the history logged so the tail can be inspected."""
    state = solve_equilibrium(pn_diode())
    assert len(state.newton.residual_history) == state.newton.iterations + 1
    assert state.newton.residual_history[-1] < state.newton.residual_history[0]


def test_equilibrium_raises_if_it_fails_to_converge() -> None:
    """A silently unconverged solve is the worst possible outcome here."""
    with pytest.raises(RuntimeError, match="converge"):
        solve_equilibrium(pn_diode(), max_iterations=1)


def test_equilibrium_accepts_a_device_with_a_bias_applied() -> None:
    """No continuation yet, but a contact voltage must still be honoured."""
    device = pn_diode(anode_voltage=-1.0)
    state = solve_equilibrium(device)
    assert state.newton.converged
    volts = state.psi.to_physical(device.scale).data
    assert volts[0] < volts[-1]


def test_device_state_is_immutable() -> None:
    state = solve_equilibrium(pn_diode())
    with pytest.raises(AttributeError):
        state.psi = None  # type: ignore[misc]


def test_solve_does_not_mutate_the_device() -> None:
    device = pn_diode()
    before = device.net_doping.data.copy()
    solve_equilibrium(device)
    np.testing.assert_array_equal(device.net_doping.data, before)


def test_device_repr_is_informative() -> None:
    text = repr(pn_diode())
    assert "Device" in text
    assert "nodes" in text


def test_build_device_defaults_to_silicon() -> None:
    device = build_device(
        mesh=uniform_mesh_1d(MICRON, 11),
        doping=Uniform(1e16),
        contacts=(OhmicContact("a", 0, 0.0),),
    )
    assert isinstance(device, Device)
    assert device.material.n_i == 1.0e10  # [cm^-3]


def test_device_rejects_duplicate_contact_names() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    with pytest.raises(ValueError, match="unique"):
        build_device(
            mesh=mesh,
            doping=Uniform(1e16),
            contacts=(OhmicContact("a", 0, 0.0), OhmicContact("a", 10, 0.0)),
        )
