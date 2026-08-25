"""The 2D path has to reproduce 1D. phases/PHASE-4.md makes it a criterion.

A device that is uniform in y is a 1D device. Solve it on a 2D mesh with
reflecting top and bottom and the answer has to be the 1D answer, because the
1D problem is not an approximation to it: it is the same problem written down
with one coordinate suppressed.

The residual is not equal to the 1D residual, and that is not a defect. Box
integration over a dual cell of height dy scales every term in the row by that
height, so what holds is

    F_2D[node(i, j)] = (dual_y[j] / x_0) * F_1D[i]

exactly, with the row factor differing between interior rows and the two
boundary rows, which carry half a dual cell each. Both sides vanish at the same
state, so the solution is identical while the residual is a row scaling of it.
Asserting the proportionality rather than equality is the sharper test anyway:
it pins the per-row factor, so a dual face taken from the wrong axis fails here
even though the solution would still come out right on a square mesh.

There is nothing to converge for this to be true. It holds at any state, solved
or not, which is why the test is written against an arbitrary perturbed state
rather than a solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.transport import TransportModels, initial_state
from ddsim.discretize.boundary import OhmicContact
from ddsim.discretize.coupled import (
    UNIFORM_1D,
    Unknown,
    coupled_residual,
    pack,
    unpack,
)
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.physics.recombination import (
    AugerRecombination,
    SRHRecombination,
    SumOfRecombination,
)

NX = 21
NY = 4
HEIGHT = 2e-5


@pytest.fixture
def setup():
    """The same 1e18 / 1e15 device, described once in 1D and once in 2D."""
    x_axis = graded_mesh_1d(
        length=1e-4, n_nodes=NX, refine_at=0.5e-4, h_min=2e-6
    )
    y_axis = uniform_mesh_1d(length=HEIGHT, n_nodes=NY)
    mesh_2d = tensor_mesh_2d(x_axis, y_axis)

    device = build_device(
        mesh=x_axis,
        doping=abrupt_junction(Na=1e18, Nd=1e15, position=0.5e-4),
        contacts=(
            OhmicContact(name="anode", node=0, voltage=0.0),
            OhmicContact(name="cathode", node=NX - 1, voltage=0.0),
        ),
    )
    # Scalar lifetimes on purpose. TransportModels.for_device builds them per
    # node from the local doping, and a per node array sized for the 1D mesh
    # cannot be evaluated on the 2D one. The models have to be the same object
    # on both sides for the comparison to mean anything, so the parts that
    # depend on the node count are made uniform and the parts that depend on
    # the edge count, the Arora diffusivities, are laid out per mesh below.
    scale = device.scale
    recombination = SumOfRecombination(
        (
            SRHRecombination(
                tau_n=C.TAU_N_MAX / scale.t_0,
                tau_p=C.TAU_P_MAX / scale.t_0,
                ni2=(device.material.n_i / scale.C_0) ** 2,
                n1=device.material.n_i / scale.C_0,
                p1=device.material.n_i / scale.C_0,
            ),
            AugerRecombination(
                C_n=C.AUGER_C_N * scale.C_0**2 * scale.t_0,
                C_p=C.AUGER_C_P * scale.C_0**2 * scale.t_0,
                ni2=(device.material.n_i / scale.C_0) ** 2,
            ),
        )
    )
    models = TransportModels.for_device(
        device, recombination=recombination, mobility="arora"
    )
    x_0 = scale.x_0

    # An arbitrary state, uniform in y. Not a solution, and it does not need
    # to be: the identity below holds at every state.
    state = initial_state(device)
    k = np.linspace(0.0, 3.0 * np.pi, NX)
    psi = state.psi.data + 0.35 * np.cos(k)
    n = state.n.data * np.exp(0.3 * np.sin(k))
    p = state.p.data * np.exp(-0.3 * np.sin(k))

    return {
        "device": device,
        "models": models,
        "mesh_2d": mesh_2d,
        "x_0": x_0,
        "psi": psi,
        "n": n,
        "p": p,
        "y_axis": y_axis,
    }


def residual_1d(setup):
    """The coupled residual on the 1D mesh."""
    device, models, x_0 = setup["device"], setup["models"], setup["x_0"]
    return coupled_residual(
        h=device.mesh.h / x_0,
        volume=device.mesh.volume / x_0,
        x=pack(setup["psi"], setup["n"], setup["p"]),
        net_doping=device.net_doping_scaled.data,
        Dn=models.Dn,
        Dp=models.Dp,
        recombination=models.recombination,
        geometry=UNIFORM_1D,
    )


def residual_2d(setup):
    """The coupled residual on the 2D mesh, with the state repeated per row.

    The scaling is the d-dimensional rule from mesh2d.py: lengths by x_0,
    faces by x_0^(d-1), volumes by x_0^d. At d = 2 that is x_0 for the dual
    face and x_0 squared for the volume.
    """
    mesh, models, x_0 = setup["mesh_2d"], setup["models"], setup["x_0"]
    device = setup["device"]

    tile = np.tile
    geometry = mesh.edge_geometry()
    scaled_geometry = type(geometry)(
        edge_nodes=geometry.edge_nodes,
        dual_face=mesh.dual_face / x_0,
        eps_r=1.0,
    )

    # Per edge diffusivity has to be laid out on the 2D edge list, not the 1D
    # one. Horizontal edges repeat the 1D per edge values row by row; vertical
    # edges join two nodes of the same column, and the doping is uniform in y,
    # so each takes the value of the column it runs along.
    Dn_1d = np.broadcast_to(np.asarray(models.Dn), (NX - 1,))
    Dp_1d = np.broadcast_to(np.asarray(models.Dp), (NX - 1,))
    Dn_node = np.concatenate([Dn_1d[:1], Dn_1d])
    Dp_node = np.concatenate([Dp_1d[:1], Dp_1d])

    Dn = np.concatenate([tile(Dn_1d, NY), tile(Dn_node, NY - 1)])
    Dp = np.concatenate([tile(Dp_1d, NY), tile(Dp_node, NY - 1)])

    return coupled_residual(
        h=mesh.h / x_0,
        volume=mesh.volume / x_0**2,
        x=pack(
            tile(setup["psi"], NY), tile(setup["n"], NY), tile(setup["p"], NY)
        ),
        net_doping=tile(device.net_doping_scaled.data, NY),
        Dn=Dn,
        Dp=Dp,
        recombination=models.recombination,
        geometry=scaled_geometry,
    )


@pytest.mark.parametrize("component", list(Unknown), ids=lambda u: u.name)
def test_the_2d_residual_is_the_1d_one_scaled_by_the_row_height(setup, component):
    """Every row of the 2D residual is the 1D residual times dual_y[j]/x_0.

    Checked per component, so a failure names which equation broke rather than
    just saying the vector moved.
    """
    one_d = unpack(residual_1d(setup))[component]
    two_d = unpack(residual_2d(setup))[component]
    dual_y = setup["y_axis"].volume / setup["x_0"]

    for row in range(NY):
        got = two_d[row * NX : (row + 1) * NX]
        np.testing.assert_allclose(
            got, dual_y[row] * one_d, rtol=1e-12, atol=1e-13 * np.abs(one_d).max()
        )


def test_the_row_factor_is_not_the_same_on_every_row(setup):
    """Guards the test above from passing for a trivial reason.

    The two boundary rows carry half a dual cell, so the factor genuinely
    varies. If it did not, the test above would be checking a single global
    constant and would not notice a dual face taken from the wrong axis.
    """
    dual_y = setup["y_axis"].volume
    assert dual_y[0] == pytest.approx(0.5 * dual_y[1], rel=1e-15)
    assert len(set(np.round(dual_y, 20))) == 2


def test_the_vertical_edges_carry_no_current_in_a_y_uniform_state(setup):
    """Nothing flows across a row boundary when the state does not vary in y.

    This is what makes the reduction work at all. If it failed, the 2D residual
    would pick up a term the 1D one has no counterpart for.
    """
    mesh = setup["mesh_2d"]
    psi = np.tile(setup["psi"], NY)

    vertical = mesh.edge_nodes[mesh.n_horizontal :]
    np.testing.assert_array_equal(
        psi[vertical[:, 0]], psi[vertical[:, 1]]
    )
