"""Tests for physics/mobility.py, the Arora doping dependent model.

docs/01-physics.md puts this in Phase 3 and is blunt about why it matters:
"Mobility is where a device simulator earns or loses its quantitative accuracy.
The PDE solve can be perfect and the answer still wrong by 3x if mobility is
wrong."

The model has four analytic limits and they are all checkable by hand:

    N -> 0      mu -> mu_min + mu_d
    N = N_ref   mu = mu_min + mu_d/2
    N -> inf    mu -> mu_min
    dmu/dN < 0  everywhere

Those pin the shape. Two literature values pin the magnitude, because the shape
would be right for any parameter set.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.physics.mobility import (
    AroraMobility,
    ConstantMobility,
    edge_diffusivity,
)

# --------------------------------------------------------- the four limits


@pytest.mark.parametrize("model", [AroraMobility.electrons(), AroraMobility.holes()])
def test_the_undoped_limit_is_mu_min_plus_mu_d(model: AroraMobility) -> None:
    """N -> 0 kills the denominator's second term."""
    assert float(model(0.0)) == pytest.approx(model.mu_min + model.mu_d, rel=1e-14)


@pytest.mark.parametrize("model", [AroraMobility.electrons(), AroraMobility.holes()])
def test_at_the_reference_doping_the_lattice_term_is_halved(
    model: AroraMobility,
) -> None:
    """N = N_ref makes (N/N_ref)^A exactly 1, whatever A is."""
    assert float(model(model.N_ref)) == pytest.approx(
        model.mu_min + model.mu_d / 2.0, rel=1e-14
    )


@pytest.mark.parametrize("model", [AroraMobility.electrons(), AroraMobility.holes()])
def test_the_heavily_doped_limit_is_mu_min(model: AroraMobility) -> None:
    """Ionized impurity scattering saturates. Approached, never reached."""
    heavy = float(model(1e25))

    assert heavy > model.mu_min
    assert heavy == pytest.approx(model.mu_min, rel=1e-3)


@pytest.mark.parametrize("model", [AroraMobility.electrons(), AroraMobility.holes()])
def test_mobility_falls_monotonically_with_doping(model: AroraMobility) -> None:
    """More scattering centres, less mobility. No exceptions in this model."""
    doping = np.logspace(10, 21, 200)

    assert np.all(np.diff(model(doping)) < 0.0)


# ------------------------------------------------------------- the magnitude


@pytest.mark.parametrize(
    "doping,expected,tolerance",
    [
        (1e16, 1230.0, 0.02),
        (1e18, 280.0, 0.05),
    ],
)
def test_electron_mobility_matches_the_literature(
    doping: float, expected: float, tolerance: float
) -> None:
    """Silicon n-type at two dopings, against the values everyone quotes.

    The four limits above fix the shape and would pass for any parameter set
    with the right form. These fix the numbers.
    """
    assert float(AroraMobility.electrons()(doping)) == pytest.approx(
        expected, rel=tolerance
    )


def test_holes_are_slower_than_electrons_at_every_doping() -> None:
    """True in silicon at every doping, and a sign check on the parameters."""
    doping = np.logspace(13, 20, 60)

    assert np.all(AroraMobility.holes()(doping) < AroraMobility.electrons()(doping))


# --------------------------------------------------------------- temperature


def test_the_parameters_reduce_to_their_tabulated_values_at_300_K() -> None:
    """Every parameter carries a (T/300)^k factor which must vanish at 300."""
    electrons = AroraMobility.electrons(T=C.T_ROOM)

    assert electrons.mu_min == pytest.approx(88.0, rel=1e-14)
    assert electrons.mu_d == pytest.approx(1252.0, rel=1e-14)
    assert electrons.N_ref == pytest.approx(1.432e17, rel=1e-14)
    assert electrons.exponent == pytest.approx(0.88, rel=1e-14)


def test_mobility_falls_as_temperature_rises_in_lightly_doped_silicon() -> None:
    """Phonon scattering dominates there, and it grows with temperature.

    Both mu_min and mu_d carry negative exponents, so this is really a check
    that neither sign was transcribed backwards.
    """
    lightly_doped = 1e14
    cold = float(AroraMobility.electrons(T=250.0)(lightly_doped))
    hot = float(AroraMobility.electrons(T=400.0)(lightly_doped))

    assert hot < cold


# ------------------------------------------------------------ constant model


def test_the_constant_model_ignores_the_doping() -> None:
    """The Phase 1 and 2 behaviour, kept so the seam has one shape."""
    model = ConstantMobility(C.MU_N_300)

    np.testing.assert_allclose(
        model(np.array([0.0, 1e15, 1e20])), C.MU_N_300, rtol=0.0
    )


def test_the_constant_model_returns_one_value_per_node() -> None:
    """A bare scalar would broadcast and then silently collapse an average."""
    got = ConstantMobility(1417.0)(np.zeros(7))

    assert got.shape == (7,)


# ------------------------------------------------- nodes to edges, and units


def test_edge_diffusivity_averages_the_two_endpoint_values() -> None:
    """Mobility is a nodal quantity and the flux needs it on the edge.

    The arithmetic mean of the two endpoints, which is what a Scharfetter-
    Gummel edge wants: the flux derivation assumes the coefficient is constant
    along the edge, so the edge value is the one thing being approximated and
    the mean is the honest choice.
    """
    mobility = np.array([1000.0, 500.0, 100.0])

    got = edge_diffusivity(mobility, V_T=0.02585)

    np.testing.assert_allclose(got, np.array([750.0, 300.0]) * 0.02585, rtol=1e-14)


def test_edge_diffusivity_applies_the_einstein_relation() -> None:
    """D = V_T * mu. Getting this wrong scales every current by 40."""
    got = edge_diffusivity(np.full(2, 1417.0), V_T=0.02585)

    assert float(got[0]) == pytest.approx(1417.0 * 0.02585, rel=1e-14)


def test_edge_diffusivity_gives_one_value_per_edge() -> None:
    got = edge_diffusivity(np.ones(11), V_T=0.02585)

    assert got.shape == (10,)


def test_edge_diffusivity_gathers_the_endpoints_an_edge_list_names() -> None:
    """A 2D mesh has to say which two nodes an edge joins.

    Its edges are not contiguous and a column of it is not a slice, so the
    1D branch would average node 3 with node 4 when the edge in question runs
    from node 3 to node 8. Nothing downstream would notice: the array is the
    right dtype and, on a square mesh, very nearly the right length.
    """
    mobility = np.array([100.0, 200.0, 400.0, 800.0])
    edge_nodes = np.array([[0, 2], [1, 3], [3, 0]], dtype=np.int64)

    got = edge_diffusivity(mobility, V_T=2.0, edge_nodes=edge_nodes)

    np.testing.assert_allclose(got, [500.0, 1000.0, 900.0], rtol=1e-14)


def test_the_edge_list_form_reproduces_the_1d_chain_exactly() -> None:
    """Bit for bit, because the 1D chain is one particular edge list.

    Every current measured in Phases 1 to 3 went through the slice branch, so
    the two have to be the same arithmetic and not merely the same answer.
    """
    mobility = np.linspace(300.0, 1400.0, 9)
    chain = np.column_stack([np.arange(8), np.arange(8) + 1]).astype(np.int64)

    np.testing.assert_array_equal(
        edge_diffusivity(mobility, V_T=0.02585, edge_nodes=chain),
        edge_diffusivity(mobility, V_T=0.02585),
    )


# --------------------------------------- the gap between the two constant sets


def test_the_arora_undoped_limit_does_not_match_the_tabulated_mobility() -> None:
    """Documented rather than reconciled, because both numbers are measured.

    docs/06-constants.md lists mu_n = 1417 for undoped silicon and separately
    gives Arora parameters whose N -> 0 limit is 88 + 1252 = 1340. They
    disagree by 5.4 percent for electrons and 1.8 percent for holes. Neither
    is wrong: they come from different fits to different data, and Arora is
    fitted over the doped range where it is used rather than at the intrinsic
    limit where nothing is ever measured.

    It matters because switching a device from the constant model to Arora
    moves its current by that much even at doping low enough that the model
    should not be doing anything, and that would otherwise read as a bug.
    """
    electrons = AroraMobility.electrons()
    holes = AroraMobility.holes()

    assert float(electrons(0.0)) == pytest.approx(1340.0, rel=1e-12)
    assert float(holes(0.0)) == pytest.approx(461.3, rel=1e-12)

    assert float(electrons(0.0)) / C.MU_N_300 == pytest.approx(0.9456, rel=1e-3)
    assert float(holes(0.0)) / C.MU_P_300 == pytest.approx(0.9815, rel=1e-3)
