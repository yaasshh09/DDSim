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
    CaugheyThomas,
    ConstantMobility,
    LombardiSurface,
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


# ------------------------------------------------------------ Caughey-Thomas
#
# The field dependent model, and the one that produces velocity saturation.
#
#     mu(E) = mu_0 / (1 + (mu_0 |E| / v_sat)^beta)^(1/beta)
#
# It lives on edges rather than on nodes, because the field it wants is the
# component along the current direction and on a box integration mesh that is
# the potential drop across an edge divided by its length. The full field
# magnitude is the common and wrong shortcut, and it is wrong by more the more
# a mesh is graded, since a graded mesh has edges of wildly different lengths
# meeting at a node.
#
# Everything below is unit free. The model is given a low field diffusivity, a
# saturation velocity and a potential drop in one consistent system, and the
# transport layer passes scaled ones. That is why these tests can use 1.0 for
# the low field value and read the answers off directly.


def edges(value: float, count: int = 5):
    """A low field diffusivity on `count` edges."""
    return np.full(count, value)


def test_at_zero_field_the_model_returns_the_low_field_value() -> None:
    """Exactly, not nearly. A model that shaved a fraction off at zero field
    would move every result taken before it existed."""
    model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)
    h = np.full(5, 0.1)

    np.testing.assert_array_equal(model(np.zeros(5), h), edges(1.0))


def test_the_drift_velocity_saturates_at_v_sat() -> None:
    """The whole point of the model. mu falls exactly fast enough that mu*E
    approaches a constant, and that constant is v_sat."""
    v_sat = 0.5
    model = CaugheyThomas(low_field=edges(1.0), v_sat=v_sat, beta=2.0)
    h = np.full(5, 0.1)

    X = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    velocity = model(X, h) * np.abs(X) / h

    assert velocity[-1] == pytest.approx(v_sat, rel=1e-6)
    assert np.all(np.diff(velocity) > 0.0)


def test_the_drift_velocity_never_exceeds_v_sat() -> None:
    """At any field at all, which is a stronger statement than the limit."""
    v_sat = 0.5
    for beta in (1.0, 2.0):
        model = CaugheyThomas(low_field=edges(1.0), v_sat=v_sat, beta=beta)
        h = np.full(5, 0.1)
        X = np.array([0.0, 1e-3, 1.0, 1e3, 1e9])

        assert np.all(model(X, h) * np.abs(X) / h <= v_sat)


def test_the_knee_is_where_the_low_field_drift_would_reach_v_sat() -> None:
    """The one point of the curve that is closed form: at mu_0 E = v_sat the
    bracket is exactly 2, so mu is mu_0 over the beta-th root of 2."""
    for beta in (1.0, 2.0):
        model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=beta)
        h = np.full(5, 0.1)
        X = np.full(5, 0.5 * 0.1)  # low_field * |X| / h == v_sat

        np.testing.assert_allclose(
            model(X, h), 1.0 / 2.0 ** (1.0 / beta), rtol=1e-14
        )


def test_mobility_falls_monotonically_with_field() -> None:
    for beta in (1.0, 2.0):
        model = CaugheyThomas(low_field=edges(1.0, 60), v_sat=0.5, beta=beta)
        h = np.full(60, 0.1)
        X = np.linspace(0.0, 30.0, 60)

        assert np.all(np.diff(model(X, h)) < 0.0)


def test_the_model_does_not_care_which_way_the_field_points() -> None:
    """A carrier slows down in a strong field whichever direction it runs, so
    only the magnitude of the potential drop enters."""
    model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)
    h = np.full(5, 0.1)
    X = np.array([0.1, 1.0, 5.0, 20.0, 100.0])

    np.testing.assert_array_equal(model(X, h), model(-X, h))


def test_electrons_hold_their_mobility_longer_than_holes_do() -> None:
    """beta = 2 against beta = 1 at the same v_sat. Below the knee the larger
    exponent keeps the bracket nearer to 1, so the electron curve stays flat
    and then turns while the hole curve starts falling straight away."""
    h = np.full(5, 0.1)
    X = np.full(5, 0.1 * 0.5 * 0.1)  # a tenth of the way to the knee
    electrons = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)
    holes = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=1.0)

    assert np.all(electrons(X, h) > holes(X, h))


def test_a_longer_edge_across_the_same_drop_is_a_weaker_field() -> None:
    """The field is the drop divided by the length, so the edge length is not
    decoration. Dropping it would make the model depend on how the mesh was
    graded rather than on the physics."""
    model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)
    X = np.full(5, 1.0)

    assert np.all(model(X, np.full(5, 1.0)) > model(X, np.full(5, 0.1)))


# ------------------------------------------ the tangent the Jacobian needs


def complex_step_dD_dX(model, X, h, step: float = 1e-30):
    """dD/dX by complex step, exact to machine precision.

    The technique phases/PHASE-3.md makes a permanent CI requirement for the
    coupled Jacobian, applied here to one term of it on its own.
    """
    return np.imag(model(X + 1j * step, h)) / step


def test_the_tangent_matches_a_complex_step_for_electrons() -> None:
    model = CaugheyThomas(low_field=edges(1.3, 7), v_sat=0.5, beta=2.0)
    h = np.linspace(0.05, 0.4, 7)
    X = np.array([-30.0, -5.0, -0.5, 0.2, 3.0, 12.0, 200.0])

    np.testing.assert_allclose(
        model.derivative(X, h), complex_step_dD_dX(model, X, h), rtol=1e-12
    )


def test_the_tangent_matches_a_complex_step_for_holes() -> None:
    """beta = 1, where the model has an absolute value in it and the tangent
    on the negative side is the one that would be easiest to get wrong."""
    model = CaugheyThomas(low_field=edges(0.4, 7), v_sat=0.3, beta=1.0)
    h = np.linspace(0.05, 0.4, 7)
    X = np.array([-30.0, -5.0, -0.5, 0.2, 3.0, 12.0, 200.0])

    np.testing.assert_allclose(
        model.derivative(X, h), complex_step_dD_dX(model, X, h), rtol=1e-12
    )


def test_the_tangent_is_negative_wherever_the_field_is_positive() -> None:
    """More field, less mobility. A sign is what a Jacobian gets wrong
    silently: Newton still converges with the wrong sign on a small term, just
    slowly, and to the same answer."""
    model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)
    h = np.full(5, 0.1)
    X = np.array([0.1, 1.0, 5.0, 20.0, 100.0])

    assert np.all(model.derivative(X, h) < 0.0)
    assert np.all(model.derivative(-X, h) > 0.0)


def test_the_tangent_is_zero_at_zero_field() -> None:
    """Zero from both sides for electrons, and by choice for holes.

    beta = 2 is smooth at the origin and its derivative there really is zero.
    beta = 1 is not: the model has |E| in it, so the two one sided derivatives
    are equal and opposite and there is no two sided one. Zero is the value
    halfway between them, and it is the right choice for a Jacobian, because
    either one sided value would claim the mobility falls when the potential
    is raised and rises when it is lowered, which is half of the truth stated
    as the whole of it.
    """
    for beta in (1.0, 2.0):
        model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=beta)

        np.testing.assert_array_equal(
            model.derivative(np.zeros(5), np.full(5, 0.1)), np.zeros(5)
        )


def test_a_complex_step_at_zero_field_takes_the_right_hand_side() -> None:
    """Which is where the Jacobian and its verification part company, and the
    only place they do.

    A complex step approaches the origin along the imaginary axis, and the
    square root of a square resolves that onto the positive branch, so it
    reports the derivative from the right. For electrons that is zero and
    agrees. For holes it is the one sided value, against zero in the Jacobian.
    Both are defensible readings of a point where the model has a corner, and
    the disagreement exists on a set of measure zero: a device with a hole
    current has a field somewhere, and an edge sitting at exactly 0.0 has no
    current across it to get wrong.
    """
    holes = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=1.0)
    electrons = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)
    h = np.full(5, 0.1)

    from_the_right = complex_step_dD_dX(holes, np.zeros(5), h)

    assert np.all(from_the_right < 0.0)
    np.testing.assert_allclose(
        from_the_right, holes.derivative(np.full(5, 1e-8), h), rtol=1e-6
    )
    np.testing.assert_allclose(
        complex_step_dD_dX(electrons, np.zeros(5), h), np.zeros(5), atol=1e-30
    )


def test_a_complex_argument_survives_the_model() -> None:
    """The block verification differentiates the coupled residual by complex
    step, so everything the residual calls has to stay analytic. abs() of a
    complex number is not, which is why the magnitude is a square root of a
    square instead."""
    model = CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=2.0)

    out = model(np.full(5, 2.0) + 1j * 1e-30, np.full(5, 0.1))

    assert np.iscomplexobj(out)
    np.testing.assert_allclose(
        np.real(out), model(np.full(5, 2.0), np.full(5, 0.1)), rtol=1e-15
    )


# ------------------------------------------------------------------ refusals


def test_a_zero_saturation_velocity_is_refused() -> None:
    """It divides the field, and a carrier that cannot move at all is not a
    slow carrier, it is a different model."""
    with pytest.raises(ValueError, match="v_sat"):
        CaugheyThomas(low_field=edges(1.0), v_sat=0.0, beta=2.0)


def test_a_non_positive_beta_is_refused() -> None:
    with pytest.raises(ValueError, match="beta"):
        CaugheyThomas(low_field=edges(1.0), v_sat=0.5, beta=0.0)


# ================================================== Lombardi surface, Phase 5
#
# docs/01-physics.md names this model and states what it is worth: "Without
# this your inversion-layer mobility is too high by a factor of 2 to 3 and
# your Id is correspondingly wrong." That factor is the acceptance test, and
# it is asserted below at a channel condition rather than assumed.
#
# The parameters are not in docs/06-constants.md. They are the enhanced
# Lombardi, or Darwish, set that DEVSIM ships in its Klaassen.py, which is the
# reference this project regresses against in tier 4. A test pins every one of
# them, so that a silent edit to a fit parameter is not something a reader has
# to notice by eye.


DEVSIM_ELECTRONS = {
    "B": 3.61e7,
    "C_ac": 1.70e4,
    "tau": 0.0233,
    "delta": 3.58e18,
    "A": 2.58,
    "alpha": 6.85e-21,
    "eta": 0.0767,
    "kappa": 1.7,
}

DEVSIM_HOLES = {
    "B": 1.51e7,
    "C_ac": 4.18e3,
    "tau": 0.0119,
    "delta": 4.10e15,
    "A": 2.18,
    "alpha": 7.82e-21,
    "eta": 0.123,
    "kappa": 0.9,
}


@pytest.mark.parametrize(
    ("build", "table"),
    [
        (LombardiSurface.electrons, DEVSIM_ELECTRONS),
        (LombardiSurface.holes, DEVSIM_HOLES),
    ],
    ids=["electrons", "holes"],
)
def test_the_parameters_are_the_devsim_ones(build, table) -> None:
    """Pins the provenance. Every one of these came out of the reference's own
    source, and a fit parameter that drifts is not visible in any result until
    a DEVSIM regression fails for a reason nobody can locate."""
    model = build(T=C.T_ROOM)
    for name, value in table.items():
        assert getattr(model, name) == value


# ------------------------------------------------------- the low field limit


@pytest.mark.parametrize(
    "build", [LombardiSurface.electrons, LombardiSurface.holes], ids=["n", "p"]
)
def test_a_vanishing_normal_field_gives_back_the_bulk_mobility(build) -> None:
    """Far from the interface there is no normal field, so there must be no
    surface scattering. This is what lets the model be applied over a whole
    region instead of inside a layer whose thickness somebody has to choose.

    Both surface terms diverge as E_perp falls, so their reciprocals vanish
    and Matthiessen leaves mu_bulk alone. Not exactly, because the floor stops
    E_perp at 1e2 V/cm, so the assertion is a percent rather than an equality.
    """
    model = build()
    mu_bulk = np.full(4, 800.0)

    mu = model(
        mu_bulk,
        E_perp=np.zeros(4),
        total_doping=np.full(4, 1e17),
        carriers=np.full(4, 1e10),
    )

    np.testing.assert_allclose(mu, mu_bulk, rtol=0.02)
    assert np.all(mu < mu_bulk), "scattering can only ever subtract"


def test_the_normal_field_is_floored_rather_than_dividing_by_zero() -> None:
    """Both terms divide by E_perp, so an unfloored zero is an inf in mu_ac
    and a nan the moment it meets the reciprocal sum. DEVSIM floors it at
    1e2 V/cm and the floor is copied rather than invented."""
    model = LombardiSurface.electrons()
    mu_bulk = np.full(3, 800.0)
    doping, carriers = np.full(3, 1e17), np.full(3, 1e10)

    at_zero = model(mu_bulk, np.zeros(3), doping, carriers)
    at_floor = model(mu_bulk, np.full(3, model.E_floor), doping, carriers)
    below = model(mu_bulk, np.full(3, 1.0), doping, carriers)

    assert np.all(np.isfinite(at_zero))
    np.testing.assert_allclose(at_zero, at_floor, rtol=1e-14)
    np.testing.assert_allclose(below, at_floor, rtol=1e-14)


# --------------------------------------------------------------- Matthiessen


@pytest.mark.parametrize(
    "build", [LombardiSurface.electrons, LombardiSurface.holes], ids=["n", "p"]
)
def test_the_three_channels_combine_by_reciprocals(build) -> None:
    """1/mu = 1/mu_bulk + 1/mu_ac + 1/mu_sr, which is the whole content of the
    model. Asserted against the components the model reports separately, so a
    sign or an association error in the combining line has somewhere to show."""
    model = build()
    mu_bulk = np.full(5, 700.0)
    E = np.array([1e3, 1e4, 1e5, 3e5, 1e6])
    doping, carriers = np.full(5, 3e17), np.full(5, 5e17)

    mu = model(mu_bulk, E, doping, carriers)
    mu_ac = model.acoustic(E, doping)
    mu_sr = model.roughness(E, doping, carriers)

    expected = 1.0 / (1.0 / mu_bulk + 1.0 / mu_ac + 1.0 / mu_sr)
    np.testing.assert_allclose(mu, expected, rtol=1e-14)
    assert np.all(mu < np.minimum(mu_bulk, np.minimum(mu_ac, mu_sr)))


@pytest.mark.parametrize(
    "build", [LombardiSurface.electrons, LombardiSurface.holes], ids=["n", "p"]
)
def test_mobility_falls_as_the_normal_field_rises(build) -> None:
    """More field pulls the carrier harder against the interface, so it
    scatters off it more. Monotone with no turning point anywhere in the range
    a MOSFET occupies."""
    model = build()
    E = np.logspace(2.0, 6.5, 60)
    mu = model(np.full(60, 800.0), E, np.full(60, 1e17), np.full(60, 1e18))

    assert np.all(np.diff(mu) < 0.0)


# --------------------------------------------------- what the model is worth


def test_the_inversion_layer_is_two_to_three_times_slower_than_bulk() -> None:
    """The acceptance test for the whole model, and the reason Phase 5 calls
    it not optional.

    docs/01-physics.md: "Without this your inversion-layer mobility is too
    high by a factor of 2 to 3 and your Id is correspondingly wrong." The
    condition is a real one: a 1e17 channel under a normal field of 5e5 V/cm,
    which is what a 1 um NMOS sees at a volt of overdrive through 10 nm of
    oxide, carrying the surface density an inversion layer holds.

    The bulk mobility is Arora's own answer at that doping rather than a round
    number, so the ratio is between two things the code computes.
    """
    channel_doping = 1e17
    mu_bulk = float(AroraMobility.electrons()(channel_doping))

    mu = LombardiSurface.electrons()(
        np.full(1, mu_bulk),
        E_perp=np.full(1, 5e5),
        total_doping=np.full(1, channel_doping),
        carriers=np.full(1, 1e18),
    )

    ratio = mu_bulk / float(mu[0])
    assert 2.0 < ratio < 3.0, f"surface mobility is {ratio:.2f}x below bulk"


def test_holes_stay_slower_than_electrons_at_the_surface() -> None:
    """True in the bulk and it had better survive the surface terms, which
    carry their own separate parameter set and could reorder them."""
    E, doping, carriers = np.full(4, 3e5), np.full(4, 1e17), np.full(4, 1e18)

    mu_n = LombardiSurface.electrons()(np.full(4, 800.0), E, doping, carriers)
    mu_p = LombardiSurface.holes()(np.full(4, 300.0), E, doping, carriers)

    assert np.all(mu_p < mu_n)


# ---------------------------------------------- the density dependent exponent


def test_a_heavier_inversion_layer_roughens_the_surface_it_sees() -> None:
    """The exponent gamma is what makes this the enhanced model rather than
    the 1988 one, and it is the only place a carrier density enters. More
    carriers in the layer means a larger exponent, so mu_sr falls faster with
    field. With gamma a constant this test cannot pass."""
    model = LombardiSurface.electrons()
    E, doping = np.full(3, 5e5), np.full(3, 1e17)

    light = model.roughness(E, doping, carriers=np.full(3, 1e14))
    heavy = model.roughness(E, doping, carriers=np.full(3, 1e20))

    assert np.all(
        model.gamma(doping, np.full(3, 1e20)) > model.gamma(doping, np.full(3, 1e14))
    )
    assert np.all(heavy < light)


def test_the_exponent_reduces_to_A_with_no_carriers_present() -> None:
    """gamma = A + alpha*(n + p)*N^(-eta), so an empty band leaves A. That is
    the 1988 Lombardi exponent, and the two models agree there."""
    model = LombardiSurface.holes()
    gamma = model.gamma(np.full(2, 1e17), np.zeros(2))

    np.testing.assert_allclose(gamma, model.A, rtol=1e-15)


# --------------------------------------------------------------- temperature


def test_the_acoustic_term_carries_the_temperature_exponent() -> None:
    """mu_ac divides its second term by (T/300)^kappa, so a hotter lattice
    scatters more. kappa differs between the carriers, 1.7 against 0.9, which
    is why it is a parameter and not a shared constant."""
    cold = LombardiSurface.electrons(T=250.0)
    hot = LombardiSurface.electrons(T=350.0)
    E, doping = np.full(3, 3e5), np.full(3, 1e17)

    assert np.all(hot.acoustic(E, doping) < cold.acoustic(E, doping))


# ------------------------------------------------------------------ refusals


def test_a_negative_normal_field_is_refused() -> None:
    """E_perp is a magnitude by construction, so a negative one means the
    caller took a signed difference and forgot the absolute value. Quietly
    taking abs() here would hide that: the exponents would accept it and the
    mobility would come out looking plausible."""
    with pytest.raises(ValueError, match="E_perp"):
        LombardiSurface.electrons()(
            np.full(3, 800.0),
            np.array([1e5, -1e5, 1e5]),
            np.full(3, 1e17),
            np.full(3, 1e18),
        )
