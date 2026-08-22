"""The Phase 3 acceptance criteria that are about convergence.

phases/PHASE-3.md grades the phase on these:

- Newton converges quadratically, with the characteristic residual drop
- Converges at 1.0 V forward bias, high injection
- Gummel and Newton agree to solver tolerance wherever both converge
- Continuation from 0 to 1 V in under 40 total solves
- No negative carrier densities at any bias

The nine block Jacobian criterion lives in tests/unit/test_coupled.py, which
is where it belongs: it is a statement about derivatives, not about solving.

One criterion is restated. The phase says 1.0 V is "where Gummel failed".
Gummel does not fail there. Phase 2 measured it converging through 1.8 V with
its per cycle rate improving as the bias rises, and that measurement is in
PROGRESS.md. The claim tested here is the one that is true and that actually
motivates the phase: Newton reaches 1.0 V in far fewer solves, and it reaches
it cold, from the Poisson guess, with no continuation at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    TransportModels,
    initial_state,
    solve_bias,
    solve_bias_newton,
)
from ddsim.solve.continuation import continue_to

N_NODES = 201
"""The Phase 2 mesh, so the two solvers are compared on the same device."""


RESIDUAL_FLOOR = 1e-14
"""Below this a row scaled residual is at its own arithmetic floor [1].

The rows are divided by the size of the terms they are built from, so every
residual entry is a relative quantity and machine epsilon is where it stops.
Entries below this are not Newton steps and carry no information about the
rate.
"""


def reduction_factors(residual_history: list[float]) -> list[float]:
    """r_k / r_{k+1} for each step still above the arithmetic floor [1].

    The rate diagnostic, in the form that distinguishes the two cases without
    needing a clean tail. A linearly convergent iteration has a constant
    reduction factor, one over its rate. A quadratic one has a factor that
    grows in proportion to the residual itself, since
    r_{k+1} = C*r_k^2 gives r_k/r_{k+1} = 1/(C*r_k).

    Estimating the exponent p from consecutive triples instead is the textbook
    route and it is too noisy to gate on here. Measured across four biases the
    per triple estimate ranges from -3.8 to 2.5 on solves whose tails are
    plainly quadratic, because the quadratic phase of a well conditioned
    coupled solve is only two or three steps long before it hits the floor,
    and one of those steps usually lands on it.
    """
    usable = [value for value in residual_history if value > RESIDUAL_FLOOR]
    return [usable[k] / usable[k + 1] for k in range(len(usable) - 1)]


# --------------------------------------------------------------- quadratic


def test_newton_converges_quadratically():
    """The characteristic drop, measured rather than eyeballed.

    A wrong Jacobian block still converges, linearly, so the rate is the only
    diagnostic that separates a correct derivative from a plausible one.
    docs/05-pitfalls.md: stagnation looks exactly like ill-conditioning.
    """
    state = solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=0.8))

    assert state.newton is not None
    assert state.newton.converged, state.newton.message

    factors = reduction_factors(state.newton.residual_history)
    history = state.newton.residual_history

    # A linear iteration cannot take four decades out of the residual in one
    # step. Measured at 2.6e4 here, against 2.9 for the first step.
    assert factors[-1] > 1e3, f"final reduction {factors[-1]:.3g} from {history}"

    # And the acceleration is the part that says quadratic rather than merely
    # fast. A linear iteration has a flat factor whatever its rate.
    assert factors[-1] > 100 * factors[0], (
        f"reduction went {factors[0]:.3g} to {factors[-1]:.3g}, "
        f"which is not accelerating, from {history}"
    )


def test_the_residual_reaches_its_arithmetic_floor():
    """Quadratic is only interesting if it arrives somewhere.

    The tail has to bottom out near machine epsilon relative to the row
    scaling, not merely satisfy a loose threshold.
    """
    state = solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=0.8))

    assert state.newton is not None
    assert state.newton.residual_history[-1] < 1e-14


def test_the_last_steps_are_not_limited():
    """A tail that is still being damped is not a quadratic tail.

    limited_steps exists for this: a solve that ends against its cap is
    taking the step the limiter chose, not the step Newton asked for.
    """
    state = solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=0.8))

    assert state.newton is not None
    assert state.newton.limited_steps < state.newton.iterations - 2


# ------------------------------------------------------- the headline result


def test_converges_at_one_volt_forward_bias():
    """The headline criterion of the phase."""
    state = solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=1.0))

    assert state.newton is not None
    assert state.newton.converged, state.newton.message


def test_converges_at_one_volt_from_a_cold_start():
    """Cold, from the Poisson guess, with no continuation.

    docs/05-pitfalls.md says there is no such thing as a good initial guess at
    1 V forward bias and to continue from equilibrium always. That is sound
    advice and it is not a requirement here: the coupled Newton reaches 1 V
    from the Poisson guess in single digit iterations. Continuation still
    earns its place at higher bias and on harder devices, and this is the
    measurement that says how much margin there is.
    """
    device = pn_diode(n_nodes=N_NODES, anode_voltage=1.0)
    state = solve_bias_newton(device, guess=initial_state(device))

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert state.newton.iterations < 15


def test_gummel_needs_far_more_cycles_than_newton_at_one_volt():
    """The comparison the phase is really asking for.

    Not that Gummel fails, because it does not. That it costs many times more
    at the bias where the coupling is strong, and that the gap widens with
    bias, which is the whole argument for the phase.
    """
    device = pn_diode(n_nodes=N_NODES, anode_voltage=1.0)

    gummel = solve_bias(device)
    newton = solve_bias_newton(device)

    assert gummel.gummel is not None and gummel.gummel.converged
    assert newton.newton is not None and newton.newton.converged
    assert newton.newton.iterations * 3 < gummel.gummel.iterations


# ---------------------------------------------------- the two solvers agree


@pytest.mark.parametrize("voltage", [0.0, 0.2, 0.4, 0.6, 0.8])
def test_gummel_and_newton_reach_the_same_solution(voltage):
    """Different algorithms, one set of equations, so one answer.

    Measured the way convergence is measured, absolutely on psi and relative
    on the densities, because an absolute comparison of a density that runs
    from 1e-6 to 1e6 says nothing.
    """
    device = pn_diode(n_nodes=N_NODES, anode_voltage=voltage)

    gummel = solve_bias(device)
    newton = solve_bias_newton(device)

    assert gummel.gummel is not None and gummel.gummel.converged
    assert newton.newton is not None and newton.newton.converged

    assert np.max(np.abs(gummel.psi.data - newton.psi.data)) < 1e-7
    assert (
        np.max(
            np.abs(gummel.n.data - newton.n.data) / (np.abs(gummel.n.data) + 1.0)
        )
        < 1e-7
    )
    assert (
        np.max(
            np.abs(gummel.p.data - newton.p.data) / (np.abs(gummel.p.data) + 1.0)
        )
        < 1e-7
    )


# ------------------------------------------------------------- continuation


def test_continuation_reaches_one_volt_inside_the_budget():
    """Under 40 total solves, per phases/PHASE-3.md. Measured at 6."""
    base = pn_diode(n_nodes=N_NODES)
    models = TransportModels.for_device(base)

    def solve(voltage, previous):
        state = solve_bias_newton(
            base.with_bias(anode=voltage, cathode=0.0),
            models=models,
            guess=previous,
        )
        assert state.newton is not None
        return state if state.newton.converged else None

    result = continue_to(
        solve,
        start=0.0,
        target=1.0,
        initial=initial_state(base),
        step=0.05,
    )

    assert result.converged, result.message
    assert len(result.events) < 40
    assert result.parameter == pytest.approx(1.0)


def test_continuation_never_has_to_retry_a_step():
    """Every attempt converges, so the ramp only ever grows.

    A retry is not a failure of the phase, and the continuation driver exists
    to absorb them. Recorded because it is the honest measure of how much
    margin the coupled solve has: the step grows by 1.5 each time and still
    never overshoots the basin between 0 and 1 V.
    """
    base = pn_diode(n_nodes=N_NODES)
    models = TransportModels.for_device(base)

    def solve(voltage, previous):
        state = solve_bias_newton(
            base.with_bias(anode=voltage, cathode=0.0),
            models=models,
            guess=previous,
        )
        assert state.newton is not None
        return state if state.newton.converged else None

    result = continue_to(
        solve, start=0.0, target=1.0, initial=initial_state(base), step=0.05
    )

    assert len(result.accepted) == len(result.events)


# --------------------------------------------------------------- positivity


@pytest.mark.parametrize("voltage", [-2.0, -0.5, 0.0, 0.3, 0.6, 0.9, 1.0])
def test_no_carrier_density_is_negative_at_any_bias(voltage):
    """phases/PHASE-3.md, and nothing here clamps to achieve it.

    The coupled matrix is not an M-matrix, unlike the two continuity matrices
    Gummel solves, so positivity is not structural here the way it is there.
    It comes from the potential update being damped and the solve staying
    inside the basin. If this ever fails the answer is a smaller max_psi_step
    or a sign error, not a clamp. docs/05-pitfalls.md.
    """
    state = solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=voltage))

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)
