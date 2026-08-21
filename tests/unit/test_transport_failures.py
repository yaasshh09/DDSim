"""What device/transport.py does when a Gummel block cannot proceed.

These are the paths that run when something has already gone wrong, which is
exactly when a broken error message costs the most. They also pin the contract
that a failure comes back as a result rather than as an exception: above
roughly 0.6 V forward bias, and anywhere continuation is skipped, failing to
converge is the expected outcome and phases/PHASE-2.md asks for it to be
measured rather than fought.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.equilibrium import MAX_PSI_STEP
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    TransportError,
    _check_positive,
    initial_state,
    poisson_block,
    solve_bias,
)

MICRON = 1e-4
"""One micron [cm]."""


def diode():
    """The Phase 2 test diode."""
    return pn_diode(
        Na=1e16,
        Nd=1e16,
        length=12 * MICRON,
        junction=6 * MICRON,
        n_nodes=101,
        h_min=5e-7,
    )


def shifted_state(device, shift: float):
    """A state whose potential is offset from the answer by `shift` [1].

    The densities are left alone, so this is the same device with both
    quasi-Fermi levels moved by the same amount. It is a perfectly consistent
    state, and it is a long way from the one the contacts demand: the potential
    has to travel `shift` to get back, and the Newton step limiter only allows
    5 per iteration.
    """
    state = initial_state(device)
    return replace(
        state,
        psi=Field(
            state.psi.data + shift,
            "V",
            ScalingState.SCALED,
            Location.NODE,
            name="psi",
        ),
    )


def test_a_stalled_poisson_block_raises_with_the_reason() -> None:
    """The block reports which of the three it was, and what Newton said."""
    device = diode()
    far_away = shifted_state(device, 400.0)

    with pytest.raises(TransportError, match="Poisson block did not converge"):
        poisson_block(device)(far_away)


def test_a_stalled_block_carries_the_state_that_failed() -> None:
    """The state at the point of failure is what wants inspecting."""
    device = diode()
    far_away = shifted_state(device, 400.0)

    try:
        poisson_block(device)(far_away)
    except TransportError as failure:
        assert failure.state is far_away
    else:  # pragma: no cover
        pytest.fail("the block should not have converged")


def test_solve_bias_reports_a_stalled_block_rather_than_raising() -> None:
    """A failure is data, not an exception, and it names itself."""
    device = diode()
    solved = solve_bias(device, guess=shifted_state(device, 400.0))

    assert solved.gummel is not None
    assert not solved.gummel.converged
    assert "Poisson" in solved.gummel.message


def test_a_non_positive_density_is_refused_rather_than_clamped() -> None:
    """docs/05-pitfalls.md: find the cause, never clamp.

    The message has to say where and by how much, because the answer to a
    negative density is always to go and look at the node.
    """
    device = diode()
    state = initial_state(device)
    density = np.full(device.mesh.n_nodes, 1.0)
    density[7] = -3.5e-9

    with pytest.raises(TransportError, match="node 7"):
        _check_positive(density, "n", state)


def test_a_positive_density_passes_the_check() -> None:
    device = diode()
    state = initial_state(device)

    _check_positive(np.full(device.mesh.n_nodes, 1e-30), "n", state)


def test_the_refusal_names_the_carrier() -> None:
    device = diode()
    state = initial_state(device)
    density = np.full(device.mesh.n_nodes, 1.0)
    density[0] = 0.0

    with pytest.raises(TransportError, match="^p came out"):
        _check_positive(density, "p", state)


def test_the_state_repr_reports_the_ranges() -> None:
    """Printed the moment anything goes wrong, so it has to be readable."""
    text = repr(initial_state(diode()))

    assert "101 nodes" in text
    assert "psi" in text
    assert "n" in text


def test_the_overflow_guard_is_unreachable_by_arithmetic() -> None:
    """Documents why one branch in poisson_block can never be taken.

    The block refuses a potential update that overflows the Boltzmann
    densities. The step limiter allows 5 V_T per Newton iteration and
    solve_poisson allows 50 of them, so one cycle can move psi by at most 250,
    and exp(250) is 3.7e108. A density would have to start above 1e199 for the
    product to overflow, which no device produces.

    The guard stays because raising either number would make it live again, and
    this is what says so out loud rather than leaving a pragma with no
    reasoning attached to it.
    """
    largest_shift = MAX_PSI_STEP * 50
    assert np.exp(largest_shift) * 1e10 < np.finfo(np.float64).max
