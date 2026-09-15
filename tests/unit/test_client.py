"""Nothing in the frontend computes a physical quantity.

phases/PHASE-7.md states this as an acceptance criterion and gives a way to
check it: grep the client for exp, log and any constant in
docs/06-constants.md, and find nothing. This file is that grep, with one
deliberate relaxation recorded in docs/07-decisions.md.

The relaxation is the log axis. The same phase asks for log scale toggles in
item 7, and a log axis cannot be drawn without a logarithm: mapping a decade
to a row of pixels is arithmetic about a screen and not about silicon. So
Math.log10 is allowed, bounded to the two helpers that do the mapping, and
everything else is not. Math.exp is refused outright, because there is no
reason to exponentiate anything in a client that is handed n and p already.

The constant names are read from ddsim/core/constants.py rather than listed
here, so a constant added there is covered the moment it exists.
"""

from __future__ import annotations

import re

import pytest

from ddsim.api.app import PAGE
from ddsim.core import constants

CLIENT = PAGE.read_text(encoding="utf-8")

SCRIPT = re.search(r"<script>(.*)</script>", CLIENT, re.S)
"""The JavaScript on its own. The CSS above it is full of numbers that are
lengths in pixels, and asking whether 1.45 is a line height or an intrinsic
carrier density is a question about the stylesheet rather than the physics."""


def constant_names() -> list[str]:
    """Every public name in ddsim/core/constants.py."""
    return [
        name
        for name in dir(constants)
        if not name.startswith("_") and name not in {"annotations", "np"}
    ]


def test_the_client_is_one_file_with_its_script_inside_it() -> None:
    """The rest of this file reads that script, so it has to be there. A
    client split across files would need this test pointed at all of them."""
    assert SCRIPT is not None
    assert len(SCRIPT.group(1)) > 1000


@pytest.mark.parametrize(
    "forbidden",
    ["Math.exp", "Math.sinh", "Math.asinh", "Math.tanh", "Math.cbrt"],
)
def test_the_client_never_evaluates_a_physical_function(forbidden) -> None:
    """Boltzmann statistics, the Bernoulli function and the equilibrium
    potential are all one of these away. None of them belongs in a page whose
    job is to draw the numbers it was sent."""
    assert forbidden not in CLIENT


def test_the_client_takes_no_natural_logarithm() -> None:
    """A decade is a screen position and Math.log10 draws one. A natural log
    is a thermal voltage away from a quasi-Fermi level, and there is no axis
    that wants one."""
    assert not re.search(r"Math\.log\b", CLIENT)
    assert not re.search(r"Math\.log2\b", CLIENT)


def test_the_client_uses_the_log_axis_only_where_the_axis_is() -> None:
    """The relaxation, bounded. Two helpers convert between a value and its
    decade and nothing else in the file mentions a logarithm."""
    script = SCRIPT.group(1)

    assert script.count("Math.log10") == 1
    assert script.count("Math.pow") == 1
    assert "const decades = (v) => Math.log10(v);" in script
    assert "const undecades = (v) => Math.pow(10, v);" in script


def test_no_silicon_constant_is_named_in_the_client() -> None:
    """The single source of truth for these is docs/06-constants.md and
    ddsim/core/constants.py. A copy in the client is a second one, and the
    browser would go on plotting after the first one changed."""
    script = SCRIPT.group(1)
    named = [
        name
        for name in constant_names()
        if re.search(rf"\b{re.escape(name)}\b", script)
    ]

    assert named == [], f"the client mentions {named}"


def test_the_client_does_not_convert_between_scaled_and_physical() -> None:
    """Everything it receives is physical, converted once in api/frames.py.
    A client that divided by a thermal voltage would be doing the one thing
    docs/03-architecture.md says never to do outside the Field type."""
    script = SCRIPT.group(1)

    for scaling in ("V_T", "0.0259", "38.7", "kT", "thermal"):
        assert scaling not in script
