---
title: The C-V sweep
summary: Capacitance against gate voltage, what its shape says about the surface, and how it is computed without differencing two solves.
docs: 02-numerics.md#Small-signal AC (Phase 4, for C-V)
---

## In plain words

Capacitance is how much the charge on the gate changes for a small change in
gate voltage. On a MOS capacitor that number moves with the surface regime. In
accumulation it sits near the oxide capacitance, because the answering charge
is right against the oxide. In depletion it falls, because the answering charge
is now the edge of a depletion region some distance into the silicon. In
inversion what happens depends on how fast the small wiggle is, and that is
the `response` knob.

`low_frequency` lets both carriers follow the wiggle. The electrons in the
inversion layer answer it right at the surface, so the capacitance climbs back
up towards the oxide value. `high_frequency` freezes the minority carrier, the
electrons in p type silicon, because in a real device they cannot be generated
fast enough to keep up. Then the depletion region does all the answering, and
the curve stays low in inversion.

The plot shows capacitance in F/cm^2 against the gate voltage in V. The default
capacitor's flatband is near -0.92 V, so the default voltage list of 0 to
0.5 V only sees depletion. The benchmark against DEVSIM sweeps -2 to 2 V.

## In more depth

docs/02-numerics.md sets out the general small signal solve

$$(J_{dc} + i\omega M)\,x = b, \qquad Y = G + i\omega C, \qquad C = \mathrm{Im}(Y)/\omega$$

For this equilibrium device the mass matrix $M$ is empty. The carriers are
functions of $\psi$ inside Poisson, so there are no $dQ/dt$ terms to put in it,
and what remains is the $\omega \to 0$ limit, which here is exact rather than
approximate.

It is still a derivative and not a difference of two DC solves. Differentiating
the converged system $F(\psi; V) = 0$ with respect to the gate bias gives

$$J\,\frac{d\psi}{dV} = -\frac{\partial F}{\partial V}$$

where $J$ is the Jacobian the DC solve ended on, and the right hand side is
nonzero only on the gate's Dirichlet rows. One sparse linear solve gives
$d\psi/dV$, and pushing it back through the unpinned rows at the gate gives
$dQ/dV$. There is no step size and no truncation error.

`high_frequency` removes one term from that Jacobian before the derivative
solve: the minority carrier's contribution to the diagonal. Minority is read
from the doping, not the local densities, so at an inverted p type surface the
electrons are still the minority carrier. The DC solution is untouched. This is
the standard approximation, a model of the high frequency limit, and it is
stated as one.

Each point also reports the gate charge in C/cm^2, read from the Poisson
residual at the gate nodes before their rows are pinned.
