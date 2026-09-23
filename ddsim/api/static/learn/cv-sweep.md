---
title: The C-V sweep
summary: Capacitance against gate voltage, what its shape says about the surface, and how it's computed without subtracting two solves.
docs: 02-numerics.md#Small-signal AC (Phase 4, for C-V)
---

## In plain words

Capacitance is how much the charge on the gate changes when you nudge the
gate voltage a tiny bit. On a MOS capacitor that number changes with the
surface regime. In accumulation it sits near the oxide capacitance, because
the charge answering the gate is right up against the oxide. In depletion it
drops, because now the answering charge is the edge of a depletion region
some distance into the silicon. In inversion, what happens depends on how
fast the nudge is, and that's the `response` knob.

`low_frequency` lets both carriers keep up with the nudge. The electrons in
the inversion layer answer it right at the surface, so the capacitance climbs
back up toward the oxide value. `high_frequency` freezes the minority carrier
(electrons, in p type silicon), because in a real device they can't be
generated fast enough to keep up. Then the depletion region does all the
answering and the curve stays low in inversion.

The plot shows capacitance in F/cm^2 against gate voltage in V. The default
capacitor's flatband is near -0.92 V, so the default voltage list from 0 to
0.5 V only sees depletion. The DEVSIM benchmark sweeps -2 to 2 V.

## In more depth

docs/02-numerics.md sets out the general small signal solve

$$(J_{dc} + i\omega M)\,x = b, \qquad Y = G + i\omega C, \qquad C = \mathrm{Im}(Y)/\omega$$

For this equilibrium device the mass matrix $M$ is empty. The carriers are
functions of $\psi$ inside Poisson, so there are no $dQ/dt$ terms to put in
it. What's left is the $\omega \to 0$ limit, which here is exact, not an
approximation.

It's still a derivative, not a difference of two DC solves. Differentiate the
converged system $F(\psi; V) = 0$ with respect to the gate bias and you get

$$J\,\frac{d\psi}{dV} = -\frac{\partial F}{\partial V}$$

where $J$ is the Jacobian the DC solve ended on. The right hand side is
nonzero only on the gate's Dirichlet rows. One sparse linear solve gives
$d\psi/dV$, and pushing it back through the unpinned rows at the gate gives
$dQ/dV$. No step size, no truncation error.

`high_frequency` removes one term from that Jacobian before the derivative
solve: the minority carrier's contribution to the diagonal. Minority is read
from the doping, not the local densities, so at an inverted p type surface
the electrons still count as the minority carrier. The DC solution isn't
touched. This is the standard approximation, a model of the high frequency
limit, and I label it as one.

Each point also reports the gate charge in C/cm^2, read from the Poisson
residual at the gate nodes before their rows get pinned.
