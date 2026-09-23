---
title: Gummel iteration
summary: Solving for the potential, the electrons and the holes one at a time, taking turns. It's what the diode I-V sweep runs.
docs: 02-numerics.md#Gummel iteration (Phase 2); 02-numerics.md#Convergence criteria; 07-decisions.md#Known deviations from reference
---

## In plain words

Three equations describe the device, and each one depends on the others. The
potential depends on where the electrons and holes are, and the electrons and
holes move according to the potential. Gummel iteration deals with that by
taking turns. Guess the carriers and solve for the potential. Hold that
potential and the holes still, and solve for the electrons. Hold the
potential and the electrons, and solve for the holes. Then go round again,
each pass starting from what the last one produced, until a whole pass
barely changes anything. One pass through all three is a cycle.

Each small problem is easy, so Gummel is very sturdy when the three equations
are only loosely tied together, like at low bias. At high forward bias the
injected carriers and the potential push on each other hard. Each turn undoes
part of what the other two just did, and the cycles pile up. The diode I-V
sweep runs Gummel at every bias point. The purple line on the residual plot
tracks its progress, one point per cycle, showing how much the biggest
quantity changed in that cycle. A healthy solve draws a straight line heading
down on the log scale. A line that flattens out more and more means the
coupling is winning.

## In more depth

One cycle is three block solves, in order.

1. Nonlinear Poisson for $\psi$, holding the quasi-Fermi potentials $\phi_n$
   and $\phi_p$ fixed. This is its own Newton solve, with the potential step
   capped at $5 V_T$, and it has to converge on both its update and its
   residual or the cycle stops with a failure. The densities then follow the
   shift, $n \leftarrow n\,e^{\Delta\psi}$ and $p \leftarrow p\,e^{-\Delta\psi}$
   in scaled units, which is what holding the quasi-Fermi potentials fixed
   means.
2. Electron continuity, linear in $n$ once $\psi$ and $p$ are held. That's one
   sparse linear solve, with the contact densities imposed afterwards. The
   matrix is an M-matrix, so a non-positive density can only mean a bug, and
   one gets refused, not clamped.
3. Hole continuity for $p$, the same way.

Each block reports its own update size in scaled units: $\max|\Delta\psi|$ in
thermal voltages for Poisson, and $\max|\Delta n| / (n + n_i)$ and
$\max|\Delta p| / (p + n_i)$ for the carriers. The cycle update is the largest
of the three, never their sum, so one slow block can't hide behind a fast
one. The cycle converges when that update drops below `update_tol`, $10^{-8}$
by default on the diode sweep, within `max_iterations` cycles, 200 by
default. There's no separate residual test on the whole system. The docs
suggest solving continuity in quasi-Fermi potentials. This code solves it in
the densities.

Convergence is linear: the update shrinks by roughly a constant factor per
cycle, which is a straight line on a log plot. At high injection that factor
is the whole story. Measured along 0.05 V continuation steps it climbs from
0.51 at 0.9 V to 0.95 at 1.8 V. On the $10^{16}$ cm$^{-3}$ diode with a 2000
cycle budget, Gummel takes 3 cycles at 0.1 V, 6 at 0.6 V, 46 at 1.0 V, 101 at
1.2 V, 224 at 1.5 V and 466 at 2.0 V. Full Newton takes 4 to 6 steps over the
same range. So Gummel doesn't fail at high bias. It just gets slower without
limit. The 224 cycles at 1.5 V are already past the default budget of 200.
An earlier reading that Gummel stalled at 1.63 V turned out to be that budget
running out, not a divergence. The Gummel path here is 1D only and can't
handle a gate, which is why the MOSFET transfer curve runs Newton instead.
