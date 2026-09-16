---
title: Gummel iteration
summary: Solving the potential, the electrons and the holes one at a time in turns, which is what the diode I-V sweep runs.
docs: 02-numerics.md#Gummel iteration (Phase 2); 02-numerics.md#Convergence criteria; 07-decisions.md#Known deviations from reference
---

## In plain words

The device is described by three equations that depend on each other: the
potential depends on where the electrons and holes are, and the electrons and
holes move according to the potential. Gummel iteration handles that by taking
them in turns. Guess the carriers and solve for the potential. Hold that
potential and the holes, and solve for the electrons. Hold the potential and
the electrons, and solve for the holes. Then go round again, each pass
starting from what the last one produced, until a whole pass changes almost
nothing. One pass through all three is a cycle.

Each of those smaller problems is easy, and that makes Gummel very robust when
the three equations are only loosely coupled, as they are at low bias. At high
forward bias the injected carriers and the potential push on each other hard,
each turn undoes part of what the other two just did, and the cycles pile up.
The diode I-V sweep runs Gummel at every bias point. The purple line on the
residual plot is its progress: one point per cycle, showing how much the
largest quantity changed in that cycle. A healthy solve draws a straight line
falling on the log scale. A line that falls ever more slowly is the coupling
winning.

## In more depth

One cycle is three block solves, in order.

1. Nonlinear Poisson for $\psi$, holding the quasi-Fermi potentials $\phi_n$
   and $\phi_p$ fixed. This is a Newton solve of its own, with the
   potential step capped at $5 V_T$, and it must converge on both its update
   and its residual or the cycle stops with a failure. The densities then
   follow the shift: $n \leftarrow n\,e^{\Delta\psi}$ and
   $p \leftarrow p\,e^{-\Delta\psi}$ in scaled units, which is what holding the
   quasi-Fermi potentials fixed means.
2. Electron continuity, linear in $n$ once $\psi$ and $p$ are held. One sparse
   linear solve, with the contact densities imposed afterwards. The matrix is
   an M-matrix, so a non-positive density can only mean a bug, and one is
   refused rather than clamped.
3. Hole continuity for $p$, the same way.

Each block reports its own update size, in scaled units: $\max|\Delta\psi|$ in
thermal voltages for Poisson, and $\max|\Delta n| / (n + n_i)$ and
$\max|\Delta p| / (p + n_i)$ for the carriers. The cycle update is the largest
of the three, never their sum, so one slow block cannot hide behind a fast
one. The cycle converges when that update falls below `update_tol`, $10^{-8}$
by default on the diode sweep, within `max_iterations` cycles, 200 by default.
There is no separate residual test on the whole system. The docs suggest
solving continuity in quasi-Fermi potentials. This code solves it in the
densities.

The convergence is linear: the update shrinks by roughly a constant factor per
cycle, which is a straight line on a log plot. That factor is the whole story
at high injection. Measured along 0.05 V continuation steps it climbs from
0.51 at 0.9 V to 0.95 at 1.8 V. On the $10^{16}$ cm$^{-3}$ diode with a 2000
cycle budget, Gummel takes 3 cycles at 0.1 V, 6 at 0.6 V, 46 at 1.0 V, 101 at
1.2 V, 224 at 1.5 V and 466 at 2.0 V. Full Newton takes 4 to 6 steps over the
same range. So Gummel does not fail at high bias, it degrades without bound.
The 224 cycles at 1.5 V are already past the default budget of 200, and an
earlier reading that Gummel stalled at 1.63 V turned out to be that budget
running out rather than a divergence. The Gummel path in this code is 1D only and
cannot take a gate, which is why the MOSFET transfer curve runs Newton
instead.
