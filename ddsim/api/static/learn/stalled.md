---
title: A stalled sweep
summary: A sweep that couldn't reach its next voltage, what the page shows when that happens, and how to read the equation families it names.
docs: 02-numerics.md#Bias continuation; 02-numerics.md#Convergence criteria
---

## In plain words

A sweep stalls when it can't get from one voltage on your list to the next.
The solver tries smaller and smaller steps toward it. When the step has
shrunk to almost nothing and still fails, it stops and keeps what it has.

A stall isn't an error, and the status still reads `done`. The red message
says where it stopped, for example stalled on the way to 1 V, and the curve
holds every voltage it reached before that, so the slider and profile work
for those points. Think of a stall as a measurement. It tells you the bias
where this solver, on this device and mesh, couldn't go any further.

On a transfer curve the message ends with a note in brackets naming the
equation with the biggest residual and the one with the biggest update in the
last rejected attempt. The families are psi (the Poisson equation) and n and
p (the electron and hole continuity equations). Each rejected attempt also
shows up as a dark dot on the residual plot, with a note saying what was done
about it.

A C-V sweep doesn't step, so a point that doesn't converge ends the sweep
right away with did not converge at that voltage, and no family note.

If the residual plot shows solves running out of iterations while still
falling, raise `max_iterations`. If it's flat or climbing, more iterations
won't help.

Here's what the named family usually means. Treat it as a first thing to try,
not a diagnosis. **psi** stuck usually means the potential had to move
further than the solver allows in one iteration, so try a smaller `step`
first. **n** or **p** stuck usually means that carrier's density changes by
many powers of ten across a few mesh cells, at a junction or an inversion
layer the mesh doesn't resolve, so try a smaller `h_min` there or more nodes.
On a device you drew, also check that every doping edge and electrode is
where you meant it.

## In more depth

Continuation tries the next step from the last converged solution. On a
failure the step gets halved, from the size actually attempted, and tried
again from the same starting point. The floor is a thousandth of `step`,
which is 5e-5 V on `iv` and 1e-4 V on `transfer` at the defaults, and it
isn't a knob on the page. Once a halved step would drop below it, the ramp
stops, because a solver that fails at that size has left its basin of
attraction and shrinking further won't bring it back. A ramp also stops after
200 attempts. The in-between bias where it stopped isn't recorded as a point.

A Newton attempt fails when either its residual or its update misses its
threshold. The page doesn't know the thresholds, so the note names the
largest of both. The update on n and p is relative, $|\Delta n|/(n + n_i)$ in
scaled terms, so a carrier far below the intrinsic density carries little
weight. Naming both matters. On one stalled MOSFET every residual family sat
at $10^{-14}$ while the n update was $1.8 \times 10^{-10}$, so pointing at the
largest residual alone would have named the wrong test.

A rejected note reads rejected at a value in V. During the cold ramp that
starts a transfer curve, that value isn't a voltage. It's the fraction of
the applied biases being attempted, between 0 and 1, even though the note
labels it in volts. A cold ramp that never arrives doesn't stall the sweep:
its final solve happens at the full bias, and if that fails the job fails.
