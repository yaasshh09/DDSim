---
title: A stalled sweep
summary: A sweep that could not reach its next voltage, what the page shows when that happens, and how to read the equation families it names.
docs: 02-numerics.md#Bias continuation; 02-numerics.md#Convergence criteria
---

## In plain words

A sweep stalls when it cannot get from one voltage on your list to the next.
The solver tries smaller and smaller steps towards it, and when the step has
shrunk to almost nothing and still fails, it stops and keeps what it has.

A stall is not an error, and the status still reads `done`. The red message
says where it stopped, for example stalled on the way to 1 V, and the curve
holds every voltage it reached before that, so the slider and profile work for
those points. A stall is a measurement: it says the bias at which this solver,
on this device and mesh, could go no further.

On a transfer curve the message ends with a note in brackets naming the
equation with the largest residual and the one with the largest update in the
last rejected attempt. The families are psi, the Poisson equation, and n and p,
the electron and hole continuity equations. Each rejected attempt also shows as
a dark dot on the residual plot, with a note saying what was done about it.

On a C-V sweep there is no stepping, so a point that does not converge ends the
sweep straight away with did not converge at that voltage, and no family note.

If the residual plot shows solves running out of iterations while still
falling, raise `max_iterations`. If it is flat or climbing, more iterations will
not help.

What the named family usually means, which is a first thing to try and not
a diagnosis. **psi** stuck usually means the potential had to move further
than the solver lets it move in one iteration, so a smaller `step` is the
first thing to try. **n** or **p** stuck usually means that carrier's density
changes by many decades across a few mesh cells, at a junction or an
inversion layer the mesh does not resolve, so a smaller `h_min` there or
more nodes comes first. On a device you drew, it is also worth checking that
every doping edge and electrode is where you meant it.

## In more depth

Continuation tries the next step from the last converged solution. On a
failure the step is halved, from the size actually attempted, and tried again
from the same starting point. The floor is a thousandth of `step`, which is
5e-5 V on `iv` and 1e-4 V on `transfer` at the defaults, and it is not a knob on
the page. Once a halved step would fall below it, the ramp stops, because a
solver that fails at that size has left its basin of attraction and shrinking
further will not bring it back. A ramp also stops after 200 attempts. The
intermediate bias where it stopped is not recorded as a point.

A Newton attempt fails when either its residual or its update misses its
threshold, and the page does not know the thresholds, so the note names the
largest of both. The update on n and p is relative, $|\Delta n|/(n + n_i)$ in
scaled terms, so a carrier far below the intrinsic density carries little
weight. Naming both matters: on one stalled MOSFET every residual family sat at
$10^{-14}$ while the n update was $1.8 \times 10^{-10}$, so pointing at the
largest residual alone would have named the wrong test.

A rejected note reads rejected at a value in V. During the cold ramp that
starts a transfer curve that value is not a voltage but the fraction of the
applied biases being attempted, between 0 and 1, even though the note labels it
in volts. A cold ramp that never arrives does not stall the sweep: its final
solve is taken at the full bias, and if that fails the job fails.
