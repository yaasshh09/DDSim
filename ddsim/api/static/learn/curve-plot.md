---
title: The curve plot
summary: Reading the I-V, transfer or C-V curve, what the log toggle does to negative values, and why every dot is a real solve.
docs: 02-numerics.md#Bias continuation
---

## In plain words

The curve plot is the measurement you ran the sweep for. The horizontal axis
is the voltage on the swept contact, in V. The vertical axis is the current
at the measured contact for `iv` and `transfer`, or the capacitance for `cv`.
The axes have numbers but no unit labels, so here they are: a diode current
is a current density in A/cm^2, a MOSFET drain current is per unit gate width
in A/cm, and a capacitance is in F/cm^2.

Each green dot is one voltage from your list, drawn the moment it's solved,
with a straight line back to the dot before it. The note under the plot counts
the points and says where the last one was. When the job finishes on its
own, the plot gets redrawn from the finished curve on the server, which has
the final say on what the sweep found.

The `log` box switches the vertical axis to a log scale. That's how you read a
diode or a subthreshold MOSFET current, since it spans many powers of ten. A
log axis can't show zero or a negative number, so those points just get left
out. A reverse biased diode current is negative, because current flows out of
the anode, so in log view the reverse half of the curve disappears instead of
getting flipped.

Points are joined in the order you listed the voltages, so an unsorted list
zigzags.

Every finished run stays on the plot as a faded blue line until you press
`clear runs`. Each one is labelled with the settings that differ from the run
after it. So `Na 2e+17` means that curve was solved with Na at 2e17 and the
next one wasn't. The label compares what the page sent, not the physics, and
a faded line is exactly the numbers that run gave. Nothing gets solved again,
so an old curve never shifts when you move a knob. A run you cancelled, or
one a moving slider replaced, never finished, so it isn't kept.

## In more depth

Every dot is a converged solve at exactly that voltage. Nothing on this plot
is interpolated. The line between two dots is just a line the canvas draws,
and no value along it was computed. If you want the shape between two points,
add more voltages to the list.

Between requested voltages the solver walks by continuation, in steps no
bigger than the `step` knob, halving any step that fails. Those in-between
solves are how it gets there, and they aren't recorded as points. On `cv`
there's no walk: each point is an independent solve.

A current is positive when conventional current flows from the contact into
the device, so a forward biased anode and an on NMOS drain both read
positive. It's read from the continuity residual at the contact nodes, not
from the edge flux next to them, which is why the terminal currents add up to
zero. Near zero current that residual is a difference of large flux terms. So
a current far below the flux sizes, like a MOSFET drain deep in its off
state, carries a floor of arithmetic noise. docs/07-decisions.md records one
near 2e-9 A/cm that can even read negative.
