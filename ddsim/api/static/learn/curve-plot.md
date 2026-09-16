---
title: The curve plot
summary: Reading the I-V, transfer or C-V curve, what the log toggle does to negative values, and why every dot is a real solve.
docs: 02-numerics.md#Bias continuation
---

## In plain words

The curve plot is the measurement the sweep was run for. The horizontal axis
is the voltage on the swept contact, in V. The vertical axis is the current at
the measured contact for `iv` and `transfer`, or the capacitance for `cv`. The
axes carry numbers but no unit labels, so here they are: a diode current is a
current density in A/cm^2, a MOSFET drain current is per unit gate width in
A/cm, and a capacitance is F/cm^2.

Each green dot is one voltage from your list, drawn as soon as it is solved,
with a straight line to the dot before it. The note under the plot counts the
points and says where the last one was. When the job finishes on its own, the plot is
redrawn from the finished curve fetched from the server, which is the
authority on what the sweep found.

The `log` box puts the vertical axis on a log scale, which is how a diode or a
subthreshold MOSFET current is read, since it spans many decades. A log axis
cannot show zero or a negative number, so those points are simply left out.
A reverse biased diode current is negative, because current flows out of the
anode, so in log view the reverse half of the curve disappears rather than
being flipped.

Points are joined in the order you listed the voltages, so a list that is not
in order zigzags.

## In more depth

Every dot is a converged solve at exactly that voltage. Nothing on this plot is
interpolated: the line between two dots is just a line the canvas draws, and
no value along it was computed. If you want the shape between two points,
put more voltages in the list.

Between requested voltages the solver walks by continuation, in steps no larger
than the `step` knob, halving a step that fails. Those intermediate solves are
how it gets there, and they are not recorded as points. On `cv` there is no
walk: each point is an independent solve.

A current is positive when conventional current flows from the contact into
the device, so a forward biased anode and an on NMOS drain both read positive.
It is read from the continuity residual at the contact nodes rather than from
the edge flux beside them, which is why the terminal currents sum to zero.
Near zero current that residual is a difference of large flux terms, so a
current far below the flux sizes, like a MOSFET drain deep in its off state,
carries a floor of arithmetic noise, and docs/07-decisions.md records one near
2e-9 A/cm that can even read negative.
