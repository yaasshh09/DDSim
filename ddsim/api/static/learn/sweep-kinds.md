---
title: The three sweeps
summary: iv, transfer and cv, which device each one is for, and which solver each one runs.
docs: 02-numerics.md#Bias continuation
---

## In plain words

A sweep is a list of voltages applied one after another to a single contact,
with something measured at each one. The voltage box takes the list, and the
contact box names the terminal it goes to.

`iv` is the diode's current against voltage curve. The current gets read at
the same contact that's swept. It only runs on a device where every contact
touches silicon, so it's meant for the diode, and the page won't run it on a
device with a gate.

`transfer` is the MOSFET's curve. The gate gets swept and the current is read
at a different terminal, the one in the `measure at` box, which is the drain
unless you pick something else. It needs a drain bias set on the device to
mean anything.

`cv` is the MOS capacitor's curve, capacitance against gate voltage. Every
point is an equilibrium solve with no current anywhere, so the models panel
is hidden for it and a request that carries models gets turned down.

When you change the device, the contact box resets to `anode` for the diode
and `gate` for the others.

## In more depth

Each sweep runs a different solver, and the knobs under the sweep belong to
that solver.

**iv** runs Gummel iteration: Poisson, then electron continuity, then hole
continuity, cycled until the update drops below `update_tol` (1e-8 by
default) within `max_iterations` (200 cycles by default). Every requested
voltage is reached by continuation from the one before, in steps of at most
`step`, 0.05 V by default. The first point, at `start`, is a Gummel solve from
the equilibrium guess, not a ramp. The residual plot shows each cycle's
update in purple.

**transfer** runs full Newton on the coupled system, all three equations at
once, with a budget of 30 iterations per solve by default and a continuation
step of 0.1 V. Newton here can handle a gate, which the Gummel blocks can't.
The first point is cold, so it's ramped in as a fraction of every applied
bias before the sweep starts. The residual plot shows one line per equation
family.

**cv** runs Newton on Poisson alone at equilibrium, 50 iterations per point
by default. It doesn't continue at all. Every point starts from the charge
neutral guess, so one bias that fails ends the sweep right there without
spoiling the points already found. Each point then takes one extra linear
solve for the capacitance.

A curve is always either complete or marked as stopped early, with the
points it reached.
