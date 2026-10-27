---
title: Why the gain falls at both ends
summary: Run an npn from a trickle to a flood and watch its current gain climb, flatten out and then collapse, with nothing in the solver that knows what a transistor is.
claims: the_collector_current_rises_a_decade_every_60_millivolts; the_gain_falls_at_both_ends
device: {"kind": "npn", "parameters": {"collector_voltage": 1.0}}
sweep: {"kind": "transfer", "contact": "base", "voltages": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], "models": {"field_dependent": true}}
---
## Steps

### The Gummel plot

A one dimensional npn: a heavily doped emitter on the left, a 0.5 um p base,
and a lightly doped collector, with the collector held at 1 V. The sweep walks
the base from 0.2 V to 1 V and plots the collector current. Press solve and
tick "log" on the curve plot.

### A thinner base

set: {"device": {"base_width": 2.5e-5}}

Same device with half the base width. Press solve and compare it with the
faded curve from the last step.

## What to look for

- On the log axis, a straight line over most of the range, ten times more
  collector current for every 60 mV on the base.
- The line bending over at the top, past about 0.8 V.
- The thinner base giving more collector current at every voltage.

## What you saw

The collector current is electrons that the emitter pushed into the base and
that made it across. How many get pushed in goes as $e^{V_{BE}/V_T}$, so the
collector current rises tenfold every 60 mV, the same as a diode. Read off this
curve the ideality is 1.006 between 0.3 and 0.6 V. Halving the base steepens
the slope the electrons diffuse down, and since the junctions' depletion
regions take the same bite out of a thinner base, the current grows by more
than double: 2.6 times at 0.2 V and 2.3 times at 0.6 V.

The gain is the collector current over the base current, and on this device
it moves with bias. It's 14 at 0.2 V, climbs to a plateau of 35 between 0.45
and 0.55 V, and falls to 2 at 1 V. Both falls come out of the physics without
anything asking for them.

At low current the base current has a second part: electrons and holes
recombining inside the emitter junction's depletion region. That part rises
only as $e^{V_{BE}/2V_T}$, half as steeply as the collector current, so it's
small at high bias but it's the bigger part at low bias. Between 0.2 and 0.3 V
the base current's ideality is 1.22, and by 0.5 V it's back to 1.00.

At high current the electrons in the base start to outnumber its doping. That's
high injection: the holes have to pile up to match them, the voltage that
pushes electrons in stops reaching the junction in full, and the collector
current bends over while the base current keeps rising. The gain falls with
it.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own device.
