---
title: Forward and reverse bias
summary: Push the junction forward and the current jumps tenfold every 60 mV. Pull it back and the depletion region widens.
claims: forward_current_rises_a_decade_every_60_millivolts; reverse_bias_widens_the_depletion_region_as_the_square_root
device: {"kind": "pn_diode", "parameters": {}}
sweep: {"kind": "iv", "contact": "anode", "voltages": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]}
---
## Steps

### Forward bias

The default diode, swept from 0 to 0.6 V on the anode. Press solve, then tick
"log" on the curve plot.

On a linear axis the current looks like nothing and then a wall. On the log
axis it's a straight line. Count how many millivolts it takes for the current
to grow ten times.

While you're here, drag the Na slider. The diode re-solves as you move it,
and the line shifts.

### Reverse bias

set: {"device": {"length": 4e-4, "junction": 2e-4, "n_nodes": 401}, "sweep": {"voltages": [0.0, -0.5, -1.0, -2.0]}}

The diode is now 4 um long, so the depletion region has room to grow without
hitting a contact. The sweep runs backwards from 0 to -2 V. Press solve, then
drag the "point" slider above the profile plot from the first point to the
last, and watch the stretch where n and p both plunge.

## What to look for

- On the forward curve with the log axis on, ten times more current for every
  60 mV or so of bias.
- In reverse, a depletion region that gets wider as the bias goes more
  negative, but by less each time: -2 V doesn't make it twice as wide as
  -1 V.

## What you saw

Forward bias lowers the barrier the built-in potential put up. The number of
carriers that can climb a barrier goes as $e^{-\text{height}/V_T}$, so every
$V_T \ln 10 \approx 60$ mV you take off the barrier lets ten times as many
across. That's the diode equation:

$$J = J_0 \left(e^{V / (n V_T)} - 1\right)$$

The ideality factor $n$ is how many times 60 mV each factor of ten costs.
There's no diode equation in the solver. The current comes out of the
continuity equations, and read off this curve $n$ sits between 1.01 and 1.04
from 0.1 V to 0.6 V. It's close to 1 because the default carrier lifetimes are
long enough that very little recombines inside the depletion region. That's
the mechanism that would push $n$ toward 2 at low bias. The small rise at the
top is high injection kicking in: the injected carriers aren't small next to
the doping anymore.

Reverse bias raises the barrier instead. A bigger step in potential needs
more uncovered ions to hold it, so the depletion region widens, and the
depletion approximation says by how much:

$$W \propto \sqrt{V_{bi} - V}$$

The square root is why each extra volt buys you less width. Measured on this
lesson's longer diode, the width follows the square root law to within 0.3
percent from 0 to -2 V. The reverse current is tiny and nearly flat, because
it's carried by the few minority carriers that wander into the field, and
raising the barrier doesn't make more of them.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own devices.
