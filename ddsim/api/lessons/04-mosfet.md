---
title: The MOSFET switching on
summary: Sweep the gate of a 1 um transistor and read its threshold and subthreshold slope straight off the transfer curve.
claims: the_subthreshold_slope_sits_above_the_body_factor_estimate; the_threshold_lands_near_the_textbook_formula
device: {"kind": "nmos", "parameters": {"drain_voltage": 0.05}}
mesh: coarse
sweep: {"kind": "transfer", "contact": "gate", "voltages": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5], "models": {"mobility": "arora", "field_dependent": true, "surface": true}}
---
## Steps

### Turn it on

A 1 um n-channel MOSFET: a $10^{17}$ p-type body, 20 nm of oxide, and n+
source and drain. The drain sits at 50 mV, low enough that the channel acts
like a resistor, and the gate sweeps from 0 to 1.5 V. It's on the coarse
mesh, so this takes around twenty seconds instead of a minute. The mobility
models are the full set: doping dependence, velocity saturation and
scattering off the oxide. Press solve and tick "log" on the curve plot.

Below about 0.7 V the curve is a straight line on the log axis. Between
0.2 V and 0.6 V, count how many millivolts of gate it takes to make the
current ten times bigger.

### Read the threshold

Untick "log". Above about 0.9 V the current climbs in a nearly straight line.
Lay a ruler along its steepest part and see where it hits zero current.
That's the threshold voltage, read the same way a datasheet reads it.

Drag the "point" slider to a point below threshold and one above, and compare
the electron density under the gate in the profile picture.

## What to look for

- On the log axis, a factor of ten in current for roughly every 100 mV of gate
  below threshold. Never less than 60 mV.
- On the linear axis, a line that hits zero current a little above 0.8 V.
- In the profile, an electron layer under the oxide that's missing below
  threshold and thick above it.

## What you saw

Below threshold there's no channel yet. The gate lowers a barrier between the
source and the body, electrons diffuse over it, and just like in a diode the
current grows tenfold for every 60 mV the barrier drops. But the gate doesn't
move the barrier one for one. The surface is tied to the gate through the
oxide capacitance and to the body through the depletion capacitance, so only
a fraction $C_{ox} / (C_{ox} + C_{dep})$ of each gate millivolt reaches it:

$$SS = V_T \ln 10 \left(1 + \frac{C_{dep}}{C_{ox}}\right)$$

With the depletion capacitance taken at the textbook surface potential of
$2\phi_F$ that's 93.9 mV per decade. The solver gives 101. It should land
above the formula, since most of the subthreshold curve happens before the
surface reaches $2\phi_F$, where the depletion layer is thinner and $C_{dep}$
is bigger. At room temperature nothing below 60 mV per decade is possible for
any device that works by carriers climbing a barrier, and the solver respects
that without being told.

Above threshold the electrons at the surface outnumber the holes that were
there. The channel is inverted, and the current grows with the charge in it,
almost in proportion to how far the gate is past threshold. The textbook
threshold is

$$V_T = V_{FB} + 2\phi_F + \frac{Q_{dep}}{C_{ox}}$$

which is 0.818 V for this device. Extrapolated off the solved curve it's
0.847 V. The formula stops at the surface potential where an inversion layer
is only just starting. The ruler reads a line through a layer that's already
carrying current, a few $V_T$ further on. The line isn't perfectly straight
either, because the surface scattering model lowers the mobility as the gate
field rises.

The coarse mesh moves this threshold by 0.7 mV compared with the converged
one. See the note under the device.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own device.
