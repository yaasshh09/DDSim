---
title: The MOS capacitor
summary: Sweep a gate over p-type silicon through accumulation, depletion and inversion, and see why the answer depends on how fast you ask.
claims: the_capacitance_never_exceeds_the_oxide; the_high_frequency_minimum_matches_the_depletion_approximation; only_a_slow_signal_sees_the_inversion_layer
device: {"kind": "mos_cap", "parameters": {}}
sweep: {"kind": "cv", "contact": "gate", "voltages": [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], "settings": {"response": "low_frequency"}}
---
## Steps

### Low frequency

A 10 nm oxide on $10^{16}$ p-type silicon with an n+ poly gate, swept from -2
to 2 V. The response is set to low frequency, which means every carrier has
time to follow the small signal. Press solve.

The curve starts high on the left, falls into a valley, and climbs back up on
the right. Drag the "point" slider to a point in each part and look at the
profile image: where are the holes, and where are the electrons?

### High frequency

set: {"sweep": {"settings": {"response": "high_frequency"}}}

Same device, same biases, but now only the majority carriers, the holes, follow
the signal. Press solve. The low frequency curve stays on the plot as a faded
line to compare against.

## What to look for

- On the far left, a capacitance close to the oxide's own, $C_{ox}$, but never
  above it.
- A valley a little below 0 V.
- On the right, the two curves part: the low frequency one climbs back towards
  $C_{ox}$, the high frequency one stays flat at the bottom.

## What you saw

The gate and the silicon are two plates with the oxide between them, and the
capacitance is how much charge moves in the silicon when the gate moves a
little. Where that charge sits decides everything.

**Accumulation**, on the left. A negative gate pulls holes up against the
oxide. The charge that responds sits right at the interface, so the device is
nearly just the oxide, $C_{ox} = \varepsilon_{ox} / t_{ox}$. Nearly, because
even an accumulation layer has a thickness, and it adds a small capacitor in
series. Two capacitors in series are always smaller than either, which is why
no point ever reaches $C_{ox}$.

**Depletion**, the valley. Past flat band the gate pushes holes away and
leaves bare acceptors behind. The charge that responds is now at the bottom
edge of that depleted layer, further from the gate, and the capacitance falls.

**Inversion**, on the right. Past threshold the surface is so strongly bent
that electrons, the minority carrier, gather in a thin layer at the oxide. If
the signal is slow enough for electrons to be generated and follow it, the
responding charge is back at the interface and the capacitance returns to near
$C_{ox}$: 0.977 of it at 2 V here. If the signal is fast, the electrons cannot
keep up, the depletion edge stops moving because the inversion layer screens
it, and the capacitance stays at its minimum:

$$C_{min} = \frac{C_{ox}}{1 + C_{ox} W_{max} / \varepsilon_{Si}}$$

That formula gives 0.090 $C_{ox}$. The solver gives 0.085, a little below,
because the formula stops the surface bending at exactly $2\phi_F$ and a real
inversion layer only forms a few $V_T$ past that, with the depletion edge a
little deeper.

One thing to say plainly: the high frequency curve here is a model, not a
frequency. The solver does not run at a frequency at all. It computes the
capacitance at zero frequency with the minority carrier either allowed to
respond or held fixed. Real measurements sit between the two, set by how fast
the silicon can generate electrons.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own device.
