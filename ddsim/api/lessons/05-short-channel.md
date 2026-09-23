---
title: Short channel effects
summary: Shrink the gate from 1 um to 50 nm on one process and watch the threshold fall and the drain start to open the channel, with no term anywhere that knows the channel is short.
claims: the_threshold_rolls_off_as_the_gate_shortens; the_drain_lowers_the_barrier_only_on_a_short_channel
device: {"kind": "nmos", "parameters": {"substrate_doping": -1e18, "sd_peak": 1e20, "x_j": 2.5e-6, "lateral_diffusion": 1e-6, "t_ox": 2e-7, "sd_length": 4e-5, "contact_length": 2e-5, "t_si": 1e-4, "L_gate": 1e-4, "drain_voltage": 0.05}}
mesh: coarse
mesh_note: This is a coarse mesh. I measured it on 2026-09-18 on this lesson's process against the converged mesh: each constant current threshold moves by 2 to 3 mV and the DIBL by under 1 mV/V, at 1 um, 100 nm and 50 nm alike, and a curve takes 15 to 20 s instead of 47 to 74 s.
sweep: {"kind": "transfer", "contact": "gate", "voltages": [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], "models": {"mobility": "arora", "field_dependent": true, "surface": true}}
---
## Steps

### A long channel

set: {"device": {"L_gate": 1e-4, "drain_voltage": 0.05}}

One process, built for short gates: 2 nm of oxide on a $10^{18}$ body, with
source and drain 25 nm deep. It starts out drawn at 1 um with the drain at
50 mV. Press solve and tick "log" on the curve plot. Each of these solves
takes fifteen to twenty seconds on the coarse mesh.

### Raise the drain

set: {"device": {"drain_voltage": 1.0}}

Same device, drain at 1 V. Press solve. On the log axis, compare the bottom
of this curve with the faded one from the last step.

### Shorten the gate to 100 nm

set: {"device": {"L_gate": 1e-5}}

Nothing else changes: same oxide, same doping, same junctions. The drain is
back at 50 mV. Press solve.

### Raise the drain again

set: {"device": {"L_gate": 1e-5, "drain_voltage": 1.0}}

The 100 nm device with 1 V on the drain. Press solve and compare with the
last step.

### Shorten it to 50 nm

set: {"device": {"L_gate": 5e-6}}

Drain at 50 mV. Press solve.

### And raise the drain

set: {"device": {"L_gate": 5e-6, "drain_voltage": 1.0}}

The 50 nm device at 1 V. Press solve. Hit "clear runs" whenever the plot gets
too busy. Every run is labelled with what changed.

## What to look for

- At 1 um, the 50 mV and 1 V curves almost on top of each other below
  threshold.
- At 100 nm, and even more at 50 nm, the 1 V curve shifted clearly left of
  the 50 mV one. The drain is helping to switch the device on.
- Each shorter device leaving the bottom of the plot at a lower gate voltage.
  A shorter channel also carries more current per unit width for the same
  charge (ten times more for a tenfold shorter gate), so read the shift
  sideways, not up.

## What you saw

In a long transistor the gate controls the barrier the source electrons have
to climb, and the source and drain junctions only matter at the two ends.
Their depletion regions reach a few tens of nanometres into the channel,
which is nothing next to a micron. At 50 nm those two depletion regions cover
a big chunk of the channel. The charge under the gate is now shared: the
junctions hold some of the body's depletion charge instead of the gate, so
the gate needs less voltage to invert what's left. That's threshold
**roll-off**. Measured here at a current of $100\,\text{nA} \times W/L$, the
threshold falls from 275 mV at 1 um to 226 mV at 100 nm and 93 mV at 50 nm.

Raise the drain on a long device and the barrier doesn't budge, because the
drain is too far away to reach it. On a short one the drain's field reaches
back to the source end of the channel and pulls the barrier down itself.
That's **drain induced barrier lowering**, or DIBL, reported as how far the
threshold moves per volt of drain: 9.1 mV/V at 1 um, 30.8 at 100 nm, 126 at
50 nm. Most of the 1 um figure isn't a barrier moving at all. At 50 mV of
drain the subthreshold current is only $1 - e^{-2}$ of what it is once the
drain is several $V_T$ up, and that alone reads as about 5 mV/V.

Neither effect is written anywhere in the solver. It solves Poisson in two
dimensions on the real geometry, and the answer comes out different when the
two junctions are close together. That's all there is to it. Against the
converged mesh, this lesson's coarse mesh moves each threshold by 2 to 3 mV
and the DIBL at every length by under 1 mV/V.

**What this solver leaves out at 50 nm**, and it matters:

- **Quantum confinement.** In a real inversion layer the electrons sit in
  discrete energy levels in a well a few nanometres wide, with their charge
  peaking a little below the oxide instead of right at it. That raises the
  threshold and makes the oxide look thicker. The solver treats electrons as
  a classical gas, so its inversion layer sits hard against the interface.
- **Velocity overshoot.** Drift-diffusion assumes each carrier's speed is set
  by the field right where it is. Below about 50 nm, carriers cross the
  channel before they settle to that speed, and real devices carry more
  current than this model can. That's why nothing shorter than 50 nm is
  offered here.
- **Gate tunnelling.** Real 2 nm oxide leaks by quantum tunnelling. Here the
  oxide is a perfect insulator.
- **Discrete dopants.** A 50 nm channel holds only a few hundred dopant atoms,
  and exactly where each one sits changes the threshold from one transistor to
  the next. Here the doping is a smooth profile, so every device is the
  average one.

The trends are real physics, and they come straight out of the equations. The
last few tens of millivolts at 50 nm aren't something to trust from this
solver.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own device.
