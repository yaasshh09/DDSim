---
title: The n-channel MOSFET
summary: Source, drain, gate and channel, what the threshold voltage is, and what drift diffusion leaves out once the gate gets short.
docs: 01-physics.md#MOS gate; 01-physics.md#Where drift-diffusion breaks down
---

## In plain words

A MOSFET is the MOS capacitor with a junction at each end. A p type body has
two heavily doped n type regions implanted into its surface, the source and
the drain. The channel is the stretch between them, under the gate. With the
gate low the channel is p type, and almost no current gets from drain to
source. Raise the gate past the threshold voltage and the surface inverts. A
thin layer of electrons links source to drain, and current flows. It's a
switch with no moving parts.

`L_gate` is the gate length, 1 um by default. The implants reach under the
gate edges by `lateral_diffusion`, 100 nm each by default, so the channel the
electrons actually cross is shorter, 800 nm here. The builder won't accept a
gate shorter than twice that reach, because the two junctions would touch.
Lower `lateral_diffusion` before you shrink the gate a lot. `sd_length` and
`contact_length` set how long the source and drain are and how much of each
the metal contact covers. `x_j` is how deep the implants go.

The `transfer` sweep moves the gate and reads the drain current in A/cm,
which is current per unit of gate width. The drain bias is whatever
`drain_voltage` says, and it defaults to 0 V, so set it first. The device
starts with Fermi-Dirac statistics on (`degenerate`), since its source and
drain peak at 1e20 cm^-3.

## In more depth

The gate is a potential with the work function folded in,
$\psi_{gate} = V_{gate} - \Phi_{MS}$, over the channel span only. The oxide
over the source and drain has no electrode on it, so there's no gate overlap.
Source and drain are ohmic plates on the silicon surface that stop short of
the gate edge. A contact pins $\psi$, $n$ and $p$, and one that reached the
junction would pin the built in potential instead of letting the solve find
it.

Drift diffusion is a local model: a carrier's velocity follows the field
right where it is. That stops being true as the gate shrinks, and
docs/01-physics.md names where.

- Below about 50 nm of channel, carriers cross almost without scattering.
  That's quasi-ballistic transport, and it needs energy balance or Monte
  Carlo. It's why the gate length sweep stops at 50 nm.
- Velocity overshoot, where carriers briefly outrun the saturation velocity
  after a sharp jump in field, can't show up here by construction.
- Quantum confinement in the inversion layer pushes the charge centroid away
  from the interface and makes the oxide look thicker. Fixing that needs a
  Schrodinger-Poisson treatment, and none is applied.
- Tunnelling, through the gate oxide or band to band, is left out.

So the short channel effects this device shows (threshold roll-off, DIBL and
velocity saturation) are what classical transport predicts on this geometry,
within the limits above.

The default mesh is 63 columns by 133 rows, 8379 nodes. The row spacing at
the silicon surface, `h_min_y`, is the one the drain current is most
sensitive to, because the inversion layer is the only structure on the device
a mesh can miss.
