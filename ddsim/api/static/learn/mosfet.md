---
title: The n-channel MOSFET
summary: Source, drain, gate and channel, what the threshold voltage is, and what drift diffusion leaves out when the gate gets short.
docs: 01-physics.md#MOS gate; 01-physics.md#Where drift-diffusion breaks down
---

## In plain words

The MOSFET is the MOS capacitor with a junction at each end. A p type body has
two heavily doped n type regions implanted into its surface, the source and
the drain. Between them, under the gate, is the channel. With the gate low the
channel is p type, each end is a reverse biased or unbiased junction, and
almost no current flows from drain to source. Raise the gate past the
threshold voltage and the surface inverts: a thin layer of electrons joins the
source to the drain, and current flows.

`L_gate` is the gate length, 1 um by default. The implants reach under the
gate edges by `lateral_diffusion`, 100 nm each by default, so the channel the
electrons actually cross is shorter, 800 nm here. The builder refuses a gate
shorter than twice that reach, because the two junctions would meet, so lower
`lateral_diffusion` before shrinking the gate far. `sd_length` and
`contact_length` set how long the source and drain regions are and how much of
each the metal contact covers. `x_j` is how deep the implants go.

The `transfer` sweep moves the gate and reads the drain current, in A/cm, which
is current per unit gate width. The drain bias is whatever `drain_voltage` is
set to, and its default is 0 V, so set it first. The device starts with
Fermi-Dirac statistics on (`degenerate`), since its source and drain peak at
1e20 cm^-3.

## In more depth

The gate is a potential with the work function folded in,
$\psi_{gate} = V_{gate} - \Phi_{MS}$, over the channel span only. The oxide
over the source and drain has no electrode on it, so there is no gate overlap.
Source and drain are ohmic plates on the silicon surface that stop short of
the gate edge, because a contact pins $\psi$, $n$ and $p$, and one reaching the
junction would pin the built in potential instead of letting the solve find
it.

Drift diffusion is a local model: a carrier's velocity follows the field where
it is right now. That breaks down as the gate shrinks, and docs/01-physics.md
names where.

- Below about 50 nm of channel, carriers cross almost without scattering. That
  is quasi-ballistic transport, and it needs energy balance or Monte Carlo.
  This is why the gate length sweep stops at 50 nm.
- Velocity overshoot, carriers briefly outrunning the saturation velocity after
  a sharp rise in field, is invisible here by construction.
- Quantum confinement in the inversion layer moves the charge centroid away
  from the interface and makes the oxide look thicker. Correcting for it needs
  a Schrodinger-Poisson treatment, and none is applied.
- Tunnelling, through the gate oxide or band to band, is absent.

So the short channel effects this device shows, threshold roll-off, DIBL and
velocity saturation, are what classical transport predicts on this geometry,
with the limits above.

The default mesh is 63 columns by 133 rows, 8379 nodes. The row spacing at the
silicon surface, `h_min_y`, is the one the drain current is most sensitive to,
because the inversion layer is the only structure on the device a mesh can
miss.
