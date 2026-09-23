---
title: Mobility
summary: How easily a carrier moves through the crystal when a field pushes it, and how doping slows it down.
docs: 01-physics.md#Mobility models; 01-physics.md#Einstein relation; 06-constants.md#Mobility, undoped silicon at 300 K; 06-constants.md#Arora model parameters
---

## In plain words

A carrier in silicon never gets far before it bumps into something, either a
vibrating atom or a charged dopant ion. Between bumps the field speeds it up,
and each bump throws that speed away. What you get is a steady average drift
speed that's proportional to the field. The constant in front is the
mobility. Electrons in silicon are about three times as mobile as holes.

The mobility knob picks how that number gets chosen. `constant` uses one
value for the whole device, the mobility of undoped silicon. `arora` makes it
depend on the local doping. Every dopant atom is a charged ion that knocks
carriers around, so heavily doped regions are slower. In a MOSFET source an
electron keeps only about a fifteenth of its undoped mobility. Two more
switches, `field_dependent` and `surface`, correct the chosen value for high
fields and for the oxide interface.

Mobility sets how much current a given field produces, so it's where a device
simulator earns or loses its accuracy. You can solve the equations perfectly
and still be off by a factor of three if the mobility is wrong.

## In more depth

Mobility enters the drift diffusion currents directly, and also through the
diffusivity. In the Boltzmann limit that follows from the Einstein relation,

$$D_n = \mu_n V_T, \qquad D_p = \mu_p V_T$$

The `constant` model uses undoped silicon at 300 K, $\mu_n = 1417$ and
$\mu_p = 470$ cm$^2$/(V s). It's the default, because every result recorded
before doping dependent mobility existed was taken with it.

The `arora` model is

$$\mu = \mu_{min} + \frac{\mu_d}{1 + (N/N_{ref})^{A}}$$

with separate parameters for each carrier, at 300 K:

| | $\mu_{min}$ | $\mu_d$ | $N_{ref}$ [cm$^{-3}$] | $A$ |
|---|---|---|---|---|
| electrons | 88 | 1252 | $1.432 \times 10^{17}$ | 0.88 |
| holes | 54.3 | 407 | $2.67 \times 10^{17}$ | 0.88 |

Each parameter has its own temperature power law. The limits are easy to read
off: $\mu_{min} + \mu_d$ with no doping, $\mu_{min} + \mu_d/2$ at
$N = N_{ref}$, and $\mu_{min}$ at very heavy doping. $N$ here is the doping
magnitude $|N_d - N_a|$, the only doping a device stores.

Arora's undoped limit is 1340 for electrons and 461.3 for holes, not the 1417
and 470 of the constant model. Both sets are measured, from different fits,
so switching a lightly doped device from `constant` to `arora` shifts its
current by about five percent even where doping isn't doing anything. That's
expected, not a bug.

Mobility is evaluated at the nodes. The diffusivity on each mesh edge is the
average of its two ends, which is the one value per edge the
Scharfetter-Gummel flux asks for. Under the nMOSFET's Fermi-Dirac statistics
the Einstein relation above no longer holds as written. The solver keeps
$D = \mu V_T$ and moves the degeneracy correction into the potential the flux
is fitted in instead, which works out to exactly the generalized Einstein
ratio.
