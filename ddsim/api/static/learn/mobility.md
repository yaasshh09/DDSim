---
title: Mobility
summary: How easily a carrier moves through the crystal when a field pushes it, and how doping slows it down.
docs: 01-physics.md#Mobility models; 01-physics.md#Einstein relation; 06-constants.md#Mobility, undoped silicon at 300 K; 06-constants.md#Arora model parameters
---

## In plain words

A carrier in silicon never gets far before it bumps into something: a
vibrating atom of the lattice, or a charged dopant ion. Between collisions a
field speeds it up, and each collision throws that speed away. The result is a
steady average drift velocity proportional to the field, and the constant of
proportionality is the mobility. Electrons in silicon are about three times as
mobile as holes.

The mobility knob picks how that number is chosen. `constant` uses one value
for the whole device, the mobility of undoped silicon. `arora` makes it depend
on the local doping: every dopant atom is a charged ion that scatters carriers,
so heavily doped regions are slower, and in a MOSFET source an electron
keeps only about a fifteenth of its undoped mobility. The two further switches,
`field_dependent` and `surface`, correct the chosen value for high fields and for
the oxide interface.

Mobility sets how much current a given field produces, so it is where a device
simulator earns or loses its accuracy. The equations can be solved perfectly and
the current still be wrong by a factor of three if the mobility is wrong.

## In more depth

Mobility enters the drift diffusion currents directly and through the
diffusivity, which in the Boltzmann limit follows from it by the Einstein
relation,

$$D_n = \mu_n V_T, \qquad D_p = \mu_p V_T$$

The `constant` model uses the undoped silicon values at 300 K,
$\mu_n = 1417$ and $\mu_p = 470$ cm$^2$/(V s). It is the default, because
every result recorded before doping dependent mobility existed was taken with
it.

The `arora` model is

$$\mu = \mu_{min} + \frac{\mu_d}{1 + (N/N_{ref})^{A}}$$

with separate parameters for each carrier, at 300 K:

| | $\mu_{min}$ | $\mu_d$ | $N_{ref}$ [cm$^{-3}$] | $A$ |
|---|---|---|---|---|
| electrons | 88 | 1252 | $1.432 \times 10^{17}$ | 0.88 |
| holes | 54.3 | 407 | $2.67 \times 10^{17}$ | 0.88 |

Each parameter carries its own temperature power law. The limits are easy to
read off: $\mu_{min} + \mu_d$ with no doping, $\mu_{min} + \mu_d/2$ at
$N = N_{ref}$, and $\mu_{min}$ at very heavy doping. $N$ here is the doping
magnitude $|N_d - N_a|$, the only doping a device stores.

The undoped limit of Arora is 1340 for electrons and 461.3 for holes, not the
1417 and 470 of the constant model. Both sets are measured, from different
fits, so switching a lightly doped device from `constant` to `arora` moves its
current by about five percent even where doping is doing nothing. That is
expected rather than a bug.

Mobility is evaluated at the nodes, and the diffusivity on each mesh edge is
the arithmetic mean of its two ends, which is the one value per edge the
Scharfetter-Gummel flux asks for. Under the nMOSFET's Fermi-Dirac statistics
the Einstein relation above no longer holds as written. The solver keeps
$D = \mu V_T$ and moves the degeneracy correction into the potential the flux
is fitted in instead, which works out to exactly the generalized Einstein
ratio.
