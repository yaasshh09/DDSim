---
title: The three devices
summary: A PN diode, a MOS capacitor and an n-channel MOSFET, what each one is for, and why none of their behaviour is fitted.
docs: 01-physics.md#What emerges, and must not be hardcoded
---

## In plain words

The device menu has three structures. Each one is built from lengths and
doping levels and nothing else.

`pn_diode` is a one dimensional bar of silicon, p type on the left and n type
on the right. It's where the basics live: the built in potential, the
depletion region, and a current that shoots up exponentially under forward
bias. Its natural sweep is `iv`.

`mos_cap` is a metal gate on a thin oxide on p type silicon, drawn in two
dimensions. No current gets through the oxide, so what it teaches is
electrostatics: how a gate voltage first pushes holes away from the surface
and then pulls electrons in. Its natural sweep is `cv`.

`nmos` is an n-channel MOSFET. It's a p type body with heavily doped n type
source and drain regions and a gate over the channel between them. It puts
the other two ideas together, a junction at each end and a gate in the
middle. Its natural sweep is `transfer`, drain current against gate voltage.

Each device's knobs come straight from the Python function that builds it,
defaults and all, so the form can't drift away from what the solver builds.

## In more depth

The list in docs/01-physics.md is the contract. The built in potential, the
depletion width, the diode ideality factor and its crossover from 2 to 1, the
threshold voltage and body effect, the subthreshold slope, DIBL, threshold
roll-off with gate length, and velocity saturation are all outputs. None of
them is typed in anywhere. If one ever showed up as a fitted parameter, that
would be a bug in the project, not a shortcut.

The MOSFET builder sticks to this in how it's drawn. The metallurgical
channel is shorter than `L_gate` because the source and drain implants spread
under the gate edge by `lateral_diffusion`. That spread lives in the doping
profile itself, so the solver sees the shorter channel without being told.
There's no halo implant, no lightly doped drain and no fixed interface
charge. Each of those exists to move a threshold or soften a field, which
would mean tuning the very answer the device is supposed to produce.

One warning. The models panel starts with constant mobility, and with field
dependent mobility and surface scattering both off. Those are the defaults of
the function that builds the models, and every result recorded before Phase 5
was taken that way. A MOSFET solved on those defaults has no velocity
saturation in it. To watch short channel behaviour come out of the physics,
switch on `field_dependent` and `surface` and pick `arora` mobility.
