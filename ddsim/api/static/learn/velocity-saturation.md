---
title: Velocity saturation
summary: Why carriers stop speeding up once the field gets strong enough, and why that shapes the current of a short MOSFET.
docs: 01-physics.md#Mobility models; 01-physics.md#What emerges, and must not be hardcoded; 06-constants.md#Caughey-Thomas; 06-constants.md#Mobility, undoped silicon at 300 K
---

## In plain words

At low field, a carrier's drift speed grows in step with the field. That
can't go on forever. Past a certain point, a carrier loses energy between
collisions almost as fast as it gains it, and its speed flattens out at a
ceiling called the saturation velocity. For electrons in silicon that's about
a hundred kilometres per second. Double the field after that and the speed
barely moves.

This matters most in short MOSFETs. Put a volt across a channel a hundred
nanometres long and the field is far past the point where carriers saturate.
In a long transistor the drain current grows roughly with the square of the
gate overdrive. Once the carriers are stuck at their ceiling it only grows
roughly in a straight line, and the device delivers less current than the
long channel formula promises.

The `field_dependent` switch turns this on. It's off by default, so a MOSFET
solved with the default models has no velocity saturation in it at all. Turn
it on for any short channel result.

## In more depth

The model is Caughey-Thomas, applied on top of whichever low field mobility
$\mu_0$ you picked:

$$\mu(E) = \frac{\mu_0}{\left(1 + \left(\dfrac{\mu_0 E_\parallel}{v_{sat}}\right)^{\beta}\right)^{1/\beta}}$$

with $\beta = 2$ for electrons and $\beta = 1$ for holes. At 300 K
$v_{sat} = 1.07 \times 10^7$ cm/s for electrons and $8.3 \times 10^6$ cm/s
for holes. At low field the bracket is 1 and $\mu = \mu_0$. At high field the
velocity $\mu E_\parallel$ tends to $v_{sat}$. The crossover sits near
$E = v_{sat}/\mu_0$, about 7.6 kV/cm for electrons at the constant undoped
mobility.

$E_\parallel$ is the field along the current, and the solver takes that
literally: on each mesh edge it's the potential drop across the edge divided
by the edge length. Using the size of the full field vector at a node instead
is a common shortcut, and a wrong one, since edges leaving the same node
don't see the same field. The mobility now depends on the unknown potential,
so its derivative goes into the Newton Jacobian. When surface scattering is
also on, the surface correction is applied first at the nodes, and
Caughey-Thomas afterwards on the edges.

Nothing in this model knows about the device. $\beta$ and $v_{sat}$ are
silicon parameters, identical for a 1 um and a 50 nm transistor, and neither
is tuned to any curve. The short channel behaviour comes out of solving
Poisson and continuity on the geometry with this mobility in place. I check
that by solving the same device with Caughey-Thomas on and off, not by
eyeballing the shape of a curve.
